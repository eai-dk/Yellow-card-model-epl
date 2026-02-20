#!/usr/bin/env python3
"""
Compute Yellow Card picks with odds from SportMonks and store in DB.
This runs every 10 minutes via EventBridge + ECS Scheduled Task.
"""

import os
import requests
import unicodedata
import json
import psycopg2
import psycopg2.extras
from datetime import datetime, timedelta
from difflib import SequenceMatcher

# Configuration
DATABASE_URL = os.environ.get("DATABASE_URL", "")
SPORTMONKS_TOKEN = os.environ.get("SPORTMONKS_TOKEN", "fd9XKsnh82xRG52vayu1ZZ1nbK8kdOk3s5Ex3ss7U2NV7MDejezJr3FNLFef")
API_FOOTBALL_KEY = os.environ.get("API_FOOTBALL_KEY", "0b8d12ae574703056b109de918c240ef")

# SportMonks Market ID for Yellow Cards
YC_MARKET_ID = 64  # Player to be booked

# Edge thresholds
VALUE_EDGE_THRESHOLD = 0.03  # 3%
LOCK_PROBABILITY_THRESHOLD = 0.50  # 50%

# Player name mapping for matching
PLAYER_NAME_MAPPING = {
    "edward nketiah": "eddie nketiah", "norberto murara neto": "neto",
    "jorgen strand larsen": "jorgen larsen", "luis guilherme": "luis guilherme lira",
    "florentino luis": "luis florentino", "jamie gittens": "jamie bynoe-gittens",
    "omari hutchinson": "omari giraud-hutchinson", "wilfried gnonto": "degnand gnonto",
    "yehor yarmolyuk": "yegor yarmolyuk", "chido obi-martin": "chidozie obi-martin",
    "tim iroegbunam": "timothy iroegbunam", "pape matar sarr": "pape sarr",
    "randal kolo muani": "randal muani", "david moller wolfe": "david wolfe",
    "yunus konak": "yunus emre konak", "valentin castellanos": "taty castellanos",
    "rayan ait-nouri": "rayan ait nouri", "amadou diallo": "amad diallo",
    "joao pedro": "joao pedro junqueira", "bruno guimaraes": "bruno guimaraes rodriguez",
    "lucas paqueta": "lucas paqueta bezerra", "darwin nunez": "darwin nunez ribeiro",
    "kiernan dewsbury-hall": "kiernan dewsbury hall",
}

TEAM_MAPPING = {
    "Brighton & Hove Albion": "Brighton", "Brighton and Hove Albion": "Brighton",
    "Newcastle United": "Newcastle", "Wolverhampton Wanderers": "Wolves",
    "Wolverhampton": "Wolves", "Nottingham Forest": "Nottingham Forest",
    "Sheffield United": "Sheffield Utd", "Leicester City": "Leicester",
    "Ipswich Town": "Ipswich", "West Ham United": "West Ham",
    "Leeds United": "Leeds", "Tottenham Hotspur": "Tottenham",
}


def normalize_name(name: str) -> str:
    """Normalize player name for matching"""
    if not name:
        return ""
    name = unicodedata.normalize("NFKD", name).encode("ASCII", "ignore").decode("ASCII")
    name = name.lower().strip()
    for remove in ["jr.", "jr", "ii", "iii", "."]:
        name = name.replace(remove, "")
    return " ".join(name.split())


def get_canonical_name(name: str) -> str:
    """Get canonical name applying mappings"""
    norm = normalize_name(name)
    return PLAYER_NAME_MAPPING.get(norm, norm)


TEAM_ALIASES = {
    "afc bournemouth": "bournemouth", "brighton & hove albion": "brighton",
    "wolverhampton wanderers": "wolves", "nottingham forest": "nott'm forest",
    "tottenham hotspur": "tottenham", "west ham united": "west ham",
    "newcastle united": "newcastle", "manchester united": "man united",
    "manchester city": "man city", "leeds united": "leeds",
}


def normalize_team_name(name):
    n = name.lower().strip()
    return TEAM_ALIASES.get(n, n)


def get_db_conn():
    return psycopg2.connect(DATABASE_URL)


def fetch_sportmonks_lineups():
    """Fetch confirmed lineups from SportMonks for upcoming EPL fixtures."""
    today = datetime.now().strftime("%Y-%m-%d")
    end_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
    url = f"https://api.sportmonks.com/v3/football/fixtures/between/{today}/{end_date}"
    params = {
        "api_token": SPORTMONKS_TOKEN,
        "filters": "fixtureLeagues:8",
        "include": "lineups.player;participants",
        "per_page": 50,
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        return resp.json().get("data", [])
    except Exception as e:
        print(f"Error fetching lineups: {e}")
        return []


def build_lineup_sets(fixtures):
    """
    Build confirmed starting XI sets per fixture. Returns {(home_norm, away_norm): set(names)}.
    Only uses lineups for fixtures within 90 minutes of kickoff (lineups are announced ~75 min
    before KO). For fixtures further out, SportMonks returns predicted lineups which are
    unreliable — those are skipped so all players pass through.
    """
    lineup_map = {}
    now = datetime.now()
    for fixture in fixtures:
        name = fixture.get("name", "")
        parts = name.split(" vs ")
        if len(parts) != 2:
            continue
        home_norm = normalize_team_name(parts[0].strip())
        away_norm = normalize_team_name(parts[1].strip())
        lineups = fixture.get("lineups", [])
        if not lineups:
            continue
        # Only trust lineups if fixture is within 90 minutes of kickoff
        starting_at = fixture.get("starting_at", "")
        if starting_at:
            try:
                ko_time = datetime.fromisoformat(starting_at.replace("Z", "+00:00"))
                ko_naive = ko_time.replace(tzinfo=None)
                minutes_until = (ko_naive - now).total_seconds() / 60
                if minutes_until > 90:
                    continue  # Too far from kickoff — lineups are predicted, not confirmed
            except (ValueError, TypeError):
                continue
        starters = set()
        for entry in lineups:
            if entry.get("type_id") != 11:
                continue
            player = entry.get("player", {}) or {}
            display_name = player.get("display_name", "")
            if display_name:
                starters.add(normalize_name(display_name))
        if starters:
            lineup_map[(home_norm, away_norm)] = starters
            print(f"  Lineup confirmed for {parts[0].strip()} vs {parts[1].strip()}: {len(starters)} starters")
    return lineup_map


def is_player_in_lineup(player_name, team, opponent, lineup_map):
    """Check if player is in confirmed starting XI. Returns True if in lineup OR no lineup available."""
    team_norm = normalize_team_name(team) if team else ""
    opp_norm = normalize_team_name(opponent) if opponent else ""

    starters = None
    for key in lineup_map:
        home_norm, away_norm = key
        team_is_home = team_norm and (team_norm in home_norm or home_norm in team_norm)
        team_is_away = team_norm and (team_norm in away_norm or away_norm in team_norm)
        opp_is_home = opp_norm and (opp_norm in home_norm or home_norm in opp_norm)
        opp_is_away = opp_norm and (opp_norm in away_norm or away_norm in opp_norm)
        if (team_is_home and opp_is_away) or (team_is_away and opp_is_home):
            starters = lineup_map[key]
            break
        if not opp_norm and (team_is_home or team_is_away):
            starters = lineup_map[key]
            break

    if starters is None:
        return True  # No lineup available — let all players through

    player_norm = normalize_name(player_name)
    player_canonical = get_canonical_name(player_name)
    if player_norm in starters or player_canonical in starters:
        return True

    # Fuzzy match on last name
    player_parts = player_norm.split()
    if len(player_parts) >= 2:
        last_name = player_parts[-1]
        for starter in starters:
            starter_parts = starter.split()
            if len(starter_parts) >= 2 and starter_parts[-1] == last_name:
                score = SequenceMatcher(None, player_norm, starter).ratio()
                if score > 0.65:
                    return True

    return False


def fetch_predictions():
    """Fetch YC predictions from DB"""
    today = datetime.now().strftime("%Y-%m-%d")
    end_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
    try:
        conn = get_db_conn()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM yc_predictions WHERE fixture_date >= %s AND fixture_date <= %s ORDER BY model_probability DESC",
                (today, end_date),
            )
            rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        print(f"Error fetching predictions: {e}")
        return []


def fetch_fixtures():
    """Fetch EPL fixtures from API-Football"""
    today = datetime.now().strftime("%Y-%m-%d")
    end_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
    
    url = "https://v3.football.api-sports.io/fixtures"
    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    params = {"league": 39, "season": 2025, "from": today, "to": end_date}
    
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    if resp.status_code != 200:
        print(f"Error fetching fixtures: {resp.status_code}")
        return []
    
    return resp.json().get("response", [])


def fetch_sportmonks_odds(fixture_ids):
    """Fetch Yellow Card odds from SportMonks"""
    all_odds = {}
    
    for fixture_id in fixture_ids:
        url = f"https://api.sportmonks.com/v3/football/odds/pre-match/fixtures/{fixture_id}"
        params = {"api_token": SPORTMONKS_TOKEN, "include": "odds"}
        
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code != 200:
                continue
            
            data = resp.json().get("data", [])
            for market in data:
                if market.get("market_id") == YC_MARKET_ID:
                    odds = market.get("odds", [])
                    for odd in odds:
                        player_name = odd.get("name", "")
                        odd_value = odd.get("value")
                        if player_name and odd_value:
                            norm = get_canonical_name(player_name)
                            all_odds[norm] = {
                                "odds": float(odd_value),
                                "player_raw": player_name
                            }
        except Exception as e:
            print(f"Error fetching odds for fixture {fixture_id}: {e}")
    
    return all_odds


def compute_picks_with_odds():
    """Main function to compute picks with odds"""
    print(f"[{datetime.now().isoformat()}] Starting YC picks computation...")
    
    # Fetch predictions
    predictions = fetch_predictions()
    print(f"Fetched {len(predictions)} predictions")

    if not predictions:
        print("No predictions found")
        return

    # Fetch fixtures from API-Football
    fixtures = fetch_fixtures()
    print(f"Fetched {len(fixtures)} fixtures")

    # Fetch confirmed lineups from SportMonks
    sm_fixtures = fetch_sportmonks_lineups()
    lineup_map = build_lineup_sets(sm_fixtures)
    print(f"Lineup data available for {len(lineup_map)} fixtures")

    picks = []
    lineup_filtered = 0
    for pred in predictions:
        player_name = pred.get("player_name", "")
        team = pred.get("team", "")
        opponent = pred.get("opponent", "")

        # Filter by confirmed lineups
        if not is_player_in_lineup(player_name, team, opponent, lineup_map):
            lineup_filtered += 1
            continue

        prob = float(pred.get("model_probability", 0) or 0)
        odds = float(pred.get("odds", 0) or 0)

        if prob <= 0:
            continue
        
        # Calculate edge if odds available
        edge = 0
        implied = 0
        tier = "PICK"
        
        if odds and odds > 1:
            implied = 1 / odds
            edge = prob - implied
            
            # Classify tier
            if edge >= VALUE_EDGE_THRESHOLD and odds >= 1.5:
                tier = "VALUE"
            elif prob >= LOCK_PROBABILITY_THRESHOLD and odds >= 1.1:
                tier = "LOCK"
        
        pick = {
            "fixture_date": pred.get("fixture_date"),
            "player_name": player_name,
            "team": team,
            "opponent": pred.get("opponent", ""),
            "position": pred.get("position", ""),
            "fixture": pred.get("fixture", ""),
            "referee": pred.get("referee", ""),
            "prob": round(prob, 4),
            "odds": round(odds, 2) if odds else None,
            "implied": round(implied, 4) if implied else None,
            "edge": round(edge, 4) if edge else None,
            "tier": tier,
            "computed_at": datetime.now().isoformat()
        }
        picks.append(pick)
    
    if lineup_filtered > 0:
        print(f"Lineup filter: removed {lineup_filtered} players not in confirmed starting XI")
    print(f"Computed {len(picks)} picks")

    # Store picks
    store_picks(picks)


def store_picks(picks):
    """Store computed picks in DB via psycopg2"""
    if not picks:
        return
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM computed_yc_picks WHERE fixture_date >= %s", (today,))
            if picks:
                cols = list(picks[0].keys())
                vals_template = ",".join(["%s"] * len(cols))
                sql = f"INSERT INTO computed_yc_picks ({','.join(cols)}) VALUES ({vals_template})"
                rows = [tuple(p.get(c) if not isinstance(p.get(c), (list, dict)) else json.dumps(p.get(c)) for c in cols) for p in picks]
                psycopg2.extras.execute_batch(cur, sql, rows)
            conn.commit()
        conn.close()
        print(f"Successfully stored {len(picks)} picks")
        value_count = len([p for p in picks if p["tier"] == "VALUE"])
        lock_count = len([p for p in picks if p["tier"] == "LOCK"])
        print(f"  VALUE picks: {value_count}")
        print(f"  LOCK picks: {lock_count}")
    except Exception as e:
        print(f"Error storing picks: {e}")


if __name__ == "__main__":
    if not DATABASE_URL:
        print("ERROR: DATABASE_URL not set")
        exit(1)
    compute_picks_with_odds()