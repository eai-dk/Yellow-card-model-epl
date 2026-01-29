#!/usr/bin/env python3
"""
Compute Yellow Card picks with odds from SportMonks and store in Supabase.
This runs every 10 minutes via GitHub Actions to keep picks fresh.
"""

import os
import requests
import unicodedata
import json
from datetime import datetime, timedelta

# Configuration
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://kijtxzvbvhgswpahmvua.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
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


def get_supabase_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }


def fetch_predictions():
    """Fetch YC predictions from Supabase"""
    today = datetime.now().strftime("%Y-%m-%d")
    end_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
    
    url = f"{SUPABASE_URL}/rest/v1/yc_predictions"
    params = f"?fixture_date=gte.{today}&fixture_date=lte.{end_date}&order=model_probability.desc"
    
    resp = requests.get(url + params, headers=get_supabase_headers(), timeout=30)
    if resp.status_code != 200:
        print(f"Error fetching predictions: {resp.status_code} - {resp.text[:200]}")
        return []
    
    return resp.json()


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
    print(f"Fetched {len(predictions)} predictions from Supabase")
    
    if not predictions:
        print("No predictions found")
        return
    
    # Fetch fixtures
    fixtures = fetch_fixtures()
    print(f"Fetched {len(fixtures)} fixtures")
    
    # Get SportMonks fixture IDs (need to map from API-Football)
    # For now, use the odds already in predictions if available
    
    picks = []
    for pred in predictions:
        player_name = pred.get("player_name", "")
        team = pred.get("team", "")
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
    
    print(f"Computed {len(picks)} picks")
    
    # Store in Supabase
    store_picks(picks)


def store_picks(picks):
    """Store computed picks in Supabase"""
    if not picks:
        return
    
    # Delete old picks first
    today = datetime.now().strftime("%Y-%m-%d")
    delete_url = f"{SUPABASE_URL}/rest/v1/computed_yc_picks?fixture_date=gte.{today}"
    requests.delete(delete_url, headers=get_supabase_headers(), timeout=30)
    
    # Insert new picks
    insert_url = f"{SUPABASE_URL}/rest/v1/computed_yc_picks"
    resp = requests.post(insert_url, headers=get_supabase_headers(), json=picks, timeout=30)
    
    if resp.status_code in [200, 201]:
        print(f"Successfully stored {len(picks)} picks")
        value_count = len([p for p in picks if p["tier"] == "VALUE"])
        lock_count = len([p for p in picks if p["tier"] == "LOCK"])
        print(f"  VALUE picks: {value_count}")
        print(f"  LOCK picks: {lock_count}")
    else:
        print(f"Error storing picks: {resp.status_code} - {resp.text[:200]}")


if __name__ == "__main__":
    if not SUPABASE_KEY:
        print("ERROR: SUPABASE_KEY not set")
        exit(1)
    compute_picks_with_odds()