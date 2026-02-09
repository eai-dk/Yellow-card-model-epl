#!/usr/bin/env python3
"""
Yellow Card Weekend Picks with Edge Calculation
Uses shared_features.FeatureEngine for Supabase-backed feature engineering
and the YC v6 CalibratedClassifierCV model for probability estimation.
Fetches real odds from SportMonks Market 64 (Player to be booked).
Only shows VALUE picks where edge > threshold.
"""

import os
import sys
import pickle
import requests
import numpy as np
import pandas as pd
import unicodedata
from datetime import datetime, timedelta

# Add parent directory to path for shared_features
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared_features import FeatureEngine
from shared_features.constants import YC_V7_FEATURES
from shared_features.ensemble import EnsembleYCModel  # needed for v10 pkl

# API Keys - use env vars if available, fallback to hardcoded
API_FOOTBALL_KEY = os.environ.get("API_FOOTBALL_KEY", "0b8d12ae574703056b109de918c240ef")
SPORTMONKS_TOKEN = os.environ.get("SPORTMONKS_TOKEN", "fd9XKsnh82xRG52vayu1ZZ1nbK8kdOk3s5Ex3ss7U2NV7MDejezJr3FNLFef")

# Minimum model probability to include in picks (filters out goalkeepers/non-starters)
MIN_PROB = 0.08  # 8% — above base rate

# Player name mapping for edge cases (our data -> SportMonks)
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
    "jota silva": "jota", "diogo jota": "jota",
    "tolu arokodare": "toluwalase arokodare",
    "kiernan dewsbury-hall": "kiernan dewsbury hall",
    "joao pedro": "joao pedro junqueira",
    "bruno guimaraes": "bruno guimaraes rodriguez",
    "lucas paqueta": "lucas paqueta bezerra",
    "darwin nunez": "darwin nunez ribeiro",
}

def normalize_player_name(name: str) -> str:
    """Normalize player name for matching (handles special characters)"""
    if not name or not isinstance(name, str):
        return ""
    special_chars = {
        '\u00f8': 'o', '\u00d8': 'O', '\u00df': 'ss', '\u0131': 'i', '\u0130': 'I',
        '\u0142': 'l', '\u0141': 'L', '\u0111': 'd', '\u0110': 'D',
        '\u00e6': 'ae', '\u00c6': 'AE', '\u0153': 'oe', '\u0152': 'OE',
    }
    for char, replacement in special_chars.items():
        name = name.replace(char, replacement)
    name = unicodedata.normalize("NFKD", name).encode("ASCII", "ignore").decode("ASCII")
    name = name.lower().strip()
    for remove in ["jr.", "jr", "ii", "iii", "."]:
        name = name.replace(remove, "")
    return " ".join(name.split())

def get_canonical_name(name: str) -> str:
    """Get canonical name, applying manual mappings"""
    norm = normalize_player_name(name)
    return PLAYER_NAME_MAPPING.get(norm, norm)

# Team name mapping (API-Football -> our data)
TEAM_MAPPING = {
    "Brighton & Hove Albion": "Brighton",
    "Brighton and Hove Albion": "Brighton",
    "Newcastle United": "Newcastle",
    "Wolverhampton Wanderers": "Wolves",
    "Wolverhampton": "Wolves",
    "Nottingham Forest": "Nottingham Forest",
    "Sheffield United": "Sheffield Utd",
    "Leicester City": "Leicester",
    "Ipswich Town": "Ipswich",
    "West Ham United": "West Ham",
    "Leeds United": "Leeds",
    "Tottenham Hotspur": "Tottenham",
}

def normalize_team(name: str) -> str:
    """Normalize team name to match our historical data"""
    return TEAM_MAPPING.get(name, name)



def get_fixtures(date_str: str) -> list:
    """Fetch EPL fixtures from API-Football"""
    url = "https://v3.football.api-sports.io/fixtures"
    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    params = {"league": 39, "season": 2025, "date": date_str}
    resp = requests.get(url, headers=headers, params=params)
    return resp.json().get("response", [])

def get_sportmonks_yc_odds(start_date: str, end_date: str) -> dict:
    """Fetch player booking odds from SportMonks (Market 64)"""
    url = f"https://api.sportmonks.com/v3/football/fixtures/between/{start_date}/{end_date}"
    params = {
        'api_token': SPORTMONKS_TOKEN,
        'filters': 'fixtureLeagues:8',
        'include': 'odds',
        'per_page': 50
    }
    resp = requests.get(url, params=params)
    data = resp.json()
    player_odds = {}
    for fixture in data.get('data', []):
        for odd in fixture.get('odds', []):
            if odd.get('market_id') == 64:
                player_name = odd.get('name')
                odds_value = odd.get('value')
                if player_name and odds_value:
                    canonical = get_canonical_name(player_name)
                    player_odds[canonical] = {
                        'odds': float(odds_value),
                        'original_name': player_name,
                        'implied_prob': 1 / float(odds_value)
                    }
    return player_odds

def upload_to_supabase(all_picks: list):
    """Upload predictions to Supabase database"""
    SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://kijtxzvbvhgswpahmvua.supabase.co")
    SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "sb_secret_8qWDEuaM0lh95i_CwBgl8A_MgxI1vQK")
    
    if not SUPABASE_KEY:
        print("   SUPABASE_KEY not set, skipping upload")
        return
    
    if not all_picks:
        print("   No picks to upload")
        return
    
    print("\n8. Uploading predictions to Supabase...")
    
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }
    
    # Clear old predictions for these dates
    dates_to_clear = list(set([p['date'] for p in all_picks]))
    for date_str in dates_to_clear:
        delete_url = f"{SUPABASE_URL}/rest/v1/yc_predictions?fixture_date=eq.{date_str}"
        requests.delete(delete_url, headers=headers)
    
    print(f"   Cleared old predictions for {dates_to_clear}")
    
    # Prepare records for upload
    records = []
    generated_at = datetime.utcnow().isoformat()
    
    for pick in all_picks:
        records.append({
            "fixture_date": pick['date'],
            "player_name": pick['player'],
            "team": pick['team'],
            "position": pick['pos'],
            "fixture": pick['game'],
            "referee": pick['ref'],
            "model_probability": round(pick['model_prob'], 4),
            "odds": round(pick['odds'], 2) if pick.get('odds') else None,
            "implied_probability": round(pick['implied_prob'], 4) if pick.get('implied_prob') else None,
            "edge": round(pick['edge'], 4) if pick.get('edge') else None,
            "tier": pick['tier'],
            "generated_at": generated_at
        })
    
    # Upload in batches
    batch_size = 50
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        insert_url = f"{SUPABASE_URL}/rest/v1/yc_predictions"
        resp = requests.post(insert_url, headers=headers, json=batch)
        if resp.status_code in [200, 201]:
            print(f"   Uploaded batch {i//batch_size + 1}: {len(batch)} records")
        else:
            print(f"   Error uploading batch: {resp.text[:200]}")
    
    print(f"   Uploaded {len(records)} predictions to Supabase")

def run_weekend_picks():
    """Generate value picks using YC v5 ML model + Supabase features"""
    today = datetime.now()

    # Generate predictions for the next 7 days (covers midweek + weekend)
    start_date = today
    end_date = today + timedelta(days=7)
    dates = [(today + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]

    print("=" * 80)
    print(f"YELLOW CARD VALUE PICKS (v6 ML) - {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print("=" * 80)

    # Load YC v10 model (LightGBM + XGBoost ensemble, best performer)
    model_path = os.path.join(os.path.dirname(__file__), "epl_yellow_cards_v10.pkl")
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    model = model_data["model"]
    model_version = model_data.get("version", "v10")
    FEATURES = model_data.get("features", YC_V7_FEATURES)
    print(f"   Loaded YC {model_version} model ({len(FEATURES)} features)")

    # Initialize shared feature engine (pulls from Supabase)
    engine = FeatureEngine()

    # Fetch odds
    print("\nFetching bookmaker odds from SportMonks...")
    yc_odds = get_sportmonks_yc_odds(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    print(f"   Found {len(yc_odds)} player booking odds")

    all_picks = []

    for date_str in dates:
        fixtures = get_fixtures(date_str)
        print(f"\n   {date_str}: {len(fixtures)} fixtures")

        for fix in fixtures:
            home_raw = fix["teams"]["home"]["name"]
            away_raw = fix["teams"]["away"]["name"]
            home = normalize_team(home_raw)
            away = normalize_team(away_raw)
            ref = fix["fixture"].get("referee", "TBD")
            ref_name = ref.split(",")[0].strip() if ref else "TBD"

            print(f"\n   {home} vs {away} (Ref: {ref_name})")

            # Process both teams
            for team, opponent, is_home in [(home, away, True), (away, home, False)]:
                try:
                    player_features_list = engine.get_fixture_player_features(
                        model='yc_v7',
                        team=team,
                        opponent=opponent,
                        is_home=is_home,
                        match_date=date_str,
                        referee=ref_name,
                        min_games=3,
                    )
                except Exception as e:
                    print(f"      Error getting features for {team}: {e}")
                    continue

                for pf in player_features_list:
                    player_name = pf.pop("_player_name", "Unknown")
                    position = pf.pop("_position", "M")
                    pf.pop("_player_id", None)

                    # Build feature array and predict
                    X = np.array([[pf[f] for f in FEATURES]])
                    prob = model.predict_proba(X)[0][1]
                    prob = min(prob, 0.45)  # clip isotonic artifacts

                    # Only include players above minimum probability
                    if prob < MIN_PROB:
                        continue

                    # Try to match odds (optional — picks are shown regardless)
                    canonical = get_canonical_name(player_name)
                    odds_data = yc_odds.get(canonical)

                    pick = {
                        'player': player_name,
                        'team': team,
                        'pos': position[0] if position else '?',
                        'game': f"{home} vs {away}",
                        'ref': ref_name,
                        'model_prob': prob,
                        'odds': odds_data['odds'] if odds_data else None,
                        'implied_prob': odds_data['implied_prob'] if odds_data else None,
                        'edge': (prob - odds_data['implied_prob']) if odds_data else None,
                        'date': date_str,
                    }

                    # Assign tier based on probability
                    if prob >= 0.20:
                        pick['tier'] = 'TOP_PICK'
                    elif prob >= 0.15:
                        pick['tier'] = 'STRONG'
                    elif prob >= 0.10:
                        pick['tier'] = 'MODERATE'
                    else:
                        pick['tier'] = 'LONGSHOT'

                    all_picks.append(pick)

    # Sort by probability (highest first)
    all_picks.sort(key=lambda x: x['model_prob'], reverse=True)

    print("\n" + "=" * 80)
    print(f"YC PREDICTIONS — Ranked by Model Probability")
    print("=" * 80)

    if not all_picks:
        print("\nNo predictions generated.")
    else:
        current_game = None
        for pick in all_picks:
            if pick['game'] != current_game:
                current_game = pick['game']
                print(f"\n   --- {pick['game']} (Ref: {pick['ref']}) ---")

            prob_pct = pick['model_prob'] * 100
            odds_str = f"@ {pick['odds']:.2f}" if pick['odds'] else ""
            edge_str = ""
            if pick['edge'] is not None:
                edge_str = f" (edge {pick['edge']*100:+.1f}%)" if pick['edge'] > 0 else ""

            tier_marker = {"TOP_PICK": "***", "STRONG": "**", "MODERATE": "*", "LONGSHOT": ""}
            marker = tier_marker.get(pick['tier'], "")

            print(f"   {prob_pct:5.1f}% {marker:3s} {pick['player'][:25]:25s} {pick['team'][:15]:15s} "
                  f"{pick['pos']} {odds_str} {edge_str}")

    if all_picks:
        df = pd.DataFrame(all_picks)
        df.to_csv("weekend_yc_picks.csv", index=False)
        print(f"\n   Saved {len(all_picks)} picks to weekend_yc_picks.csv")

    # Summary by tier
    tiers = {}
    for p in all_picks:
        t = p['tier']
        tiers[t] = tiers.get(t, 0) + 1

    print("\n" + "=" * 80)
    print(f"SUMMARY: {len(all_picks)} predictions across {len(set(p['game'] for p in all_picks))} matches")
    for tier in ['TOP_PICK', 'STRONG', 'MODERATE', 'LONGSHOT']:
        if tier in tiers:
            print(f"   {tier:12s}: {tiers[tier]} players")
    print(f"Strategy: YC v7 model + Supabase features (ranked by probability)")
    print("=" * 80)

    return all_picks

if __name__ == "__main__":
    picks = run_weekend_picks()
    upload_to_supabase(picks)
