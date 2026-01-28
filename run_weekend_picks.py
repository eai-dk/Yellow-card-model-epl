#!/usr/bin/env python3
"""
🔥 Yellow Card Weekend Picks with Edge Calculation 🔥
Fetches real odds from SportMonks Market 64 (Player to be booked)
Only shows VALUE picks where edge > threshold
"""

import requests
import pandas as pd
import sys
from datetime import datetime, timedelta
from config import VERY_STRICT_REFS, STRICT_REFS, HIGH_CARD_OPPONENTS

# API Keys
API_FOOTBALL_KEY = "0b8d12ae574703056b109de918c240ef"
SPORTMONKS_TOKEN = "fd9XKsnh82xRG52vayu1ZZ1nbK8kdOk3s5Ex3ss7U2NV7MDejezJr3FNLFef"

# Edge threshold (only show picks with edge > this)
MIN_EDGE = 0.05  # 5%

def normalize_name(name: str) -> str:
    """Normalize player name for matching"""
    import unicodedata
    if not name:
        return ""
    name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('ASCII')
    return name.lower().strip()

def get_ref_tier(ref_name: str) -> int:
    """0=normal, 1=very strict, 2=strict"""
    if not ref_name or ref_name == "TBD":
        return 0
    for ref in VERY_STRICT_REFS:
        if ref.lower() in ref_name.lower() or ref_name.lower() in ref.lower():
            return 1
    for ref in STRICT_REFS:
        if ref.lower() in ref_name.lower() or ref_name.lower() in ref.lower():
            return 2
    return 0

def load_historical_data() -> pd.DataFrame:
    return pd.read_csv("data/complete_yc_data.csv")

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
    
    player_odds = {}  # {normalized_name: {'odds': X, 'original_name': Y}}
    
    for fixture in data.get('data', []):
        for odd in fixture.get('odds', []):
            if odd.get('market_id') == 64:  # Player to be booked
                player_name = odd.get('name')
                odds_value = odd.get('value')
                if player_name and odds_value:
                    norm = normalize_name(player_name)
                    player_odds[norm] = {
                        'odds': float(odds_value),
                        'original_name': player_name,
                        'implied_prob': 1 / float(odds_value)
                    }
    
    return player_odds

def run_weekend_picks():
    """Generate value picks with edge calculation"""
    today = datetime.now()
    
    # Get fixtures for Saturday and Sunday
    saturday = today + timedelta(days=(5 - today.weekday()) % 7)
    sunday = saturday + timedelta(days=1)
    
    print("=" * 80)
    print(f"🔥 YELLOW CARD VALUE PICKS - Weekend of {saturday.strftime('%Y-%m-%d')}")
    print("=" * 80)
    
    # Load data
    historical = load_historical_data()
    
    # Fetch odds
    print("\n📊 Fetching bookmaker odds from SportMonks...")
    yc_odds = get_sportmonks_yc_odds(saturday.strftime('%Y-%m-%d'), sunday.strftime('%Y-%m-%d'))
    print(f"   Found {len(yc_odds)} player booking odds")
    
    all_picks = []
    
    for date_str in [saturday.strftime('%Y-%m-%d'), sunday.strftime('%Y-%m-%d')]:
        fixtures = get_fixtures(date_str)
        
        for fix in fixtures:
            home = fix["teams"]["home"]["name"]
            away = fix["teams"]["away"]["name"]
            time = fix["fixture"]["date"][11:16]
            ref = fix["fixture"].get("referee", "TBD")
            ref_name = ref.split(",")[0].strip() if ref else "TBD"
            tier = get_ref_tier(ref_name)
            
            if tier == 0:
                continue  # Skip non-strict refs
            
            very_strict = tier == 1
            vs_high_card = home in HIGH_CARD_OPPONENTS
            
            # Get away DEF/MID picks (primary)
            away_data = historical[historical["team"] == away]
            away_defmid = away_data[away_data["position"].isin(["D", "M"])]
            
            if len(away_defmid) > 0:
                players = away_defmid.groupby("player_name").agg({
                    "yellow_card": ["mean", "count"],
                    "position": "first"
                }).reset_index()
                players.columns = ["name", "yc_rate", "games", "pos"]
                players = players[players["games"] >= 3]
                
                for _, p in players.iterrows():
                    # Adjust rate for ref strictness
                    adj_rate = min(p['yc_rate'] * 1.2, 0.55) if very_strict else p['yc_rate']
                    if vs_high_card:
                        adj_rate = min(adj_rate * 1.1, 0.55)
                    
                    # Match to odds
                    norm = normalize_name(p['name'])
                    odds_data = yc_odds.get(norm)
                    
                    if odds_data:
                        implied = odds_data['implied_prob']
                        edge = adj_rate - implied
                        
                        if edge >= MIN_EDGE:
                            all_picks.append({
                                'player': p['name'],
                                'team': away,
                                'pos': p['pos'],
                                'game': f"{home} vs {away}",
                                'ref': ref_name,
                                'model_prob': adj_rate,
                                'odds': odds_data['odds'],
                                'implied_prob': implied,
                                'edge': edge,
                                'tier': 'PRIMARY',
                                'date': date_str
                            })

            # Secondary: Home DEF/MID only if VERY STRICT or vs high-card away
            away_is_high_card = away in HIGH_CARD_OPPONENTS
            if very_strict or away_is_high_card:
                home_data = historical[historical["team"] == home]
                home_defmid = home_data[home_data["position"].isin(["D", "M"])]

                if len(home_defmid) > 0:
                    players = home_defmid.groupby("player_name").agg({
                        "yellow_card": ["mean", "count"],
                        "position": "first"
                    }).reset_index()
                    players.columns = ["name", "yc_rate", "games", "pos"]
                    players = players[players["games"] >= 3]

                    for _, p in players.iterrows():
                        adj_rate = min(p['yc_rate'] * 1.15, 0.50) if very_strict else p['yc_rate']

                        norm = normalize_name(p['name'])
                        odds_data = yc_odds.get(norm)

                        if odds_data:
                            implied = odds_data['implied_prob']
                            edge = adj_rate - implied

                            if edge >= MIN_EDGE:
                                all_picks.append({
                                    'player': p['name'],
                                    'team': home,
                                    'pos': p['pos'],
                                    'game': f"{home} vs {away}",
                                    'ref': ref_name,
                                    'model_prob': adj_rate,
                                    'odds': odds_data['odds'],
                                    'implied_prob': implied,
                                    'edge': edge,
                                    'tier': 'SECONDARY',
                                    'date': date_str
                                })

    # Sort by edge and display
    all_picks.sort(key=lambda x: x['edge'], reverse=True)

    print("\n" + "=" * 80)
    print(f"🔥 VALUE PICKS (Edge > {MIN_EDGE*100:.0f}%)")
    print("=" * 80)

    if not all_picks:
        print("\n❌ No value picks found. Either:")
        print("   - No strict ref games this weekend")
        print("   - Odds not available yet (check 24-48 hrs before kickoff)")
        print("   - No edge vs bookmaker prices")
    else:
        for pick in all_picks:
            edge_pct = pick['edge'] * 100
            prob_pct = pick['model_prob'] * 100
            implied_pct = pick['implied_prob'] * 100
            tier_icon = "💰" if pick['tier'] == 'PRIMARY' else "⚡"

            print(f"\n{tier_icon} {pick['player']} ({pick['team']}) - {pick['pos']}")
            print(f"   Game: {pick['game']} | Ref: {pick['ref']}")
            print(f"   Model: {prob_pct:.1f}% | Odds: {pick['odds']} (implied {implied_pct:.1f}%)")
            print(f"   🎯 EDGE: +{edge_pct:.1f}%")

    # Save to CSV
    if all_picks:
        df = pd.DataFrame(all_picks)
        df.to_csv("weekend_yc_picks.csv", index=False)
        print(f"\n✅ Saved {len(all_picks)} picks to weekend_yc_picks.csv")

    print("\n" + "=" * 80)
    print(f"📊 SUMMARY: {len(all_picks)} value picks with edge > {MIN_EDGE*100:.0f}%")
    print(f"💡 Strategy: STRICT REF + AWAY DEF/MID = 43% hit rate")
    print("=" * 80)

    return all_picks

if __name__ == "__main__":
    run_weekend_picks()

