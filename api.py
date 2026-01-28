#!/usr/bin/env python3
"""
🔥 Yellow Card Prediction API with Edge Calculation 🔥
FastAPI server for real-time YC value picks

Endpoints:
    GET /                    - Health check
    GET /picks/weekend       - Get weekend value picks with edge
    GET /predictions/<date>  - Get predictions for a specific date
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import requests
from datetime import datetime, timedelta
import unicodedata
import os
import subprocess
import json

app = FastAPI(
    title="Yellow Card Model API",
    description="EPL Yellow Card predictions with edge calculation",
    version="2.0.0"
)

# Config
API_FOOTBALL_KEY = os.environ.get('API_FOOTBALL_KEY', '0b8d12ae574703056b109de918c240ef')
SPORTMONKS_TOKEN = os.environ.get('SPORTMONKS_TOKEN', 'fd9XKsnh82xRG52vayu1ZZ1nbK8kdOk3s5Ex3ss7U2NV7MDejezJr3FNLFef')
DATA_PATH = os.environ.get('DATA_PATH', 'data/complete_yc_data.csv')
MIN_EDGE = 0.05  # 5% minimum edge

# Referee strictness ratings (YC per game)
VERY_STRICT_REFS = ['S. Allison', 'Tim Robinson', 'C. Kavanagh', 'Chris Kavanagh', 'T. Robinson', 'S. Barrott', 'Samuel Barrott']
STRICT_REFS_LIST = ['Michael Salisbury', 'J. Brooks', 'John Brooks', 'Stuart Attwell', 'T. Bramall', 'D. Bond', 'Robert Jones', 'Jarred Gillett', 'Anthony Taylor', 'Andy Madley', 'Michael Oliver']

STRICT_REFS = {
    'S. Allison': 6.5, 'Simon Allison': 6.5,
    'Tim Robinson': 5.2, 'T. Robinson': 5.2,
    'Chris Kavanagh': 5.2, 'C. Kavanagh': 5.2,
    'Sam Barrott': 5.1, 'S. Barrott': 5.1, 'Samuel Barrott': 5.1,
    'Michael Salisbury': 4.9, 'M. Salisbury': 4.9,
    'John Brooks': 4.8, 'J. Brooks': 4.8,
    'Stuart Attwell': 4.7, 'S. Attwell': 4.7,
    'Anthony Taylor': 4.2, 'A. Taylor': 4.2,
    'Michael Oliver': 4.0, 'M. Oliver': 4.0,
}

HIGH_CARD_OPPONENTS = ["Newcastle", "Aston Villa", "Chelsea", "Brighton", "Manchester City", "Tottenham"]

def load_data():
    return pd.read_csv(DATA_PATH)

def get_fixtures(date_str):
    """Fetch fixtures from API-Football"""
    url = "https://v3.football.api-sports.io/fixtures"
    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    params = {"league": 39, "season": 2025, "date": date_str}
    resp = requests.get(url, headers=headers, params=params)
    return resp.json().get('response', [])

def get_ref_strictness(ref_name):
    """Get referee strictness rating"""
    if not ref_name:
        return 3.5, False
    ref_clean = ref_name.split(',')[0].strip()
    yc_rate = STRICT_REFS.get(ref_clean, 3.5)
    return yc_rate, yc_rate >= 4.0

def get_player_picks(df, team_name, is_strict, is_away):
    """Get DEF/MID picks for a team"""
    team_df = df[df['team'].str.contains(team_name.split()[0], case=False, na=False)]
    if len(team_df) == 0:
        return []
    
    stats = team_df.groupby(['player_id', 'player_name', 'position']).agg({
        'yellow_card': ['sum', 'count', 'mean'],
        'fouls_committed': 'mean',
        'minutes': 'mean'
    }).reset_index()
    stats.columns = ['player_id', 'name', 'position', 'total_yc', 'games', 'yc_rate', 'avg_fouls', 'avg_mins']
    stats = stats[(stats['games'] >= 5) & (stats['avg_mins'] >= 45)]
    defmid = stats[stats['position'].isin(['D', 'M'])].sort_values('yc_rate', ascending=False)
    
    # Determine tier
    if is_strict and is_away:
        tier = 1
    elif is_strict or is_away:
        tier = 2
    else:
        tier = 3
    
    picks = []
    for _, p in defmid.head(10).iterrows():
        adj_rate = min(p['yc_rate'] * 1.2, 0.55) if is_strict else p['yc_rate']
        fair_odds = 1 / adj_rate if adj_rate > 0 else 99
        picks.append({
            'player_id': int(p['player_id']),
            'player_name': p['name'],
            'position': p['position'],
            'yc_rate': round(p['yc_rate'], 3),
            'total_yc': int(p['total_yc']),
            'games': int(p['games']),
            'avg_fouls': round(p['avg_fouls'], 2),
            'fair_odds': round(fair_odds, 2),
            'bet_threshold': round(fair_odds * 0.85, 2),
            'tier': tier
        })
    return picks

def normalize_name(name: str) -> str:
    """Normalize player name for matching"""
    if not name:
        return ""
    name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('ASCII')
    return name.lower().strip()

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

def get_ref_tier(ref_name: str) -> int:
    """0=normal, 1=very strict, 2=strict"""
    if not ref_name or ref_name == "TBD":
        return 0
    for ref in VERY_STRICT_REFS:
        if ref.lower() in ref_name.lower() or ref_name.lower() in ref.lower():
            return 1
    for ref in STRICT_REFS_LIST:
        if ref.lower() in ref_name.lower() or ref_name.lower() in ref.lower():
            return 2
    return 0

@app.get("/")
def root():
    return {"status": "ok", "model": "Yellow Card v2.0", "timestamp": datetime.utcnow().isoformat()}

@app.get("/picks/weekend")
def weekend_picks():
    """Get weekend value picks with edge calculation"""
    today = datetime.now()
    saturday = today + timedelta(days=(5 - today.weekday()) % 7)
    sunday = saturday + timedelta(days=1)

    df = load_data()
    yc_odds = get_sportmonks_yc_odds(saturday.strftime('%Y-%m-%d'), sunday.strftime('%Y-%m-%d'))

    all_picks = []

    for date_str in [saturday.strftime('%Y-%m-%d'), sunday.strftime('%Y-%m-%d')]:
        fixtures = get_fixtures(date_str)

        for fix in fixtures:
            home = fix["teams"]["home"]["name"]
            away = fix["teams"]["away"]["name"]
            ref = fix["fixture"].get("referee", "TBD")
            ref_name = ref.split(",")[0].strip() if ref else "TBD"
            tier = get_ref_tier(ref_name)

            if tier == 0:
                continue

            very_strict = tier == 1
            vs_high_card = home in HIGH_CARD_OPPONENTS

            # Away DEF/MID (primary picks)
            away_data = df[df["team"] == away]
            away_defmid = away_data[away_data["position"].isin(["D", "M"])]

            if len(away_defmid) > 0:
                players = away_defmid.groupby("player_name").agg({
                    "yellow_card": ["mean", "count"],
                    "position": "first"
                }).reset_index()
                players.columns = ["name", "yc_rate", "games", "pos"]
                players = players[players["games"] >= 3]

                for _, p in players.iterrows():
                    adj_rate = min(p['yc_rate'] * 1.2, 0.55) if very_strict else p['yc_rate']
                    if vs_high_card:
                        adj_rate = min(adj_rate * 1.1, 0.55)

                    norm = normalize_name(p['name'])
                    odds_data = yc_odds.get(norm)

                    if odds_data:
                        implied = odds_data['implied_prob']
                        edge = adj_rate - implied

                        if edge >= MIN_EDGE:
                            all_picks.append({
                                'player_name': p['name'],
                                'team': away,
                                'position': p['pos'],
                                'fixture': f"{home} vs {away}",
                                'referee': ref_name,
                                'probability': round(adj_rate, 4),  # Decimal for dashboard
                                'odds': odds_data['odds'],
                                'implied_probability': round(implied, 4),
                                'edge': round(edge, 4),
                                'tier': 'PRIMARY'
                            })

    all_picks.sort(key=lambda x: x['edge'], reverse=True)

    return {
        "weekend_start": saturday.strftime('%Y-%m-%d'),
        "weekend_end": sunday.strftime('%Y-%m-%d'),
        "min_edge": MIN_EDGE,
        "total_picks": len(all_picks),
        "odds_found": len(yc_odds),
        "picks": all_picks,
        "strategy": "STRICT REF + AWAY DEF/MID = 43% hit rate",
        "generated_at": datetime.utcnow().isoformat()
    }

@app.get("/predictions/{date}")
def predictions(date: str):
    """Get predictions for a specific date (legacy endpoint)"""
    try:
        df = load_data()
        fixtures = get_fixtures(date)

        results = []
        for fix in fixtures:
            home = fix['teams']['home']['name']
            away = fix['teams']['away']['name']
            ref = fix['fixture'].get('referee', '')
            kickoff = fix['fixture']['date']

            ref_yc, is_strict = get_ref_strictness(ref)

            game = {
                'fixture_id': fix['fixture']['id'],
                'home': home,
                'away': away,
                'kickoff': kickoff,
                'referee': ref,
                'referee_yc_per_game': ref_yc,
                'is_strict_ref': is_strict,
                'home_picks': get_player_picks(df, home, is_strict, is_away=False),
                'away_picks': get_player_picks(df, away, is_strict, is_away=True),
            }
            results.append(game)

        return {
            'date': date,
            'fixtures': len(results),
            'games': results,
            'strategy': 'STRICT_REF + AWAY + DEF/MID = 43% hit rate',
            'generated_at': datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('PORT', 5000))
    uvicorn.run(app, host='0.0.0.0', port=port)

