#!/usr/bin/env python3
"""
Yellow Card Prediction API
Deploy this on your backend to get weekly predictions

Usage:
    pip install flask pandas requests
    python api.py
    
Endpoints:
    GET /predictions/<date>  - Get predictions for a specific date (YYYY-MM-DD)
    GET /health              - Health check
"""

from flask import Flask, jsonify, request
import pandas as pd
import requests
from datetime import datetime, timedelta
import os

app = Flask(__name__)

# Config
API_FOOTBALL_KEY = os.environ.get('API_FOOTBALL_KEY', '0b8d12ae574703056b109de918c240ef')
DATA_PATH = os.environ.get('DATA_PATH', 'data/complete_yc_data.csv')

# Referee strictness ratings (YC per game)
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

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'timestamp': datetime.utcnow().isoformat()})

@app.route('/predictions/<date>')
def predictions(date):
    """Get predictions for a specific date"""
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
        
        return jsonify({
            'date': date,
            'fixtures': len(results),
            'games': results,
            'strategy': 'STRICT_REF + AWAY + DEF/MID = 43% hit rate',
            'generated_at': datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

