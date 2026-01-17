#!/usr/bin/env python3
"""
🔒 LOCKED BASELINE STRATEGY - v1.0
==================================
Locked: 17th January 2026
Backtested: 4 seasons, 26,000+ player-games

RESULTS ON 17th JAN 2026:
- 6/20 hits at avg odds 6.5
- +£193 profit on £200 staked  
- +97% ROI

THE EDGE: Bookies don't adjust player YC odds for referee strictness.
"""

import requests
import pandas as pd
import sys
from datetime import datetime
from config import (
    API_KEY, BASE_URL, LEAGUE_ID, SEASON,
    VERY_STRICT_REFS, STRICT_REFS,
    TARGET_POSITIONS, MIN_GAMES, TOP_N_PLAYERS
)


def get_ref_tier(ref_name: str) -> int:
    """Determine referee tier (1=very strict, 2=strict, 0=normal)"""
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
    """Load historical player YC data"""
    return pd.read_csv("data/complete_yc_data.csv")


def get_fixtures(date_str: str) -> list:
    """Fetch fixtures for a given date from API-Football"""
    url = f"{BASE_URL}/fixtures"
    headers = {"x-apisports-key": API_KEY}
    params = {"league": LEAGUE_ID, "season": SEASON, "date": date_str}
    resp = requests.get(url, headers=headers, params=params)
    return resp.json().get("response", [])


def get_picks_for_game(historical: pd.DataFrame, team: str, is_home: bool, tier: int) -> list:
    """Get player picks for a team based on historical data"""
    team_data = historical[historical["team"] == team]
    defmid = team_data[team_data["position"].isin(TARGET_POSITIONS)]
    
    if len(defmid) == 0:
        return []
    
    player_stats = defmid.groupby("player_id").agg({
        "yellow_card": ["mean", "count"],
        "player_name": "first",
        "position": "first"
    }).reset_index()
    player_stats.columns = ["pid", "yc_rate", "games", "name", "pos"]
    player_stats = player_stats[player_stats["games"] >= MIN_GAMES]
    player_stats = player_stats.nlargest(TOP_N_PLAYERS, "games")
    
    picks = []
    for _, p in player_stats.iterrows():
        # Tier 2 = away team only
        if tier == 2 and is_home:
            continue
        picks.append({
            "name": p["name"],
            "pos": p["pos"],
            "rate": p["yc_rate"],
            "team": team,
            "home": is_home
        })
    return picks


def run_strategy(date_str: str):
    """Run the locked baseline strategy for a given date"""
    
    # Load data
    historical = load_historical_data()
    fixtures = get_fixtures(date_str)
    
    print("=" * 80)
    print(f"🔒 LOCKED BASELINE STRATEGY - {date_str}")
    print("=" * 80)
    print("Edge: Bookies don't adjust for ref strictness. We do.")
    print()
    
    tier1_games = []
    tier2_games = []
    
    for fix in fixtures:
        home = fix["teams"]["home"]["name"]
        away = fix["teams"]["away"]["name"]
        time = fix["fixture"]["date"][11:16]
        ref = fix["fixture"].get("referee", "TBD")
        ref_name = ref.split(",")[0].strip() if ref else "TBD"
        tier = get_ref_tier(ref_name)
        
        if tier == 0:
            continue
        
        # Get picks for both teams
        picks = []
        picks.extend(get_picks_for_game(historical, home, True, tier))
        picks.extend(get_picks_for_game(historical, away, False, tier))
        
        game = {
            "home": home, "away": away, "time": time,
            "ref": ref_name, "tier": tier, "picks": picks
        }
        
        if tier == 1:
            tier1_games.append(game)
        else:
            tier2_games.append(game)
    
    # Display TIER 1
    if tier1_games:
        print("🔥🔥 TIER 1 - VERY STRICT REFS (+25% ROI expected)")
        print("-" * 80)
        for g in tier1_games:
            print(f"\n⚽ {g['time']} {g['home']} vs {g['away']}")
            print(f"   Ref: {g['ref']} 🔥🔥 VERY STRICT")
            sorted_picks = sorted(g["picks"], key=lambda x: x["rate"], reverse=True)
            for p in sorted_picks[:6]:
                print(f"   • {p['name']:<25} ({p['pos']}) - {p['rate']*100:.0f}%")
    
    # Display TIER 2
    if tier2_games:
        print(f"\n{'=' * 80}")
        print("🔥 TIER 2 - STRICT REFS + AWAY ONLY (+12% ROI expected)")
        print("-" * 80)
        for g in tier2_games:
            print(f"\n⚽ {g['time']} {g['home']} vs {g['away']}")
            print(f"   Ref: {g['ref']} 🔥 STRICT")
            sorted_picks = sorted(g["picks"], key=lambda x: x["rate"], reverse=True)
            for p in sorted_picks[:5]:
                print(f"   • {p['name']:<25} ({p['pos']}) - {p['rate']*100:.0f}% (AWAY)")
    
    # No games
    if not tier1_games and not tier2_games:
        print("❌ No strict ref games today. Wait for refs to be assigned.")
    
    print(f"\n{'=' * 80}")
    print("💡 BET: Target odds 4.0-10.0 on these DEF/MID players")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        date = sys.argv[1]
    else:
        date = datetime.now().strftime("%Y-%m-%d")
    run_strategy(date)

