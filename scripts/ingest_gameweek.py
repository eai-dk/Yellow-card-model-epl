#!/usr/bin/env python3
"""
ingest_gameweek.py — Fetch recent EPL results into Supabase
===========================================================

Fetches finished EPL fixtures from the last 10 days and their player stats,
then upserts into Supabase `fixtures` and `player_match_stats` tables.

Designed to run weekly via GitHub Actions after each gameweek.
Uses ~20 API calls per gameweek (well within the 100/day free limit).
Upsert is idempotent — safe to re-run.

Usage:
    python scripts/ingest_gameweek.py
"""

import os, sys, json, time, math, requests
from datetime import datetime, timedelta

# ── Config ──────────────────────────────────────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://kijtxzvbvhgswpahmvua.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "sb_secret_8qWDEuaM0lh95i_CwBgl8A_MgxI1vQK")
API_FOOTBALL_KEY = os.environ.get("API_FOOTBALL_KEY", "0b8d12ae574703056b109de918c240ef")

AF_BASE = "https://v3.football.api-sports.io"
EPL_LEAGUE_ID = 39
CURRENT_SEASON = 2025  # API-Football season year (2025 = 2025-26)
LOOKBACK_DAYS = 10     # fetch fixtures from the last N days
BATCH_SIZE = 500
RATE_LIMIT_PAUSE = 6.5  # seconds between API-Football calls


def sb_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=minimal",
    }


def af_headers():
    return {"x-apisports-key": API_FOOTBALL_KEY}


def upsert(table, rows, on_conflict=""):
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    if on_conflict:
        url += f"?on_conflict={on_conflict}"
    total = len(rows)
    batches = math.ceil(total / BATCH_SIZE)
    ok = 0
    for i in range(batches):
        chunk = rows[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        resp = requests.post(url, headers=sb_headers(), json=chunk)
        if resp.status_code in (200, 201, 204):
            ok += len(chunk)
        else:
            print(f"   FAIL batch {i+1}/{batches} ({resp.status_code}): {resp.text[:300]}")
        time.sleep(0.3)
    print(f"   {ok}/{total} rows upserted into `{table}`")


def _season_str(year):
    return f"{year}-{str(year + 1)[-2:]}"


def fetch_recent_fixtures():
    """Fetch finished EPL fixtures from the last LOOKBACK_DAYS days."""
    today = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    season = _season_str(CURRENT_SEASON)

    print(f"   Fetching fixtures from {from_date} to {today} (season {season})...")

    resp = requests.get(
        f"{AF_BASE}/fixtures",
        headers=af_headers(),
        params={
            "league": EPL_LEAGUE_ID,
            "season": CURRENT_SEASON,
            "from": from_date,
            "to": today,
            "status": "FT",
        },
        timeout=15,
    )
    if resp.status_code != 200:
        print(f"   API error {resp.status_code}: {resp.text[:200]}")
        return []

    fixtures = resp.json().get("response", [])
    print(f"   Found {len(fixtures)} finished fixtures")

    rows = []
    for fix in fixtures:
        f = fix["fixture"]
        teams = fix["teams"]
        goals = fix["goals"]
        league = fix.get("league", {})
        venue = f.get("venue", {})
        rows.append({
            "af_fixture_id": f["id"],
            "season": season,
            "match_date": f["date"][:10],
            "kickoff": f["date"],
            "home_team": teams["home"]["name"],
            "away_team": teams["away"]["name"],
            "home_team_id": teams["home"]["id"],
            "away_team_id": teams["away"]["id"],
            "home_goals": goals.get("home"),
            "away_goals": goals.get("away"),
            "status": f.get("status", {}).get("short", "FT"),
            "referee": f.get("referee"),
            "venue": venue.get("name") if isinstance(venue, dict) else None,
            "round": league.get("round"),
            "raw_json": json.dumps(fix),
        })

    if rows:
        upsert("fixtures", rows, on_conflict="af_fixture_id")

    return rows


def fetch_player_stats(fixture_row):
    """Fetch all player stats for a single fixture."""
    af_id = fixture_row["af_fixture_id"]
    resp = requests.get(
        f"{AF_BASE}/fixtures/players",
        headers=af_headers(),
        params={"fixture": af_id},
        timeout=15,
    )
    if resp.status_code != 200:
        print(f"   Fixture {af_id}: API error {resp.status_code}")
        return []

    teams_data = resp.json().get("response", [])
    rows = []

    for team_block in teams_data:
        team_info = team_block.get("team", {})
        team_name = team_info.get("name", "")
        team_id = team_info.get("id")
        is_home = (team_name == fixture_row.get("home_team"))
        opponent = fixture_row["away_team"] if is_home else fixture_row["home_team"]
        opponent_id = fixture_row.get("away_team_id") if is_home else fixture_row.get("home_team_id")

        for p in team_block.get("players", []):
            s = p["statistics"][0] if p.get("statistics") else {}
            games = s.get("games", {})
            shots = s.get("shots", {})
            goals_s = s.get("goals", {})
            passes = s.get("passes", {})
            duels = s.get("duels", {})
            fouls = s.get("fouls", {})
            cards = s.get("cards", {})
            tackles = s.get("tackles", {})
            dribbles = s.get("dribbles", {})

            pos_raw = games.get("position", "")
            pos_map = {"Goalkeeper": "G", "Defender": "D", "Midfielder": "M", "Attacker": "F"}
            pos = pos_map.get(pos_raw, pos_raw[:1] if pos_raw else None)

            rows.append({
                "af_fixture_id": af_id,
                "af_player_id": p["player"]["id"],
                "season": fixture_row["season"],
                "match_date": fixture_row["match_date"],
                "player_name": p["player"]["name"],
                "team": team_name,
                "team_id": team_id,
                "opponent": opponent,
                "opponent_id": opponent_id,
                "is_home": is_home,
                "position": pos,
                "minutes_played": int(games.get("minutes") or 0),
                "rating": float(games["rating"]) if games.get("rating") else None,
                "shots_total": int(shots.get("total") or 0),
                "shots_on_target": int(shots.get("on") or 0),
                "goals": int(goals_s.get("total") or 0),
                "assists": int(goals_s.get("assists") or 0),
                "passes_total": int(passes.get("total") or 0),
                "passes_accuracy": int(passes.get("accuracy") or 0),
                "duels_total": int(duels.get("total") or 0),
                "duels_won": int(duels.get("won") or 0),
                "fouls_drawn": int(fouls.get("drawn") or 0),
                "fouls_committed": int(fouls.get("committed") or 0),
                "yellow_cards": int(cards.get("yellow") or 0),
                "red_cards": int(cards.get("red") or 0),
                "tackles_total": int(tackles.get("total") or 0),
                "interceptions": int(tackles.get("interceptions") or 0),
                "dribbles_attempts": int(dribbles.get("attempts") or 0),
                "dribbles_success": int(dribbles.get("success") or 0),
                "referee": fixture_row.get("referee"),
                "raw_json": json.dumps(p),
            })

    return rows


def main():
    if not SUPABASE_KEY:
        print("SUPABASE_KEY not set"); sys.exit(1)
    if not API_FOOTBALL_KEY:
        print("API_FOOTBALL_KEY not set"); sys.exit(1)

    print("=" * 60)
    print("  GAMEWEEK DATA INGESTION")
    print(f"  Lookback: {LOOKBACK_DAYS} days | Season: {_season_str(CURRENT_SEASON)}")
    print("=" * 60)

    # Step 1: Fetch recent fixtures
    print("\n1. Fetching recent fixtures...")
    fixtures = fetch_recent_fixtures()

    if not fixtures:
        print("   No new fixtures found. Done.")
        return

    # Step 2: Fetch player stats for each fixture
    print(f"\n2. Fetching player stats for {len(fixtures)} fixtures...")
    print(f"   Estimated time: ~{len(fixtures) * RATE_LIMIT_PAUSE / 60:.1f} minutes")

    all_player_rows = []
    for idx, fix in enumerate(fixtures, 1):
        print(f"   [{idx}/{len(fixtures)}] {fix['home_team']} vs {fix['away_team']} ({fix['match_date']})")
        player_rows = fetch_player_stats(fix)
        all_player_rows.extend(player_rows)

        if len(all_player_rows) >= 5000 or idx == len(fixtures):
            upsert("player_match_stats", all_player_rows, on_conflict="af_fixture_id,af_player_id")
            all_player_rows = []

        time.sleep(RATE_LIMIT_PAUSE)

    print(f"\n{'='*60}")
    print(f"  INGESTION COMPLETE — {len(fixtures)} fixtures processed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
