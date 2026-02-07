"""
Team-level feature computation for Team Goals v2 model.
Pulls from team_match_xg (Understat) and fixtures tables.
"""

import numpy as np
from typing import Dict, List, Optional
from datetime import datetime


def compute_team_goals_features(
    home_xg_history: List[Dict],
    away_xg_history: List[Dict],
    home_fixtures: List[Dict],
    away_fixtures: List[Dict],
    match_date: str,
    n: int = 5,
) -> Dict:
    """
    Compute the 19 features needed by Team Goals v2 model.

    Args:
        home_xg_history: team_match_xg rows for home team, most recent first
        away_xg_history: team_match_xg rows for away team, most recent first
        home_fixtures: fixture rows for home team (for rest days)
        away_fixtures: fixture rows for away team (for rest days)
        match_date: target match date string
        n: number of games to average over (default 5)
    """
    home_l = home_xg_history[:n]
    away_l = away_xg_history[:n]

    # Home team stats (last n)
    home_xG_l5 = _avg(home_l, "xg", 1.35)
    home_npxG_l5 = _avg(home_l, "npxg", 1.20)
    home_xGA_l5 = _avg(home_l, "xga", 1.30)
    home_npxGA_l5 = _avg(home_l, "npxga", 1.15)
    home_ppda_l5 = _avg(home_l, "ppda", 10.5)
    home_scored_l5 = _avg(home_l, "scored", 1.3)
    home_xG_overperf_l5 = home_scored_l5 - home_xG_l5

    # Away team stats (last n)
    away_xG_l5 = _avg(away_l, "xg", 1.35)
    away_npxG_l5 = _avg(away_l, "npxg", 1.20)
    away_xGA_l5 = _avg(away_l, "xga", 1.30)
    away_npxGA_l5 = _avg(away_l, "npxga", 1.15)
    away_ppda_l5 = _avg(away_l, "ppda", 10.5)
    away_scored_l5 = _avg(away_l, "scored", 1.1)
    away_xG_overperf_l5 = away_scored_l5 - away_xG_l5

    # Derived
    ppda_diff = home_ppda_l5 - away_ppda_l5
    combined_xG_overperf = home_xG_overperf_l5 + away_xG_overperf_l5
    total_xG_l5 = home_xG_l5 + away_xG_l5
    home_attack_vs_away_defense = home_xG_l5 - away_xGA_l5
    away_attack_vs_home_defense = away_xG_l5 - home_xGA_l5

    # Rest days
    rest_days_home = _compute_rest_days(home_fixtures, match_date)
    rest_days_away = _compute_rest_days(away_fixtures, match_date)

    return {
        "home_xG_l5": home_xG_l5,
        "home_npxG_l5": home_npxG_l5,
        "away_xG_l5": away_xG_l5,
        "away_npxG_l5": away_npxG_l5,
        "home_xGA_l5": home_xGA_l5,
        "home_npxGA_l5": home_npxGA_l5,
        "away_xGA_l5": away_xGA_l5,
        "away_npxGA_l5": away_npxGA_l5,
        "home_ppda_l5": home_ppda_l5,
        "away_ppda_l5": away_ppda_l5,
        "ppda_diff": ppda_diff,
        "home_xG_overperf_l5": home_xG_overperf_l5,
        "away_xG_overperf_l5": away_xG_overperf_l5,
        "combined_xG_overperf": combined_xG_overperf,
        "total_xG_l5": total_xG_l5,
        "home_attack_vs_away_defense": home_attack_vs_away_defense,
        "away_attack_vs_home_defense": away_attack_vs_home_defense,
        "rest_days_home": rest_days_home,
        "rest_days_away": rest_days_away,
    }


def _avg(rows: List[Dict], key: str, default: float) -> float:
    """Average a numeric field from rows, falling back to default."""
    vals = []
    for r in rows:
        v = r.get(key)
        if v is not None:
            try:
                vals.append(float(v))
            except (ValueError, TypeError):
                pass
    return float(np.mean(vals)) if vals else default


def _compute_rest_days(fixtures: List[Dict], match_date: str) -> int:
    """Compute days since last fixture."""
    try:
        target = datetime.strptime(str(match_date)[:10], "%Y-%m-%d")
    except (ValueError, TypeError):
        return 7

    for f in fixtures:
        try:
            fd = datetime.strptime(str(f.get("match_date", ""))[:10], "%Y-%m-%d")
            delta = (target - fd).days
            if delta > 0:
                return delta
        except (ValueError, TypeError):
            continue
    return 7

