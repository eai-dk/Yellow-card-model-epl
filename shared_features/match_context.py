"""
Match context features: referee, rivalry, home/away, rest days, season stage.
"""

from typing import Dict, Optional, List
from datetime import datetime, timedelta

from .constants import (
    POSITION_CODES, POSITION_SCORES, BIG_SIX, RIVALRIES,
    DEFAULT_DEFENSE_TIERS, DEFAULT_PRESSING,
)


def compute_match_context(
    player_position: str,
    is_home: bool,
    team: str,
    opponent: str,
    match_date: str,
    referee_data: Optional[Dict],
    team_data: Optional[Dict],
    opponent_team_data: Optional[Dict],
    round_str: str = "",
) -> Dict:
    """Compute all match-context features needed by all models."""

    pos = (player_position or "M")[0].upper()

    # Position flags
    is_defender = 1 if pos == "D" else 0
    is_midfielder = 1 if pos == "M" else 0
    is_forward = 1 if pos == "F" else 0
    is_goalkeeper = 1 if pos == "G" else 0
    position_code = POSITION_CODES.get(pos, 2)
    position_score = POSITION_SCORES.get(pos, 2)

    # Home/Away
    is_home_val = 1 if is_home else 0
    is_away_val = 1 - is_home_val

    # Rivalry / Big 6
    pair = frozenset({team, opponent})
    is_rivalry = 1 if pair in RIVALRIES else 0
    team_is_big6 = team in BIG_SIX
    opp_is_big6 = opponent in BIG_SIX
    is_big6_match = 1 if (team_is_big6 and opp_is_big6) else 0
    is_non_big6 = 0 if team_is_big6 else 1

    # Round / high stakes
    round_num = _parse_round(round_str)
    high_stakes = 1 if (round_num >= 30 or is_rivalry) else 0

    # Late season (March–May)
    try:
        dt = datetime.strptime(str(match_date)[:10], "%Y-%m-%d")
        late_season = 1 if dt.month in (3, 4, 5) else 0
    except (ValueError, TypeError):
        late_season = 0

    # Referee
    # Z-score parameters derived from referee_stats.csv (47 EPL referees)
    _REF_YPG_MEAN = 4.0504
    _REF_YPG_STD = 0.8485
    if referee_data:
        raw_ypg = float(referee_data.get("yellows_per_match", _REF_YPG_MEAN) or _REF_YPG_MEAN)
        referee_strictness = (raw_ypg - _REF_YPG_MEAN) / _REF_YPG_STD   # z-scored
        cards_per_game = raw_ypg                                          # raw yellows/match
    else:
        referee_strictness = 0.0          # z-score mean → average referee
        cards_per_game = _REF_YPG_MEAN    # league average raw value

    # Opponent quality
    if opponent_team_data:
        opp_def_tier = opponent_team_data.get("defense_tier") or DEFAULT_DEFENSE_TIERS.get(opponent, 3)
        opp_pressing_raw = opponent_team_data.get("pressing_intensity", "medium") or "medium"
        opp_pressing = _pressing_str_to_int(opp_pressing_raw, opponent)
    else:
        opp_def_tier = DEFAULT_DEFENSE_TIERS.get(opponent, 3)
        opp_pressing = DEFAULT_PRESSING.get(opponent, 2)

    opponent_weakness = opp_def_tier * 0.7 + (6 - opp_pressing) * 0.3
    vs_weak_defense = 1 if opp_def_tier >= 4 else 0

    return {
        # Position
        "is_defender": is_defender, "is_midfielder": is_midfielder,
        "is_forward": is_forward, "is_goalkeeper": is_goalkeeper,
        "position_code": position_code, "position_score": position_score,
        # Home/Away
        "is_home": is_home_val, "is_away": is_away_val,
        # Rivalry / Big6
        "is_rivalry_match": is_rivalry, "is_big6_match": is_big6_match,
        "is_non_big6": is_non_big6,
        # Match context
        "high_stakes_match": high_stakes, "late_season": late_season,
        # Referee
        "referee_strictness": referee_strictness, "cards_per_game": cards_per_game,
        # Opponent quality
        "opponent_defense_tier": opp_def_tier,
        "opponent_pressing": opp_pressing,
        "opponent_weakness": opponent_weakness,
        "vs_weak_defense": vs_weak_defense,
    }


def compute_rest_features(
    match_date: str,
    player_history: List[Dict],
) -> Dict:
    """Compute rest/fatigue features from fixture dates."""
    try:
        target = datetime.strptime(str(match_date)[:10], "%Y-%m-%d")
    except (ValueError, TypeError):
        return {
            "days_since_last_match": 7, "well_rested": 1, "rest_factor": 1.0,
            "matches_last_7_days": 0, "minutes_last_7_days": 0,
            "is_short_rest": 0, "is_congested_week": 0,
        }

    days_since = 7
    matches_7d = 0
    minutes_7d = 0

    for g in player_history:
        try:
            gd = datetime.strptime(str(g.get("match_date", ""))[:10], "%Y-%m-%d")
        except (ValueError, TypeError):
            continue
        delta = (target - gd).days
        if 0 < delta < days_since:
            days_since = delta
        if 0 < delta <= 7:
            matches_7d += 1
            minutes_7d += g.get("minutes_played", 0) or 0

    well_rested = 1 if days_since >= 6 else 0
    rest_factor = _rest_factor(days_since)
    is_short_rest = 1 if days_since < 4 else 0
    is_congested = 1 if matches_7d >= 2 else 0

    return {
        "days_since_last_match": days_since, "well_rested": well_rested,
        "rest_factor": rest_factor,
        "matches_last_7_days": matches_7d, "minutes_last_7_days": minutes_7d,
        "is_short_rest": is_short_rest, "is_congested_week": is_congested,
    }


def _rest_factor(days: int) -> float:
    if days <= 2: return 0.85
    if days <= 3: return 0.92
    if days <= 6: return 1.00
    if days <= 10: return 0.97
    return 0.95


def _parse_round(round_str: str) -> int:
    """Extract round number from 'Regular Season - 20' etc."""
    try:
        return int(round_str.split("-")[-1].strip())
    except (ValueError, IndexError, AttributeError):
        return 15  # mid-season default


def _pressing_str_to_int(val, team: str) -> int:
    """Convert pressing string or int to integer."""
    if isinstance(val, (int, float)):
        return int(val)
    mapping = {"high": 5, "medium": 3, "low": 1}
    return mapping.get(str(val).lower(), DEFAULT_PRESSING.get(team, 2))

