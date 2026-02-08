"""
Player-level feature computation from Supabase data.
Covers: SOT v5, Shots Combo v3, Foul v2, YC v5
"""

import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from .constants import (
    POSITION_CODES, POSITION_SCORES, BIG_SIX, RIVALRIES,
    DEFAULT_DEFENSE_TIERS, DEFAULT_PRESSING,
)


def _safe_mean(values: list, default: float = 0.0) -> float:
    return float(np.mean(values)) if values else default


def _safe_sum(values: list, default: float = 0.0) -> float:
    return float(np.sum(values)) if values else default


# ═══════════════════════════════════════════════════════════════════════
# ROLLING STATS CORE (used by all player-level models)
# ═══════════════════════════════════════════════════════════════════════

def compute_rolling_stats(history: List[Dict]) -> Dict:
    """
    Given a player's match history (most recent first, BEFORE the target game),
    compute all rolling stats needed by every model.
    """
    if not history:
        return {k: 0 for k in [
            "sot_l5", "sot_l10", "sot_season_avg", "shots_l5", "shots_l10",
            "shots_season_avg", "minutes_l5", "minutes_l10", "games_season",
            "goals_l5", "sot_streak_l3", "shots_streak_l3", "is_hot", "form_surge",
            "is_consistent", "minutes_stable", "shots_per_90_l5", "shot_accuracy_l5",
            "confidence_score", "shots_trend", "shot_variance_l10",
            "fouls_drawn_last_3", "fouls_drawn_last_5", "fouls_drawn_last_10",
            "fouls_drawn_avg_3", "fouls_drawn_avg_5", "fouls_drawn_avg_10",
            "fouls_committed_last_3", "fouls_committed_last_5", "fouls_committed_last_10",
            "fouls_committed_avg_3", "fouls_committed_avg_5", "fouls_committed_avg_10",
            "career_fouls_drawn_rate", "career_fouls_committed_rate", "career_games",
            "career_card_rate", "cards_last_10", "card_rate_last_10",
            "games_since_last_card", "yellow_cards_last_5", "yellow_cards_last_10",
            "recent_card_intensity",
        ]}

    l3 = history[:3]
    l5 = history[:5]
    l10 = history[:10]
    all_games = history

    # ── SOT / Shots stats ─────────────────────────────────────────
    sot_l5 = _safe_mean([g.get("shots_on_target", 0) or 0 for g in l5])
    sot_l10 = _safe_mean([g.get("shots_on_target", 0) or 0 for g in l10])
    shots_l5 = _safe_mean([g.get("shots_total", 0) or 0 for g in l5])
    shots_l10 = _safe_mean([g.get("shots_total", 0) or 0 for g in l10])
    shots_season_avg = _safe_mean([g.get("shots_total", 0) or 0 for g in all_games])
    sot_season_avg = _safe_mean([g.get("shots_on_target", 0) or 0 for g in all_games])

    # Minutes
    minutes_l5 = _safe_mean([g.get("minutes_played", 0) or 0 for g in l5])
    minutes_l10 = _safe_mean([g.get("minutes_played", 0) or 0 for g in l10])

    # Goals
    goals_l5 = _safe_mean([g.get("goals", 0) or 0 for g in l5])

    # ── Fouls stats ───────────────────────────────────────────────
    fd_l3 = _safe_sum([g.get("fouls_drawn", 0) or 0 for g in l3])
    fd_l5 = _safe_sum([g.get("fouls_drawn", 0) or 0 for g in l5])
    fd_l10 = _safe_sum([g.get("fouls_drawn", 0) or 0 for g in l10])
    fc_l3 = _safe_sum([g.get("fouls_committed", 0) or 0 for g in l3])
    fc_l5 = _safe_sum([g.get("fouls_committed", 0) or 0 for g in l5])
    fc_l10 = _safe_sum([g.get("fouls_committed", 0) or 0 for g in l10])
    fd_avg3 = _safe_mean([g.get("fouls_drawn", 0) or 0 for g in l3])
    fd_avg5 = _safe_mean([g.get("fouls_drawn", 0) or 0 for g in l5])
    fd_avg10 = _safe_mean([g.get("fouls_drawn", 0) or 0 for g in l10])
    fc_avg3 = _safe_mean([g.get("fouls_committed", 0) or 0 for g in l3])
    fc_avg5 = _safe_mean([g.get("fouls_committed", 0) or 0 for g in l5])
    fc_avg10 = _safe_mean([g.get("fouls_committed", 0) or 0 for g in l10])

    # Career rates
    career_games = len(all_games)
    career_fouls_drawn = sum(g.get("fouls_drawn", 0) or 0 for g in all_games)
    career_fouls_committed = sum(g.get("fouls_committed", 0) or 0 for g in all_games)
    career_yellows = sum(g.get("yellow_cards", 0) or 0 for g in all_games)
    career_fouls_drawn_rate = career_fouls_drawn / max(career_games, 1)
    career_fouls_committed_rate = career_fouls_committed / max(career_games, 1)
    career_card_rate = career_yellows / max(career_games, 1)

    # ── Tackles / Duels / Interceptions stats ─────────────────────
    tackles_avg_5 = _safe_mean([g.get("tackles_total", 0) or 0 for g in l5])
    duels_avg_5 = _safe_mean([g.get("duels_total", 0) or 0 for g in l5])
    interceptions_avg_5 = _safe_mean([g.get("interceptions", 0) or 0 for g in l5])
    duels_total_5 = _safe_sum([g.get("duels_total", 0) or 0 for g in l5])
    duels_won_5 = _safe_sum([g.get("duels_won", 0) or 0 for g in l5])
    duel_win_pct_l5 = duels_won_5 / max(duels_total_5, 1)

    # ── YC stats ──────────────────────────────────────────────────
    cards_last_10 = _safe_sum([g.get("yellow_cards", 0) or 0 for g in l10])
    card_rate_last_10 = _safe_mean([g.get("yellow_cards", 0) or 0 for g in l10])
    yc_last_5 = _safe_sum([g.get("yellow_cards", 0) or 0 for g in l5])
    yc_last_10 = _safe_sum([g.get("yellow_cards", 0) or 0 for g in l10])

    # Games since last card
    games_since_last_card = career_games  # default: never carded
    for i, g in enumerate(history):
        if (g.get("yellow_cards", 0) or 0) > 0:
            games_since_last_card = i
            break

    # SOT streak (last 3 games)
    sot_streak_l3 = sum(1 for g in l3 if (g.get("shots_on_target", 0) or 0) >= 1)
    shots_streak_l3 = 1 if shots_l5 >= 1 else 0
    is_hot = 1 if sot_streak_l3 >= 2 else 0

    # Form surge
    l3_sot_avg = _safe_mean([g.get("shots_on_target", 0) or 0 for g in l3])
    form_surge = 1 if sot_l10 > 0 and l3_sot_avg > sot_l10 * 1.2 else 0

    # Consistency
    sot_values = [g.get("shots_on_target", 0) or 0 for g in l10]
    sot_std = float(np.std(sot_values)) if len(sot_values) >= 5 else 1.0
    is_consistent = 1 if sot_std < 0.8 else 0

    # Minutes stability
    mins_values = [g.get("minutes_played", 0) or 0 for g in l5]
    mins_std = float(np.std(mins_values)) if len(mins_values) >= 3 else 20.0
    minutes_stable = 1 if mins_std < 15 else 0

    # Shot variance
    shot_values_l10 = [g.get("shots_total", 0) or 0 for g in l10]
    shot_variance_l10 = float(np.std(shot_values_l10)) if len(shot_values_l10) >= 5 else 0.5

    # Derived
    shots_per_90_l5 = min((shots_l5 / (minutes_l5 / 90)), 10) if minutes_l5 > 0 else 0
    shot_accuracy_l5 = min(sot_l5 / max(shots_l5, 0.1), 1.0)
    confidence_score = min(career_games / 10, 1.0)

    # Recent card intensity (YC model)
    recent_card_intensity = card_rate_last_10 * cards_last_10

    # ── v8 derived features ───────────────────────────────────────

    # card_per_foul_rate: career yellows / career fouls (foul severity)
    card_per_foul_rate = career_yellows / max(career_fouls_committed, 1)

    # fouls_per_90_l5: normalize fouls by minutes
    fc_avg_5 = _safe_mean([g.get("fouls_committed", 0) or 0 for g in l5])
    fouls_per_90_l5 = (fc_avg_5 / max(minutes_l5, 1)) * 90

    # fouls_committed_trend: fc_avg_3 - fc_avg_10 (getting dirtier?)
    fc_avg_3 = _safe_mean([g.get("fouls_committed", 0) or 0 for g in l3])
    fc_avg_10 = _safe_mean([g.get("fouls_committed", 0) or 0 for g in l10])
    fouls_committed_trend = fc_avg_3 - fc_avg_10

    # season_yellows_accumulated: count of YCs in current season
    # Approximate: count yellows across all available history in the same season
    current_season_games = [g for g in all_games if g.get("season") == (all_games[0].get("season") if all_games else "")]
    season_yellows_accumulated = sum(g.get("yellow_cards", 0) or 0 for g in current_season_games)

    # Recency-weighted career stats (exponential decay, 0.95 factor)
    _DECAY = 0.95
    if career_games > 0:
        yc_vals = [g.get("yellow_cards", 0) or 0 for g in all_games]  # most-recent first
        fc_vals = [g.get("fouls_committed", 0) or 0 for g in all_games]
        weights = [_DECAY ** j for j in range(len(yc_vals))]
        w_sum = sum(weights)
        career_card_rate_weighted = sum(y * w for y, w in zip(yc_vals, weights)) / w_sum
        career_fc_rate_weighted = sum(f * w for f, w in zip(fc_vals, weights)) / w_sum
    else:
        career_card_rate_weighted = 0
        career_fc_rate_weighted = 0

    return {
        # SOT / Shots
        "sot_l5": sot_l5, "sot_l10": sot_l10, "sot_season_avg": sot_season_avg,
        "shots_l5": shots_l5, "shots_l10": shots_l10, "shots_season_avg": shots_season_avg,
        "minutes_l5": minutes_l5, "minutes_l10": minutes_l10,
        "games_season": career_games,
        "goals_l5": goals_l5,
        "sot_streak_l3": sot_streak_l3, "shots_streak_l3": shots_streak_l3,
        "is_hot": is_hot, "form_surge": form_surge,
        "is_consistent": is_consistent, "minutes_stable": minutes_stable,
        "shots_per_90_l5": shots_per_90_l5, "shot_accuracy_l5": shot_accuracy_l5,
        "confidence_score": confidence_score,
        "shots_trend": shots_l5 - shots_l10,
        "shot_variance_l10": shot_variance_l10,
        # Fouls
        "fouls_drawn_last_3": fd_l3, "fouls_drawn_last_5": fd_l5, "fouls_drawn_last_10": fd_l10,
        "fouls_drawn_avg_3": fd_avg3, "fouls_drawn_avg_5": fd_avg5, "fouls_drawn_avg_10": fd_avg10,
        "fouls_committed_last_3": fc_l3, "fouls_committed_last_5": fc_l5, "fouls_committed_last_10": fc_l10,
        "fouls_committed_avg_3": fc_avg3, "fouls_committed_avg_5": fc_avg5, "fouls_committed_avg_10": fc_avg10,
        "career_fouls_drawn_rate": career_fouls_drawn_rate,
        "career_fouls_committed_rate": career_fouls_committed_rate,
        "career_games": career_games,
        # Tackles / Duels / Interceptions
        "tackles_avg_5": tackles_avg_5, "duels_avg_5": duels_avg_5,
        "interceptions_avg_5": interceptions_avg_5, "duel_win_pct_l5": duel_win_pct_l5,
        # v8 derived
        "card_per_foul_rate": card_per_foul_rate,
        "fouls_per_90_l5": fouls_per_90_l5,
        "fouls_committed_trend": fouls_committed_trend,
        "season_yellows_accumulated": season_yellows_accumulated,
        "career_card_rate_weighted": career_card_rate_weighted,
        "career_fc_rate_weighted": career_fc_rate_weighted,
        # YC
        "career_card_rate": career_card_rate,
        "cards_last_10": cards_last_10, "card_rate_last_10": card_rate_last_10,
        "games_since_last_card": games_since_last_card,
        "yellow_cards_last_5": yc_last_5, "yellow_cards_last_10": yc_last_10,
        "recent_card_intensity": recent_card_intensity,
    }

