"""
FeatureEngine — the main orchestrator.
Pulls data from Supabase, computes features, returns model-ready dicts/DataFrames.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from .supabase_client import SupabaseClient
from .player_features import compute_rolling_stats
from .match_context import compute_match_context, compute_rest_features
from .team_features import compute_team_goals_features
from .constants import (
    YC_V5_FEATURES, FOUL_DRAWN_FEATURES, FOUL_COMMITTED_FEATURES,
    SOT_V5_FEATURES, SHOTS_COMBO_V3_FEATURES, TEAM_GOALS_V2_FEATURES,
    POSITION_CODES, BIG_SIX,
)


# Map model names to their feature lists
MODEL_FEATURES = {
    "yc_v5": YC_V5_FEATURES,
    "foul_drawn": FOUL_DRAWN_FEATURES,
    "foul_committed": FOUL_COMMITTED_FEATURES,
    "sot_v5": SOT_V5_FEATURES,
    "shots_combo_v3": SHOTS_COMBO_V3_FEATURES,
    "team_goals_v2": TEAM_GOALS_V2_FEATURES,
}


class FeatureEngine:
    """
    Main interface for feature computation.

    Usage:
        engine = FeatureEngine()
        features = engine.get_player_features(player_id=526, ...)
    """

    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        self.db = SupabaseClient(url=supabase_url, key=supabase_key)
        # Cache for referee/team lookups within a session
        self._referee_cache: Dict[str, Dict] = {}
        self._team_cache: Dict[str, Dict] = {}
        # Cache for aggregate + xG features (keyed by (team, opponent, date))
        self._agg_cache: Dict[tuple, Dict] = {}
        self._xg_cache: Dict[tuple, Dict] = {}

    # ═══════════════════════════════════════════════════════════════
    # PLAYER-LEVEL FEATURES (SOT, Shots Combo, Foul, YC)
    # ═══════════════════════════════════════════════════════════════

    def get_player_features(
        self,
        model: str,
        player_id: int = None,
        player_name: str = None,
        team: str = "",
        opponent: str = "",
        is_home: bool = True,
        match_date: str = "",
        referee: str = "",
        position: str = None,
        round_str: str = "",
        history_limit: int = 50,
    ) -> Dict:
        """
        Compute all features for a single player for a given model.

        Args:
            model: One of 'yc_v5', 'foul_drawn', 'foul_committed', 'sot_v5', 'shots_combo_v3'
            player_id: API-Football player ID
            player_name: Player name (fallback if no ID)
            team: Player's team (canonical name)
            opponent: Opposing team (canonical name)
            is_home: Whether player's team is home
            match_date: Target match date (YYYY-MM-DD)
            referee: Referee name for this match
            position: Player position (G/D/M/F) — auto-detected from history if None
            round_str: e.g. 'Regular Season - 25'
            history_limit: How many past games to fetch

        Returns:
            Dict with all features for the specified model (exact keys/order).
        """
        if model not in MODEL_FEATURES or model == "team_goals_v2":
            raise ValueError(f"Use get_team_goals_features() for team_goals_v2. Valid: {list(MODEL_FEATURES.keys())}")

        # 1. Fetch player history from Supabase
        history = self.db.get_player_history(
            player_id=player_id,
            player_name=player_name,
            limit=history_limit,
            before_date=match_date if match_date else None,
        )

        # Auto-detect position from most recent game
        if position is None and history:
            position = history[0].get("position", "M") or "M"
        elif position is None:
            position = "M"

        # 2. Compute rolling stats
        rolling = compute_rolling_stats(history)

        # 3. Compute match context
        referee_data = self._get_referee(referee)
        team_data = self._get_team(team)
        opponent_data = self._get_team(opponent)

        context = compute_match_context(
            player_position=position,
            is_home=is_home,
            team=team,
            opponent=opponent,
            match_date=match_date,
            referee_data=referee_data,
            team_data=team_data,
            opponent_team_data=opponent_data,
            round_str=round_str,
        )

        # 4. Compute rest features
        rest = compute_rest_features(match_date, history)

        # 5. Compute opponent/team aggregate features
        agg = self._compute_aggregate_features(
            team=team, opponent=opponent, match_date=match_date, history=history,
        )

        # 6. Compute xG features (for shots_combo_v3)
        xg = {}
        if model == "shots_combo_v3":
            xg = self._compute_xg_features(player_name or "", opponent, match_date)

        # 7. Merge all feature dicts
        all_features = {**rolling, **context, **rest, **agg, **xg}

        # 8. Add shot-specific derived features
        pos_char = (position or "M")[0].upper()
        expected_by_pos = {"F": 2.5, "M": 1.5, "D": 0.5, "G": 0}
        all_features["expected_shots"] = expected_by_pos.get(pos_char, 1.5)
        shots_l5 = all_features.get("shots_l5", 0)
        if shots_l5 >= 2.5:
            all_features["shot_volume_tier"] = 3
        elif shots_l5 >= 1.5:
            all_features["shot_volume_tier"] = 2
        else:
            all_features["shot_volume_tier"] = 1

        # 9. Select only the features this model needs, in order
        feature_list = MODEL_FEATURES[model]
        result = {}
        for f in feature_list:
            result[f] = all_features.get(f, 0)

        return result

    # ═══════════════════════════════════════════════════════════════
    # TEAM-LEVEL FEATURES (Team Goals v2)
    # ═══════════════════════════════════════════════════════════════

    def get_team_goals_features(
        self, home_team: str, away_team: str, match_date: str = "",
    ) -> Dict:
        """Compute 19 features for Team Goals v2 model."""
        home_xg = self.db.get_team_xg_history(home_team, limit=10, before_date=match_date or None)
        away_xg = self.db.get_team_xg_history(away_team, limit=10, before_date=match_date or None)
        home_fix = self.db.get_fixtures(home_team, limit=5, before_date=match_date or None)
        away_fix = self.db.get_fixtures(away_team, limit=5, before_date=match_date or None)
        raw = compute_team_goals_features(home_xg, away_xg, home_fix, away_fix, match_date)
        return {f: raw.get(f, 0) for f in TEAM_GOALS_V2_FEATURES}

    # ═══════════════════════════════════════════════════════════════
    # BATCH: all players for a fixture
    # ═══════════════════════════════════════════════════════════════

    def get_fixture_player_features(
        self, model: str, team: str, opponent: str, is_home: bool,
        match_date: str, referee: str = "", round_str: str = "", min_games: int = 3,
    ) -> List[Dict]:
        """Get features for ALL players on a team for a fixture.

        OPTIMIZED: Fetches all player histories in 1 bulk query instead of
        N individual queries. Aggregate and xG features are cached per fixture.
        """
        # 1. Bulk-fetch ALL player histories for this team in ONE query
        all_histories = self.db.get_all_team_player_histories(
            team=team, limit_per_player=50, before_date=match_date or None,
        )

        # 2. Pre-compute shared data (cached across players)
        referee_data = self._get_referee(referee)
        team_data = self._get_team(team)
        opponent_data = self._get_team(opponent)
        # Aggregate features — same for every player on this team
        # (uses caching internally, so only 2 API calls total for the fixture)
        dummy_history = []  # not used by _compute_aggregate_features
        agg = self._compute_aggregate_features(team, opponent, match_date, dummy_history)

        # Pre-compute xG opponent data if shots_combo model (cached)
        if model == "shots_combo_v3":
            # Warm the xG cache for this opponent
            self._compute_xg_features("", opponent, match_date)

        # 3. Identify unique players from the bulk data
        player_meta = {}
        for key, history in all_histories.items():
            if history:
                latest = history[0]
                pid = latest.get("af_player_id")
                name = latest.get("player_name", "")
                pos = latest.get("position", "M")
                if pid and pid not in player_meta:
                    player_meta[pid] = {"name": name, "pos": pos, "history": history}

        # 4. Compute features for each player using pre-fetched data
        results = []
        feature_list = MODEL_FEATURES[model]

        for pid, meta in player_meta.items():
            history = meta["history"]
            name = meta["name"]
            position = meta["pos"] or "M"

            # Rolling stats from pre-fetched history
            rolling = compute_rolling_stats(history)

            # Match context (referee/team data already cached)
            context = compute_match_context(
                player_position=position, is_home=is_home, team=team,
                opponent=opponent, match_date=match_date,
                referee_data=referee_data, team_data=team_data,
                opponent_team_data=opponent_data, round_str=round_str,
            )

            # Rest features from pre-fetched history
            rest = compute_rest_features(match_date, history)

            # xG features (player xG still per-player, opponent xG cached)
            xg = {}
            if model == "shots_combo_v3":
                xg = self._compute_xg_features(name, opponent, match_date)

            # Merge all
            all_features = {**rolling, **context, **rest, **agg, **xg}

            # Shot-specific derived features
            pos_char = (position or "M")[0].upper()
            expected_by_pos = {"F": 2.5, "M": 1.5, "D": 0.5, "G": 0}
            all_features["expected_shots"] = expected_by_pos.get(pos_char, 1.5)
            shots_l5 = all_features.get("shots_l5", 0)
            if shots_l5 >= 2.5:
                all_features["shot_volume_tier"] = 3
            elif shots_l5 >= 1.5:
                all_features["shot_volume_tier"] = 2
            else:
                all_features["shot_volume_tier"] = 1

            # Select model features
            feats = {f: all_features.get(f, 0) for f in feature_list}

            if feats.get("career_games", feats.get("games_season", 0)) >= min_games:
                feats["_player_id"] = pid
                feats["_player_name"] = name
                feats["_position"] = position
                results.append(feats)

        return results

    # ═══════════════════════════════════════════════════════════════
    # INTERNAL HELPERS
    # ═══════════════════════════════════════════════════════════════

    def _get_referee(self, name: str) -> Optional[Dict]:
        if not name:
            return None
        if name not in self._referee_cache:
            self._referee_cache[name] = self.db.get_referee(name)
        return self._referee_cache[name]

    def _get_team(self, name: str) -> Optional[Dict]:
        if not name:
            return None
        if name not in self._team_cache:
            self._team_cache[name] = self.db.get_team(name)
        return self._team_cache[name]

    def _compute_aggregate_features(
        self, team: str, opponent: str, match_date: str, history: List[Dict],
    ) -> Dict:
        """Team-level aggregates: defensive pressure, opponent tendencies.
        Results are cached per (team, opponent, match_date) to avoid repeat API calls."""
        cache_key = (team, opponent, match_date)
        if cache_key in self._agg_cache:
            return self._agg_cache[cache_key]

        team_rows = self.db.get_team_match_history(team, limit=5, before_date=match_date or None)
        team_yc = [r.get("yellow_cards", 0) or 0 for r in team_rows]
        team_defensive_pressure = np.mean(team_yc) if team_yc else 0.1
        fixture_yc = {}
        for r in team_rows:
            fid = r.get("af_fixture_id")
            if fid not in fixture_yc:
                fixture_yc[fid] = 0
            fixture_yc[fid] += r.get("yellow_cards", 0) or 0
        last_5_totals = list(fixture_yc.values())[:5]
        # Normalize: divide total team YCs by unique players seen
        # Training data scale ≈ 0.54 (tree splits [0.42, 0.68])
        num_unique_players = len(set(
            r.get("af_player_id") or r.get("player_name") for r in team_rows
        )) or 1
        team_cards_last_5 = sum(last_5_totals) / num_unique_players if last_5_totals else 0.5

        opp_rows = self.db.get_opponent_stats(opponent, limit=5)
        opp_fc = [r.get("fouls_committed", 0) or 0 for r in opp_rows]
        opp_fd = [r.get("fouls_drawn", 0) or 0 for r in opp_rows]
        opponent_fouls_tendency = np.mean(opp_fc) if opp_fc else 1.0
        opponent_fouls_drawn_tendency = np.mean(opp_fd) if opp_fd else 1.0
        opp_yc = [r.get("yellow_cards", 0) or 0 for r in opp_rows]
        opponent_avg_cards = np.mean(opp_yc) if opp_yc else 0.1

        team_shots = [r.get("shots_total", 0) or 0 for r in team_rows]
        team_offensive_strength = np.mean(team_shots) * 11 if team_shots else 12
        opponent_shots_conceded = np.mean([r.get("shots_total", 0) or 0 for r in opp_rows]) * 11 if opp_rows else 10

        result = {
            "team_defensive_pressure": team_defensive_pressure,
            "team_cards_last_5": team_cards_last_5,
            "opponent_fouls_tendency": opponent_fouls_tendency,
            "opponent_fouls_drawn_tendency": opponent_fouls_drawn_tendency,
            "opponent_avg_cards": opponent_avg_cards,
            "team_offensive_strength": team_offensive_strength,
            "opponent_shots_conceded": opponent_shots_conceded,
        }
        self._agg_cache[cache_key] = result
        return result

    def _compute_xg_features(self, player_name: str, opponent: str, match_date: str) -> Dict:
        """xG + PPDA features for Shots Combo v3.
        Opponent-level xG data is cached per (opponent, match_date)."""
        pxg = self.db.get_player_xg(player_name) if player_name else None
        if pxg:
            shots = max(int(pxg.get("shots", 1) or 1), 1)
            player_xg_per_shot = float(pxg.get("xg", 0) or 0) / shots
            player_npxg_season = float(pxg.get("npxg", 0) or 0)
            player_xg_overperform = int(pxg.get("goals", 0) or 0) - float(pxg.get("xg", 0) or 0)
        else:
            player_xg_per_shot, player_npxg_season, player_xg_overperform = 0.1, 2.0, 0

        # Cache opponent xG data
        opp_cache_key = (opponent, match_date)
        if opp_cache_key in self._xg_cache:
            opp_xg = self._xg_cache[opp_cache_key]
        else:
            opp_xg_rows = self.db.get_team_xg_history(opponent, limit=5, before_date=match_date or None)
            if opp_xg_rows:
                opp_xg = {
                    "opponent_ppda": np.mean([float(r.get("ppda", 12) or 12) for r in opp_xg_rows]),
                    "opponent_xga_avg": np.mean([float(r.get("xga", 1.5) or 1.5) for r in opp_xg_rows]),
                }
            else:
                opp_xg = {"opponent_ppda": 12.0, "opponent_xga_avg": 1.5}
            self._xg_cache[opp_cache_key] = opp_xg

        return {
            "player_xg_per_shot": player_xg_per_shot,
            "player_npxg_season": player_npxg_season,
            "player_xg_overperform": player_xg_overperform,
            "opponent_ppda": opp_xg["opponent_ppda"],
            "opponent_xga_avg": opp_xg["opponent_xga_avg"],
            "opponent_is_high_press": 1 if opp_xg["opponent_ppda"] < 10 else 0,
        }
