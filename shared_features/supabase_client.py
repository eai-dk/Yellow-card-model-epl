"""
Thin Supabase REST API client.
All data fetching happens here — no other module talks to Supabase directly.
"""

import os
import requests
from typing import List, Dict, Optional
from .constants import SUPABASE_URL, SUPABASE_KEY


class SupabaseClient:
    """Lightweight Supabase REST client for read-only queries."""

    def __init__(self, url: str = None, key: str = None):
        self.url = url or os.environ.get("SUPABASE_URL", SUPABASE_URL)
        self.key = key or os.environ.get("SUPABASE_KEY", SUPABASE_KEY)
        self.headers = {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
        }

    def _get(self, table: str, params: Dict) -> List[Dict]:
        """Execute a GET request against Supabase REST API."""
        resp = requests.get(
            f"{self.url}/rest/v1/{table}",
            headers=self.headers,
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    # ─── player_match_stats ───────────────────────────────────────────

    def get_player_history(
        self,
        player_id: int = None,
        player_name: str = None,
        limit: int = 100,
        before_date: str = None,
    ) -> List[Dict]:
        """
        Get a player's match history, most recent first.
        Use player_id (af_player_id) or player_name.
        If before_date is set, only returns games before that date.
        """
        params = {
            "select": "*",
            "order": "match_date.desc",
            "limit": str(limit),
        }
        if player_id:
            params["af_player_id"] = f"eq.{player_id}"
        elif player_name:
            params["player_name"] = f"eq.{player_name}"
        else:
            raise ValueError("Must provide player_id or player_name")

        if before_date:
            params["match_date"] = f"lt.{before_date}"

        return self._get("player_match_stats", params)

    def get_team_players_latest(self, team: str, min_games: int = 3) -> List[Dict]:
        """Get all players for a team with their most recent stats."""
        params = {
            "select": "*",
            "team": f"eq.{team}",
            "order": "match_date.desc",
            "limit": "5000",
        }
        return self._get("player_match_stats", params)

    def get_team_match_history(
        self, team: str, limit: int = 20, before_date: str = None
    ) -> List[Dict]:
        """Get all player_match_stats rows for a team's recent fixtures."""
        params = {
            "select": "*",
            "team": f"eq.{team}",
            "order": "match_date.desc",
            "limit": str(limit * 20),  # ~20 players per match
        }
        if before_date:
            params["match_date"] = f"lt.{before_date}"
        return self._get("player_match_stats", params)

    def get_opponent_stats(self, opponent: str, limit: int = 10) -> List[Dict]:
        """Get recent player_match_stats where team == opponent (their players)."""
        params = {
            "select": "*",
            "team": f"eq.{opponent}",
            "order": "match_date.desc",
            "limit": str(limit * 20),
        }
        return self._get("player_match_stats", params)

    def get_all_team_player_histories(
        self, team: str, limit_per_player: int = 50, before_date: str = None,
        season_start: str = "2025-08-01",
    ) -> Dict[str, List[Dict]]:
        """
        Batch-fetch ALL player histories for a team in a single query.
        Returns dict keyed by (af_player_id or player_name) -> list of match rows.
        Only includes matches from season_start onwards to exclude transferred players.
        """
        params = {
            "select": "*",
            "team": f"eq.{team}",
            "order": "match_date.desc",
            "limit": "5000",
        }
        # Build date filter: after season_start AND before match_date
        if before_date and season_start:
            params["and"] = f"(match_date.gte.{season_start},match_date.lt.{before_date})"
        elif before_date:
            params["match_date"] = f"lt.{before_date}"
        elif season_start:
            params["match_date"] = f"gte.{season_start}"

        rows = self._get("player_match_stats", params)

        # Group by player
        grouped: Dict[str, List[Dict]] = {}
        for r in rows:
            pid = r.get("af_player_id") or r.get("player_name", "")
            key = str(pid)
            if key not in grouped:
                grouped[key] = []
            if len(grouped[key]) < limit_per_player:
                grouped[key].append(r)

        return grouped

    # ─── referees ─────────────────────────────────────────────────────

    def get_referee(self, referee_name: str) -> Optional[Dict]:
        """Get referee stats by name."""
        params = {"referee_name": f"eq.{referee_name}", "limit": "1"}
        rows = self._get("referees", params)
        return rows[0] if rows else None

    def get_all_referees(self) -> List[Dict]:
        """Get all referees."""
        return self._get("referees", {"select": "*", "limit": "100"})

    # ─── teams ────────────────────────────────────────────────────────

    def get_team(self, team_name: str) -> Optional[Dict]:
        """Get team info (defense_tier, pressing, is_big_six)."""
        params = {"team_name": f"eq.{team_name}", "limit": "1"}
        rows = self._get("teams", params)
        return rows[0] if rows else None

    # ─── fixtures ─────────────────────────────────────────────────────

    def get_fixtures(self, team: str = None, limit: int = 20, before_date: str = None) -> List[Dict]:
        """Get fixtures for a team."""
        params = {"select": "*", "order": "match_date.desc", "limit": str(limit)}
        if team:
            params["or"] = f"(home_team.eq.{team},away_team.eq.{team})"
        if before_date:
            params["match_date"] = f"lt.{before_date}"
        return self._get("fixtures", params)

    # ─── team_match_xg (Understat) ────────────────────────────────────

    def get_team_xg_history(
        self, team: str, limit: int = 10, before_date: str = None
    ) -> List[Dict]:
        """Get team's xG/PPDA history from Understat."""
        params = {
            "select": "*",
            "team": f"eq.{team}",
            "order": "match_date.desc",
            "limit": str(limit),
        }
        if before_date:
            params["match_date"] = f"lt.{before_date}"
        return self._get("team_match_xg", params)

    # ─── player_xg_data (Understat season totals) ─────────────────────

    def get_player_xg(self, player_name: str, season: str = None) -> Optional[Dict]:
        """Get player's season xG data from Understat."""
        params = {
            "player_name": f"eq.{player_name}",
            "order": "season.desc",
            "limit": "1",
        }
        if season:
            params["season"] = f"eq.{season}"
        rows = self._get("player_xg_data", params)
        return rows[0] if rows else None

