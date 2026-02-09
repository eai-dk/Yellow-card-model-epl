"""
Squad Validator - Centralized 25/26 EPL Squad Data
===================================================
Loads current squad data from local epl-squads-full.json file
and validates/corrects player-team mappings in predictions.
"""

import json
import os
from typing import Dict, List, Optional
from difflib import SequenceMatcher
import unicodedata

_squad_cache: Dict = {}

LOCAL_SQUAD_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "epl-squads-full.json")


def normalize_name(name: str) -> str:
    if not name:
        return ""
    name = unicodedata.normalize("NFKD", name).encode("ASCII", "ignore").decode("ASCII")
    return name.lower().strip()


def get_squad_data() -> Dict[str, List[Dict]]:
    global _squad_cache

    if _squad_cache:
        return _squad_cache

    # Load from local file
    if os.path.exists(LOCAL_SQUAD_FILE):
        try:
            with open(LOCAL_SQUAD_FILE, "r", encoding="utf-8") as f:
                _squad_cache = json.load(f)
            print(f"[SquadValidator] Loaded {len(_squad_cache)} teams from local file")
            return _squad_cache
        except Exception as e:
            print(f"[SquadValidator] Error loading local squad file: {e}")

    print(f"[SquadValidator] Squad file not found at {LOCAL_SQUAD_FILE}")
    return {}


def build_player_team_map() -> Dict[str, str]:
    squad_data = get_squad_data()
    if not squad_data:
        return {}

    player_map = {}
    for team, players in squad_data.items():
        for player in players:
            name = player.get("name", "")
            if name:
                player_map[normalize_name(name)] = team
                # Also map short name (e.g. "A. Semenyo" -> "Semenyo")
                if ". " in name:
                    short_name = name.split(". ", 1)[1]
                    player_map[normalize_name(short_name)] = team
                # Also map full first + last for common patterns
                parts = name.split()
                if len(parts) >= 2:
                    # Map last name only if unique enough (3+ chars)
                    last = parts[-1]
                    if len(last) >= 4:
                        key = normalize_name(last)
                        # Don't overwrite if already mapped to different team
                        if key not in player_map:
                            player_map[key] = team
    return player_map


def find_player_team(player_name: str) -> Optional[str]:
    player_map = build_player_team_map()
    if not player_map:
        return None

    normalized = normalize_name(player_name)

    # Exact match
    if normalized in player_map:
        return player_map[normalized]

    # Try short name match (e.g. "A. Semenyo" -> look up "semenyo")
    if ". " in player_name:
        short_name = player_name.split(". ", 1)[1]
        if normalize_name(short_name) in player_map:
            return player_map[normalize_name(short_name)]

    # Try last name match
    parts = player_name.split()
    if len(parts) >= 2:
        last = normalize_name(parts[-1])
        if last in player_map:
            return player_map[last]

    # Fuzzy match as last resort
    best_match, best_score = None, 0
    for name, team in player_map.items():
        score = SequenceMatcher(None, normalized, name).ratio()
        if score > best_score and score > 0.85:
            best_score, best_match = score, team
    return best_match


def validate_predictions(predictions: List[Dict]) -> List[Dict]:
    """
    Validate predictions against squad data.
    If squad data unavailable, return predictions unchanged.
    """
    squad_data = get_squad_data()
    if not squad_data:
        print("[SquadValidator] Squad data unavailable, returning predictions unchanged")
        return predictions

    corrections = 0
    validated = []
    for pred in predictions:
        player_name = pred.get("player_name") or pred.get("player", "")
        if not player_name:
            validated.append(pred)
            continue

        current_team = find_player_team(player_name)
        if current_team is None:
            validated.append(pred)
        else:
            claimed_team = pred.get("team", "")
            if normalize_name(current_team) != normalize_name(claimed_team):
                corrected = pred.copy()
                corrected["team"] = current_team
                print(f"[SquadValidator] Corrected {player_name}: {claimed_team} -> {current_team}")
                corrections += 1
                validated.append(corrected)
            else:
                validated.append(pred)

    print(f"[SquadValidator] Validated {len(validated)} predictions, {corrections} corrections made")
    return validated