"""
Squad Validator - Centralized 25/26 EPL Squad Data
===================================================
Fetches current squad data from PitchPredict frontend repo
and validates/corrects player-team mappings in predictions.
"""

import requests
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher
import unicodedata
import time

SQUAD_DATA_URL = "https://raw.githubusercontent.com/EagleAI-Research/Pitchpredict/main/src/data/epl-squads-full.json"

_squad_cache: Dict = {}
_cache_timestamp: float = 0
CACHE_TTL = 3600


def normalize_name(name: str) -> str:
    if not name:
        return ""
    name = unicodedata.normalize("NFKD", name).encode("ASCII", "ignore").decode("ASCII")
    return name.lower().strip()


def get_squad_data() -> Dict[str, List[Dict]]:
    global _squad_cache, _cache_timestamp
    if _squad_cache and (time.time() - _cache_timestamp) < CACHE_TTL:
        return _squad_cache
    try:
        resp = requests.get(SQUAD_DATA_URL, timeout=10)
        if resp.status_code == 200:
            _squad_cache = resp.json()
            _cache_timestamp = time.time()
            print(f"[SquadValidator] Loaded {len(_squad_cache)} teams")
            return _squad_cache
    except Exception as e:
        print(f"[SquadValidator] Error: {e}")
    return _squad_cache


def build_player_team_map() -> Dict[str, str]:
    squad_data = get_squad_data()
    player_map = {}
    for team, players in squad_data.items():
        for player in players:
            name = player.get("name", "")
            if name:
                player_map[normalize_name(name)] = team
                if ". " in name:
                    short_name = name.split(". ", 1)[1]
                    player_map[normalize_name(short_name)] = team
    return player_map


def find_player_team(player_name: str) -> Optional[str]:
    player_map = build_player_team_map()
    normalized = normalize_name(player_name)
    if normalized in player_map:
        return player_map[normalized]
    if ". " in player_name:
        short_name = player_name.split(". ", 1)[1]
        if normalize_name(short_name) in player_map:
            return player_map[normalize_name(short_name)]
    best_match, best_score = None, 0
    for name, team in player_map.items():
        score = SequenceMatcher(None, normalized, name).ratio()
        if score > best_score and score > 0.85:
            best_score, best_match = score, team
    return best_match


def validate_prediction(prediction: Dict) -> Tuple[Dict, bool]:
    """
    Validate a prediction. Returns (prediction, is_valid).
    - If player found: correct team if needed, return True
    - If player NOT found: keep original, return True (don't filter)
    - Only return False for empty player names
    """
    player_name = prediction.get("player_name") or prediction.get("player", "")
    claimed_team = prediction.get("team", "")
    if not player_name:
        return prediction, False
    
    current_team = find_player_team(player_name)
    
    # Player not found in squad data - keep as-is (don't filter)
    if current_team is None:
        print(f"[SquadValidator] Player not found, keeping: {player_name} ({claimed_team})")
        return prediction, True
    
    # Player found - correct team if different
    if normalize_name(current_team) != normalize_name(claimed_team):
        corrected = prediction.copy()
        corrected["team"] = current_team
        print(f"[SquadValidator] Corrected: {player_name} {claimed_team} -> {current_team}")
        return corrected, True
    
    return prediction, True


def validate_predictions(predictions: List[Dict]) -> List[Dict]:
    validated = []
    for pred in predictions:
        corrected, is_valid = validate_prediction(pred)
        if is_valid:
            validated.append(corrected)
    return validated
