"""
Squad Validator - Centralized 25/26 EPL Squad Data
=====================================================
Loads current squad data from local epl-squads-full.json file
and validates/corrects player-team mappings in predictions.

Key behavior:
  - Corrects team name if player transferred (e.g. Semenyo -> Man City)
  - REMOVES predictions where corrected team doesn't match either team
    in the fixture (player shouldn't be predicted for that match at all)
"""

import json
import os
from typing import Dict, List, Optional
from difflib import SequenceMatcher
import unicodedata

_squad_cache: Dict = {}

LOCAL_SQUAD_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "epl-squads-full.json")

# Team name aliases for matching fixture strings
TEAM_ALIASES = {
    "Manchester City": ["manchester city", "man city", "man. city"],
    "Manchester United": ["manchester united", "man united", "man utd", "man. united"],
    "Newcastle": ["newcastle", "newcastle united"],
    "Tottenham": ["tottenham", "tottenham hotspur", "spurs"],
    "Wolves": ["wolves", "wolverhampton", "wolverhampton wanderers"],
    "Brighton": ["brighton", "brighton & hove albion", "brighton and hove albion"],
    "Nottingham Forest": ["nottingham forest", "nott'm forest", "nottm forest"],
    "West Ham": ["west ham", "west ham united"],
    "Crystal Palace": ["crystal palace"],
    "Aston Villa": ["aston villa"],
    "Leicester": ["leicester", "leicester city"],
    "Bournemouth": ["bournemouth", "afc bournemouth"],
    "Fulham": ["fulham"],
    "Everton": ["everton"],
    "Liverpool": ["liverpool"],
    "Arsenal": ["arsenal"],
    "Chelsea": ["chelsea"],
    "Ipswich": ["ipswich", "ipswich town"],
    "Sunderland": ["sunderland"],
    "Burnley": ["burnley"],
    "Leeds": ["leeds", "leeds united"],
    "Sheffield Utd": ["sheffield utd", "sheffield united", "sheffield"],
    "Southampton": ["southampton"],
}


def normalize_name(name: str) -> str:
    if not name:
        return ""
    name = unicodedata.normalize("NFKD", name).encode("ASCII", "ignore").decode("ASCII")
    return name.lower().strip()


def team_in_fixture(team_name: str, fixture_str: str) -> bool:
    """Check if team_name matches any part of the fixture string."""
    if not team_name or not fixture_str:
        return False
    fixture_lower = fixture_str.lower()
    team_lower = team_name.lower()

    # Direct check
    if team_lower in fixture_lower:
        return True

    # Check aliases
    for canonical, aliases in TEAM_ALIASES.items():
        if normalize_name(canonical) == normalize_name(team_name) or normalize_name(team_name) in [normalize_name(a) for a in aliases]:
            for alias in aliases:
                if alias in fixture_lower:
                    return True
            if normalize_name(canonical) in fixture_lower:
                return True
    return False


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
    - Corrects team name if player transferred
    - REMOVES predictions where corrected team doesn't match fixture
    If squad data unavailable, return predictions unchanged.
    """
    squad_data = get_squad_data()
    if not squad_data:
        print("[SquadValidator] Squad data unavailable, returning predictions unchanged")
        return predictions

    corrections = 0
    removals = 0
    validated = []
    for pred in predictions:
        player_name = pred.get("player_name") or pred.get("player", "")
        if not player_name:
            validated.append(pred)
            continue

        current_team = find_player_team(player_name)
        if current_team is None:
            validated.append(pred)
            continue

        claimed_team = pred.get("team", "")
        fixture = pred.get("game") or pred.get("fixture", "")

        if normalize_name(current_team) != normalize_name(claimed_team):
            # Player's real team differs from prediction team
            # Check if their REAL team is in this fixture
            if fixture and not team_in_fixture(current_team, fixture):
                # Player shouldn't be in this fixture at all - REMOVE
                print(f"[SquadValidator] REMOVED {player_name}: plays for {current_team}, not in fixture '{fixture}'")
                removals += 1
                continue
            else:
                # Real team IS in the fixture - just correct the team name
                corrected = pred.copy()
                corrected["team"] = current_team
                print(f"[SquadValidator] Corrected {player_name}: {claimed_team} -> {current_team}")
                corrections += 1
                validated.append(corrected)
        else:
            validated.append(pred)

    print(f"[SquadValidator] Validated {len(validated)} predictions, {corrections} corrections, {removals} removals")
    return validated
