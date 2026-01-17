"""
Configuration for YC Model
"""

# API-Football Pro Key
API_KEY = "YOUR_API_KEY_HERE"  # Replace with your key

# API Base URL
BASE_URL = "https://v3.football.api-sports.io"

# EPL League ID
LEAGUE_ID = 39
SEASON = 2025

# ============================================================================
# REFEREE CLASSIFICATIONS
# ============================================================================

# VERY STRICT REFS (5+ yellows per game) - TIER 1
VERY_STRICT_REFS = [
    "S. Allison",       # 6.5 yc/game
    "Tim Robinson",     # 5.2 yc/game
    "C. Kavanagh",      # 5.2 yc/game
    "Chris Kavanagh",   # 5.2 yc/game
    "T. Robinson",      # 5.1 yc/game
    "S. Barrott",       # 5.1 yc/game
    "Samuel Barrott",   # 5.1 yc/game
]

# STRICT REFS (4+ yellows per game) - TIER 2
STRICT_REFS = [
    "Michael Salisbury",  # 4.9 yc/game
    "J. Brooks",          # 4.8 yc/game
    "John Brooks",        # 4.8 yc/game
    "Stuart Attwell",     # 4.7 yc/game
    "T. Bramall",         # 4.7 yc/game
    "D. Bond",            # 4.6 yc/game
    "Robert Jones",       # 4.5 yc/game
    "Jarred Gillett",     # 4.3 yc/game
    "Anthony Taylor",     # 4.2 yc/game
    "Andy Madley",        # 4.1 yc/game
    "Michael Oliver",     # 4.0 yc/game
]

# Target positions (DEF and MID get more cards)
TARGET_POSITIONS = ["D", "M"]

# Minimum games for player to be considered
MIN_GAMES = 3

# Top N players per team to show
TOP_N_PLAYERS = 5

