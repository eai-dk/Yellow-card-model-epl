"""
🔥🔥🔥 ULTIMATE YC MODEL v2.0 - Configuration 🔥🔥🔥

THE FORMULA: STRICT REF + AWAY + DEF/MID = 43% hit rate (+160% ROI)
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
# These refs = bet on BOTH home and away DEF/MID
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
# These refs = bet on AWAY DEF/MID only
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

# ============================================================================
# HIGH-CARD OPPONENTS
# ============================================================================
# Playing AGAINST these teams = +42% more cards for your players
# These teams cause opponents to foul more

HIGH_CARD_OPPONENTS = [
    "Newcastle",        # 12.8% YC rate (+23% lift)
    "Aston Villa",      # 12.4% YC rate (+19% lift)
    "Chelsea",          # 12.1% YC rate (+17% lift)
    "Brighton",         # 12.1% YC rate (+17% lift)
    "Manchester City",  # 12.1% YC rate (+17% lift)
    "Tottenham",        # 11.7% YC rate (+13% lift)
]

# ============================================================================
# FOUL DRAWERS - Players who make opponents get carded
# ============================================================================
# When these players are on the pitch, opposing DEF/MID get carded more

FOUL_DRAWERS = [
    "Bruno Guimarães",   # +26% lift on opponents
    "Jérémy Doku",       # +24% lift
    "Anthony Gordon",    # +20% lift
    "Bukayo Saka",       # +9% lift (but 13.7% on opp DEF!)
    "Jack Grealish",     # +10% lift
    "Jordan Ayew",       # +15% lift
]

# ============================================================================
# HEAD-TO-HEAD CARD MACHINES
# ============================================================================
# These players get carded 75-100% of the time vs specific foul-drawers

H2H_MACHINES = {
    "Ola Aina": ["Bukayo Saka"],           # 100% (3/3)
    "Tyrick Mitchell": ["Bruno Guimarães", "Anthony Gordon"],  # 100%, 75%
    "Marc Cucurella": ["Bukayo Saka"],     # 75% (3/4), 100% with strict ref
    "Rodrigo Bentancur": ["Bruno Guimarães"],  # 100% (3/3)
    "Moisés Caicedo": ["Jack Grealish"],   # 100% (3/3)
}

