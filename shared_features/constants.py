"""
Constants shared across all models.
"""

# Position codes used by SOT and Shots Combo models
POSITION_CODES = {"G": 0, "D": 1, "M": 2, "F": 3}

# Position scores used by Foul model
POSITION_SCORES = {"F": 3, "M": 2, "D": 1, "G": 0}

# Big Six teams (canonical API-Football names)
BIG_SIX = {"Arsenal", "Chelsea", "Liverpool", "Manchester City", "Manchester United", "Tottenham"}

# Rivalry pairs (both directions) — used by YC model
RIVALRIES = {
    frozenset({"Arsenal", "Tottenham"}),
    frozenset({"Arsenal", "Chelsea"}),
    frozenset({"Liverpool", "Manchester United"}),
    frozenset({"Liverpool", "Everton"}),
    frozenset({"Manchester City", "Manchester United"}),
    frozenset({"Chelsea", "Tottenham"}),
    frozenset({"Crystal Palace", "Brighton"}),
    frozenset({"Newcastle", "Sunderland"}),
    frozenset({"Aston Villa", "Wolves"}),
    frozenset({"West Ham", "Tottenham"}),
}

# Default defense tiers (1=elite, 5=worst) — can be overridden from teams table
DEFAULT_DEFENSE_TIERS = {
    "Arsenal": 1, "Liverpool": 1, "Manchester City": 1,
    "Chelsea": 2, "Manchester United": 2, "Tottenham": 2, "Newcastle": 2,
    "Aston Villa": 3, "Brighton": 3, "West Ham": 3, "Everton": 3,
    "Bournemouth": 4, "Fulham": 4, "Brentford": 4, "Crystal Palace": 4,
    "Nottingham Forest": 4, "Wolves": 4, "Leicester": 4,
    "Ipswich": 5, "Southampton": 5,
}

# Default pressing intensity (1=low, 5=high)
DEFAULT_PRESSING = {
    "Liverpool": 5, "Arsenal": 5, "Manchester City": 4, "Brighton": 4,
    "Tottenham": 4, "Chelsea": 3, "Newcastle": 3, "Aston Villa": 3,
    "Manchester United": 3, "Bournemouth": 3, "Brentford": 3,
    "West Ham": 2, "Crystal Palace": 2, "Fulham": 2, "Everton": 2,
    "Wolves": 2, "Nottingham Forest": 2, "Leicester": 2,
    "Ipswich": 1, "Southampton": 1,
}

# Supabase connection
SUPABASE_URL = "https://kijtxzvbvhgswpahmvua.supabase.co"
SUPABASE_KEY = "sb_secret_8qWDEuaM0lh95i_CwBgl8A_MgxI1vQK"

# Feature lists per model (exact order matters for pkl compatibility)
YC_V5_FEATURES = [
    'career_card_rate', 'cards_last_10', 'card_rate_last_10', 'career_games',
    'games_since_last_card',
    'is_defender', 'is_midfielder', 'is_forward', 'is_goalkeeper',
    'is_home', 'is_away', 'is_rivalry_match', 'is_big6_match', 'high_stakes_match',
    'referee_strictness', 'cards_per_game',
    'team_defensive_pressure', 'team_cards_last_5',
    'opponent_avg_cards',
    'late_season', 'recent_card_intensity',
]

FOUL_DRAWN_FEATURES = [
    'fouls_drawn_last_3', 'fouls_drawn_last_5', 'fouls_drawn_last_10',
    'fouls_drawn_avg_3', 'fouls_drawn_avg_5', 'fouls_drawn_avg_10',
    'career_fouls_drawn_rate', 'career_games', 'position_score',
    'is_forward', 'is_midfielder', 'is_defender', 'is_home', 'opponent_fouls_tendency',
]

FOUL_COMMITTED_FEATURES = [
    'fouls_committed_last_3', 'fouls_committed_last_5', 'fouls_committed_last_10',
    'fouls_committed_avg_3', 'fouls_committed_avg_5', 'fouls_committed_avg_10',
    'career_fouls_committed_rate', 'career_games',
    'yellow_cards_last_5', 'yellow_cards_last_10',
    'position_score', 'is_forward', 'is_midfielder', 'is_defender', 'is_home',
    'opponent_fouls_drawn_tendency',
]

SOT_V5_FEATURES = [
    "sot_l5", "sot_l10", "sot_season_avg",
    "shots_l5", "shots_l10",
    "minutes_l5", "minutes_l10",
    "games_season", "position_code", "is_home",
    "opponent_defense_tier", "opponent_pressing", "opponent_weakness",
    "is_non_big6", "is_consistent", "form_surge", "vs_weak_defense",
    "well_rested", "sot_streak_l3", "is_hot",
    "shots_per_90_l5", "shot_accuracy_l5", "rest_factor",
    "team_offensive_strength", "opponent_shots_conceded",
    "minutes_stable", "confidence_score",
]

SHOTS_COMBO_V3_FEATURES = [
    "shots_l5", "shots_l10", "shots_season_avg",
    "sot_l5", "sot_l10",
    "minutes_l5", "minutes_l10",
    "games_season", "position_code", "is_home",
    "opponent_defense_tier", "opponent_pressing", "opponent_weakness",
    "is_non_big6", "is_consistent", "form_surge", "vs_weak_defense",
    "well_rested", "shots_streak_l3", "is_hot",
    "shots_per_90_l5", "shot_accuracy_l5", "goals_l5",
    "expected_shots", "shot_volume_tier",
    "minutes_stable", "confidence_score",
    "days_since_last_match", "matches_last_7_days", "minutes_last_7_days",
    "is_short_rest", "is_congested_week",
    "player_xg_per_shot", "player_npxg_season", "player_xg_overperform",
    "opponent_ppda", "opponent_xga_avg", "opponent_is_high_press",
    "shots_trend", "shot_variance_l10",
]

TEAM_GOALS_V2_FEATURES = [
    'home_xG_l5', 'home_npxG_l5', 'away_xG_l5', 'away_npxG_l5',
    'home_xGA_l5', 'home_npxGA_l5', 'away_xGA_l5', 'away_npxGA_l5',
    'home_ppda_l5', 'away_ppda_l5', 'ppda_diff',
    'home_xG_overperf_l5', 'away_xG_overperf_l5', 'combined_xG_overperf',
    'total_xG_l5', 'home_attack_vs_away_defense', 'away_attack_vs_home_defense',
    'rest_days_home', 'rest_days_away',
]

