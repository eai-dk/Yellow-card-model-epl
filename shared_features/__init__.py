"""
PitchPredict Shared Feature Engineering Module
================================================
Single source of truth for all 5 models' feature computation.
Pulls data from Supabase tables and computes rolling stats, career rates,
referee strictness, opponent tendencies, etc.

Usage:
    from shared_features import FeatureEngine
    engine = FeatureEngine()
    features = engine.get_player_features(player_id=526, opponent="Arsenal", ...)
"""

from .engine import FeatureEngine

__all__ = ["FeatureEngine"]
__version__ = "1.0.0"

