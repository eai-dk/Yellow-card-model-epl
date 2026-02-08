"""
Ensemble model wrapper for YC v10.
Averages calibrated LightGBM + XGBoost probabilities.
"""

import numpy as np


class EnsembleYCModel:
    """Wrapper that averages LightGBM + XGBoost calibrated probabilities."""

    def __init__(self, lgbm_cal, xgb_cal, weight_lgbm=0.5):
        self.lgbm = lgbm_cal
        self.xgb = xgb_cal
        self.w = weight_lgbm

    def predict_proba(self, X):
        p1 = self.lgbm.predict_proba(X)[:, 1]
        p2 = self.xgb.predict_proba(X)[:, 1]
        blended = self.w * p1 + (1 - self.w) * p2
        return np.column_stack([1 - blended, blended])
