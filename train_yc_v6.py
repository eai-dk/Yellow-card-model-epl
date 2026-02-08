#!/usr/bin/env python3
"""
train_yc_v6.py — Yellow Card Model v6 Training Pipeline
========================================================

Pulls audited data from Supabase, computes features using the EXACT same
logic as the production FeatureEngine, trains LightGBM with isotonic
calibration.

Usage:
    python3 train_yc_v6.py

Output:
    - epl_yellow_cards_v6.pkl   (model + features + metadata)
    - yc_v6_training_report.txt (evaluation metrics)

Features (21 total, matching YC_V5_FEATURES order):
    Player rolling:  career_card_rate, cards_last_10, card_rate_last_10,
                     career_games, games_since_last_card
    Position:        is_defender, is_midfielder, is_forward, is_goalkeeper
    Home/Away:       is_home, is_away
    Match context:   is_rivalry_match, is_big6_match, high_stakes_match
    Referee:         referee_strictness, cards_per_game
    Team:            team_defensive_pressure, team_cards_last_5
    Opponent:        opponent_avg_cards
    Season:          late_season
    Derived:         recent_card_intensity
"""

import os
import sys
import pickle
import warnings
import time
import numpy as np
import pandas as pd
import requests
from collections import defaultdict
from datetime import datetime

import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, log_loss,
    classification_report, precision_recall_curve, f1_score,
)

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS — must match production shared_features exactly
# ═══════════════════════════════════════════════════════════════════════

SUPABASE_URL = "https://kijtxzvbvhgswpahmvua.supabase.co"
SUPABASE_KEY = "sb_secret_8qWDEuaM0lh95i_CwBgl8A_MgxI1vQK"

YC_V5_FEATURES = [
    "career_card_rate", "cards_last_10", "card_rate_last_10", "career_games",
    "games_since_last_card",
    "is_defender", "is_midfielder", "is_forward", "is_goalkeeper",
    "is_home", "is_away", "is_rivalry_match", "is_big6_match", "high_stakes_match",
    "referee_strictness", "cards_per_game",
    "team_defensive_pressure", "team_cards_last_5",
    "opponent_avg_cards",
    "late_season", "recent_card_intensity",
]

BIG_SIX = {
    "Arsenal", "Chelsea", "Liverpool",
    "Manchester City", "Manchester United", "Tottenham",
}

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

# Referee z-score parameters — from production match_context.py
_REF_YPG_MEAN = 4.0504
_REF_YPG_STD = 0.8485

# Max history window per player (matches production history_limit=50)
MAX_HISTORY = 50

# Team aggregate lookback (matches production limit=5 in get_team_match_history)
TEAM_LOOKBACK_FIXTURES = 5


# ═══════════════════════════════════════════════════════════════════════
# 1. DATA PULL — Supabase with pagination
# ═══════════════════════════════════════════════════════════════════════

def _supabase_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }


def pull_all_data():
    """Pull all player_match_stats from Supabase (paginated, 1000/request)."""
    columns = (
        "af_player_id,player_name,team,opponent,match_date,season,"
        "position,is_home,minutes_played,yellow_cards,"
        "fouls_committed,fouls_drawn,af_fixture_id,referee"
    )
    all_rows = []
    offset = 0
    batch_size = 1000

    print("1. Pulling player_match_stats from Supabase...")
    t0 = time.time()
    while True:
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/player_match_stats",
            headers=_supabase_headers(),
            params={
                "select": columns,
                "order": "match_date.asc,af_fixture_id.asc",
                "limit": str(batch_size),
                "offset": str(offset),
            },
            timeout=60,
        )
        resp.raise_for_status()
        rows = resp.json()
        all_rows.extend(rows)
        print(f"   Fetched {len(all_rows):,} rows...", end="\r")
        if len(rows) < batch_size:
            break
        offset += batch_size

    elapsed = time.time() - t0
    print(f"   Total: {len(all_rows):,} rows in {elapsed:.1f}s")

    df = pd.DataFrame(all_rows)
    # Clean types
    df["yellow_cards"] = pd.to_numeric(df["yellow_cards"], errors="coerce").fillna(0).astype(int)
    df["is_home"] = df["is_home"].astype(bool).astype(int)
    df["minutes_played"] = pd.to_numeric(df["minutes_played"], errors="coerce").fillna(0).astype(int)
    df["fouls_committed"] = pd.to_numeric(df["fouls_committed"], errors="coerce").fillna(0).astype(int)
    df["fouls_drawn"] = pd.to_numeric(df["fouls_drawn"], errors="coerce").fillna(0).astype(int)
    df["position"] = df["position"].fillna("M").str[0].str.upper()
    df["referee"] = df["referee"].fillna("")
    df["match_date"] = pd.to_datetime(df["match_date"])

    # Sort by date for chronological processing
    df = df.sort_values(["match_date", "af_fixture_id", "af_player_id"]).reset_index(drop=True)

    print(f"   Seasons: {sorted(df['season'].unique())}")
    print(f"   Date range: {df['match_date'].min().date()} → {df['match_date'].max().date()}")
    print(f"   YC base rate: {df['yellow_cards'].mean():.4f} ({df['yellow_cards'].sum():,}/{len(df):,})")
    print(f"   Unique players: {df['af_player_id'].nunique():,}")
    print(f"   Unique fixtures: {df['af_fixture_id'].nunique():,}")
    return df


def pull_referees():
    """Pull referee yellows_per_match mapping from Supabase."""
    print("\n2. Pulling referee data...")
    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/referees",
        headers=_supabase_headers(),
        params={"select": "referee_name,yellows_per_match", "limit": "200"},
        timeout=30,
    )
    resp.raise_for_status()
    ref_map = {}
    for r in resp.json():
        name = r.get("referee_name", "")
        ypg = r.get("yellows_per_match")
        if name and ypg is not None:
            ref_map[name] = float(ypg)
    print(f"   {len(ref_map)} referees loaded")
    return ref_map


# ═══════════════════════════════════════════════════════════════════════
# 2. FEATURE COMPUTATION
# ═══════════════════════════════════════════════════════════════════════

def compute_player_rolling_features(df):
    """
    Compute per-player rolling/expanding features using only prior games.

    Replicates shared_features/player_features.py::compute_rolling_stats()
    with the same 50-game history window and identical calculations.
    """
    print("\n3. Computing player rolling features...")
    n = len(df)
    career_card_rate = np.zeros(n)
    cards_last_10 = np.zeros(n)
    card_rate_last_10 = np.zeros(n)
    career_games = np.zeros(n)
    games_since_last_card = np.zeros(n)
    recent_card_intensity = np.zeros(n)

    player_count = 0
    for pid, group in df.groupby("af_player_id"):
        player_count += 1
        if player_count % 500 == 0:
            print(f"   Processing player {player_count}...", end="\r")

        idxs = group.index.values
        ycs = group["yellow_cards"].values  # chronological order

        # Running state
        all_ycs = []  # full history of YC values (most recent last)

        for i, idx in enumerate(idxs):
            # History: prior games only, capped at MAX_HISTORY, most-recent first
            history_ycs = all_ycs[-MAX_HISTORY:][::-1]  # reverse for most-recent first
            n_prior = len(history_ycs)

            if n_prior > 0:
                career_games[idx] = n_prior
                career_card_rate[idx] = sum(history_ycs) / n_prior

                # Last 10 (most recent 10 from history)
                l10 = history_ycs[:10]
                cards_last_10[idx] = sum(l10)
                card_rate_last_10[idx] = np.mean(l10)

                # Games since last card (index in most-recent-first history)
                found = False
                for j, yc_val in enumerate(history_ycs):
                    if yc_val > 0:
                        games_since_last_card[idx] = j
                        found = True
                        break
                if not found:
                    games_since_last_card[idx] = n_prior  # never carded

                # Derived
                recent_card_intensity[idx] = card_rate_last_10[idx] * cards_last_10[idx]
            else:
                # No prior games
                career_games[idx] = 0
                career_card_rate[idx] = 0
                cards_last_10[idx] = 0
                card_rate_last_10[idx] = 0
                games_since_last_card[idx] = 0
                recent_card_intensity[idx] = 0

            # Update state AFTER computing features for this row
            all_ycs.append(int(ycs[i]))

    print(f"   Processed {player_count:,} players")

    df["career_card_rate"] = career_card_rate
    df["cards_last_10"] = cards_last_10
    df["card_rate_last_10"] = card_rate_last_10
    df["career_games"] = career_games
    df["games_since_last_card"] = games_since_last_card
    df["recent_card_intensity"] = recent_card_intensity
    return df


def compute_match_context_features(df, ref_map):
    """
    Compute match context features: position, home/away, rivalry, referee.

    Replicates shared_features/match_context.py::compute_match_context()
    """
    print("\n4. Computing match context features...")

    # Position flags (vectorized)
    df["is_defender"] = (df["position"] == "D").astype(int)
    df["is_midfielder"] = (df["position"] == "M").astype(int)
    df["is_forward"] = (df["position"] == "F").astype(int)
    df["is_goalkeeper"] = (df["position"] == "G").astype(int)

    # Home / Away
    df["is_away"] = 1 - df["is_home"]

    # Rivalry (vectorized via apply)
    df["is_rivalry_match"] = df.apply(
        lambda r: int(frozenset({r["team"], r["opponent"]}) in RIVALRIES), axis=1
    )

    # Big 6 match
    df["is_big6_match"] = (
        df["team"].isin(BIG_SIX) & df["opponent"].isin(BIG_SIX)
    ).astype(int)

    # Late season (March, April, May)
    df["late_season"] = df["match_date"].dt.month.isin([3, 4, 5]).astype(int)

    # High stakes: approximate round from fixture ordering within season
    # For each team+season, count their fixture number
    print("   Computing approximate round numbers...")
    fixture_dates = (
        df.groupby(["af_fixture_id", "team", "season"])["match_date"]
        .first()
        .reset_index()
        .sort_values("match_date")
    )
    fixture_dates["round_num"] = fixture_dates.groupby(["team", "season"]).cumcount() + 1
    round_map = dict(zip(
        zip(fixture_dates["af_fixture_id"], fixture_dates["team"]),
        fixture_dates["round_num"],
    ))
    df["_round_num"] = df.apply(
        lambda r: round_map.get((r["af_fixture_id"], r["team"]), 15), axis=1
    )
    df["high_stakes_match"] = ((df["_round_num"] >= 30) | (df["is_rivalry_match"] == 1)).astype(int)
    df.drop(columns=["_round_num"], inplace=True)

    # Referee features
    print("   Computing referee features...")
    def _ref_features(ref_name):
        if ref_name and ref_name in ref_map:
            raw_ypg = ref_map[ref_name]
        else:
            raw_ypg = _REF_YPG_MEAN
        strictness = (raw_ypg - _REF_YPG_MEAN) / _REF_YPG_STD
        return strictness, raw_ypg

    ref_results = df["referee"].apply(_ref_features)
    df["referee_strictness"] = ref_results.apply(lambda x: x[0])
    df["cards_per_game"] = ref_results.apply(lambda x: x[1])

    return df


def compute_team_aggregate_features(df):
    """
    Compute team-level aggregate features using only prior fixtures.

    Replicates shared_features/engine.py::_compute_aggregate_features()

    For each fixture, computes:
      - team_defensive_pressure: per-player avg YC rate in team's last 5 fixtures
      - team_cards_last_5: total YCs / unique players in team's last 5 fixtures
      - opponent_avg_cards: per-player avg YC rate in opponent's last 5 fixtures
    """
    print("\n5. Computing team aggregate features...")

    # Step 1: Build per-fixture, per-team summary
    fixture_team = df.groupby(["af_fixture_id", "team"]).agg(
        match_date=("match_date", "first"),
        total_yc=("yellow_cards", "sum"),
        n_players=("af_player_id", "nunique"),
        n_rows=("af_player_id", "count"),  # total player-match rows
    ).reset_index()

    # Step 2: For each team, sort fixtures chronologically and compute rolling stats
    team_fixture_agg = {}  # (fixture_id, team) -> {tdp, tcl5}

    for team_name, group in fixture_team.groupby("team"):
        group = group.sort_values("match_date").reset_index(drop=True)

        for i in range(len(group)):
            # Prior fixtures (up to TEAM_LOOKBACK_FIXTURES before this one)
            start = max(0, i - TEAM_LOOKBACK_FIXTURES)
            prior = group.iloc[start:i]

            if len(prior) > 0:
                total_yc = prior["total_yc"].sum()
                total_rows = prior["n_rows"].sum()
                total_unique = prior["n_players"].sum()  # approximate unique players

                # team_defensive_pressure: mean(yc) across all player rows
                # In production: np.mean([r.get("yellow_cards", 0) for r in team_rows])
                tdp = total_yc / max(total_rows, 1)

                # team_cards_last_5: sum(fixture_yc_totals) / unique_players
                # In production: sum(last_5_totals) / num_unique_players
                tcl5 = total_yc / max(total_unique, 1)
            else:
                tdp = 0.1
                tcl5 = 0.5

            fid = group.iloc[i]["af_fixture_id"]
            team_fixture_agg[(fid, team_name)] = {"tdp": tdp, "tcl5": tcl5}

    # Step 3: Map back to player rows
    print("   Mapping team aggregates to player rows...")
    tdp_arr = np.full(len(df), 0.1)
    tcl5_arr = np.full(len(df), 0.5)
    oac_arr = np.full(len(df), 0.1)

    for idx, row in df.iterrows():
        fid = row["af_fixture_id"]
        team = row["team"]
        opponent = row["opponent"]

        team_agg = team_fixture_agg.get((fid, team))
        if team_agg:
            tdp_arr[idx] = team_agg["tdp"]
            tcl5_arr[idx] = team_agg["tcl5"]

        opp_agg = team_fixture_agg.get((fid, opponent))
        if opp_agg:
            oac_arr[idx] = opp_agg["tdp"]

    df["team_defensive_pressure"] = tdp_arr
    df["team_cards_last_5"] = tcl5_arr
    df["opponent_avg_cards"] = oac_arr
    return df


# ═══════════════════════════════════════════════════════════════════════
# 3. MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════

def train_model(X_train, y_train, X_cal, y_cal):
    """
    Train LightGBM classifier + isotonic calibration.

    Architecture:
      1. LGBMClassifier with conservative hyperparameters
      2. CalibratedClassifierCV (isotonic, prefit) for proper probability estimation
    """
    print("\n6. Training LightGBM classifier...")

    lgbm = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.03,
        max_depth=5,
        num_leaves=20,
        min_child_samples=50,
        reg_alpha=0.5,
        reg_lambda=0.5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,  # let calibration handle imbalance
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
    )

    lgbm.fit(X_train, y_train)

    # Raw predictions on calibration set
    raw_probs = lgbm.predict_proba(X_cal)[:, 1]
    raw_auc = roc_auc_score(y_cal, raw_probs)
    raw_brier = brier_score_loss(y_cal, raw_probs)
    print(f"   Raw LightGBM — AUC: {raw_auc:.4f}, Brier: {raw_brier:.4f}")
    print(f"   Raw prob stats — mean: {raw_probs.mean():.4f}, "
          f"median: {np.median(raw_probs):.4f}, "
          f"max: {raw_probs.max():.4f}")

    # Calibrate with isotonic regression
    print("   Applying isotonic calibration...")
    calibrated = CalibratedClassifierCV(lgbm, cv="prefit", method="isotonic")
    calibrated.fit(X_cal, y_cal)

    cal_probs = calibrated.predict_proba(X_cal)[:, 1]
    cal_brier = brier_score_loss(y_cal, cal_probs)
    print(f"   Calibrated — Brier: {cal_brier:.4f}")
    print(f"   Cal prob stats — mean: {cal_probs.mean():.4f}, "
          f"median: {np.median(cal_probs):.4f}, "
          f"max: {cal_probs.max():.4f}")

    # Feature importance
    print("\n   Feature importance (gain):")
    importances = lgbm.feature_importances_
    feat_imp = sorted(zip(YC_V5_FEATURES, importances), key=lambda x: -x[1])
    for feat, imp in feat_imp:
        bar = "█" * int(imp / max(importances) * 30)
        print(f"     {feat:30s} {imp:6.0f}  {bar}")

    return calibrated, lgbm


def evaluate_model(model, X_test, y_test, report_path=None):
    """Comprehensive evaluation on held-out test set."""
    print("\n7. Evaluating on test set...")

    probs = model.predict_proba(X_test)[:, 1]

    # Metrics
    auc = roc_auc_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    logloss = log_loss(y_test, probs)

    print(f"   AUC-ROC:     {auc:.4f}")
    print(f"   Brier Score: {brier:.4f}")
    print(f"   Log Loss:    {logloss:.4f}")

    # Probability distribution
    print(f"\n   P(YC) Distribution on test set:")
    print(f"     Mean:   {probs.mean():.4f}  (base rate: {y_test.mean():.4f})")
    print(f"     Median: {np.median(probs):.4f}")
    print(f"     Std:    {probs.std():.4f}")
    print(f"     Min:    {probs.min():.4f}")
    print(f"     Max:    {probs.max():.4f}")

    # Percentiles
    for p in [5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"     p{p:2d}:    {np.percentile(probs, p):.4f}")

    # Calibration check: bin predictions and compare to actual rate
    print(f"\n   Calibration (predicted vs actual):")
    bins = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 1.0]
    for i in range(len(bins) - 1):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        n_bin = mask.sum()
        if n_bin > 0:
            actual_rate = y_test[mask].mean()
            pred_mean = probs[mask].mean()
            print(f"     [{bins[i]:.2f}, {bins[i+1]:.2f}): n={n_bin:5d}, "
                  f"pred={pred_mean:.4f}, actual={actual_rate:.4f}, "
                  f"diff={pred_mean - actual_rate:+.4f}")

    # Classification report at 0.10 threshold (near base rate)
    preds_010 = (probs >= 0.10).astype(int)
    print(f"\n   Classification Report (threshold=0.10):")
    print(classification_report(y_test, preds_010, target_names=["No YC", "YC"], digits=4))

    # Save report
    if report_path:
        with open(report_path, "w") as f:
            f.write("YC v6 Model Training Report\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"AUC-ROC:     {auc:.4f}\n")
            f.write(f"Brier Score: {brier:.4f}\n")
            f.write(f"Log Loss:    {logloss:.4f}\n\n")
            f.write(f"P(YC) Distribution:\n")
            f.write(f"  Mean:   {probs.mean():.4f}\n")
            f.write(f"  Median: {np.median(probs):.4f}\n")
            f.write(f"  Max:    {probs.max():.4f}\n\n")
            f.write(f"Base rate: {y_test.mean():.4f}\n")
            f.write(f"Test set size: {len(y_test):,}\n")
        print(f"   Report saved to {report_path}")

    return {"auc": auc, "brier": brier, "logloss": logloss, "probs": probs}


# ═══════════════════════════════════════════════════════════════════════
# 4. MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    print("=" * 70)
    print("YC v6 MODEL TRAINING — From Audited Supabase Data")
    print("=" * 70)

    # ── Pull data ──────────────────────────────────────────────────
    df = pull_all_data()
    ref_map = pull_referees()

    # ── Compute features ───────────────────────────────────────────
    df = compute_player_rolling_features(df)
    df = compute_match_context_features(df, ref_map)
    df = compute_team_aggregate_features(df)

    # ── Filter: require at least 3 prior games ─────────────────────
    print("\n   Filtering: requiring career_games >= 3...")
    before = len(df)
    df = df[df["career_games"] >= 3].reset_index(drop=True)
    print(f"   {before:,} → {len(df):,} rows ({before - len(df):,} removed)")

    # ── Binarize target (some rows have yellow_cards > 1) ─────────
    df["yellow_card_binary"] = (df["yellow_cards"] >= 1).astype(int)
    multi_yc = (df["yellow_cards"] > 1).sum()
    if multi_yc > 0:
        print(f"   Note: {multi_yc} rows had yellow_cards > 1, binarized to 1")

    # ── Verify feature ranges ──────────────────────────────────────
    print("\n   Feature ranges (training data):")
    X_all = df[YC_V5_FEATURES].values
    y_all = df["yellow_card_binary"].values
    for i, feat in enumerate(YC_V5_FEATURES):
        vals = X_all[:, i]
        print(f"     {feat:30s}  [{vals.min():.4f}, {vals.max():.4f}]  "
              f"mean={vals.mean():.4f}  median={np.median(vals):.4f}")

    # ── Time-based split: 60% train / 20% calibrate / 20% test ────
    print("\n   Time-based split (60/20/20)...")
    n = len(df)
    split1 = int(n * 0.60)
    split2 = int(n * 0.80)

    X_train = df[YC_V5_FEATURES].values[:split1]
    y_train = df["yellow_card_binary"].values[:split1]
    X_cal = df[YC_V5_FEATURES].values[split1:split2]
    y_cal = df["yellow_card_binary"].values[split1:split2]
    X_test = df[YC_V5_FEATURES].values[split2:]
    y_test = df["yellow_card_binary"].values[split2:]

    print(f"   Train: {len(y_train):,} rows ({y_train.mean():.4f} base rate), "
          f"dates: {df.iloc[0]['match_date'].date()} → {df.iloc[split1-1]['match_date'].date()}")
    print(f"   Cal:   {len(y_cal):,} rows ({y_cal.mean():.4f} base rate), "
          f"dates: {df.iloc[split1]['match_date'].date()} → {df.iloc[split2-1]['match_date'].date()}")
    print(f"   Test:  {len(y_test):,} rows ({y_test.mean():.4f} base rate), "
          f"dates: {df.iloc[split2]['match_date'].date()} → {df.iloc[-1]['match_date'].date()}")

    # ── Train ──────────────────────────────────────────────────────
    calibrated_model, raw_lgbm = train_model(X_train, y_train, X_cal, y_cal)

    # ── Evaluate ───────────────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.join(script_dir, "yc_v6_training_report.txt")
    metrics = evaluate_model(calibrated_model, X_test, y_test, report_path)

    # ── Sanity check: sample predictions ───────────────────────────
    print("\n8. Sanity check — sample predictions:")
    test_df = df.iloc[split2:].copy()
    test_df["pred_prob"] = metrics["probs"]

    # Show top 20 highest predictions
    top = test_df.nlargest(20, "pred_prob")
    print(f"\n   Top 20 predictions:")
    print(f"   {'Player':25s} {'Team':15s} {'Pos':3s} {'Prob':>6s} {'Actual':>6s}")
    print(f"   {'-'*60}")
    for _, row in top.iterrows():
        print(f"   {row['player_name'][:25]:25s} {row['team'][:15]:15s} "
              f"{row['position']:3s} {row['pred_prob']:.4f} "
              f"{'YES' if row['yellow_card_binary'] else 'no'}")

    # Show predictions by position
    print(f"\n   Mean P(YC) by position:")
    for pos in ["D", "M", "F", "G"]:
        mask = test_df["position"] == pos
        if mask.sum() > 0:
            mean_p = test_df.loc[mask, "pred_prob"].mean()
            actual = test_df.loc[mask, "yellow_card_binary"].mean()
            print(f"     {pos}: pred={mean_p:.4f}, actual={actual:.4f}, n={mask.sum()}")

    # ── Save model ─────────────────────────────────────────────────
    model_path = os.path.join(script_dir, "epl_yellow_cards_v6.pkl")
    model_data = {
        "model": calibrated_model,
        "features": YC_V5_FEATURES,
        "version": "v6",
        "trained_at": datetime.now().isoformat(),
        "training_rows": len(df),
        "base_rate": float(y_all.mean()),
        "test_auc": metrics["auc"],
        "test_brier": metrics["brier"],
        "ref_ypg_mean": _REF_YPG_MEAN,
        "ref_ypg_std": _REF_YPG_STD,
    }
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE — {elapsed:.1f}s total")
    print(f"Model saved: {model_path}")
    print(f"Report saved: {report_path}")
    print(f"{'='*70}")

    return calibrated_model, metrics


if __name__ == "__main__":
    main()
