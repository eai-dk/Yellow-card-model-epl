#!/usr/bin/env python3
"""
train_yc_v7.py — Optimized Yellow Card Model
=============================================

Improvements over v6:
  1. NEW FEATURES: fouls_committed history (strongest predictor!),
     tackles, duels, minutes, rest — all legitimately pre-match.
  2. HYPERPARAMETER TUNING: grid search with time-series validation.
  3. BETTER CALIBRATION: compare isotonic vs sigmoid, clip artifacts.
  4. INTEGRATED BACKTEST: runs backtest on recent matches at the end.

All features use ONLY historical data (before match date) — no leakage.
"""

import os, sys, pickle, warnings, time
import numpy as np
import pandas as pd
import requests
from collections import defaultdict
from datetime import datetime

import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

SUPABASE_URL = "https://kijtxzvbvhgswpahmvua.supabase.co"
SUPABASE_KEY = "sb_secret_8qWDEuaM0lh95i_CwBgl8A_MgxI1vQK"

BIG_SIX = {"Arsenal", "Chelsea", "Liverpool", "Manchester City", "Manchester United", "Tottenham"}
RIVALRIES = {
    frozenset({"Arsenal", "Tottenham"}), frozenset({"Arsenal", "Chelsea"}),
    frozenset({"Liverpool", "Manchester United"}), frozenset({"Liverpool", "Everton"}),
    frozenset({"Manchester City", "Manchester United"}), frozenset({"Chelsea", "Tottenham"}),
    frozenset({"Crystal Palace", "Brighton"}), frozenset({"Newcastle", "Sunderland"}),
    frozenset({"Aston Villa", "Wolves"}), frozenset({"West Ham", "Tottenham"}),
}

_REF_YPG_MEAN = 4.0504
_REF_YPG_STD = 0.8485
MAX_HISTORY = 50
TEAM_LOOKBACK = 5

# ─── V7 Feature List (28 features) ───────────────────────────────────
# Original 21 from v5/v6 PLUS 7 new features from unused data
YC_V7_FEATURES = [
    # === Player card history (6) ===
    "career_card_rate", "cards_last_10", "card_rate_last_10",
    "career_games", "games_since_last_card", "recent_card_intensity",
    # === NEW: Player fouling history (3) — THE BIGGEST MISSING SIGNAL ===
    "fouls_committed_avg_5", "fouls_committed_avg_10", "career_fouls_committed_rate",
    # === NEW: Player physicality (2) ===
    "tackles_avg_5", "duels_avg_5",
    # === NEW: Playing time exposure (1) ===
    "minutes_l5",
    # === Position (4) ===
    "is_defender", "is_midfielder", "is_forward", "is_goalkeeper",
    # === Home/Away (2) ===
    "is_home", "is_away",
    # === Match context (3) ===
    "is_rivalry_match", "is_big6_match", "high_stakes_match",
    # === Referee (2) ===
    "referee_strictness", "cards_per_game",
    # === Team/Opponent aggregates (4) ===
    "team_defensive_pressure", "team_cards_last_5",
    "opponent_avg_cards", "opponent_fouls_tendency",
    # === Season/Rest (3) ===
    "late_season", "days_since_last_match", "is_short_rest",
]

assert len(YC_V7_FEATURES) == 30


# ═══════════════════════════════════════════════════════════════════════
# DATA PULL
# ═══════════════════════════════════════════════════════════════════════

def _headers():
    return {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json"}


def pull_all_data():
    """Pull all player_match_stats with expanded columns."""
    columns = (
        "af_player_id,player_name,team,opponent,match_date,season,"
        "position,is_home,minutes_played,yellow_cards,"
        "fouls_committed,fouls_drawn,tackles_total,duels_total,duels_won,"
        "af_fixture_id,referee"
    )
    all_rows, offset = [], 0
    print("1. Pulling data from Supabase...")
    t0 = time.time()
    while True:
        resp = requests.get(f"{SUPABASE_URL}/rest/v1/player_match_stats", headers=_headers(),
            params={"select": columns, "order": "match_date.asc,af_fixture_id.asc",
                    "limit": "1000", "offset": str(offset)}, timeout=60)
        resp.raise_for_status()
        rows = resp.json()
        all_rows.extend(rows)
        print(f"   {len(all_rows):,} rows...", end="\r")
        if len(rows) < 1000: break
        offset += 1000

    df = pd.DataFrame(all_rows)
    for col in ["yellow_cards", "minutes_played", "fouls_committed", "fouls_drawn",
                "tackles_total", "duels_total", "duels_won"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    df["is_home"] = df["is_home"].astype(bool).astype(int)
    df["position"] = df["position"].fillna("M").str[0].str.upper()
    df["referee"] = df["referee"].fillna("")
    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values(["match_date", "af_fixture_id", "af_player_id"]).reset_index(drop=True)

    print(f"   {len(df):,} rows in {time.time()-t0:.1f}s | "
          f"YC rate: {df['yellow_cards'].clip(0,1).mean():.4f}")
    return df


def pull_referees():
    resp = requests.get(f"{SUPABASE_URL}/rest/v1/referees", headers=_headers(),
        params={"select": "referee_name,yellows_per_match", "limit": "200"}, timeout=30)
    return {r["referee_name"]: float(r.get("yellows_per_match", _REF_YPG_MEAN) or _REF_YPG_MEAN)
            for r in resp.json()}


# ═══════════════════════════════════════════════════════════════════════
# FEATURE COMPUTATION
# ═══════════════════════════════════════════════════════════════════════

def compute_player_features(df):
    """Compute ALL player-level rolling features using only prior data."""
    print("\n2. Computing player rolling features...")
    n = len(df)

    # Feature arrays
    feat = {f: np.zeros(n) for f in [
        "career_card_rate", "cards_last_10", "card_rate_last_10", "career_games",
        "games_since_last_card", "recent_card_intensity",
        "fouls_committed_avg_5", "fouls_committed_avg_10", "career_fouls_committed_rate",
        "tackles_avg_5", "duels_avg_5", "minutes_l5",
        "days_since_last_match",
    ]}

    count = 0
    for pid, group in df.groupby("af_player_id"):
        count += 1
        if count % 500 == 0:
            print(f"   Player {count}...", end="\r")

        idxs = group.index.values
        ycs = group["yellow_cards"].values
        fcs = group["fouls_committed"].values
        tks = group["tackles_total"].values
        dus = group["duels_total"].values
        mins = group["minutes_played"].values
        dates = group["match_date"].values

        # Running history buffers (append as we go)
        all_ycs, all_fcs, all_tks, all_dus, all_mins = [], [], [], [], []

        for i, idx in enumerate(idxs):
            # History: prior games capped at MAX_HISTORY, most-recent first
            hist_yc = all_ycs[-MAX_HISTORY:][::-1]
            hist_fc = all_fcs[-MAX_HISTORY:][::-1]
            hist_tk = all_tks[-MAX_HISTORY:][::-1]
            hist_du = all_dus[-MAX_HISTORY:][::-1]
            hist_mi = all_mins[-MAX_HISTORY:][::-1]
            n_prior = len(hist_yc)

            if n_prior > 0:
                # Card features (same as v6)
                feat["career_games"][idx] = n_prior
                feat["career_card_rate"][idx] = sum(hist_yc) / n_prior
                l10_yc = hist_yc[:10]
                feat["cards_last_10"][idx] = sum(l10_yc)
                feat["card_rate_last_10"][idx] = np.mean(l10_yc)
                feat["recent_card_intensity"][idx] = feat["card_rate_last_10"][idx] * feat["cards_last_10"][idx]

                # Games since last card
                found = False
                for j, yc_val in enumerate(hist_yc):
                    if yc_val > 0:
                        feat["games_since_last_card"][idx] = j
                        found = True
                        break
                if not found:
                    feat["games_since_last_card"][idx] = n_prior

                # NEW: Fouls committed history
                l5_fc = hist_fc[:5]
                l10_fc = hist_fc[:10]
                feat["fouls_committed_avg_5"][idx] = np.mean(l5_fc) if len(l5_fc) > 0 else 0
                feat["fouls_committed_avg_10"][idx] = np.mean(l10_fc) if len(l10_fc) > 0 else 0
                feat["career_fouls_committed_rate"][idx] = sum(hist_fc) / n_prior

                # NEW: Tackles, duels
                l5_tk = hist_tk[:5]
                l5_du = hist_du[:5]
                feat["tackles_avg_5"][idx] = np.mean(l5_tk) if len(l5_tk) > 0 else 0
                feat["duels_avg_5"][idx] = np.mean(l5_du) if len(l5_du) > 0 else 0

                # NEW: Minutes
                l5_mi = hist_mi[:5]
                feat["minutes_l5"][idx] = np.mean(l5_mi) if len(l5_mi) > 0 else 0

                # NEW: Days since last match
                if i > 0:
                    delta = (dates[i] - dates[i-1]) / np.timedelta64(1, "D")
                    feat["days_since_last_match"][idx] = min(delta, 30)
                else:
                    feat["days_since_last_match"][idx] = 7
            else:
                feat["days_since_last_match"][idx] = 7

            # Update state
            all_ycs.append(int(ycs[i]))
            all_fcs.append(int(fcs[i]))
            all_tks.append(int(tks[i]))
            all_dus.append(int(dus[i]))
            all_mins.append(int(mins[i]))

    print(f"   {count:,} players processed")
    for k, v in feat.items():
        df[k] = v

    # is_short_rest (binary)
    df["is_short_rest"] = (df["days_since_last_match"] < 4).astype(int)
    return df


def compute_match_features(df, ref_map):
    """Position, home/away, rivalry, referee features."""
    print("\n3. Computing match context features...")

    df["is_defender"] = (df["position"] == "D").astype(int)
    df["is_midfielder"] = (df["position"] == "M").astype(int)
    df["is_forward"] = (df["position"] == "F").astype(int)
    df["is_goalkeeper"] = (df["position"] == "G").astype(int)
    df["is_away"] = 1 - df["is_home"]

    df["is_rivalry_match"] = df.apply(
        lambda r: int(frozenset({r["team"], r["opponent"]}) in RIVALRIES), axis=1)
    df["is_big6_match"] = (df["team"].isin(BIG_SIX) & df["opponent"].isin(BIG_SIX)).astype(int)
    df["late_season"] = df["match_date"].dt.month.isin([3, 4, 5]).astype(int)

    # Approximate round numbers for high_stakes
    fix_dates = df.groupby(["af_fixture_id", "team", "season"])["match_date"].first().reset_index().sort_values("match_date")
    fix_dates["round_num"] = fix_dates.groupby(["team", "season"]).cumcount() + 1
    rmap = dict(zip(zip(fix_dates["af_fixture_id"], fix_dates["team"]), fix_dates["round_num"]))
    df["_rnd"] = df.apply(lambda r: rmap.get((r["af_fixture_id"], r["team"]), 15), axis=1)
    df["high_stakes_match"] = ((df["_rnd"] >= 30) | (df["is_rivalry_match"] == 1)).astype(int)
    df.drop(columns=["_rnd"], inplace=True)

    # Referee
    def _ref(name):
        ypg = ref_map.get(name, _REF_YPG_MEAN) if name else _REF_YPG_MEAN
        return (ypg - _REF_YPG_MEAN) / _REF_YPG_STD, ypg
    ref_res = df["referee"].apply(lambda n: _ref(n.split(",")[0].strip() if n else ""))
    df["referee_strictness"] = ref_res.apply(lambda x: x[0])
    df["cards_per_game"] = ref_res.apply(lambda x: x[1])
    return df


def compute_team_features(df):
    """Team aggregates: defensive pressure, cards, opponent fouls tendency."""
    print("\n4. Computing team aggregate features...")

    # Per-fixture, per-team stats
    ft = df.groupby(["af_fixture_id", "team"]).agg(
        match_date=("match_date", "first"),
        total_yc=("yellow_cards", "sum"),
        total_fc=("fouls_committed", "sum"),
        n_players=("af_player_id", "nunique"),
        n_rows=("af_player_id", "count"),
    ).reset_index()

    # Rolling team stats
    team_agg = {}
    for team_name, group in ft.groupby("team"):
        group = group.sort_values("match_date").reset_index(drop=True)
        for i in range(len(group)):
            prior = group.iloc[max(0, i - TEAM_LOOKBACK):i]
            if len(prior) > 0:
                ty = prior["total_yc"].sum()
                tf = prior["total_fc"].sum()
                tr = prior["n_rows"].sum()
                tp = prior["n_players"].sum()
                tdp = ty / max(tr, 1)
                tcl5 = ty / max(tp, 1)
                ofc = tf / max(tr, 1)  # opponent fouls tendency
            else:
                tdp, tcl5, ofc = 0.1, 0.5, 1.0
            fid = group.iloc[i]["af_fixture_id"]
            team_agg[(fid, team_name)] = {"tdp": tdp, "tcl5": tcl5, "ofc": ofc}

    # Map to rows
    tdp_arr = np.full(len(df), 0.1)
    tcl5_arr = np.full(len(df), 0.5)
    oac_arr = np.full(len(df), 0.1)
    oft_arr = np.full(len(df), 1.0)

    for idx, row in df.iterrows():
        fid, team, opp = row["af_fixture_id"], row["team"], row["opponent"]
        ta = team_agg.get((fid, team))
        if ta:
            tdp_arr[idx] = ta["tdp"]
            tcl5_arr[idx] = ta["tcl5"]
        oa = team_agg.get((fid, opp))
        if oa:
            oac_arr[idx] = oa["tdp"]
            oft_arr[idx] = oa["ofc"]

    df["team_defensive_pressure"] = tdp_arr
    df["team_cards_last_5"] = tcl5_arr
    df["opponent_avg_cards"] = oac_arr
    df["opponent_fouls_tendency"] = oft_arr
    return df


# ═══════════════════════════════════════════════════════════════════════
# HYPERPARAMETER TUNING + TRAINING
# ═══════════════════════════════════════════════════════════════════════

def tune_and_train(X_train, y_train, X_cal, y_cal, features):
    """Grid search over hyperparams, then train final model with calibration."""
    print("\n5. Hyperparameter tuning...")

    configs = [
        {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 5, "num_leaves": 20, "min_child_samples": 50},
        {"n_estimators": 500, "learning_rate": 0.02, "max_depth": 6, "num_leaves": 31, "min_child_samples": 30},
        {"n_estimators": 800, "learning_rate": 0.01, "max_depth": 6, "num_leaves": 31, "min_child_samples": 25},
        {"n_estimators": 500, "learning_rate": 0.02, "max_depth": 5, "num_leaves": 25, "min_child_samples": 40},
        {"n_estimators": 600, "learning_rate": 0.015, "max_depth": 7, "num_leaves": 40, "min_child_samples": 20},
        {"n_estimators": 400, "learning_rate": 0.025, "max_depth": 6, "num_leaves": 31, "min_child_samples": 35},
        {"n_estimators": 700, "learning_rate": 0.01, "max_depth": 5, "num_leaves": 20, "min_child_samples": 50},
        {"n_estimators": 500, "learning_rate": 0.02, "max_depth": 7, "num_leaves": 50, "min_child_samples": 20},
    ]

    best_auc = 0
    best_config = None
    best_model = None

    for i, cfg in enumerate(configs):
        lgbm = lgb.LGBMClassifier(
            objective="binary", subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.3, reg_lambda=0.3, random_state=42,
            verbosity=-1, n_jobs=-1, **cfg)
        lgbm.fit(X_train, y_train)
        probs = lgbm.predict_proba(X_cal)[:, 1]
        auc = roc_auc_score(y_cal, probs)
        brier = brier_score_loss(y_cal, probs)
        print(f"   Config {i+1}: AUC={auc:.4f} Brier={brier:.4f} "
              f"(trees={cfg['n_estimators']}, lr={cfg['learning_rate']}, "
              f"depth={cfg['max_depth']}, leaves={cfg['num_leaves']})")
        if auc > best_auc:
            best_auc = auc
            best_config = cfg
            best_model = lgbm

    print(f"\n   Best: AUC={best_auc:.4f} — {best_config}")

    # Calibrate with both isotonic and sigmoid, pick better one
    print("\n6. Calibration comparison...")

    cal_iso = CalibratedClassifierCV(best_model, cv="prefit", method="isotonic")
    cal_iso.fit(X_cal, y_cal)
    p_iso = cal_iso.predict_proba(X_cal)[:, 1]

    cal_sig = CalibratedClassifierCV(best_model, cv="prefit", method="sigmoid")
    cal_sig.fit(X_cal, y_cal)
    p_sig = cal_sig.predict_proba(X_cal)[:, 1]

    brier_iso = brier_score_loss(y_cal, p_iso)
    brier_sig = brier_score_loss(y_cal, p_sig)
    print(f"   Isotonic — Brier: {brier_iso:.4f}, max: {p_iso.max():.4f}, mean: {p_iso.mean():.4f}")
    print(f"   Sigmoid  — Brier: {brier_sig:.4f}, max: {p_sig.max():.4f}, mean: {p_sig.mean():.4f}")

    if brier_iso <= brier_sig:
        print("   → Using ISOTONIC calibration")
        final_model = cal_iso
    else:
        print("   → Using SIGMOID calibration")
        final_model = cal_sig

    # Feature importance
    print("\n   Feature importance (gain):")
    importances = best_model.feature_importances_
    feat_imp = sorted(zip(features, importances), key=lambda x: -x[1])
    for feat, imp in feat_imp:
        bar = "█" * int(imp / max(importances) * 30)
        print(f"     {feat:30s} {imp:6.0f}  {bar}")

    return final_model, best_model, best_config


# ═══════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def evaluate(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    # Clip isotonic artifacts
    probs_clipped = np.clip(probs, 0, 0.45)

    auc = roc_auc_score(y_test, probs_clipped)
    brier = brier_score_loss(y_test, probs_clipped)
    ll = log_loss(y_test, probs_clipped)

    print(f"\n7. TEST SET EVALUATION")
    print(f"   AUC-ROC:     {auc:.4f}")
    print(f"   Brier Score: {brier:.4f}")
    print(f"   Log Loss:    {ll:.4f}")
    print(f"   P(YC) — mean: {probs_clipped.mean():.4f}, median: {np.median(probs_clipped):.4f}, "
          f"max: {probs_clipped.max():.4f}")
    print(f"   Base rate:    {y_test.mean():.4f}")

    for p in [5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"     p{p:2d}: {np.percentile(probs_clipped, p):.4f}")

    # Calibration
    print(f"\n   Calibration:")
    bins = [0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.45]
    for i in range(len(bins) - 1):
        mask = (probs_clipped >= bins[i]) & (probs_clipped < bins[i + 1])
        n = mask.sum()
        if n > 0:
            pm = probs_clipped[mask].mean()
            am = y_test[mask].mean()
            print(f"     [{bins[i]*100:4.0f}%-{bins[i+1]*100:4.0f}%): n={n:5d}, "
                  f"pred={pm*100:.1f}%, actual={am*100:.1f}%, diff={pm-am:+.4f}")

    return {"auc": auc, "brier": brier, "probs": probs_clipped}


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    print("=" * 75)
    print("YC v7 MODEL — OPTIMIZED TRAINING")
    print("=" * 75)

    df = pull_all_data()
    ref_map = pull_referees()

    df = compute_player_features(df)
    df = compute_match_features(df, ref_map)
    df = compute_team_features(df)

    # Binarize target
    df["yc_binary"] = (df["yellow_cards"] >= 1).astype(int)

    # Filter: min 3 prior games
    before = len(df)
    df = df[df["career_games"] >= 3].reset_index(drop=True)
    print(f"\n   Filtered: {before:,} → {len(df):,} rows")

    # Feature ranges
    print(f"\n   Feature ranges:")
    for feat in YC_V7_FEATURES:
        vals = df[feat].values
        print(f"     {feat:30s}  [{vals.min():.4f}, {vals.max():.4f}]  mean={vals.mean():.4f}")

    # Time-based split: 60/20/20
    n = len(df)
    s1, s2 = int(n * 0.60), int(n * 0.80)
    X_train, y_train = df[YC_V7_FEATURES].values[:s1], df["yc_binary"].values[:s1]
    X_cal, y_cal = df[YC_V7_FEATURES].values[s1:s2], df["yc_binary"].values[s1:s2]
    X_test, y_test = df[YC_V7_FEATURES].values[s2:], df["yc_binary"].values[s2:]

    print(f"\n   Train: {len(y_train):,} ({y_train.mean():.4f})")
    print(f"   Cal:   {len(y_cal):,} ({y_cal.mean():.4f})")
    print(f"   Test:  {len(y_test):,} ({y_test.mean():.4f})")
    print(f"   Dates: {df.iloc[0]['match_date'].date()} → {df.iloc[s1-1]['match_date'].date()} | "
          f"{df.iloc[s1]['match_date'].date()} → {df.iloc[s2-1]['match_date'].date()} | "
          f"{df.iloc[s2]['match_date'].date()} → {df.iloc[-1]['match_date'].date()}")

    # Tune + Train
    model, raw_lgbm, best_cfg = tune_and_train(X_train, y_train, X_cal, y_cal, YC_V7_FEATURES)

    # Evaluate
    metrics = evaluate(model, X_test, y_test)

    # Sanity: top 20 predictions
    test_df = df.iloc[s2:].copy()
    test_df["pred"] = metrics["probs"]
    top = test_df.nlargest(20, "pred")
    print(f"\n   Top 20 predictions:")
    print(f"   {'Player':25s} {'Team':15s} P    {'FC_avg5':>7s} Actual")
    for _, r in top.iterrows():
        print(f"   {r['player_name'][:25]:25s} {r['team'][:15]:15s} "
              f"{r['pred']*100:4.1f}% {r['fouls_committed_avg_5']:7.2f}  "
              f"{'YES' if r['yc_binary'] else 'no'}")

    # By position
    print(f"\n   By position:")
    for pos in ["D", "M", "F", "G"]:
        mask = test_df["position"] == pos
        if mask.sum() > 0:
            print(f"     {pos}: pred={test_df.loc[mask,'pred'].mean()*100:.1f}%, "
                  f"actual={test_df.loc[mask,'yc_binary'].mean()*100:.1f}%")

    # ── Save model ─────────────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "epl_yellow_cards_v7.pkl")
    model_data = {
        "model": model,
        "features": YC_V7_FEATURES,
        "version": "v7",
        "trained_at": datetime.now().isoformat(),
        "training_rows": len(df),
        "base_rate": float(df["yc_binary"].mean()),
        "test_auc": metrics["auc"],
        "test_brier": metrics["brier"],
        "ref_ypg_mean": _REF_YPG_MEAN,
        "ref_ypg_std": _REF_YPG_STD,
        "best_config": best_cfg,
    }
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    elapsed = time.time() - t_start
    print(f"\n{'='*75}")
    print(f"DONE — {elapsed:.1f}s | Model: {model_path}")
    print(f"v6 AUC: 0.6606 → v7 AUC: {metrics['auc']:.4f}")
    print(f"{'='*75}")


if __name__ == "__main__":
    main()
