#!/usr/bin/env python3
"""
train_yc_v11.py — CatBoost with Native Categorical Features
============================================================

Research shows CatBoost outperforms LightGBM/XGBoost for soccer prediction
because it handles categorical features (team, opponent, referee) NATIVELY
using ordered target encoding. This captures interaction effects like:
  - "Ref X cards midfielders more than average"
  - "Team Y gets more cards when playing away"
  - "Opponent Z's style causes more cards for defenders"

Approach:
  1. Same 30 numerical features as v7
  2. ADD team, opponent, referee as raw categorical features
  3. CatBoost handles encoding internally (no information leakage)
  4. Compare against v7 on full 270-match backtest
"""

import os, sys, pickle, warnings, time
import numpy as np
import pandas as pd
import requests
from datetime import datetime

from catboost import CatBoostClassifier, Pool
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")

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
_REF_YPG_MEAN, _REF_YPG_STD = 4.0504, 0.8485
MAX_HISTORY, TEAM_LOOKBACK = 50, 5

# v11 features: v7 numerical + 3 categoricals
YC_V11_NUM_FEATURES = [
    "career_card_rate", "cards_last_10", "card_rate_last_10",
    "career_games", "games_since_last_card", "recent_card_intensity",
    "fouls_committed_avg_5", "fouls_committed_avg_10", "career_fouls_committed_rate",
    "tackles_avg_5", "duels_avg_5", "minutes_l5",
    "is_defender", "is_midfielder", "is_forward", "is_goalkeeper",
    "is_home", "is_away",
    "is_rivalry_match", "is_big6_match", "high_stakes_match",
    "referee_strictness", "cards_per_game",
    "team_defensive_pressure", "team_cards_last_5",
    "opponent_avg_cards", "opponent_fouls_tendency",
    "late_season", "days_since_last_match", "is_short_rest",
]
YC_V11_CAT_FEATURES = ["team", "opponent", "referee_name"]
YC_V11_ALL_FEATURES = YC_V11_NUM_FEATURES + YC_V11_CAT_FEATURES


# ═══════════════════════════════════════════════════════════════════════
# DATA PULL + FEATURES (same as v7)
# ═══════════════════════════════════════════════════════════════════════

def _headers():
    return {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json"}

def _paginate(table, columns, order="match_date.asc", extra=None):
    all_rows, offset = [], 0
    while True:
        p = {"select": columns, "order": order, "limit": "1000", "offset": str(offset)}
        if extra: p.update(extra)
        r = requests.get(f"{SUPABASE_URL}/rest/v1/{table}", headers=_headers(), params=p, timeout=60)
        r.raise_for_status()
        rows = r.json()
        all_rows.extend(rows)
        if len(rows) < 1000: break
        offset += 1000
    return all_rows

def pull_data():
    print("1. Pulling data...")
    t0 = time.time()
    cols = ("af_player_id,player_name,team,opponent,match_date,season,position,is_home,"
            "minutes_played,yellow_cards,fouls_committed,fouls_drawn,tackles_total,"
            "duels_total,duels_won,af_fixture_id,referee")
    rows = _paginate("player_match_stats", cols, "match_date.asc,af_fixture_id.asc")
    df = pd.DataFrame(rows)
    for c in ["yellow_cards", "minutes_played", "fouls_committed", "fouls_drawn",
              "tackles_total", "duels_total", "duels_won"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    df["is_home"] = df["is_home"].astype(bool).astype(int)
    df["position"] = df["position"].fillna("M").str[0].str.upper()
    df["referee"] = df["referee"].fillna("")
    df["match_date"] = pd.to_datetime(df["match_date"])
    # Clean referee name for categorical use
    df["referee_name"] = df["referee"].apply(lambda n: n.split(",")[0].strip() if n else "Unknown")
    df = df.sort_values(["match_date", "af_fixture_id", "af_player_id"]).reset_index(drop=True)
    print(f"   {len(df):,} rows in {time.time()-t0:.1f}s")
    return df

def pull_referees():
    r = requests.get(f"{SUPABASE_URL}/rest/v1/referees", headers=_headers(),
        params={"select": "referee_name,yellows_per_match", "limit": "200"}, timeout=30)
    return {x["referee_name"]: float(x.get("yellows_per_match", _REF_YPG_MEAN) or _REF_YPG_MEAN) for x in r.json()}

def compute_player_features(df):
    print("\n2. Player features...")
    n = len(df)
    feat = {f: np.zeros(n) for f in [
        "career_card_rate", "cards_last_10", "card_rate_last_10", "career_games",
        "games_since_last_card", "recent_card_intensity",
        "fouls_committed_avg_5", "fouls_committed_avg_10", "career_fouls_committed_rate",
        "tackles_avg_5", "duels_avg_5", "minutes_l5", "days_since_last_match"]}
    ct = 0
    for pid, grp in df.groupby("af_player_id"):
        ct += 1
        if ct % 500 == 0: print(f"   {ct}...", end="\r")
        idx = grp.index.values
        yc, fc, tk, du, mi, dt = (grp["yellow_cards"].values, grp["fouls_committed"].values,
            grp["tackles_total"].values, grp["duels_total"].values,
            grp["minutes_played"].values, grp["match_date"].values)
        ay, af, at, ad, am = [], [], [], [], []
        for i, ix in enumerate(idx):
            hy = ay[-MAX_HISTORY:][::-1]; hf = af[-MAX_HISTORY:][::-1]
            ht = at[-MAX_HISTORY:][::-1]; hd = ad[-MAX_HISTORY:][::-1]
            hm = am[-MAX_HISTORY:][::-1]; np_ = len(hy)
            if np_ > 0:
                feat["career_games"][ix] = np_
                feat["career_card_rate"][ix] = sum(hy) / np_
                l10 = hy[:10]
                feat["cards_last_10"][ix] = sum(l10)
                feat["card_rate_last_10"][ix] = np.mean(l10)
                feat["recent_card_intensity"][ix] = feat["card_rate_last_10"][ix] * feat["cards_last_10"][ix]
                f_ = False
                for j, v in enumerate(hy):
                    if v > 0: feat["games_since_last_card"][ix] = j; f_ = True; break
                if not f_: feat["games_since_last_card"][ix] = np_
                feat["fouls_committed_avg_5"][ix] = np.mean(hf[:5])
                feat["fouls_committed_avg_10"][ix] = np.mean(hf[:10])
                feat["career_fouls_committed_rate"][ix] = sum(hf) / np_
                feat["tackles_avg_5"][ix] = np.mean(ht[:5])
                feat["duels_avg_5"][ix] = np.mean(hd[:5])
                feat["minutes_l5"][ix] = np.mean(hm[:5])
                if i > 0:
                    feat["days_since_last_match"][ix] = min((dt[i] - dt[i-1]) / np.timedelta64(1, "D"), 30)
                else:
                    feat["days_since_last_match"][ix] = 7
            else:
                feat["days_since_last_match"][ix] = 7
            ay.append(int(yc[i])); af.append(int(fc[i]))
            at.append(int(tk[i])); ad.append(int(du[i])); am.append(int(mi[i]))
    print(f"   {ct:,} players")
    for k, v in feat.items(): df[k] = v
    df["is_short_rest"] = (df["days_since_last_match"] < 4).astype(int)
    return df

def compute_match_features(df, ref_map):
    print("3. Match features...")
    df["is_defender"] = (df["position"] == "D").astype(int)
    df["is_midfielder"] = (df["position"] == "M").astype(int)
    df["is_forward"] = (df["position"] == "F").astype(int)
    df["is_goalkeeper"] = (df["position"] == "G").astype(int)
    df["is_away"] = 1 - df["is_home"]
    df["is_rivalry_match"] = df.apply(lambda r: int(frozenset({r["team"], r["opponent"]}) in RIVALRIES), axis=1)
    df["is_big6_match"] = (df["team"].isin(BIG_SIX) & df["opponent"].isin(BIG_SIX)).astype(int)
    df["late_season"] = df["match_date"].dt.month.isin([3, 4, 5]).astype(int)
    fd = df.groupby(["af_fixture_id", "team", "season"])["match_date"].first().reset_index().sort_values("match_date")
    fd["rn"] = fd.groupby(["team", "season"]).cumcount() + 1
    rm = dict(zip(zip(fd["af_fixture_id"], fd["team"]), fd["rn"]))
    df["_r"] = df.apply(lambda r: rm.get((r["af_fixture_id"], r["team"]), 15), axis=1)
    df["high_stakes_match"] = ((df["_r"] >= 30) | (df["is_rivalry_match"] == 1)).astype(int)
    df.drop(columns=["_r"], inplace=True)
    def _ref(n):
        y = ref_map.get(n, _REF_YPG_MEAN) if n else _REF_YPG_MEAN
        return (y - _REF_YPG_MEAN) / _REF_YPG_STD, y
    rr = df["referee_name"].apply(_ref)
    df["referee_strictness"] = rr.apply(lambda x: x[0])
    df["cards_per_game"] = rr.apply(lambda x: x[1])
    return df

def compute_team_features(df):
    print("4. Team features...")
    ft = df.groupby(["af_fixture_id", "team"]).agg(
        match_date=("match_date", "first"), total_yc=("yellow_cards", "sum"),
        total_fc=("fouls_committed", "sum"), n_players=("af_player_id", "nunique"),
        n_rows=("af_player_id", "count")).reset_index()
    ta = {}
    for tn, g in ft.groupby("team"):
        g = g.sort_values("match_date").reset_index(drop=True)
        for i in range(len(g)):
            p = g.iloc[max(0, i - TEAM_LOOKBACK):i]
            if len(p) > 0:
                ty, tf, tr, tp = p["total_yc"].sum(), p["total_fc"].sum(), p["n_rows"].sum(), p["n_players"].sum()
                ta[(g.iloc[i]["af_fixture_id"], tn)] = {"tdp": ty/max(tr,1), "tcl5": ty/max(tp,1), "ofc": tf/max(tr,1)}
            else:
                ta[(g.iloc[i]["af_fixture_id"], tn)] = {"tdp": 0.1, "tcl5": 0.5, "ofc": 1.0}
    a1, a2, a3, a4 = np.full(len(df), 0.1), np.full(len(df), 0.5), np.full(len(df), 0.1), np.full(len(df), 1.0)
    for ix, r in df.iterrows():
        t = ta.get((r["af_fixture_id"], r["team"]))
        if t: a1[ix], a2[ix] = t["tdp"], t["tcl5"]
        o = ta.get((r["af_fixture_id"], r["opponent"]))
        if o: a3[ix], a4[ix] = o["tdp"], o["ofc"]
    df["team_defensive_pressure"] = a1
    df["team_cards_last_5"] = a2
    df["opponent_avg_cards"] = a3
    df["opponent_fouls_tendency"] = a4
    return df


# ═══════════════════════════════════════════════════════════════════════
# CATBOOST TRAINING
# ═══════════════════════════════════════════════════════════════════════

def train_catboost(df_train, df_cal, df_test, features, cat_features):
    """Train CatBoost with native categorical handling."""

    cat_indices = [features.index(f) for f in cat_features]

    X_train = df_train[features].copy()
    y_train = df_train["yc_binary"].values
    X_cal = df_cal[features].copy()
    y_cal = df_cal["yc_binary"].values

    # Ensure categoricals are strings
    for cf in cat_features:
        X_train[cf] = X_train[cf].astype(str)
        X_cal[cf] = X_cal[cf].astype(str)

    print("\n5. Training CatBoost with native categoricals...")
    print(f"   Categorical features: {cat_features}")
    print(f"   Categorical indices: {cat_indices}")

    configs = [
        {"iterations": 500, "learning_rate": 0.03, "depth": 5, "l2_leaf_reg": 3},
        {"iterations": 800, "learning_rate": 0.02, "depth": 6, "l2_leaf_reg": 5},
        {"iterations": 1000, "learning_rate": 0.01, "depth": 5, "l2_leaf_reg": 3},
        {"iterations": 500, "learning_rate": 0.03, "depth": 4, "l2_leaf_reg": 7},
        {"iterations": 700, "learning_rate": 0.02, "depth": 5, "l2_leaf_reg": 5},
        {"iterations": 1200, "learning_rate": 0.008, "depth": 5, "l2_leaf_reg": 3},
    ]

    best_auc, best_model = 0, None
    for i, cfg in enumerate(configs):
        cb = CatBoostClassifier(
            cat_features=cat_indices,
            auto_class_weights="Balanced",
            random_seed=42,
            verbose=0,
            **cfg,
        )
        cb.fit(X_train, y_train)
        probs = cb.predict_proba(X_cal)[:, 1]
        auc = roc_auc_score(y_cal, probs)
        brier = brier_score_loss(y_cal, probs)
        marker = " *" if auc > best_auc else ""
        print(f"   Config {i+1}: AUC={auc:.4f} Brier={brier:.4f} "
              f"(iter={cfg['iterations']}, lr={cfg['learning_rate']}, "
              f"depth={cfg['depth']}, l2={cfg['l2_leaf_reg']}){marker}")
        if auc > best_auc:
            best_auc, best_model = auc, cb

    print(f"\n   Best CatBoost AUC: {best_auc:.4f}")

    # Calibrate
    print("\n6. Calibration...")
    cal = CalibratedClassifierCV(best_model, cv="prefit", method="isotonic")
    cal.fit(X_cal, y_cal)
    cal_probs = cal.predict_proba(X_cal)[:, 1]
    print(f"   Calibrated Brier: {brier_score_loss(y_cal, cal_probs):.4f}")
    print(f"   Calibrated max: {cal_probs.max():.4f}, mean: {cal_probs.mean():.4f}")

    # Feature importance
    imp = best_model.feature_importances_
    print("\n   Feature importance:")
    for f, v in sorted(zip(features, imp), key=lambda x: -x[1])[:20]:
        bar = "█" * int(v / max(imp) * 25)
        print(f"     {f:30s} {v:6.1f}  {bar}")

    return cal, best_model


def evaluate_full(model, df_test, features, cat_features, label=""):
    """Full evaluation + per-match top-5 hit rate."""
    X = df_test[features].copy()
    for cf in cat_features:
        X[cf] = X[cf].astype(str)
    y = df_test["yc_binary"].values
    probs = np.clip(model.predict_proba(X)[:, 1], 0, 0.50)

    auc = roc_auc_score(y, probs)
    brier = brier_score_loss(y, probs)

    print(f"\n   {label} — AUC={auc:.4f}, Brier={brier:.4f}")
    print(f"   P(YC): mean={probs.mean():.4f}, max={probs.max():.4f}")

    # Calibration
    bins = [0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    for i in range(len(bins) - 1):
        mask = (probs >= bins[i]) & (probs < bins[i+1])
        n = mask.sum()
        if n > 0:
            print(f"     [{bins[i]*100:.0f}%-{bins[i+1]*100:.0f}%): n={n:5d}, "
                  f"pred={probs[mask].mean()*100:.1f}%, actual={y[mask].mean()*100:.1f}%")

    # Per-match backtest
    df_bt = df_test.copy()
    df_bt["pred"] = probs
    total_hits, total_picks, total_booked = 0, 0, 0
    for fid, match in df_bt.groupby("af_fixture_id"):
        played = match[match["minutes_played"] > 0]
        if len(played) < 5: continue
        booked = set(played[played["yc_binary"] == 1]["player_name"].values)
        top5 = played.nlargest(5, "pred")
        hits = set(top5["player_name"].values) & booked
        total_hits += len(hits)
        total_picks += len(top5)
        total_booked += len(booked)

    hit_rate = total_hits / max(total_picks, 1)
    base_rate = total_booked / max(len(df_bt[df_bt["minutes_played"] > 0]), 1)
    n_matches = df_bt["af_fixture_id"].nunique()
    print(f"   Backtest ({n_matches} matches): {total_hits}/{total_picks} = {hit_rate*100:.1f}% "
          f"({hit_rate/max(base_rate,0.001):.2f}x vs random)")

    return {"auc": auc, "brier": brier, "hit_rate": hit_rate,
            "total_hits": total_hits, "total_picks": total_picks}


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    print("=" * 75)
    print("YC v11 — CATBOOST WITH NATIVE CATEGORICALS")
    print("=" * 75)

    df = pull_data()
    ref_map = pull_referees()
    df = compute_player_features(df)
    df = compute_match_features(df, ref_map)
    df = compute_team_features(df)
    df["yc_binary"] = (df["yellow_cards"] >= 1).astype(int)
    df = df[df["career_games"] >= 3].reset_index(drop=True)

    n = len(df)
    s1, s2 = int(n * 0.60), int(n * 0.80)
    df_train = df.iloc[:s1]
    df_cal = df.iloc[s1:s2]
    df_test = df.iloc[s2:]

    print(f"\n   Train: {len(df_train):,} | Cal: {len(df_cal):,} | Test: {len(df_test):,}")
    print(f"   Unique teams: {df['team'].nunique()}, referees: {df['referee_name'].nunique()}")

    # Train CatBoost
    catboost_cal, catboost_raw = train_catboost(
        df_train, df_cal, df_test, YC_V11_ALL_FEATURES, YC_V11_CAT_FEATURES)

    # Evaluate
    print("\n7. EVALUATION:")
    cb_metrics = evaluate_full(catboost_cal, df_test, YC_V11_ALL_FEATURES, YC_V11_CAT_FEATURES,
                               "CatBoost + categoricals")

    # Compare
    print(f"\n{'='*75}")
    print(f"COMPARISON:")
    print(f"  v7  LGBM (30 num):    AUC=0.6968 | 272/1350 = 20.1% (1.67x)")
    print(f"  v10 Ensemble:         AUC=0.6986 | 278/1350 = 20.6% (1.71x)")
    print(f"  v11 CatBoost+cat:     AUC={cb_metrics['auc']:.4f} | "
          f"{cb_metrics['total_hits']}/{cb_metrics['total_picks']} = "
          f"{cb_metrics['hit_rate']*100:.1f}%")

    # Save
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "epl_yellow_cards_v11.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": catboost_cal, "features": YC_V11_ALL_FEATURES,
            "cat_features": YC_V11_CAT_FEATURES,
            "version": "v11-catboost",
            "trained_at": datetime.now().isoformat(),
            "test_auc": cb_metrics["auc"], "test_brier": cb_metrics["brier"],
            "hit_rate": cb_metrics["hit_rate"],
        }, f)

    print(f"\n{'='*75}")
    print(f"DONE — {time.time()-t_start:.1f}s | {model_path}")
    print(f"{'='*75}")


if __name__ == "__main__":
    main()
