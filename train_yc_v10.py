#!/usr/bin/env python3
"""
train_yc_v10.py — Ensemble Yellow Card Model
=============================================

Key insight: v7-v9 showed that adding features doesn't help.
Different approach: ENSEMBLE of LightGBM + XGBoost on the same v7 features.

Different algorithms find different patterns:
  - LightGBM: leaf-wise splitting, fast, good at sparse features
  - XGBoost: level-wise splitting, stronger regularization

Final prediction = average of both calibrated probabilities.

Also: FULL backtest on 250+ matches instead of 10-match samples.
"""

import os, sys, pickle, warnings, time
import numpy as np
import pandas as pd
import requests
from collections import defaultdict
from datetime import datetime

import lightgbm as lgb
import xgboost as xgb
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

# Same 30 features as v7 — proven to be the optimal set
YC_V10_FEATURES = [
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

# ═══════════════════════════════════════════════════════════════════════
# DATA PULL + FEATURE COMPUTATION (identical to v7)
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
    for c in ["yellow_cards","minutes_played","fouls_committed","fouls_drawn",
              "tackles_total","duels_total","duels_won"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    df["is_home"] = df["is_home"].astype(bool).astype(int)
    df["position"] = df["position"].fillna("M").str[0].str.upper()
    df["referee"] = df["referee"].fillna("")
    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values(["match_date","af_fixture_id","af_player_id"]).reset_index(drop=True)
    print(f"   {len(df):,} rows in {time.time()-t0:.1f}s")
    return df

def pull_referees():
    r = requests.get(f"{SUPABASE_URL}/rest/v1/referees", headers=_headers(),
        params={"select":"referee_name,yellows_per_match","limit":"200"}, timeout=30)
    return {x["referee_name"]: float(x.get("yellows_per_match",_REF_YPG_MEAN) or _REF_YPG_MEAN) for x in r.json()}

def compute_player_features(df):
    print("\n2. Player features...")
    n = len(df)
    feat = {f: np.zeros(n) for f in [
        "career_card_rate","cards_last_10","card_rate_last_10","career_games",
        "games_since_last_card","recent_card_intensity",
        "fouls_committed_avg_5","fouls_committed_avg_10","career_fouls_committed_rate",
        "tackles_avg_5","duels_avg_5","minutes_l5","days_since_last_match"]}
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
                feat["career_card_rate"][ix] = sum(hy)/np_
                l10 = hy[:10]
                feat["cards_last_10"][ix] = sum(l10)
                feat["card_rate_last_10"][ix] = np.mean(l10)
                feat["recent_card_intensity"][ix] = feat["card_rate_last_10"][ix]*feat["cards_last_10"][ix]
                f_ = False
                for j,v in enumerate(hy):
                    if v>0: feat["games_since_last_card"][ix]=j; f_=True; break
                if not f_: feat["games_since_last_card"][ix]=np_
                feat["fouls_committed_avg_5"][ix]=np.mean(hf[:5])
                feat["fouls_committed_avg_10"][ix]=np.mean(hf[:10])
                feat["career_fouls_committed_rate"][ix]=sum(hf)/np_
                feat["tackles_avg_5"][ix]=np.mean(ht[:5])
                feat["duels_avg_5"][ix]=np.mean(hd[:5])
                feat["minutes_l5"][ix]=np.mean(hm[:5])
                if i>0:
                    feat["days_since_last_match"][ix]=min((dt[i]-dt[i-1])/np.timedelta64(1,"D"),30)
                else: feat["days_since_last_match"][ix]=7
            else: feat["days_since_last_match"][ix]=7
            ay.append(int(yc[i])); af.append(int(fc[i]))
            at.append(int(tk[i])); ad.append(int(du[i])); am.append(int(mi[i]))
    print(f"   {ct:,} players")
    for k,v in feat.items(): df[k]=v
    df["is_short_rest"]=(df["days_since_last_match"]<4).astype(int)
    return df

def compute_match_features(df, ref_map):
    print("3. Match features...")
    df["is_defender"]=(df["position"]=="D").astype(int)
    df["is_midfielder"]=(df["position"]=="M").astype(int)
    df["is_forward"]=(df["position"]=="F").astype(int)
    df["is_goalkeeper"]=(df["position"]=="G").astype(int)
    df["is_away"]=1-df["is_home"]
    df["is_rivalry_match"]=df.apply(lambda r:int(frozenset({r["team"],r["opponent"]})in RIVALRIES),axis=1)
    df["is_big6_match"]=(df["team"].isin(BIG_SIX)&df["opponent"].isin(BIG_SIX)).astype(int)
    df["late_season"]=df["match_date"].dt.month.isin([3,4,5]).astype(int)
    fd=df.groupby(["af_fixture_id","team","season"])["match_date"].first().reset_index().sort_values("match_date")
    fd["rn"]=fd.groupby(["team","season"]).cumcount()+1
    rm=dict(zip(zip(fd["af_fixture_id"],fd["team"]),fd["rn"]))
    df["_r"]=df.apply(lambda r:rm.get((r["af_fixture_id"],r["team"]),15),axis=1)
    df["high_stakes_match"]=((df["_r"]>=30)|(df["is_rivalry_match"]==1)).astype(int)
    df.drop(columns=["_r"],inplace=True)
    def _ref(n):
        y=ref_map.get(n,_REF_YPG_MEAN)if n else _REF_YPG_MEAN
        return(y-_REF_YPG_MEAN)/_REF_YPG_STD,y
    rr=df["referee"].apply(lambda n:_ref(n.split(",")[0].strip()if n else""))
    df["referee_strictness"]=rr.apply(lambda x:x[0])
    df["cards_per_game"]=rr.apply(lambda x:x[1])
    return df

def compute_team_features(df):
    print("4. Team features...")
    ft=df.groupby(["af_fixture_id","team"]).agg(
        match_date=("match_date","first"),total_yc=("yellow_cards","sum"),
        total_fc=("fouls_committed","sum"),n_players=("af_player_id","nunique"),
        n_rows=("af_player_id","count")).reset_index()
    ta={}
    for tn,g in ft.groupby("team"):
        g=g.sort_values("match_date").reset_index(drop=True)
        for i in range(len(g)):
            p=g.iloc[max(0,i-TEAM_LOOKBACK):i]
            if len(p)>0:
                ty,tf,tr,tp=p["total_yc"].sum(),p["total_fc"].sum(),p["n_rows"].sum(),p["n_players"].sum()
                ta[(g.iloc[i]["af_fixture_id"],tn)]={"tdp":ty/max(tr,1),"tcl5":ty/max(tp,1),"ofc":tf/max(tr,1)}
            else:
                ta[(g.iloc[i]["af_fixture_id"],tn)]={"tdp":0.1,"tcl5":0.5,"ofc":1.0}
    a1,a2,a3,a4=np.full(len(df),0.1),np.full(len(df),0.5),np.full(len(df),0.1),np.full(len(df),1.0)
    for ix,r in df.iterrows():
        t=ta.get((r["af_fixture_id"],r["team"]))
        if t: a1[ix],a2[ix]=t["tdp"],t["tcl5"]
        o=ta.get((r["af_fixture_id"],r["opponent"]))
        if o: a3[ix],a4[ix]=o["tdp"],o["ofc"]
    df["team_defensive_pressure"]=a1; df["team_cards_last_5"]=a2
    df["opponent_avg_cards"]=a3; df["opponent_fouls_tendency"]=a4
    return df

# ═══════════════════════════════════════════════════════════════════════
# ENSEMBLE TRAINING
# ═══════════════════════════════════════════════════════════════════════

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
        # Return as 2-column array for sklearn compatibility
        return np.column_stack([1 - blended, blended])


def train_ensemble(X_train, y_train, X_cal, y_cal, features):
    print("\n5. Training ENSEMBLE (LightGBM + XGBoost)...")

    # ── LightGBM ──────────────────────────────────────────────────
    print("\n   === LightGBM ===")
    lgbm_configs = [
        {"n_estimators": 700, "learning_rate": 0.01, "max_depth": 5, "num_leaves": 20, "min_child_samples": 50},
        {"n_estimators": 1000, "learning_rate": 0.008, "max_depth": 5, "num_leaves": 20, "min_child_samples": 40},
        {"n_estimators": 500, "learning_rate": 0.02, "max_depth": 6, "num_leaves": 31, "min_child_samples": 30},
    ]
    best_lgbm_auc, best_lgbm = 0, None
    for i, cfg in enumerate(lgbm_configs):
        m = lgb.LGBMClassifier(objective="binary", subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.3, reg_lambda=0.3, random_state=42, verbosity=-1, n_jobs=-1, **cfg)
        m.fit(X_train, y_train)
        auc = roc_auc_score(y_cal, m.predict_proba(X_cal)[:, 1])
        print(f"   LGBM {i+1}: AUC={auc:.4f}")
        if auc > best_lgbm_auc: best_lgbm_auc, best_lgbm = auc, m

    lgbm_cal = CalibratedClassifierCV(best_lgbm, cv="prefit", method="isotonic")
    lgbm_cal.fit(X_cal, y_cal)
    lgbm_brier = brier_score_loss(y_cal, lgbm_cal.predict_proba(X_cal)[:, 1])
    print(f"   Best LGBM: AUC={best_lgbm_auc:.4f}, cal Brier={lgbm_brier:.4f}")

    # ── XGBoost ───────────────────────────────────────────────────
    print("\n   === XGBoost ===")
    xgb_configs = [
        {"n_estimators": 500, "learning_rate": 0.02, "max_depth": 4, "min_child_weight": 50, "gamma": 0.1},
        {"n_estimators": 800, "learning_rate": 0.01, "max_depth": 5, "min_child_weight": 30, "gamma": 0.05},
        {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 4, "min_child_weight": 40, "gamma": 0.1},
        {"n_estimators": 1000, "learning_rate": 0.008, "max_depth": 4, "min_child_weight": 50, "gamma": 0.15},
    ]
    best_xgb_auc, best_xgb = 0, None
    for i, cfg in enumerate(xgb_configs):
        m = xgb.XGBClassifier(objective="binary:logistic", eval_metric="auc",
            subsample=0.8, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=1.0,
            random_state=42, verbosity=0, n_jobs=-1, **cfg)
        m.fit(X_train, y_train)
        auc = roc_auc_score(y_cal, m.predict_proba(X_cal)[:, 1])
        print(f"   XGB {i+1}: AUC={auc:.4f}")
        if auc > best_xgb_auc: best_xgb_auc, best_xgb = auc, m

    xgb_cal = CalibratedClassifierCV(best_xgb, cv="prefit", method="isotonic")
    xgb_cal.fit(X_cal, y_cal)
    xgb_brier = brier_score_loss(y_cal, xgb_cal.predict_proba(X_cal)[:, 1])
    print(f"   Best XGB: AUC={best_xgb_auc:.4f}, cal Brier={xgb_brier:.4f}")

    # ── Find optimal blend weight ─────────────────────────────────
    print("\n   === Blending ===")
    p_lgbm = lgbm_cal.predict_proba(X_cal)[:, 1]
    p_xgb = xgb_cal.predict_proba(X_cal)[:, 1]

    best_w, best_blend_auc = 0.5, 0
    for w in np.arange(0.3, 0.8, 0.05):
        blended = w * p_lgbm + (1 - w) * p_xgb
        auc = roc_auc_score(y_cal, blended)
        if auc > best_blend_auc:
            best_blend_auc, best_w = auc, w

    print(f"   Optimal weight: LGBM={best_w:.2f}, XGB={1-best_w:.2f}")
    print(f"   Ensemble cal AUC: {best_blend_auc:.4f}")
    print(f"   vs LGBM alone:    {best_lgbm_auc:.4f}")
    print(f"   vs XGB alone:     {best_xgb_auc:.4f}")

    ensemble = EnsembleYCModel(lgbm_cal, xgb_cal, best_w)

    # Feature importance (from LightGBM)
    imp = best_lgbm.feature_importances_
    print("\n   Feature importance (LGBM):")
    for f, v in sorted(zip(features, imp), key=lambda x: -x[1])[:15]:
        print(f"     {f:30s} {v:6.0f}")

    return ensemble, best_lgbm, best_xgb, best_w


# ═══════════════════════════════════════════════════════════════════════
# EVALUATION + FULL BACKTEST
# ═══════════════════════════════════════════════════════════════════════

def evaluate_and_backtest(model, df_test, features):
    """Full evaluation + per-match top-5 hit rate on ALL test matches."""
    X = df_test[features].values
    y = df_test["yc_binary"].values
    probs = np.clip(model.predict_proba(X)[:, 1], 0, 0.50)

    auc = roc_auc_score(y, probs)
    brier = brier_score_loss(y, probs)

    print(f"\n6. TEST SET (global):")
    print(f"   AUC={auc:.4f}, Brier={brier:.4f}")
    print(f"   P(YC): mean={probs.mean():.4f}, max={probs.max():.4f}")

    # Calibration
    bins = [0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    for i in range(len(bins) - 1):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        n = mask.sum()
        if n > 0:
            print(f"     [{bins[i]*100:.0f}%-{bins[i+1]*100:.0f}%): n={n:5d}, "
                  f"pred={probs[mask].mean()*100:.1f}%, actual={y[mask].mean()*100:.1f}%")

    # ── FULL per-match backtest ──────────────────────────────────
    print(f"\n7. FULL PER-MATCH BACKTEST ({df_test['af_fixture_id'].nunique()} matches):")

    df_test = df_test.copy()
    df_test["pred"] = probs

    total_hits, total_picks, total_booked = 0, 0, 0
    match_results = []

    for fid, match_df in df_test.groupby("af_fixture_id"):
        played = match_df[match_df["minutes_played"] > 0]
        if len(played) < 5:
            continue

        booked = set(played[played["yc_binary"] == 1]["player_name"].values)
        top5 = played.nlargest(5, "pred")
        top5_names = set(top5["player_name"].values)
        hits = top5_names & booked

        total_hits += len(hits)
        total_picks += len(top5)
        total_booked += len(booked)

    hit_rate = total_hits / max(total_picks, 1)
    base_rate = total_booked / max(len(df_test[df_test["minutes_played"] > 0]), 1)
    n_matches = df_test["af_fixture_id"].nunique()

    print(f"   Matches: {n_matches}")
    print(f"   Top-5 hits: {total_hits}/{total_picks} = {hit_rate*100:.1f}%")
    print(f"   Bookings caught: {total_hits}/{total_booked} = {total_hits/max(total_booked,1)*100:.1f}%")
    print(f"   Base rate: {base_rate*100:.1f}%")
    print(f"   Edge vs random: {hit_rate/max(base_rate,0.001):.2f}x")

    return {"auc": auc, "brier": brier, "hit_rate": hit_rate, "probs": probs,
            "total_hits": total_hits, "total_picks": total_picks}


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    print("=" * 75)
    print("YC v10 MODEL — ENSEMBLE (LightGBM + XGBoost)")
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
    X_tr, y_tr = df[YC_V10_FEATURES].values[:s1], df["yc_binary"].values[:s1]
    X_cal, y_cal = df[YC_V10_FEATURES].values[s1:s2], df["yc_binary"].values[s1:s2]
    df_test = df.iloc[s2:].copy()

    print(f"\n   Train: {len(y_tr):,} | Cal: {len(y_cal):,} | Test: {len(df_test):,}")

    # Train ensemble
    ensemble, lgbm, xgb_model, blend_w = train_ensemble(X_tr, y_tr, X_cal, y_cal, YC_V10_FEATURES)

    # Also evaluate v7-style (LGBM only) for comparison
    print("\n   --- v7-style LGBM-only baseline ---")
    lgbm_only = CalibratedClassifierCV(lgbm, cv="prefit", method="isotonic")
    lgbm_only.fit(X_cal, y_cal)
    lgbm_metrics = evaluate_and_backtest(lgbm_only, df_test, YC_V10_FEATURES)

    print("\n   --- v10 ENSEMBLE ---")
    ensemble_metrics = evaluate_and_backtest(ensemble, df_test, YC_V10_FEATURES)

    # Compare
    print(f"\n{'='*75}")
    print(f"COMPARISON:")
    print(f"  LGBM only:  AUC={lgbm_metrics['auc']:.4f} | "
          f"Top-5 hits: {lgbm_metrics['total_hits']}/{lgbm_metrics['total_picks']} "
          f"({lgbm_metrics['hit_rate']*100:.1f}%)")
    print(f"  ENSEMBLE:   AUC={ensemble_metrics['auc']:.4f} | "
          f"Top-5 hits: {ensemble_metrics['total_hits']}/{ensemble_metrics['total_picks']} "
          f"({ensemble_metrics['hit_rate']*100:.1f}%)")

    # Save the better model
    if ensemble_metrics["auc"] >= lgbm_metrics["auc"]:
        save_model = ensemble
        save_version = "v10-ensemble"
        save_metrics = ensemble_metrics
        print(f"\n  → ENSEMBLE wins!")
    else:
        save_model = lgbm_only
        save_version = "v10-lgbm"
        save_metrics = lgbm_metrics
        print(f"\n  → LGBM wins (ensemble didn't help)")

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "epl_yellow_cards_v10.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": save_model, "features": YC_V10_FEATURES, "version": save_version,
            "trained_at": datetime.now().isoformat(),
            "test_auc": save_metrics["auc"], "test_brier": save_metrics["brier"],
            "hit_rate": save_metrics["hit_rate"],
            "ref_ypg_mean": _REF_YPG_MEAN, "ref_ypg_std": _REF_YPG_STD,
        }, f)

    print(f"\n{'='*75}")
    print(f"DONE — {time.time()-t_start:.1f}s | {model_path}")
    print(f"{'='*75}")


if __name__ == "__main__":
    main()
