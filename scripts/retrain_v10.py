#!/usr/bin/env python3
"""
retrain_v10.py — Automated weekly retraining of YC v10 ensemble
===============================================================

Pulls all data from Supabase, computes features, trains LightGBM + XGBoost
ensemble with the exact same architecture as the original v10, saves pkl.

Designed to run weekly via GitHub Actions after ingest_gameweek.py.

Usage:
    python scripts/retrain_v10.py
"""

import os, sys, pickle, warnings, time
import numpy as np
import pandas as pd
import requests
from datetime import datetime

import lightgbm as lgb
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")

# Add repo root to path for shared_features
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from shared_features.ensemble import EnsembleYCModel

# ── Config ──────────────────────────────────────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://kijtxzvbvhgswpahmvua.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "sb_secret_8qWDEuaM0lh95i_CwBgl8A_MgxI1vQK")

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

FEATURES = [
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

# v10 best hyperparameters (frozen from original tuning)
LGBM_PARAMS = {
    "n_estimators": 700, "learning_rate": 0.01, "max_depth": 5,
    "num_leaves": 20, "min_child_samples": 50,
    "objective": "binary", "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.3, "reg_lambda": 0.3, "random_state": 42,
    "verbosity": -1, "n_jobs": -1,
}
XGB_PARAMS = {
    "n_estimators": 500, "learning_rate": 0.02, "max_depth": 4,
    "min_child_weight": 50, "gamma": 0.1,
    "objective": "binary:logistic", "eval_metric": "auc",
    "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.5, "reg_lambda": 1.0, "random_state": 42,
    "verbosity": 0, "n_jobs": -1,
}
BLEND_WEIGHT = 0.55  # LGBM weight


# ── Data pull ───────────────────────────────────────────────────────────

def _headers():
    return {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json"}

def _paginate(table, columns, order="match_date.asc"):
    all_rows, offset = [], 0
    while True:
        r = requests.get(f"{SUPABASE_URL}/rest/v1/{table}", headers=_headers(),
            params={"select": columns, "order": order, "limit": "1000", "offset": str(offset)},
            timeout=60)
        r.raise_for_status()
        rows = r.json()
        all_rows.extend(rows)
        if len(rows) < 1000: break
        offset += 1000
    return all_rows

def pull_data():
    print("1. Pulling data from Supabase...")
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


# ── Feature computation (identical to v7/v10 training) ──────────────────

def compute_player_features(df):
    print("2. Computing player features...")
    n = len(df)
    feat = {f: np.zeros(n) for f in [
        "career_card_rate","cards_last_10","card_rate_last_10","career_games",
        "games_since_last_card","recent_card_intensity",
        "fouls_committed_avg_5","fouls_committed_avg_10","career_fouls_committed_rate",
        "tackles_avg_5","duels_avg_5","minutes_l5","days_since_last_match"]}
    for pid, grp in df.groupby("af_player_id"):
        idx = grp.index.values
        yc,fc,tk,du,mi,dt = (grp["yellow_cards"].values,grp["fouls_committed"].values,
            grp["tackles_total"].values,grp["duels_total"].values,
            grp["minutes_played"].values,grp["match_date"].values)
        ay,af,at,ad,am = [],[],[],[],[]
        for i, ix in enumerate(idx):
            hy=ay[-MAX_HISTORY:][::-1]; hf=af[-MAX_HISTORY:][::-1]
            ht=at[-MAX_HISTORY:][::-1]; hd=ad[-MAX_HISTORY:][::-1]
            hm=am[-MAX_HISTORY:][::-1]; np_=len(hy)
            if np_>0:
                feat["career_games"][ix]=np_
                feat["career_card_rate"][ix]=sum(hy)/np_
                l10=hy[:10]
                feat["cards_last_10"][ix]=sum(l10)
                feat["card_rate_last_10"][ix]=np.mean(l10)
                feat["recent_card_intensity"][ix]=feat["card_rate_last_10"][ix]*feat["cards_last_10"][ix]
                f_=False
                for j,v in enumerate(hy):
                    if v>0: feat["games_since_last_card"][ix]=j; f_=True; break
                if not f_: feat["games_since_last_card"][ix]=np_
                feat["fouls_committed_avg_5"][ix]=np.mean(hf[:5])
                feat["fouls_committed_avg_10"][ix]=np.mean(hf[:10])
                feat["career_fouls_committed_rate"][ix]=sum(hf)/np_
                feat["tackles_avg_5"][ix]=np.mean(ht[:5])
                feat["duels_avg_5"][ix]=np.mean(hd[:5])
                feat["minutes_l5"][ix]=np.mean(hm[:5])
                if i>0: feat["days_since_last_match"][ix]=min((dt[i]-dt[i-1])/np.timedelta64(1,"D"),30)
                else: feat["days_since_last_match"][ix]=7
            else: feat["days_since_last_match"][ix]=7
            ay.append(int(yc[i]));af.append(int(fc[i]))
            at.append(int(tk[i]));ad.append(int(du[i]));am.append(int(mi[i]))
    for k,v in feat.items(): df[k]=v
    df["is_short_rest"]=(df["days_since_last_match"]<4).astype(int)
    return df

def compute_match_features(df, ref_map):
    print("3. Computing match features...")
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
        y=ref_map.get(n,_REF_YPG_MEAN) if n else _REF_YPG_MEAN
        return(y-_REF_YPG_MEAN)/_REF_YPG_STD,y
    rr=df["referee"].apply(lambda n:_ref(n.split(",")[0].strip()if n else""))
    df["referee_strictness"]=rr.apply(lambda x:x[0])
    df["cards_per_game"]=rr.apply(lambda x:x[1])
    return df

def compute_team_features(df):
    print("4. Computing team features...")
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
    df["team_defensive_pressure"]=a1;df["team_cards_last_5"]=a2
    df["opponent_avg_cards"]=a3;df["opponent_fouls_tendency"]=a4
    return df


# ── Training ────────────────────────────────────────────────────────────

def train_ensemble(X_train, y_train, X_cal, y_cal):
    """Train LightGBM + XGBoost ensemble with fixed v10 hyperparameters."""
    print("\n5. Training ensemble...")

    # LightGBM
    lgbm = lgb.LGBMClassifier(**LGBM_PARAMS)
    lgbm.fit(X_train, y_train)
    lgbm_auc = roc_auc_score(y_cal, lgbm.predict_proba(X_cal)[:, 1])
    lgbm_cal = CalibratedClassifierCV(lgbm, cv="prefit", method="isotonic")
    lgbm_cal.fit(X_cal, y_cal)
    print(f"   LGBM AUC: {lgbm_auc:.4f}")

    # XGBoost
    xgb_model = xgb.XGBClassifier(**XGB_PARAMS)
    xgb_model.fit(X_train, y_train)
    xgb_auc = roc_auc_score(y_cal, xgb_model.predict_proba(X_cal)[:, 1])
    xgb_cal = CalibratedClassifierCV(xgb_model, cv="prefit", method="isotonic")
    xgb_cal.fit(X_cal, y_cal)
    print(f"   XGB  AUC: {xgb_auc:.4f}")

    # Ensemble
    ensemble = EnsembleYCModel(lgbm_cal, xgb_cal, BLEND_WEIGHT)
    p_ens = ensemble.predict_proba(X_cal)[:, 1]
    ens_auc = roc_auc_score(y_cal, p_ens)
    print(f"   Ensemble AUC: {ens_auc:.4f}")

    return ensemble, ens_auc


# ── Main ────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    print("=" * 60)
    print("  YC v10 WEEKLY RETRAIN")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    df = pull_data()
    ref_map = pull_referees()
    df = compute_player_features(df)
    df = compute_match_features(df, ref_map)
    df = compute_team_features(df)
    df["yc_binary"] = (df["yellow_cards"] >= 1).astype(int)
    df = df[df["career_games"] >= 3].reset_index(drop=True)

    # 75/25 split (more training data for weekly retrain)
    n = len(df)
    s1 = int(n * 0.75)
    X_train = df[FEATURES].values[:s1]
    y_train = df["yc_binary"].values[:s1]
    X_cal = df[FEATURES].values[s1:]
    y_cal = df["yc_binary"].values[s1:]

    print(f"\n   Total rows: {n:,} | Train: {len(y_train):,} | Cal: {len(y_cal):,}")
    print(f"   YC base rate: {df['yc_binary'].mean():.4f}")

    ensemble, auc = train_ensemble(X_train, y_train, X_cal, y_cal)

    # Save
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "epl_yellow_cards_v10.pkl")
    model_data = {
        "model": ensemble,
        "features": FEATURES,
        "version": "v10-ensemble",
        "trained_at": datetime.now().isoformat(),
        "training_rows": n,
        "base_rate": float(df["yc_binary"].mean()),
        "test_auc": auc,
        "ref_ypg_mean": _REF_YPG_MEAN,
        "ref_ypg_std": _REF_YPG_STD,
    }
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  RETRAIN COMPLETE — {elapsed:.1f}s")
    print(f"  AUC: {auc:.4f} | Rows: {n:,}")
    print(f"  Saved: {os.path.basename(model_path)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
