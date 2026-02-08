#!/usr/bin/env python3
"""
train_yc_v12.py — Cross-Model YC with Underdog Factor + xG
===========================================================

Research-backed: "statistically significant negative correlation between
relative strength and number of bookings" (Hargreaves & Powell, York).
"Foul play induced by losing position is an important influence on cards."

New features over v7:
  1. team_underdog_factor — log(opp_win_prob / team_win_prob) from match odds
     Underdogs foul more → more cards
  2. team_xga_l5 — rolling xG against (goals expected to concede)
     Teams conceding more play more desperately → more fouls
  3. opponent_xg_l5 — opponent's rolling xG (how dangerous they are)
"""

import os, sys, pickle, warnings, time
import numpy as np
import pandas as pd
import requests
from datetime import datetime

import lightgbm as lgb
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

# v12 = v7 (30) + 3 new cross-model features
YC_V12_FEATURES = [
    # v7 core (30)
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
    # NEW v12 (3)
    "team_underdog_factor",  # >0 = underdog, <0 = favourite
    "team_xga_l5",           # rolling expected goals conceded
    "opponent_xg_l5",        # opponent's rolling xG scored
]

assert len(YC_V12_FEATURES) == 33

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

# ═══════════════════════════════════════════════════════════════════════
# DATA PULL
# ═══════════════════════════════════════════════════════════════════════

def pull_data():
    print("1. Pulling player_match_stats...")
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

def pull_match_odds():
    """Pull fulltime result odds → compute underdog factor per fixture."""
    print("   Pulling match result odds...")
    t0 = time.time()
    # Paginate by season to avoid timeouts
    all_rows = []
    for start, end in [("2022-08-01","2023-06-30"),("2023-07-01","2024-06-30"),
                       ("2024-07-01","2025-06-30"),("2025-07-01","2026-06-30")]:
        offset = 0
        while True:
            r = requests.get(f"{SUPABASE_URL}/rest/v1/historical_odds", headers=_headers(),
                params={"select":"match_date,home_team,away_team,label,odds_value",
                        "market_id":"eq.1",
                        "and":f"(match_date.gte.{start},match_date.lte.{end})",
                        "order":"match_date.asc","limit":"1000","offset":str(offset)}, timeout=60)
            if r.status_code != 200: break
            rows = r.json()
            all_rows.extend(rows)
            if len(rows) < 1000: break
            offset += 1000
    print(f"   {len(all_rows):,} match odds rows in {time.time()-t0:.1f}s")

    # Compute avg implied probability per (date, home, away, label)
    odds_df = pd.DataFrame(all_rows)
    if len(odds_df) == 0:
        return {}
    odds_df["odds_value"] = pd.to_numeric(odds_df["odds_value"], errors="coerce")
    odds_df = odds_df[odds_df["odds_value"] > 1].copy()
    odds_df["implied"] = 1.0 / odds_df["odds_value"]

    # Average across bookmakers
    avg = odds_df.groupby(["match_date", "home_team", "away_team", "label"])["implied"].mean().reset_index()
    avg = avg.pivot_table(index=["match_date", "home_team", "away_team"],
                          columns="label", values="implied", fill_value=0.33).reset_index()

    # Compute underdog factor for each team
    # For home team: log(away_win / home_win). Positive = home is underdog
    # For away team: log(home_win / away_win). Positive = away is underdog
    result = {}
    for _, row in avg.iterrows():
        date = str(row["match_date"])[:10]
        home = row["home_team"]
        away = row["away_team"]
        home_win = row.get("Home", 0.33)
        away_win = row.get("Away", 0.33)
        # Avoid log(0)
        home_win = max(home_win, 0.05)
        away_win = max(away_win, 0.05)
        # Underdog factor: positive means you're the underdog
        result[(date, home)] = np.log(away_win / home_win)  # home's factor
        result[(date, away)] = np.log(home_win / away_win)  # away's factor

    print(f"   {len(result):,} (date, team) underdog factors computed")
    return result

def pull_xg_data():
    """Pull match_xg_data for rolling xG/xGA per team."""
    print("   Pulling match xG data...")
    rows = _paginate("match_xg_data",
                     "match_date,home_team,away_team,home_xg,away_xg",
                     "match_date.asc")
    # Convert to per-team rows
    team_xg = []
    for r in rows:
        hxg = r.get("home_xg")
        axg = r.get("away_xg")
        if hxg is None or axg is None: continue
        date = r["match_date"]
        team_xg.append({"team": r["home_team"], "date": date, "xg": float(hxg), "xga": float(axg)})
        team_xg.append({"team": r["away_team"], "date": date, "xg": float(axg), "xga": float(hxg)})

    xg_df = pd.DataFrame(team_xg)
    xg_df["date"] = pd.to_datetime(xg_df["date"])
    xg_df = xg_df.sort_values("date")

    # Compute rolling 5-match xG and xGA per team
    xg_lookup = {}  # (team, date_str) -> {xg_l5, xga_l5}
    for team, group in xg_df.groupby("team"):
        group = group.sort_values("date").reset_index(drop=True)
        xgs = group["xg"].values
        xgas = group["xga"].values
        dates = group["date"].values
        for i in range(len(group)):
            prior_xg = xgs[max(0, i-5):i]
            prior_xga = xgas[max(0, i-5):i]
            d = str(dates[i])[:10]
            xg_lookup[(team, d)] = {
                "xg_l5": float(np.mean(prior_xg)) if len(prior_xg) > 0 else 1.3,
                "xga_l5": float(np.mean(prior_xga)) if len(prior_xga) > 0 else 1.3,
            }

    print(f"   {len(xg_lookup):,} (team, date) xG entries")
    return xg_lookup

# ═══════════════════════════════════════════════════════════════════════
# FEATURE COMPUTATION
# ═══════════════════════════════════════════════════════════════════════

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
                    if v>0: feat["games_since_last_card"][ix]=j;f_=True;break
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
        y=ref_map.get(n,_REF_YPG_MEAN) if n else _REF_YPG_MEAN
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
    df["team_defensive_pressure"]=a1;df["team_cards_last_5"]=a2
    df["opponent_avg_cards"]=a3;df["opponent_fouls_tendency"]=a4
    return df

def compute_cross_model_features(df, underdog_map, xg_lookup):
    """Add underdog factor + xG features."""
    print("\n5. Cross-model features (underdog + xG)...")

    # Team name mapping for odds table names → our names
    ODDS_TO_OUR = {
        "Newcastle United": "Newcastle", "West Ham United": "West Ham",
        "Wolverhampton Wanderers": "Wolves", "Brighton & Hove Albion": "Brighton",
        "Brighton and Hove Albion": "Brighton", "Tottenham Hotspur": "Tottenham",
        "Leicester City": "Leicester", "Ipswich Town": "Ipswich",
        "Leeds United": "Leeds", "Sheffield United": "Sheffield Utd",
        "Nottingham Forest": "Nottingham Forest",
    }

    # Build normalized underdog map
    norm_underdog = {}
    for (date, team), val in underdog_map.items():
        norm_team = ODDS_TO_OUR.get(team, team)
        norm_underdog[(date, norm_team)] = val

    df["_date_str"] = df["match_date"].dt.strftime("%Y-%m-%d")

    # Underdog factor
    udf = np.zeros(len(df))
    matched = 0
    for ix, row in df.iterrows():
        key = (row["_date_str"], row["team"])
        if key in norm_underdog:
            udf[ix] = norm_underdog[key]
            matched += 1
    df["team_underdog_factor"] = udf
    print(f"   Underdog factor matched: {matched:,}/{len(df):,} ({matched/len(df)*100:.0f}%)")

    # xG features — team's xGA and opponent's xG
    xga = np.full(len(df), 1.3)
    oxg = np.full(len(df), 1.3)
    xg_matched = 0
    for ix, row in df.iterrows():
        d = row["_date_str"]
        # Team xGA (goals this team is expected to concede)
        tk = (row["team"], d)
        if tk in xg_lookup:
            xga[ix] = xg_lookup[tk]["xga_l5"]
            xg_matched += 1
        # Opponent xG (how dangerous is the opponent)
        ok = (row["opponent"], d)
        if ok in xg_lookup:
            oxg[ix] = xg_lookup[ok]["xg_l5"]

    df["team_xga_l5"] = xga
    df["opponent_xg_l5"] = oxg
    df.drop(columns=["_date_str"], inplace=True)
    print(f"   xG matched: {xg_matched:,}/{len(df):,} ({xg_matched/len(df)*100:.0f}%)")
    print(f"   team_underdog_factor: mean={df['team_underdog_factor'].mean():.3f}, "
          f"std={df['team_underdog_factor'].std():.3f}")
    print(f"   team_xga_l5: mean={df['team_xga_l5'].mean():.3f}")
    print(f"   opponent_xg_l5: mean={df['opponent_xg_l5'].mean():.3f}")
    return df

# ═══════════════════════════════════════════════════════════════════════
# TRAINING + EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def train_and_evaluate(df, features):
    df["yc_binary"] = (df["yellow_cards"] >= 1).astype(int)
    df = df[df["career_games"] >= 3].reset_index(drop=True)

    n = len(df)
    s1, s2 = int(n * 0.60), int(n * 0.80)
    X_tr, y_tr = df[features].values[:s1], df["yc_binary"].values[:s1]
    X_cal, y_cal = df[features].values[s1:s2], df["yc_binary"].values[s1:s2]
    df_test = df.iloc[s2:].copy()
    X_te, y_te = df_test[features].values, df_test["yc_binary"].values

    print(f"\n   Train: {len(y_tr):,} | Cal: {len(y_cal):,} | Test: {len(y_te):,}")

    # Tune
    print("\n6. Training LightGBM...")
    configs = [
        {"n_estimators":700,"learning_rate":0.01,"max_depth":5,"num_leaves":20,"min_child_samples":50},
        {"n_estimators":1000,"learning_rate":0.008,"max_depth":5,"num_leaves":20,"min_child_samples":40},
        {"n_estimators":500,"learning_rate":0.02,"max_depth":6,"num_leaves":31,"min_child_samples":30},
        {"n_estimators":800,"learning_rate":0.01,"max_depth":6,"num_leaves":31,"min_child_samples":25},
    ]
    best_auc, best_model = 0, None
    for i, cfg in enumerate(configs):
        m = lgb.LGBMClassifier(objective="binary",subsample=0.8,colsample_bytree=0.8,
            reg_alpha=0.3,reg_lambda=0.3,random_state=42,verbosity=-1,n_jobs=-1,**cfg)
        m.fit(X_tr, y_tr)
        auc = roc_auc_score(y_cal, m.predict_proba(X_cal)[:,1])
        marker = " *" if auc > best_auc else ""
        print(f"   Config {i+1}: AUC={auc:.4f}{marker}")
        if auc > best_auc: best_auc, best_model = auc, m

    # Calibrate
    cal = CalibratedClassifierCV(best_model, cv="prefit", method="isotonic")
    cal.fit(X_cal, y_cal)

    # Feature importance
    imp = best_model.feature_importances_
    print(f"\n   Feature importance (top 15):")
    for f, v in sorted(zip(features, imp), key=lambda x: -x[1])[:15]:
        bar = "█" * int(v / max(imp) * 25)
        print(f"     {f:30s} {v:6.0f}  {bar}")

    # Test
    probs = np.clip(cal.predict_proba(X_te)[:,1], 0, 0.50)
    auc = roc_auc_score(y_te, probs)
    brier = brier_score_loss(y_te, probs)
    print(f"\n7. TEST: AUC={auc:.4f}, Brier={brier:.4f}")

    # Full backtest
    df_test["pred"] = probs
    total_hits, total_picks, total_booked = 0, 0, 0
    for fid, match in df_test.groupby("af_fixture_id"):
        played = match[match["minutes_played"] > 0]
        if len(played) < 5: continue
        booked = set(played[played["yc_binary"]==1]["player_name"].values)
        top5 = played.nlargest(5, "pred")
        hits = set(top5["player_name"].values) & booked
        total_hits += len(hits); total_picks += len(top5); total_booked += len(booked)

    hr = total_hits / max(total_picks, 1)
    br = total_booked / max(len(df_test[df_test["minutes_played"]>0]), 1)
    nm = df_test["af_fixture_id"].nunique()

    print(f"\n8. BACKTEST ({nm} matches):")
    print(f"   Top-5 hits: {total_hits}/{total_picks} = {hr*100:.1f}% ({hr/max(br,0.001):.2f}x)")

    print(f"\n{'='*75}")
    print(f"COMPARISON:")
    print(f"  v7  LGBM:       AUC=0.6968 | 272/1350 = 20.1% (1.67x)")
    print(f"  v10 Ensemble:   AUC=0.6986 | 278/1350 = 20.6% (1.71x)")
    print(f"  v12 +underdog:  AUC={auc:.4f} | {total_hits}/{total_picks} = {hr*100:.1f}% ({hr/max(br,0.001):.2f}x)")

    # Save
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "epl_yellow_cards_v12.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": cal, "features": features, "version": "v12",
                     "trained_at": datetime.now().isoformat(),
                     "test_auc": auc, "test_brier": brier, "hit_rate": hr}, f)
    print(f"\n   Saved: {model_path}")

def main():
    t_start = time.time()
    print("="*75)
    print("YC v12 — CROSS-MODEL (Underdog Factor + xG)")
    print("="*75)

    df = pull_data()
    ref_map = pull_referees()
    underdog_map = pull_match_odds()
    xg_lookup = pull_xg_data()

    df = compute_player_features(df)
    df = compute_match_features(df, ref_map)
    df = compute_team_features(df)
    df = compute_cross_model_features(df, underdog_map, xg_lookup)

    train_and_evaluate(df, YC_V12_FEATURES)

    print(f"\nDONE — {time.time()-t_start:.1f}s")

if __name__ == "__main__":
    main()
