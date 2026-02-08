#!/usr/bin/env python3
"""
train_yc_v9.py — Market-Aware Yellow Card Model
================================================

Key innovation: uses bookmaker implied probability as a FEATURE.
The model learns "when is the market wrong?" instead of just
"who gets booked?"

New over v7:
  1. market_implied_prob — bookmaker's YC pricing as input (52K odds rows)
  2. is_starter — minutes_l5 >= 60 (filters sub risk)
  3. Bigger evaluation: full test set + 10-match detailed backtest

Architecture:
  v7 (30 features) + market_implied_prob + is_starter = 32 features
"""

import os, sys, pickle, warnings, time
import unicodedata
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

# v9 = v7 features + market_implied_prob + is_starter
YC_V9_FEATURES = [
    # === v7 core (30) ===
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
    # === NEW v9 (2) ===
    "market_implied_prob",  # bookmaker's YC pricing — the market's opinion
    "is_starter",           # minutes_l5 >= 60 — starter vs sub risk
]

assert len(YC_V9_FEATURES) == 32


# ═══════════════════════════════════════════════════════════════════════
# DATA PULL
# ═══════════════════════════════════════════════════════════════════════

def _headers():
    return {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json"}


def _paginate(table, columns, order="match_date.asc", extra_params=None):
    all_rows, offset = [], 0
    while True:
        params = {"select": columns, "order": order, "limit": "1000", "offset": str(offset)}
        if extra_params:
            params.update(extra_params)
        resp = requests.get(f"{SUPABASE_URL}/rest/v1/{table}", headers=_headers(),
                            params=params, timeout=60)
        resp.raise_for_status()
        rows = resp.json()
        all_rows.extend(rows)
        if len(rows) < 1000:
            break
        offset += 1000
    return all_rows


def _normalize_name(name):
    """Normalize player name for matching between API-Football and SportMonks."""
    if not name:
        return ""
    for c, r in {'\u00f8': 'o', '\u00df': 'ss', '\u0131': 'i', '\u0142': 'l',
                 '\u0111': 'd', '\u00e6': 'ae', '\u0153': 'oe'}.items():
        name = name.replace(c, r)
    name = unicodedata.normalize("NFKD", name).encode("ASCII", "ignore").decode("ASCII")
    return " ".join(name.lower().strip().split())


def pull_all_data():
    columns = (
        "af_player_id,player_name,team,opponent,match_date,season,"
        "position,is_home,minutes_played,yellow_cards,"
        "fouls_committed,fouls_drawn,tackles_total,duels_total,duels_won,"
        "af_fixture_id,referee"
    )
    print("1. Pulling player_match_stats...")
    t0 = time.time()
    all_rows = _paginate("player_match_stats", columns, "match_date.asc,af_fixture_id.asc")

    df = pd.DataFrame(all_rows)
    for col in ["yellow_cards", "minutes_played", "fouls_committed", "fouls_drawn",
                "tackles_total", "duels_total", "duels_won"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    df["is_home"] = df["is_home"].astype(bool).astype(int)
    df["position"] = df["position"].fillna("M").str[0].str.upper()
    df["referee"] = df["referee"].fillna("")
    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values(["match_date", "af_fixture_id", "af_player_id"]).reset_index(drop=True)
    print(f"   {len(df):,} rows in {time.time()-t0:.1f}s")
    return df


def pull_yc_odds():
    """Pull all historical YC booking odds (market_id=64), paginated by season."""
    print("   Pulling YC odds (market_id=64)...")
    t0 = time.time()
    all_rows = []
    # Paginate by season to avoid Supabase offset limits on large tables
    seasons = [
        ("2022-08-01", "2023-06-30"),
        ("2023-07-01", "2024-06-30"),
        ("2024-07-01", "2025-06-30"),
        ("2025-07-01", "2026-06-30"),
    ]
    for start, end in seasons:
        offset = 0
        while True:
            resp = requests.get(f"{SUPABASE_URL}/rest/v1/historical_odds", headers=_headers(),
                params={"select": "player_name,odds_value,match_date",
                        "market_id": "eq.64",
                        "and": f"(match_date.gte.{start},match_date.lte.{end})",
                        "order": "match_date.asc",
                        "limit": "1000", "offset": str(offset)}, timeout=60)
            if resp.status_code != 200:
                break
            rows = resp.json()
            all_rows.extend(rows)
            print(f"   {len(all_rows):,} odds rows...", end="\r")
            if len(rows) < 1000:
                break
            offset += 1000

    # Build lookup: (match_date, normalized_name) -> avg implied probability
    odds_lookup = {}
    for row in all_rows:
        date = str(row.get("match_date", ""))[:10]
        name = _normalize_name(row.get("player_name", ""))
        odds = row.get("odds_value")
        if not name or not odds or odds <= 1:
            continue
        key = (date, name)
        if key not in odds_lookup:
            odds_lookup[key] = []
        odds_lookup[key].append(1.0 / odds)  # implied probability

    # Average if multiple odds per player-match
    final = {}
    for key, probs in odds_lookup.items():
        final[key] = float(np.mean(probs))

    print(f"   {len(all_rows):,} odds rows → {len(final):,} unique (player, date) pairs in {time.time()-t0:.1f}s")
    return final


def pull_referees():
    resp = requests.get(f"{SUPABASE_URL}/rest/v1/referees", headers=_headers(),
        params={"select": "referee_name,yellows_per_match", "limit": "200"}, timeout=30)
    return {r["referee_name"]: float(r.get("yellows_per_match", _REF_YPG_MEAN) or _REF_YPG_MEAN)
            for r in resp.json()}


# ═══════════════════════════════════════════════════════════════════════
# FEATURE COMPUTATION (same as v7 + market + starter)
# ═══════════════════════════════════════════════════════════════════════

def compute_player_features(df):
    """v7 rolling features."""
    print("\n2. Computing player rolling features...")
    n = len(df)
    feat = {f: np.zeros(n) for f in [
        "career_card_rate", "cards_last_10", "card_rate_last_10", "career_games",
        "games_since_last_card", "recent_card_intensity",
        "fouls_committed_avg_5", "fouls_committed_avg_10", "career_fouls_committed_rate",
        "tackles_avg_5", "duels_avg_5", "minutes_l5", "days_since_last_match",
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

        all_ycs, all_fcs, all_tks, all_dus, all_mins = [], [], [], [], []

        for i, idx in enumerate(idxs):
            hist_yc = all_ycs[-MAX_HISTORY:][::-1]
            hist_fc = all_fcs[-MAX_HISTORY:][::-1]
            hist_tk = all_tks[-MAX_HISTORY:][::-1]
            hist_du = all_dus[-MAX_HISTORY:][::-1]
            hist_mi = all_mins[-MAX_HISTORY:][::-1]
            n_prior = len(hist_yc)

            if n_prior > 0:
                feat["career_games"][idx] = n_prior
                feat["career_card_rate"][idx] = sum(hist_yc) / n_prior
                l10 = hist_yc[:10]
                feat["cards_last_10"][idx] = sum(l10)
                feat["card_rate_last_10"][idx] = np.mean(l10)
                feat["recent_card_intensity"][idx] = feat["card_rate_last_10"][idx] * feat["cards_last_10"][idx]

                found = False
                for j, yc_val in enumerate(hist_yc):
                    if yc_val > 0:
                        feat["games_since_last_card"][idx] = j
                        found = True
                        break
                if not found:
                    feat["games_since_last_card"][idx] = n_prior

                feat["fouls_committed_avg_5"][idx] = np.mean(hist_fc[:5])
                feat["fouls_committed_avg_10"][idx] = np.mean(hist_fc[:10])
                feat["career_fouls_committed_rate"][idx] = sum(hist_fc) / n_prior
                feat["tackles_avg_5"][idx] = np.mean(hist_tk[:5])
                feat["duels_avg_5"][idx] = np.mean(hist_du[:5])
                feat["minutes_l5"][idx] = np.mean(hist_mi[:5])

                if i > 0:
                    delta = (dates[i] - dates[i-1]) / np.timedelta64(1, "D")
                    feat["days_since_last_match"][idx] = min(delta, 30)
                else:
                    feat["days_since_last_match"][idx] = 7
            else:
                feat["days_since_last_match"][idx] = 7

            all_ycs.append(int(ycs[i]))
            all_fcs.append(int(fcs[i]))
            all_tks.append(int(tks[i]))
            all_dus.append(int(dus[i]))
            all_mins.append(int(mins[i]))

    print(f"   {count:,} players processed")
    for k, v in feat.items():
        df[k] = v

    df["is_short_rest"] = (df["days_since_last_match"] < 4).astype(int)

    # NEW v9: is_starter (minutes_l5 >= 60)
    df["is_starter"] = (df["minutes_l5"] >= 60).astype(int)

    return df


def compute_match_features(df, ref_map):
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

    fix_dates = df.groupby(["af_fixture_id", "team", "season"])["match_date"].first().reset_index().sort_values("match_date")
    fix_dates["round_num"] = fix_dates.groupby(["team", "season"]).cumcount() + 1
    rmap = dict(zip(zip(fix_dates["af_fixture_id"], fix_dates["team"]), fix_dates["round_num"]))
    df["_rnd"] = df.apply(lambda r: rmap.get((r["af_fixture_id"], r["team"]), 15), axis=1)
    df["high_stakes_match"] = ((df["_rnd"] >= 30) | (df["is_rivalry_match"] == 1)).astype(int)
    df.drop(columns=["_rnd"], inplace=True)

    def _ref(name):
        ypg = ref_map.get(name, _REF_YPG_MEAN) if name else _REF_YPG_MEAN
        return (ypg - _REF_YPG_MEAN) / _REF_YPG_STD, ypg
    ref_res = df["referee"].apply(lambda n: _ref(n.split(",")[0].strip() if n else ""))
    df["referee_strictness"] = ref_res.apply(lambda x: x[0])
    df["cards_per_game"] = ref_res.apply(lambda x: x[1])
    return df


def compute_team_features(df):
    print("\n4. Computing team aggregate features...")
    ft = df.groupby(["af_fixture_id", "team"]).agg(
        match_date=("match_date", "first"),
        total_yc=("yellow_cards", "sum"), total_fc=("fouls_committed", "sum"),
        n_players=("af_player_id", "nunique"), n_rows=("af_player_id", "count"),
    ).reset_index()

    team_agg = {}
    for team_name, group in ft.groupby("team"):
        group = group.sort_values("match_date").reset_index(drop=True)
        for i in range(len(group)):
            prior = group.iloc[max(0, i - TEAM_LOOKBACK):i]
            if len(prior) > 0:
                ty, tf, tr, tp = prior["total_yc"].sum(), prior["total_fc"].sum(), prior["n_rows"].sum(), prior["n_players"].sum()
                tdp, tcl5, ofc = ty / max(tr, 1), ty / max(tp, 1), tf / max(tr, 1)
            else:
                tdp, tcl5, ofc = 0.1, 0.5, 1.0
            team_agg[(group.iloc[i]["af_fixture_id"], team_name)] = {"tdp": tdp, "tcl5": tcl5, "ofc": ofc}

    tdp_arr = np.full(len(df), 0.1)
    tcl5_arr = np.full(len(df), 0.5)
    oac_arr = np.full(len(df), 0.1)
    oft_arr = np.full(len(df), 1.0)
    for idx, row in df.iterrows():
        fid, team, opp = row["af_fixture_id"], row["team"], row["opponent"]
        ta = team_agg.get((fid, team))
        if ta:
            tdp_arr[idx], tcl5_arr[idx] = ta["tdp"], ta["tcl5"]
        oa = team_agg.get((fid, opp))
        if oa:
            oac_arr[idx], oft_arr[idx] = oa["tdp"], oa["ofc"]
    df["team_defensive_pressure"] = tdp_arr
    df["team_cards_last_5"] = tcl5_arr
    df["opponent_avg_cards"] = oac_arr
    df["opponent_fouls_tendency"] = oft_arr
    return df


def join_market_odds(df, odds_lookup):
    """Join bookmaker implied probability to each player-match row."""
    print("\n5. Joining market odds...")

    # Normalize player names for matching
    df["_norm_name"] = df["player_name"].apply(_normalize_name)
    df["_date_str"] = df["match_date"].dt.strftime("%Y-%m-%d")

    # Direct match
    matched = 0
    implied = np.full(len(df), np.nan)
    for idx, row in df.iterrows():
        key = (row["_date_str"], row["_norm_name"])
        if key in odds_lookup:
            implied[idx] = odds_lookup[key]
            matched += 1
        else:
            # Last-name fallback
            last_name = row["_norm_name"].split()[-1] if row["_norm_name"] else ""
            if last_name:
                for (d, n), prob in odds_lookup.items():
                    if d == row["_date_str"] and n.split()[-1] == last_name:
                        implied[idx] = prob
                        matched += 1
                        break

    # Fill missing with position-based defaults
    df["market_implied_prob"] = implied
    pos_defaults = df.groupby("position")["market_implied_prob"].transform("median")
    global_default = 0.10  # base rate
    df["market_implied_prob"] = df["market_implied_prob"].fillna(pos_defaults).fillna(global_default)

    coverage = (~np.isnan(implied)).sum()
    print(f"   Matched: {matched:,}/{len(df):,} ({matched/len(df)*100:.1f}%)")
    print(f"   Filled with position defaults: {len(df) - coverage:,}")
    print(f"   market_implied_prob: mean={df['market_implied_prob'].mean():.4f}, "
          f"median={df['market_implied_prob'].median():.4f}")

    df.drop(columns=["_norm_name", "_date_str"], inplace=True)
    return df


# ═══════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════

def tune_and_train(X_train, y_train, X_cal, y_cal, features):
    print("\n6. Hyperparameter tuning...")
    configs = [
        {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 5, "num_leaves": 20, "min_child_samples": 50},
        {"n_estimators": 700, "learning_rate": 0.01, "max_depth": 5, "num_leaves": 20, "min_child_samples": 50},
        {"n_estimators": 500, "learning_rate": 0.02, "max_depth": 6, "num_leaves": 31, "min_child_samples": 30},
        {"n_estimators": 800, "learning_rate": 0.01, "max_depth": 6, "num_leaves": 31, "min_child_samples": 25},
        {"n_estimators": 1000, "learning_rate": 0.008, "max_depth": 5, "num_leaves": 20, "min_child_samples": 40},
        {"n_estimators": 600, "learning_rate": 0.015, "max_depth": 5, "num_leaves": 25, "min_child_samples": 35},
    ]
    best_auc, best_cfg, best_model = 0, None, None
    for i, cfg in enumerate(configs):
        lgbm = lgb.LGBMClassifier(
            objective="binary", subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.3, reg_lambda=0.3, random_state=42,
            verbosity=-1, n_jobs=-1, **cfg)
        lgbm.fit(X_train, y_train)
        probs = lgbm.predict_proba(X_cal)[:, 1]
        auc = roc_auc_score(y_cal, probs)
        brier = brier_score_loss(y_cal, probs)
        marker = " *" if auc > best_auc else ""
        print(f"   Config {i+1}: AUC={auc:.4f} Brier={brier:.4f}{marker}")
        if auc > best_auc:
            best_auc, best_cfg, best_model = auc, cfg, lgbm

    print(f"\n   Best: AUC={best_auc:.4f}")

    # Calibrate
    cal_iso = CalibratedClassifierCV(best_model, cv="prefit", method="isotonic")
    cal_iso.fit(X_cal, y_cal)
    cal_sig = CalibratedClassifierCV(best_model, cv="prefit", method="sigmoid")
    cal_sig.fit(X_cal, y_cal)
    b_iso = brier_score_loss(y_cal, cal_iso.predict_proba(X_cal)[:, 1])
    b_sig = brier_score_loss(y_cal, cal_sig.predict_proba(X_cal)[:, 1])
    print(f"   Isotonic Brier={b_iso:.4f} | Sigmoid Brier={b_sig:.4f}")
    final = cal_iso if b_iso <= b_sig else cal_sig

    # Feature importance
    print("\n   Feature importance:")
    imp = best_model.feature_importances_
    for feat, v in sorted(zip(features, imp), key=lambda x: -x[1]):
        bar = "█" * int(v / max(imp) * 25)
        print(f"     {feat:30s} {v:6.0f}  {bar}")

    return final, best_model, best_cfg


def evaluate(model, X_test, y_test):
    probs = np.clip(model.predict_proba(X_test)[:, 1], 0, 0.50)
    auc = roc_auc_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)

    print(f"\n7. TEST SET EVALUATION")
    print(f"   AUC-ROC:     {auc:.4f}")
    print(f"   Brier Score: {brier:.4f}")
    print(f"   P(YC) — mean={probs.mean():.4f}, max={probs.max():.4f}")

    bins = [0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.50]
    print(f"\n   Calibration:")
    for i in range(len(bins) - 1):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        n = mask.sum()
        if n > 0:
            pm, am = probs[mask].mean(), y_test[mask].mean()
            print(f"     [{bins[i]*100:4.0f}%-{bins[i+1]*100:4.0f}%): n={n:5d}, "
                  f"pred={pm*100:.1f}%, actual={am*100:.1f}%")
    return {"auc": auc, "brier": brier, "probs": probs}


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    print("=" * 75)
    print("YC v9 MODEL — MARKET-AWARE")
    print("=" * 75)

    df = pull_all_data()
    odds_lookup = pull_yc_odds()
    ref_map = pull_referees()

    df = compute_player_features(df)
    df = compute_match_features(df, ref_map)
    df = compute_team_features(df)
    df = join_market_odds(df, odds_lookup)

    df["yc_binary"] = (df["yellow_cards"] >= 1).astype(int)
    before = len(df)
    df = df[df["career_games"] >= 3].reset_index(drop=True)
    print(f"\n   Filtered: {before:,} → {len(df):,} rows")

    # Split
    n = len(df)
    s1, s2 = int(n * 0.60), int(n * 0.80)
    X_train = df[YC_V9_FEATURES].values[:s1]
    y_train = df["yc_binary"].values[:s1]
    X_cal = df[YC_V9_FEATURES].values[s1:s2]
    y_cal = df["yc_binary"].values[s1:s2]
    X_test = df[YC_V9_FEATURES].values[s2:]
    y_test = df["yc_binary"].values[s2:]

    print(f"\n   Train: {len(y_train):,} | Cal: {len(y_cal):,} | Test: {len(y_test):,}")

    # Train
    model, raw_lgbm, best_cfg = tune_and_train(X_train, y_train, X_cal, y_cal, YC_V9_FEATURES)
    metrics = evaluate(model, X_test, y_test)

    # Top predictions
    test_df = df.iloc[s2:].copy()
    test_df["pred"] = metrics["probs"]
    top = test_df.nlargest(20, "pred")
    print(f"\n   Top 20 predictions:")
    print(f"   {'Player':25s} {'Team':15s} {'P':>5s} {'Mkt':>5s} {'FC5':>4s} {'St':>2s} Actual")
    for _, r in top.iterrows():
        print(f"   {r['player_name'][:25]:25s} {r['team'][:15]:15s} "
              f"{r['pred']*100:4.1f}% {r['market_implied_prob']*100:4.0f}% "
              f"{r['fouls_committed_avg_5']:.1f} {int(r['is_starter']):2d}  "
              f"{'YES' if r['yc_binary'] else 'no'}")

    # Save
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "epl_yellow_cards_v9.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model, "features": YC_V9_FEATURES, "version": "v9",
            "trained_at": datetime.now().isoformat(),
            "training_rows": len(df), "base_rate": float(df["yc_binary"].mean()),
            "test_auc": metrics["auc"], "test_brier": metrics["brier"],
            "ref_ypg_mean": _REF_YPG_MEAN, "ref_ypg_std": _REF_YPG_STD,
            "best_config": best_cfg,
        }, f)

    elapsed = time.time() - t_start
    print(f"\n{'='*75}")
    print(f"DONE — {elapsed:.1f}s | {model_path}")
    print(f"v7 AUC: 0.6968 → v9 AUC: {metrics['auc']:.4f}")
    print(f"{'='*75}")


if __name__ == "__main__":
    main()
