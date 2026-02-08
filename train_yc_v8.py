#!/usr/bin/env python3
"""
train_yc_v8.py — Yellow Card Model v8: Full Optimization
=========================================================

Improvements over v7 (AUC 0.6968, 29.2% hit rate):
  1. card_per_foul_rate — separates cynical foulers from routine foulers
  2. fouls_per_90_l5 — normalize fouls by minutes played
  3. fouls_committed_trend — fc_avg_3 - fc_avg_10 (getting dirtier?)
  4. season_yellows_accumulated — suspension avoidance behavior
  5. interceptions_avg_5, duel_win_pct_l5 — defensive aggression
  6. Recency-weighted career stats (exponential decay)
  7. opponent_ppda_l5 — dynamic pressing from team_match_xg
  8. match_card_intensity — match-level expected cards meta-feature
  9. Actual round numbers from fixtures table
  10. Expanded hyperparameter search

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
DECAY = 0.95  # exponential decay factor for recency weighting

# ─── V8 Feature List (41 features) ───────────────────────────────────
YC_V8_FEATURES = [
    # === Player card history (6) ===
    "career_card_rate", "cards_last_10", "card_rate_last_10",
    "career_games", "games_since_last_card", "recent_card_intensity",
    # === Player fouling history (3) ===
    "fouls_committed_avg_5", "fouls_committed_avg_10", "career_fouls_committed_rate",
    # === NEW v8: Foul quality + trends (4) ===
    "card_per_foul_rate",       # career yellows / career fouls — foul severity
    "fouls_per_90_l5",          # fouls normalized by minutes
    "fouls_committed_trend",    # fc_avg_3 - fc_avg_10 — getting dirtier?
    "season_yellows_accumulated",  # running YC count this season
    # === NEW v8: Recency-weighted career stats (2) ===
    "career_card_rate_weighted",   # exponential decay weighted
    "career_fc_rate_weighted",     # exponential decay weighted
    # === Player physicality (4) ===
    "tackles_avg_5", "duels_avg_5",
    "interceptions_avg_5",      # NEW v8: defensive positioning
    "duel_win_pct_l5",          # NEW v8: losing duels → desperate fouls
    # === Playing time (1) ===
    "minutes_l5",
    # === Position (4) ===
    "is_defender", "is_midfielder", "is_forward", "is_goalkeeper",
    # === Home/Away (2) ===
    "is_home", "is_away",
    # === Match context (4) ===
    "is_rivalry_match", "is_big6_match", "high_stakes_match",
    "round_number",             # NEW v8: actual round from fixtures table
    # === Referee (2) ===
    "referee_strictness", "cards_per_game",
    # === Team/Opponent aggregates (5) ===
    "team_defensive_pressure", "team_cards_last_5",
    "opponent_avg_cards", "opponent_fouls_tendency",
    "opponent_ppda_l5",         # NEW v8: dynamic pressing from xG data
    # === Match-level intelligence (1) ===
    "match_card_intensity",     # NEW v8: expected total cards in this match
    # === Season/Rest (3) ===
    "late_season", "days_since_last_match", "is_short_rest",
]

assert len(YC_V8_FEATURES) == 41, f"Expected 41, got {len(YC_V8_FEATURES)}"


# ═══════════════════════════════════════════════════════════════════════
# DATA PULL
# ═══════════════════════════════════════════════════════════════════════

def _headers():
    return {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json"}


def _paginate(table, columns, order="match_date.asc", extra_params=None):
    """Pull all rows from a Supabase table with pagination."""
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


def pull_all_data():
    """Pull player_match_stats with all columns needed for v8."""
    columns = (
        "af_player_id,player_name,team,opponent,match_date,season,"
        "position,is_home,minutes_played,yellow_cards,"
        "fouls_committed,fouls_drawn,tackles_total,duels_total,duels_won,"
        "interceptions,af_fixture_id,referee"
    )
    print("1. Pulling player_match_stats...")
    t0 = time.time()
    all_rows = _paginate("player_match_stats", columns, "match_date.asc,af_fixture_id.asc")

    df = pd.DataFrame(all_rows)
    int_cols = ["yellow_cards", "minutes_played", "fouls_committed", "fouls_drawn",
                "tackles_total", "duels_total", "duels_won", "interceptions"]
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    df["is_home"] = df["is_home"].astype(bool).astype(int)
    df["position"] = df["position"].fillna("M").str[0].str.upper()
    df["referee"] = df["referee"].fillna("")
    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values(["match_date", "af_fixture_id", "af_player_id"]).reset_index(drop=True)

    print(f"   {len(df):,} rows in {time.time()-t0:.1f}s | "
          f"YC rate: {df['yellow_cards'].clip(0,1).mean():.4f}")
    return df


def pull_fixtures():
    """Pull fixtures table for actual round numbers."""
    print("   Pulling fixtures (round data)...")
    rows = _paginate("fixtures", "af_fixture_id,round,match_date", "match_date.asc")
    fix_df = pd.DataFrame(rows)
    # Parse round: "Regular Season - 25" → 25
    def parse_round(r):
        try:
            return int(str(r).split("-")[-1].strip())
        except (ValueError, IndexError, AttributeError):
            return 15
    fix_df["round_num"] = fix_df["round"].apply(parse_round)
    return dict(zip(fix_df["af_fixture_id"].astype(int), fix_df["round_num"]))


def pull_ppda():
    """Pull team_match_xg for dynamic PPDA data."""
    print("   Pulling team_match_xg (PPDA)...")
    rows = _paginate("team_match_xg", "team,match_date,ppda", "match_date.asc")
    ppda_df = pd.DataFrame(rows)
    ppda_df["ppda"] = pd.to_numeric(ppda_df["ppda"], errors="coerce").fillna(12.0)
    ppda_df["match_date"] = pd.to_datetime(ppda_df["match_date"])
    ppda_df = ppda_df.sort_values("match_date")
    print(f"   {len(ppda_df):,} PPDA rows")
    return ppda_df


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

    feat_names = [
        "career_card_rate", "cards_last_10", "card_rate_last_10", "career_games",
        "games_since_last_card", "recent_card_intensity",
        "fouls_committed_avg_5", "fouls_committed_avg_10", "career_fouls_committed_rate",
        "tackles_avg_5", "duels_avg_5", "minutes_l5", "days_since_last_match",
        # NEW v8
        "card_per_foul_rate", "fouls_per_90_l5", "fouls_committed_trend",
        "season_yellows_accumulated", "interceptions_avg_5", "duel_win_pct_l5",
        "career_card_rate_weighted", "career_fc_rate_weighted",
    ]
    feat = {f: np.zeros(n) for f in feat_names}

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
        duw = group["duels_won"].values
        mins = group["minutes_played"].values
        dates = group["match_date"].values
        seasons = group["season"].values
        ints = group["interceptions"].values

        # Running history buffers
        all_ycs, all_fcs, all_tks, all_dus, all_duw, all_mins, all_ints = \
            [], [], [], [], [], [], []

        # Season-level YC accumulator
        current_season = None
        season_yc_count = 0

        for i, idx in enumerate(idxs):
            # Track season YC accumulation
            s = seasons[i]
            if s != current_season:
                current_season = s
                season_yc_count = 0
            feat["season_yellows_accumulated"][idx] = season_yc_count

            # History: prior games capped at MAX_HISTORY, most-recent first
            hist_yc = all_ycs[-MAX_HISTORY:][::-1]
            hist_fc = all_fcs[-MAX_HISTORY:][::-1]
            hist_tk = all_tks[-MAX_HISTORY:][::-1]
            hist_du = all_dus[-MAX_HISTORY:][::-1]
            hist_dw = all_duw[-MAX_HISTORY:][::-1]
            hist_mi = all_mins[-MAX_HISTORY:][::-1]
            hist_in = all_ints[-MAX_HISTORY:][::-1]
            n_prior = len(hist_yc)

            if n_prior > 0:
                # ── v7 features (unchanged) ──────────────────────
                feat["career_games"][idx] = n_prior
                total_yc = sum(hist_yc)
                total_fc = sum(hist_fc)
                feat["career_card_rate"][idx] = total_yc / n_prior
                feat["career_fouls_committed_rate"][idx] = total_fc / n_prior

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

                l5_fc = hist_fc[:5]
                l10_fc = hist_fc[:10]
                l3_fc = hist_fc[:3]
                feat["fouls_committed_avg_5"][idx] = np.mean(l5_fc)
                feat["fouls_committed_avg_10"][idx] = np.mean(l10_fc)

                l5_tk = hist_tk[:5]
                l5_du = hist_du[:5]
                feat["tackles_avg_5"][idx] = np.mean(l5_tk)
                feat["duels_avg_5"][idx] = np.mean(l5_du)

                l5_mi = hist_mi[:5]
                feat["minutes_l5"][idx] = np.mean(l5_mi)

                if i > 0:
                    delta = (dates[i] - dates[i-1]) / np.timedelta64(1, "D")
                    feat["days_since_last_match"][idx] = min(delta, 30)
                else:
                    feat["days_since_last_match"][idx] = 7

                # ── NEW v8 features ──────────────────────────────

                # card_per_foul_rate: career yellows / career fouls
                feat["card_per_foul_rate"][idx] = total_yc / max(total_fc, 1)

                # fouls_per_90_l5: normalize by minutes
                avg_mins = feat["minutes_l5"][idx]
                avg_fc5 = feat["fouls_committed_avg_5"][idx]
                feat["fouls_per_90_l5"][idx] = (avg_fc5 / max(avg_mins, 1)) * 90

                # fouls_committed_trend: fc_avg_3 - fc_avg_10
                fc_avg_3 = np.mean(l3_fc) if len(l3_fc) > 0 else 0
                feat["fouls_committed_trend"][idx] = fc_avg_3 - feat["fouls_committed_avg_10"][idx]

                # interceptions_avg_5
                l5_in = hist_in[:5]
                feat["interceptions_avg_5"][idx] = np.mean(l5_in)

                # duel_win_pct_l5
                l5_dw = hist_dw[:5]
                total_du_5 = sum(l5_du)
                total_dw_5 = sum(l5_dw)
                feat["duel_win_pct_l5"][idx] = total_dw_5 / max(total_du_5, 1)

                # Recency-weighted career stats (exponential decay)
                # weight[0] = DECAY^0 = 1.0 (most recent), weight[1] = DECAY^1, ...
                weights = np.array([DECAY ** j for j in range(n_prior)])
                w_sum = weights.sum()
                feat["career_card_rate_weighted"][idx] = np.dot(hist_yc[:n_prior], weights) / w_sum
                feat["career_fc_rate_weighted"][idx] = np.dot(hist_fc[:n_prior], weights) / w_sum

            else:
                feat["days_since_last_match"][idx] = 7

            # Update state AFTER computing features
            all_ycs.append(int(ycs[i]))
            all_fcs.append(int(fcs[i]))
            all_tks.append(int(tks[i]))
            all_dus.append(int(dus[i]))
            all_duw.append(int(duw[i]))
            all_mins.append(int(mins[i]))
            all_ints.append(int(ints[i]))

            # Update season accumulator AFTER computing features
            season_yc_count += int(min(ycs[i], 1))

    print(f"   {count:,} players processed")
    for k, v in feat.items():
        df[k] = v

    df["is_short_rest"] = (df["days_since_last_match"] < 4).astype(int)
    return df


def compute_match_features(df, ref_map, round_map):
    """Position, home/away, rivalry, referee, round features."""
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

    # Actual round number from fixtures table
    df["round_number"] = df["af_fixture_id"].map(round_map).fillna(15).astype(int)
    df["high_stakes_match"] = ((df["round_number"] >= 30) | (df["is_rivalry_match"] == 1)).astype(int)

    # Referee
    def _ref(name):
        ypg = ref_map.get(name, _REF_YPG_MEAN) if name else _REF_YPG_MEAN
        return (ypg - _REF_YPG_MEAN) / _REF_YPG_STD, ypg
    ref_res = df["referee"].apply(lambda n: _ref(n.split(",")[0].strip() if n else ""))
    df["referee_strictness"] = ref_res.apply(lambda x: x[0])
    df["cards_per_game"] = ref_res.apply(lambda x: x[1])
    return df


def compute_team_features(df, ppda_df):
    """Team aggregates + dynamic PPDA from xG data."""
    print("\n4. Computing team aggregate features...")

    # ── Per-fixture, per-team stats (same as v7) ─────────────────
    ft = df.groupby(["af_fixture_id", "team"]).agg(
        match_date=("match_date", "first"),
        total_yc=("yellow_cards", "sum"),
        total_fc=("fouls_committed", "sum"),
        n_players=("af_player_id", "nunique"),
        n_rows=("af_player_id", "count"),
    ).reset_index()

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
                ofc = tf / max(tr, 1)
            else:
                tdp, tcl5, ofc = 0.1, 0.5, 1.0
            fid = group.iloc[i]["af_fixture_id"]
            team_agg[(fid, team_name)] = {"tdp": tdp, "tcl5": tcl5, "ofc": ofc}

    # ── Dynamic PPDA from team_match_xg ──────────────────────────
    print("   Computing dynamic PPDA...")
    ppda_by_team = {}  # (team, match_date) -> rolling avg ppda
    for team_name, group in ppda_df.groupby("team"):
        group = group.sort_values("match_date").reset_index(drop=True)
        dates = group["match_date"].values
        ppdas = group["ppda"].values
        team_ppda_list = []
        for i in range(len(group)):
            prior = ppdas[max(0, i - 5):i]
            avg_ppda = float(np.mean(prior)) if len(prior) > 0 else 12.0
            team_ppda_list.append((dates[i], avg_ppda))
        ppda_by_team[team_name] = team_ppda_list

    # Build a lookup: for a given opponent + match_date, find their latest PPDA avg
    def get_opponent_ppda(opponent, match_date):
        if opponent not in ppda_by_team:
            return 12.0
        entries = ppda_by_team[opponent]
        # Find the last entry before match_date
        best = 12.0
        for d, p in entries:
            if d < match_date:
                best = p
            else:
                break
        return best

    # ── Map to rows ──────────────────────────────────────────────
    print("   Mapping to player rows...")
    tdp_arr = np.full(len(df), 0.1)
    tcl5_arr = np.full(len(df), 0.5)
    oac_arr = np.full(len(df), 0.1)
    oft_arr = np.full(len(df), 1.0)
    ppda_arr = np.full(len(df), 12.0)

    for idx, row in df.iterrows():
        fid, team, opp = row["af_fixture_id"], row["team"], row["opponent"]
        md = row["match_date"]
        ta = team_agg.get((fid, team))
        if ta:
            tdp_arr[idx] = ta["tdp"]
            tcl5_arr[idx] = ta["tcl5"]
        oa = team_agg.get((fid, opp))
        if oa:
            oac_arr[idx] = oa["tdp"]
            oft_arr[idx] = oa["ofc"]
        ppda_arr[idx] = get_opponent_ppda(opp, md)

    df["team_defensive_pressure"] = tdp_arr
    df["team_cards_last_5"] = tcl5_arr
    df["opponent_avg_cards"] = oac_arr
    df["opponent_fouls_tendency"] = oft_arr
    df["opponent_ppda_l5"] = ppda_arr
    return df


def compute_match_card_intensity(df, ref_map):
    """
    Match-level expected total cards = f(referee, rivalry, team foul rates, round).
    This is a meta-feature that captures whether THIS match is likely to be card-heavy.
    """
    print("\n5. Computing match_card_intensity...")

    # Per-fixture stats
    fixture_stats = df.groupby("af_fixture_id").agg(
        match_date=("match_date", "first"),
        referee=("referee", "first"),
        is_rivalry=("is_rivalry_match", "first"),
        round_num=("round_number", "first"),
        team_fouls_tend=("opponent_fouls_tendency", "mean"),  # avg of both teams' foul tendencies
    ).reset_index()

    mci = {}
    for _, row in fixture_stats.iterrows():
        ref_name = str(row["referee"]).split(",")[0].strip() if row["referee"] else ""
        ref_ypg = ref_map.get(ref_name, _REF_YPG_MEAN)

        # Components: referee cards/game + rivalry bonus + late-season bonus + team fouls
        intensity = ref_ypg  # base: referee's avg cards per game
        if row["is_rivalry"]:
            intensity += 0.8
        if row["round_num"] >= 30:
            intensity += 0.3
        # Scale team fouls tendency (mean ~0.55, higher = more fouls)
        fouls_factor = row["team_fouls_tend"] / 0.55 if row["team_fouls_tend"] > 0 else 1.0
        intensity *= fouls_factor

        mci[row["af_fixture_id"]] = intensity

    df["match_card_intensity"] = df["af_fixture_id"].map(mci).fillna(_REF_YPG_MEAN)
    return df


# ═══════════════════════════════════════════════════════════════════════
# HYPERPARAMETER TUNING + TRAINING
# ═══════════════════════════════════════════════════════════════════════

def tune_and_train(X_train, y_train, X_cal, y_cal, features):
    print("\n6. Hyperparameter tuning...")

    configs = [
        {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 5, "num_leaves": 20, "min_child_samples": 50},
        {"n_estimators": 500, "learning_rate": 0.02, "max_depth": 6, "num_leaves": 31, "min_child_samples": 30},
        {"n_estimators": 800, "learning_rate": 0.01, "max_depth": 6, "num_leaves": 31, "min_child_samples": 25},
        {"n_estimators": 700, "learning_rate": 0.01, "max_depth": 5, "num_leaves": 20, "min_child_samples": 50},
        {"n_estimators": 600, "learning_rate": 0.015, "max_depth": 7, "num_leaves": 40, "min_child_samples": 20},
        {"n_estimators": 400, "learning_rate": 0.025, "max_depth": 6, "num_leaves": 31, "min_child_samples": 35},
        {"n_estimators": 1000, "learning_rate": 0.008, "max_depth": 5, "num_leaves": 20, "min_child_samples": 40},
        {"n_estimators": 500, "learning_rate": 0.02, "max_depth": 5, "num_leaves": 25, "min_child_samples": 40},
        {"n_estimators": 800, "learning_rate": 0.01, "max_depth": 7, "num_leaves": 50, "min_child_samples": 20},
        {"n_estimators": 600, "learning_rate": 0.015, "max_depth": 6, "num_leaves": 31, "min_child_samples": 30},
    ]

    best_auc, best_config, best_model = 0, None, None

    for i, cfg in enumerate(configs):
        lgbm = lgb.LGBMClassifier(
            objective="binary", subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.3, reg_lambda=0.3, random_state=42,
            verbosity=-1, n_jobs=-1, **cfg)
        lgbm.fit(X_train, y_train)
        probs = lgbm.predict_proba(X_cal)[:, 1]
        auc = roc_auc_score(y_cal, probs)
        brier = brier_score_loss(y_cal, probs)
        marker = " ★" if auc > best_auc else ""
        print(f"   Config {i+1:2d}: AUC={auc:.4f} Brier={brier:.4f} "
              f"(n={cfg['n_estimators']}, lr={cfg['learning_rate']}, "
              f"d={cfg['max_depth']}, l={cfg['num_leaves']}){marker}")
        if auc > best_auc:
            best_auc, best_config, best_model = auc, cfg, lgbm

    print(f"\n   Best: AUC={best_auc:.4f} — {best_config}")

    # Calibration
    print("\n7. Calibration...")
    cal_iso = CalibratedClassifierCV(best_model, cv="prefit", method="isotonic")
    cal_iso.fit(X_cal, y_cal)
    p_iso = cal_iso.predict_proba(X_cal)[:, 1]

    cal_sig = CalibratedClassifierCV(best_model, cv="prefit", method="sigmoid")
    cal_sig.fit(X_cal, y_cal)
    p_sig = cal_sig.predict_proba(X_cal)[:, 1]

    brier_iso = brier_score_loss(y_cal, p_iso)
    brier_sig = brier_score_loss(y_cal, p_sig)
    print(f"   Isotonic — Brier={brier_iso:.4f}, max={p_iso.max():.4f}")
    print(f"   Sigmoid  — Brier={brier_sig:.4f}, max={p_sig.max():.4f}")

    final_model = cal_iso if brier_iso <= brier_sig else cal_sig
    print(f"   → Using {'ISOTONIC' if brier_iso <= brier_sig else 'SIGMOID'}")

    # Feature importance
    print("\n   Feature importance (gain):")
    importances = best_model.feature_importances_
    feat_imp = sorted(zip(features, importances), key=lambda x: -x[1])
    for feat, imp in feat_imp:
        bar = "█" * int(imp / max(importances) * 30)
        print(f"     {feat:32s} {imp:6.0f}  {bar}")

    return final_model, best_model, best_config


# ═══════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def evaluate(model, X_test, y_test):
    probs = np.clip(model.predict_proba(X_test)[:, 1], 0, 0.50)

    auc = roc_auc_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)

    print(f"\n8. TEST SET EVALUATION")
    print(f"   AUC-ROC:     {auc:.4f}")
    print(f"   Brier Score: {brier:.4f}")
    print(f"   P(YC) — mean={probs.mean():.4f}, median={np.median(probs):.4f}, max={probs.max():.4f}")
    print(f"   Base rate:    {y_test.mean():.4f}")

    for p in [5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"     p{p:2d}: {np.percentile(probs, p):.4f}")

    print(f"\n   Calibration:")
    bins = [0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.50]
    for i in range(len(bins) - 1):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        n_bin = mask.sum()
        if n_bin > 0:
            pm, am = probs[mask].mean(), y_test[mask].mean()
            print(f"     [{bins[i]*100:4.0f}%-{bins[i+1]*100:4.0f}%): n={n_bin:5d}, "
                  f"pred={pm*100:.1f}%, actual={am*100:.1f}%, diff={pm-am:+.4f}")

    return {"auc": auc, "brier": brier, "probs": probs}


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    print("=" * 75)
    print("YC v8 MODEL — FULL OPTIMIZATION")
    print("=" * 75)

    # ── Data pull ────────────────────────────────────────────────
    df = pull_all_data()
    round_map = pull_fixtures()
    ppda_df = pull_ppda()
    ref_map = pull_referees()

    # ── Feature computation ──────────────────────────────────────
    df = compute_player_features(df)
    df = compute_match_features(df, ref_map, round_map)
    df = compute_team_features(df, ppda_df)
    df = compute_match_card_intensity(df, ref_map)

    # ── Binarize + filter ────────────────────────────────────────
    df["yc_binary"] = (df["yellow_cards"] >= 1).astype(int)
    before = len(df)
    df = df[df["career_games"] >= 3].reset_index(drop=True)
    print(f"\n   Filtered: {before:,} → {len(df):,} rows")

    # ── Feature ranges ───────────────────────────────────────────
    print(f"\n   Feature ranges:")
    for feat in YC_V8_FEATURES:
        vals = df[feat].values
        print(f"     {feat:32s}  [{vals.min():.4f}, {vals.max():.4f}]  mean={vals.mean():.4f}")

    # ── Time-based split: 60/20/20 ───────────────────────────────
    n = len(df)
    s1, s2 = int(n * 0.60), int(n * 0.80)
    X_train, y_train = df[YC_V8_FEATURES].values[:s1], df["yc_binary"].values[:s1]
    X_cal, y_cal = df[YC_V8_FEATURES].values[s1:s2], df["yc_binary"].values[s1:s2]
    X_test, y_test = df[YC_V8_FEATURES].values[s2:], df["yc_binary"].values[s2:]

    print(f"\n   Train: {len(y_train):,} ({y_train.mean():.4f})")
    print(f"   Cal:   {len(y_cal):,} ({y_cal.mean():.4f})")
    print(f"   Test:  {len(y_test):,} ({y_test.mean():.4f})")

    # ── Tune + Train ─────────────────────────────────────────────
    model, raw_lgbm, best_cfg = tune_and_train(X_train, y_train, X_cal, y_cal, YC_V8_FEATURES)

    # ── Evaluate ─────────────────────────────────────────────────
    metrics = evaluate(model, X_test, y_test)

    # ── Sanity check ─────────────────────────────────────────────
    test_df = df.iloc[s2:].copy()
    test_df["pred"] = metrics["probs"]
    top = test_df.nlargest(20, "pred")
    print(f"\n   Top 20 predictions:")
    print(f"   {'Player':25s} {'Team':15s} {'P':>5s} {'CPF':>5s} {'FC5':>5s} {'Actual':>6s}")
    for _, r in top.iterrows():
        print(f"   {r['player_name'][:25]:25s} {r['team'][:15]:15s} "
              f"{r['pred']*100:4.1f}% {r['card_per_foul_rate']:.2f}  "
              f"{r['fouls_committed_avg_5']:.1f}  "
              f"{'YES' if r['yc_binary'] else 'no'}")

    print(f"\n   By position:")
    for pos in ["D", "M", "F", "G"]:
        mask = test_df["position"] == pos
        if mask.sum() > 0:
            print(f"     {pos}: pred={test_df.loc[mask,'pred'].mean()*100:.1f}%, "
                  f"actual={test_df.loc[mask,'yc_binary'].mean()*100:.1f}%")

    # ── Save ─────────────────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "epl_yellow_cards_v8.pkl")
    model_data = {
        "model": model,
        "features": YC_V8_FEATURES,
        "version": "v8",
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
    print(f"v7 AUC: 0.6968 → v8 AUC: {metrics['auc']:.4f}")
    print(f"{'='*75}")


if __name__ == "__main__":
    main()
