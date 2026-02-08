#!/usr/bin/env python3
"""
backtest_yc_v6.py — Backtest YC v6 model on completed matches.

For each completed fixture:
  1. Compute features using only pre-match data (FeatureEngine with before_date)
  2. Predict P(YC) with the v6 model
  3. Compare against actual yellow card outcomes
  4. Show calibration, accuracy, and per-match breakdowns
"""

import sys
import os
import pickle
import warnings
import numpy as np
import requests
from datetime import datetime

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared_features import FeatureEngine
from shared_features.constants import YC_V5_FEATURES, SUPABASE_URL, SUPABASE_KEY

# ─── Config ───────────────────────────────────────────────────────────

MAX_PROB = 0.40  # clip isotonic artifacts

# ─── Helpers ──────────────────────────────────────────────────────────

def _supabase_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }


def get_completed_fixtures(start_date, end_date):
    """Get unique fixtures from Supabase between two dates."""
    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/player_match_stats",
        headers=_supabase_headers(),
        params={
            "select": "af_fixture_id,match_date,team,opponent,referee,is_home",
            "and": f"(match_date.gte.{start_date},match_date.lte.{end_date},is_home.eq.true)",
            "order": "match_date.asc",
            "limit": "200",
        },
        timeout=30,
    )
    resp.raise_for_status()
    rows = resp.json()

    # Deduplicate by fixture_id (keep home team row)
    seen = set()
    fixtures = []
    for r in rows:
        fid = r["af_fixture_id"]
        if fid not in seen:
            seen.add(fid)
            fixtures.append({
                "fixture_id": fid,
                "date": r["match_date"],
                "home": r["team"],
                "away": r["opponent"],
                "referee": r["referee"].split(",")[0].strip() if r.get("referee") else "",
            })
    return fixtures


def get_actual_yellows(fixture_id):
    """Get actual YC outcomes for all players in a fixture."""
    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/player_match_stats",
        headers=_supabase_headers(),
        params={
            "select": "player_name,team,position,yellow_cards,minutes_played",
            "af_fixture_id": f"eq.{fixture_id}",
            "limit": "100",
        },
        timeout=30,
    )
    resp.raise_for_status()
    actuals = {}
    for r in resp.json():
        name = r["player_name"]
        actuals[name] = {
            "yc": int(r.get("yellow_cards", 0) or 0),
            "team": r["team"],
            "pos": (r.get("position") or "M")[0],
            "mins": int(r.get("minutes_played", 0) or 0),
        }
    return actuals


# ─── Main Backtest ────────────────────────────────────────────────────

def backtest(start_date, end_date):
    print("=" * 85)
    print(f"YC v6 BACKTEST — {start_date} to {end_date}")
    print("=" * 85)

    # Load model
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "epl_yellow_cards_v6.pkl")
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    model = model_data["model"]
    print(f"Model: {model_data.get('version', '?')} | Test AUC: {model_data.get('test_auc', 0):.4f}")

    engine = FeatureEngine()
    fixtures = get_completed_fixtures(start_date, end_date)
    print(f"Fixtures found: {len(fixtures)}\n")

    if not fixtures:
        print("No fixtures found in date range.")
        return

    # Collect all predictions and actuals
    all_preds = []    # (prob, actual_yc, name, team, pos, fixture_str)
    match_results = []

    for fix in fixtures:
        fixture_str = f"{fix['home']} vs {fix['away']}"
        print(f"--- {fix['date']} | {fixture_str} | Ref: {fix['referee']} ---")

        # Get actual outcomes
        actuals = get_actual_yellows(fix["fixture_id"])
        actual_names_booked = [n for n, a in actuals.items() if a["yc"] > 0]
        total_yc = sum(a["yc"] for a in actuals.values())

        # Predict for both teams
        match_preds = []
        for team, opponent, is_home in [
            (fix["home"], fix["away"], True),
            (fix["away"], fix["home"], False),
        ]:
            try:
                features_list = engine.get_fixture_player_features(
                    model="yc_v5",
                    team=team,
                    opponent=opponent,
                    is_home=is_home,
                    match_date=fix["date"],
                    referee=fix["referee"],
                    min_games=3,
                )
            except Exception as e:
                print(f"  Error: {team}: {e}")
                continue

            for pf in features_list:
                name = pf.pop("_player_name", "Unknown")
                pos = pf.pop("_position", "M")
                pf.pop("_player_id", None)
                X = np.array([[pf[f] for f in YC_V5_FEATURES]])
                prob = min(model.predict_proba(X)[0][1], MAX_PROB)

                actual = actuals.get(name, {})
                actual_yc = actual.get("yc", 0)
                mins = actual.get("mins", 0)

                # Only count players who actually played (>0 mins)
                if name in actuals and mins > 0:
                    match_preds.append((prob, actual_yc, name, team, pos))
                    all_preds.append((prob, actual_yc, name, team, pos, fixture_str))

        # Sort by probability descending
        match_preds.sort(key=lambda x: -x[0])

        # Show top 10 predictions for this match
        correct_flags = 0
        for prob, yc, name, team, pos in match_preds[:12]:
            flag = "YES" if yc > 0 else "   "
            marker = " <-- BOOKED" if yc > 0 else ""
            print(f"  {prob*100:5.1f}%  {name[:25]:25s} {team[:15]:15s} {pos} {flag}{marker}")

        if len(match_preds) > 12:
            rest_booked = sum(1 for _, yc, _, _, _ in match_preds[12:] if yc > 0)
            if rest_booked:
                print(f"  ... +{rest_booked} more booked players below top 12")

        # Match-level stats
        n_played = len(match_preds)
        n_booked = sum(1 for _, yc, _, _, _ in match_preds if yc > 0)
        sum_probs = sum(p for p, _, _, _, _ in match_preds)

        # Did the model rank booked players higher?
        if n_booked > 0 and n_played > 0:
            booked_ranks = []
            for i, (_, yc, _, _, _) in enumerate(match_preds):
                if yc > 0:
                    booked_ranks.append(i + 1)
            avg_rank = np.mean(booked_ranks)
            print(f"  >> {n_booked} booked / {n_played} players | "
                  f"Expected {sum_probs:.1f} YCs (actual {total_yc}) | "
                  f"Avg rank of booked: {avg_rank:.1f}/{n_played}")
            match_results.append({
                "fixture": fixture_str,
                "predicted_yc": sum_probs,
                "actual_yc": total_yc,
                "avg_rank": avg_rank,
                "n_players": n_played,
            })
        print()

    # ─── Aggregate Results ─────────────────────────────────────────
    if not all_preds:
        print("No predictions to evaluate.")
        return

    probs = np.array([p[0] for p in all_preds])
    actuals_arr = np.array([p[1] for p in all_preds])
    actuals_binary = (actuals_arr >= 1).astype(int)

    print("=" * 85)
    print("AGGREGATE BACKTEST RESULTS")
    print("=" * 85)

    print(f"\nTotal player-match predictions: {len(probs)}")
    print(f"Total yellow cards:             {actuals_binary.sum()}")
    print(f"Actual YC rate:                 {actuals_binary.mean()*100:.1f}%")
    print(f"Mean predicted P(YC):           {probs.mean()*100:.1f}%")
    print(f"Predicted total YCs:            {probs.sum():.1f}")
    print(f"Actual total YCs:               {actuals_binary.sum()}")

    # AUC
    from sklearn.metrics import roc_auc_score, brier_score_loss
    try:
        auc = roc_auc_score(actuals_binary, probs)
        brier = brier_score_loss(actuals_binary, probs)
        print(f"\nAUC-ROC:     {auc:.4f}")
        print(f"Brier Score: {brier:.4f}")
    except Exception:
        print("\nCould not compute AUC (likely single class in actuals)")

    # Calibration buckets
    print(f"\nCalibration:")
    print(f"  {'Bucket':15s} {'Count':>6s} {'Predicted':>10s} {'Actual':>8s} {'Diff':>8s}")
    print(f"  {'-'*50}")
    bins = [0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.40]
    for i in range(len(bins) - 1):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        n = mask.sum()
        if n > 0:
            pred_mean = probs[mask].mean()
            actual_rate = actuals_binary[mask].mean()
            diff = pred_mean - actual_rate
            print(f"  [{bins[i]*100:4.0f}%-{bins[i+1]*100:4.0f}%) "
                  f"{n:6d} {pred_mean*100:9.1f}% {actual_rate*100:7.1f}% {diff*100:+7.1f}%")

    # Top-N accuracy: if we picked the top N players per match, how many were booked?
    print(f"\nTop-N Precision (across all matches):")
    for n_top in [3, 5, 10]:
        top_preds = []
        for fix in fixtures:
            fixture_str = f"{fix['home']} vs {fix['away']}"
            match_data = [(p, yc) for p, yc, _, _, _, fs in all_preds if fs == fixture_str]
            match_data.sort(key=lambda x: -x[0])
            for p, yc in match_data[:n_top]:
                top_preds.append((p, yc))
        if top_preds:
            n_correct = sum(1 for _, yc in top_preds if yc > 0)
            n_total = len(top_preds)
            precision = n_correct / n_total if n_total > 0 else 0
            print(f"  Top {n_top:2d} per match: {n_correct}/{n_total} booked "
                  f"({precision*100:.1f}% precision)")

    # Per-match expected vs actual YCs
    print(f"\nPer-Match: Expected vs Actual YCs:")
    print(f"  {'Match':40s} {'Pred':>5s} {'Actual':>7s} {'Diff':>6s}")
    print(f"  {'-'*62}")
    total_pred = 0
    total_actual = 0
    for mr in match_results:
        diff = mr["predicted_yc"] - mr["actual_yc"]
        print(f"  {mr['fixture'][:40]:40s} {mr['predicted_yc']:5.1f} {mr['actual_yc']:7d} {diff:+5.1f}")
        total_pred += mr["predicted_yc"]
        total_actual += mr["actual_yc"]
    print(f"  {'TOTAL':40s} {total_pred:5.1f} {total_actual:7d} {total_pred-total_actual:+5.1f}")


if __name__ == "__main__":
    # Backtest on the most recent gameweek(s)
    backtest("2026-01-25", "2026-02-06")
