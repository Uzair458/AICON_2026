"""
OPTIONAL: train a second, harder model without the PaySim "account-drain" shortcut features.

Run from the project folder (after the CSV is in place):
    python optional/train_hard_mode.py

It writes cache/scored_hard.parquet. Restart the app and a "Model" switch appears in the
sidebar so you can compare your XGBoost model with this stricter one.

Needs:  pip install -r optional/requirements-optional.txt
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
REQUIRED = ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig", "nameDest",
            "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud"]

# Balance-based features are left out on purpose: in PaySim they restate how the simulator
# creates fraud (the sender's account is emptied). Features that simply restate "the sender's account was emptied" are removed on purpose.
FEATURES = ["is_transfer", "amount_log", "hour_of_day", "dest_prior_txns", "orig_prior_txns"]
TRAIN_Q, VAL_Q = 0.65, 0.80
TOP_EXPLAIN, TOP_FACTORS = 1000, 4


def find_csv():
    for folder in (ROOT, ROOT / "data"):
        for p in sorted(folder.glob("*.csv"), key=lambda p: -p.stat().st_size):
            try:
                if set(REQUIRED).issubset(pd.read_csv(p, nrows=1).columns):
                    return p
            except Exception:
                pass
    return None


def build(df):
    df = df.sort_values(["step", "tx_id"]).reset_index(drop=True)
    df["is_transfer"] = (df["type"] == "TRANSFER").astype(int)
    df["amount_log"] = np.log1p(df["amount"])
    df["hour_of_day"] = df["step"] % 24
    df["dest_prior_txns"] = df.groupby("nameDest").cumcount()      # backward-looking only
    df["orig_prior_txns"] = df.groupby("nameOrig").cumcount()
    return df


def main():
    csv = find_csv()
    if csv is None:
        sys.exit(f"No PaySim CSV found in {ROOT} or {ROOT / 'data'}.")
    print(f"Reading {csv.name} ...")
    df = pd.read_csv(csv, usecols=REQUIRED)
    df["tx_id"] = df.index
    df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()
    df = build(df)

    c1, c2 = int(df["step"].quantile(TRAIN_Q)), int(df["step"].quantile(VAL_Q))
    train, val, test = df[df.step <= c1], df[(df.step > c1) & (df.step <= c2)], df[df.step > c2].copy()
    print(f"Time split: train step<={c1}, validation {c1 + 1}-{c2}, test step>{c2} "
          f"({len(test):,} rows, {int(test.isFraud.sum()):,} fraud)")

    params = dict(objective="binary", metric="average_precision", learning_rate=0.05, num_leaves=31,
                  min_child_samples=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  scale_pos_weight=10, seed=42, verbose=-1)
    dtr = lgb.Dataset(train[FEATURES], train["isFraud"])
    dva = lgb.Dataset(val[FEATURES], val["isFraud"], reference=dtr)
    model = lgb.train(params, dtr, 1000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])

    margin = model.predict(test[FEATURES], num_iteration=model.best_iteration, raw_score=True)
    test["score"] = 1.0 / (1.0 + np.exp(-margin))

    top = np.argsort(-margin, kind="stable")[:TOP_EXPLAIN]
    contrib = model.predict(test[FEATURES].iloc[top], num_iteration=model.best_iteration, pred_contrib=True)
    shap_json = np.full(len(test), "", dtype=object)
    X = test[FEATURES].to_numpy()
    for r, idx in enumerate(top):
        order = np.argsort(-np.abs(contrib[r, :-1]))[:TOP_FACTORS]
        shap_json[idx] = json.dumps([{"feature": FEATURES[j], "value": round(float(X[idx, j]), 4),
                                      "shap": round(float(contrib[r, j]), 4)} for j in order])
    test["shap_json"] = shap_json

    CACHE.mkdir(exist_ok=True)
    cols = ["tx_id", "step", "type", "amount", "nameOrig", "nameDest", "oldbalanceOrg", "newbalanceOrig",
            "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud", "score", "shap_json"]
    test[cols].to_parquet(CACHE / "scored_hard.parquet", index=False)
    print(f"Saved {CACHE / 'scored_hard.parquet'}. Restart the app to see the Model switch.")


if __name__ == "__main__":
    main()
