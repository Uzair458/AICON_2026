"""
Mobile-Money Fraud Triage (PaySim) - single-file app.

    python app.py --prepare        score the dataset once (cached in ./cache)
    streamlit run app.py           open the review-queue UI

Everything lives here: loading the PaySim CSV, rebuilding the model's features,
scoring with the XGBoost model in fraud_model.py, evaluation, per-transaction
explanations, grounded text generation (with a number checker) and the UI.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / "cache"
SCORED_PATH = CACHE_DIR / "scored_eval.parquet"
META_PATH = CACHE_DIR / "scored_eval.json"
HARD_PATH = CACHE_DIR / "scored_hard.parquet"      # optional second model (see README)
CACHE_VERSION = 3

KEEP_TYPES = ["TRANSFER", "CASH_OUT"]   # the only types where PaySim has fraud
TOP_EXPLAIN = 1000                      # rows that get a per-transaction explanation
TOP_FACTORS = 4
QUEUE_MAX_ROWS = 500
FALLBACK_CUTOFF = 500                   # used only if the model never splits on `step`
CSV_COLUMNS = ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
               "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud"]

FEATURE_LABELS = {
    "step": "Time of transaction",
    "amount": "Amount",
    "log_amount": "Amount (log scale)",
    "oldbalanceOrg": "Sender balance before",
    "newbalanceOrig": "Sender balance after",
    "oldbalanceDest": "Receiver balance before",
    "newbalanceDest": "Receiver balance after",
    "origin_balance_delta": "Change in sender balance",
    "destination_balance_delta": "Change in receiver balance",
    "amount_to_origin_balance": "Amount / sender balance",
    "amount_to_destination_balance": "Amount / receiver balance",
    "origin_zeroed": "Sender account emptied",
    "destination_zeroed": "Receiver balance zeroed",
    "origin_balance_zero": "Sender balance was zero",
    "destination_balance_zero": "Receiver balance was zero",
    "is_transfer": "Transaction is a transfer",
}
BINARY_FEATURES = {"origin_zeroed", "destination_zeroed", "origin_balance_zero",
                   "destination_balance_zero", "is_transfer"}
RATIO_FEATURES = {"amount_to_origin_balance", "amount_to_destination_balance"}


def log(msg: str) -> None:
    print(msg, flush=True)


# =============================================================================
# 1. Locate the dataset
# =============================================================================
def find_csv() -> Path | None:
    env = os.environ.get("PAYSIM_CSV")
    if env and Path(env).is_file():
        return Path(env)
    found = []
    for folder in (APP_DIR, APP_DIR / "data"):
        if not folder.is_dir():
            continue
        for p in folder.glob("*.csv"):
            try:
                head = pd.read_csv(p, nrows=2)
            except Exception:
                continue
            if set(CSV_COLUMNS).issubset(head.columns):
                found.append(p)
    return max(found, key=lambda p: p.stat().st_size) if found else None


# =============================================================================
# 2. Model + feature engineering (must match how the model was trained)
# =============================================================================
def load_model():
    warnings.filterwarnings("ignore", category=UserWarning)
    sys.path.insert(0, str(APP_DIR))
    try:
        import fraud_model                      # unpickles the embedded XGBClassifier
    except Exception as exc:                    # pragma: no cover
        raise RuntimeError(
            "Could not load fraud_model.py. This usually means xgboost or numpy is missing "
            f"or too old (need xgboost>=2.0, numpy>=2.0). Original error: {exc}") from exc
    model = fraud_model.model
    try:
        model.set_params(device="cpu")          # it was trained on a GPU
    except Exception:
        pass
    return model


def model_signature() -> str:
    p = APP_DIR / "fraud_model.py"
    return f"{p.stat().st_size}-{hashlib.md5(p.read_bytes()[:200000]).hexdigest()[:10]}"


def best_iteration(model) -> int:
    bi = getattr(model, "best_iteration", None)
    return int(bi) if bi is not None else model.get_booster().num_boosted_rounds() - 1


def infer_train_cutoff(booster) -> int | None:
    """Largest `step` value any tree splits on = the end of the model's training window."""
    mx = None
    for tree in booster.get_dump(dump_format="json"):
        stack = [json.loads(tree)]
        while stack:
            node = stack.pop()
            if "split" in node:
                if node["split"] == "step":
                    mx = max(mx if mx is not None else 0.0, float(node["split_condition"]))
                stack.extend(node["children"])
    return int(round(mx)) if mx is not None else None


# The model's 4 binary flags could be defined in a couple of ways; the pipeline picks
# the definition that best reproduces the model on its own training window.
FLAG_OPTIONS = {
    "origin_zeroed": ["old>0&new==0", "new==0"],
    "destination_zeroed": ["old>0&new==0", "new==0"],
    "origin_balance_zero": ["old==0", "old==0&new==0"],
    "destination_balance_zero": ["old==0", "old==0&new==0"],
}
DEFAULT_FLAGS = {k: v[0] for k, v in FLAG_OPTIONS.items()}


def _flag(expr: str, old: np.ndarray, new: np.ndarray) -> np.ndarray:
    if expr == "old>0&new==0":
        return (old > 0) & (new == 0)
    if expr == "new==0":
        return new == 0
    if expr == "old==0":
        return old == 0
    if expr == "old==0&new==0":
        return (old == 0) & (new == 0)
    raise ValueError(expr)


def build_features(df: pd.DataFrame, feature_names: list[str], flags: dict | None = None) -> pd.DataFrame:
    flags = flags or DEFAULT_FLAGS
    amt = df["amount"].to_numpy(np.float64)
    oo = df["oldbalanceOrg"].to_numpy(np.float64)
    no = df["newbalanceOrig"].to_numpy(np.float64)
    od = df["oldbalanceDest"].to_numpy(np.float64)
    nd = df["newbalanceDest"].to_numpy(np.float64)
    feats = {
        "step": df["step"].to_numpy(np.float64),
        "amount": amt,
        "log_amount": np.log1p(amt),
        "oldbalanceOrg": oo,
        "newbalanceOrig": no,
        "oldbalanceDest": od,
        "newbalanceDest": nd,
        "origin_balance_delta": no - oo,
        "destination_balance_delta": nd - od,
        "amount_to_origin_balance": amt / (oo + 1.0),
        "amount_to_destination_balance": amt / (od + 1.0),
        "origin_zeroed": _flag(flags["origin_zeroed"], oo, no),
        "destination_zeroed": _flag(flags["destination_zeroed"], od, nd),
        "origin_balance_zero": _flag(flags["origin_balance_zero"], oo, no),
        "destination_balance_zero": _flag(flags["destination_balance_zero"], od, nd),
        "is_transfer": (df["type"] == "TRANSFER").to_numpy(),
    }
    X = pd.DataFrame({k: np.asarray(v, dtype=np.float64) for k, v in feats.items()})
    missing = [f for f in feature_names if f not in X.columns]
    if missing:
        raise RuntimeError(f"The model expects features this app cannot build: {missing}")
    return X[feature_names]


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="stable")
    ys = y[order]
    tp = np.cumsum(ys)
    prec = tp / (np.arange(len(ys)) + 1)
    return float(prec[ys == 1].sum() / y.sum())


def _margin(booster, X: pd.DataFrame, best_it: int) -> np.ndarray:
    import xgboost as xgb
    dm = xgb.DMatrix(X, feature_names=list(X.columns))
    return booster.predict(dm, output_margin=True, iteration_range=(0, best_it + 1)).astype(np.float64)


def calibrate_flags(booster, feature_names, best_it, sample: pd.DataFrame) -> tuple[dict, list]:
    """Pick the flag definitions that best reproduce the model on rows it was trained on."""
    keys = list(FLAG_OPTIONS)
    rows = []
    for combo in itertools.product(*[FLAG_OPTIONS[k] for k in keys]):
        flags = dict(zip(keys, combo))
        X = build_features(sample, feature_names, flags)
        ap = average_precision(sample["isFraud"].to_numpy(), _margin(booster, X, best_it))
        rows.append({"flags": flags, "train_ap": ap})
    top = max(r["train_ap"] for r in rows)
    best = next(r for r in rows if r["train_ap"] >= top - 1e-4)     # first (default-ordered) near-best
    return best["flags"], rows


# =============================================================================
# 3. Prepare: read CSV -> score the hold-out window -> explanations -> cache
# =============================================================================
def prepare(csv_path: Path, force: bool = False) -> dict:
    CACHE_DIR.mkdir(exist_ok=True)
    sig = {"csv": str(csv_path), "csv_size": csv_path.stat().st_size,
           "csv_mtime": int(csv_path.stat().st_mtime), "model": model_signature(),
           "cache_version": CACHE_VERSION}
    if not force and SCORED_PATH.exists() and META_PATH.exists():
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        if all(meta.get(k) == v for k, v in sig.items()):
            log("Cache is up to date - nothing to do.")
            return meta

    t0 = time.time()
    log("Loading model ...")
    model = load_model()
    booster = model.get_booster()
    feature_names = list(booster.feature_names)
    best_it = best_iteration(model)
    cutoff = infer_train_cutoff(booster)
    if cutoff is None:
        cutoff = FALLBACK_CUTOFF
        log(f"  The model never splits on step; assuming it was trained on steps <= {cutoff}.")
    else:
        log(f"  The model was trained on steps <= {cutoff}; scoring steps > {cutoff} as the hold-out window.")

    log(f"Reading {csv_path.name} (this takes ~20-60 seconds) ...")
    dtypes = {"step": "int32", "type": "category", "amount": "float64", "oldbalanceOrg": "float64",
              "newbalanceOrig": "float64", "oldbalanceDest": "float64", "newbalanceDest": "float64",
              "isFraud": "int8", "isFlaggedFraud": "int8"}
    ev_parts, tr_parts, offset, total = [], [], 0, 0
    for chunk in pd.read_csv(csv_path, usecols=CSV_COLUMNS, dtype=dtypes, chunksize=1_000_000):
        chunk["tx_id"] = np.arange(offset, offset + len(chunk), dtype=np.int64)
        offset += len(chunk)
        total = offset
        chunk = chunk[chunk["type"].isin(KEEP_TYPES)]
        chunk["type"] = chunk["type"].astype(str)
        ev_parts.append(chunk[chunk["step"] > cutoff])
        tr_parts.append(chunk[chunk["step"] <= cutoff].drop(columns=["nameOrig", "nameDest"]))
    ev = pd.concat(ev_parts, ignore_index=True)
    tr = pd.concat(tr_parts, ignore_index=True)
    del ev_parts, tr_parts
    if ev.empty:
        raise RuntimeError(f"No TRANSFER/CASH_OUT rows after step {cutoff}. Is this the full PaySim file "
                           "(about 6.3 million rows, steps 1-743)?")
    log(f"  {total:,} rows read; hold-out window has {len(ev):,} transactions "
        f"and {int(ev['isFraud'].sum()):,} fraud cases.")

    log("Checking feature definitions against the model's training window ...")
    fraud_rows = tr[tr["isFraud"] == 1]
    other = tr[tr["isFraud"] == 0]
    sample = pd.concat([fraud_rows.sample(min(len(fraud_rows), 4000), random_state=0),
                        other.sample(min(len(other), 60000), random_state=0)], ignore_index=True)
    flags, cal_rows = calibrate_flags(booster, feature_names, best_it, sample)
    cal_ap = max(r["train_ap"] for r in cal_rows)
    log(f"  Best match: PR-AUC {cal_ap:.4f} on training-window rows. Flags: {flags}")
    if cal_ap < 0.9:
        log("  WARNING: the model does not reproduce well on its own training rows. Its features may be "
            "defined differently from this app's - see README, section 'If the numbers look wrong'.")
    del tr, sample

    log("Scoring the hold-out window ...")
    X = build_features(ev, feature_names, flags)
    margin = _margin(booster, X, best_it)
    score = 1.0 / (1.0 + np.exp(-margin))

    log(f"Explaining the {TOP_EXPLAIN} riskiest transactions ...")
    import xgboost as xgb
    order = np.argsort(-margin, kind="stable")[:TOP_EXPLAIN]
    dm = xgb.DMatrix(X.iloc[order], feature_names=feature_names)
    contrib = booster.predict(dm, pred_contribs=True, iteration_range=(0, best_it + 1))
    shap_json = np.full(len(ev), "", dtype=object)
    for r, idx in enumerate(order):
        top = np.argsort(-np.abs(contrib[r, :-1]))[:TOP_FACTORS]
        shap_json[idx] = json.dumps([{"feature": feature_names[j],
                                      "value": round(float(X.iloc[idx, j]), 4),
                                      "shap": round(float(contrib[r, j]), 4)} for j in top])

    out = ev[["tx_id", "step", "type", "amount", "nameOrig", "nameDest", "oldbalanceOrg", "newbalanceOrig",
              "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud"]].copy()
    out["score"] = score
    out["shap_json"] = shap_json
    out.to_parquet(SCORED_PATH, index=False)

    gain = booster.get_score(importance_type="gain")
    tot = sum(gain.values()) or 1.0
    meta = dict(sig)
    meta.update({
        "train_cutoff": cutoff, "flags": flags, "calibration_ap": cal_ap,
        "calibration": [{"flags": r["flags"], "train_ap": r["train_ap"]} for r in cal_rows],
        "eval_rows": int(len(out)), "eval_fraud": int(out["isFraud"].sum()),
        "total_rows": int(total), "best_iteration": best_it,
        "importance": {k: v / tot for k, v in sorted(gain.items(), key=lambda kv: -kv[1])},
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"Done in {time.time() - t0:.0f}s. Saved {SCORED_PATH.name}.")
    return meta


# =============================================================================
# 4. Grounded text: facts -> template or Claude -> number check
# =============================================================================
def _money(x: float) -> str:
    return f"{x:,.2f}"


def factor_value_text(feat: str, val: float) -> str:
    if feat in BINARY_FEATURES:
        return "yes" if val >= 0.5 else "no"
    if feat == "step":
        return f"day {int((val - 1) // 24) + 1}, hour {int((val - 1) % 24)}"
    if feat in RATIO_FEATURES:
        return f"{val:,.2f}x"
    if feat == "log_amount":
        return f"{val:.2f}"
    return _money(val)


def pretty(feat: str) -> str:
    return FEATURE_LABELS.get(feat, feat.replace("_", " ").capitalize())


def build_facts(tx: pd.Series, rank: int, total: int, factors: list[dict]) -> dict:
    step = int(tx["step"])
    return {
        "transaction": {
            "type": "transfer" if tx["type"] == "TRANSFER" else "cash-out",
            "amount": _money(float(tx["amount"])),
            "day": str((step - 1) // 24 + 1), "hour": str((step - 1) % 24),
            "sender_balance_before": _money(float(tx["oldbalanceOrg"])),
            "sender_balance_after": _money(float(tx["newbalanceOrig"])),
            "receiver_balance_before": _money(float(tx["oldbalanceDest"])),
            "receiver_balance_after": _money(float(tx["newbalanceDest"])),
        },
        "risk": {"score": f"{float(tx['score']):.2f}", "queue_rank": f"{rank:,}", "queue_size": f"{total:,}"},
        "top_factors": [{"factor": pretty(f["feature"]), "value": factor_value_text(f["feature"], f["value"]),
                         "effect": "raises risk" if f["shap"] >= 0 else "lowers risk"} for f in factors],
    }


def template_texts(facts: dict) -> dict:
    t, r = facts["transaction"], facts["risk"]
    drivers = "; ".join(f"{f['factor']} ({f['value']}) {f['effect']}" for f in facts["top_factors"][:3])
    summary = (f"A {t['type']} of {t['amount']} on day {t['day']} at hour {t['hour']}. "
               f"Risk score {r['score']}, ranked {r['queue_rank']} of {r['queue_size']} in the review queue. "
               f"Sender balance went from {t['sender_balance_before']} to {t['sender_balance_after']}; "
               f"receiver balance from {t['receiver_balance_before']} to {t['receiver_balance_after']}. "
               f"Main drivers: {drivers or 'no explanation cached for this row'}. "
               "Recommended action: hold and review before releasing.")
    alert = (f"We noticed a {t['type']} of {t['amount']} from your account that looks unusual. "
             "If you did not make it, please contact us right away so we can protect your account. "
             "If it was you, no action is needed.")
    return {"summary": summary, "alert": alert}


_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> list[float]:
    out = []
    for m in _NUM.findall(text):
        try:
            out.append(float(m.rstrip(",").replace(",", "")))
        except ValueError:
            pass
    return out


def allowed_numbers(facts: dict) -> list[float]:
    base = _numbers(json.dumps(facts))
    allowed = set()
    for v in base:
        allowed.update({v, round(v), round(v, 1), round(v, 2), v * 100, round(v * 100), round(v * 100, 1)})
    return sorted(allowed)


def verify_text(text: str, facts: dict) -> tuple[bool, list[float]]:
    allowed = allowed_numbers(facts)
    bad = [x for x in _numbers(text)
           if not any(abs(x - a) <= max(0.011, 0.002 * abs(a)) for a in allowed)]
    return (len(bad) == 0), bad


def llm_ready() -> bool:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
        return True
    except Exception:
        return False


LLM_SYSTEM = (
    "You assist a bank fraud-review team. You are given JSON facts about one flagged transaction. "
    "Write (1) an analyst summary of at most 3 sentences and (2) a short, calm message to the customer. "
    "Use ONLY the facts provided. Copy every number exactly as it is written in the JSON. "
    "Do not invent amounts, dates, names, reasons or phone numbers. Do not mention the model or scores "
    "in the customer message. Reply with a single JSON object with keys \"summary\" and \"alert\"."
)


def llm_texts(facts: dict) -> dict:
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5"), max_tokens=600, system=LLM_SYSTEM,
        messages=[{"role": "user", "content": json.dumps(facts)}])
    raw = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    obj = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
    return {"summary": str(obj["summary"]), "alert": str(obj["alert"])}


# =============================================================================
# 5. UI
# =============================================================================
def load_scored(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.rename(columns={"newbalanceOrg": "newbalanceOrig"})
    for c in ("shap_json",):
        if c not in df.columns:
            df[c] = ""
    return df


def top_reason(shap_json: str) -> str:
    if not shap_json:
        return "-"
    f = max(json.loads(shap_json), key=lambda x: abs(x["shap"]))
    return pretty(f["feature"])


def risk_label(score: float) -> str:
    return "High" if score >= 0.8 else "Medium" if score >= 0.5 else "Low"


def main() -> None:
    import altair as alt
    import streamlit as st

    st.set_page_config(page_title="Fraud Triage", page_icon="🛡️", layout="wide")
    try:
        major, minor = [int(x) for x in st.__version__.split(".")[:2]]
        S = {"width": "stretch"} if (major, minor) >= (1, 50) else {"use_container_width": True}
    except Exception:
        S = {"use_container_width": True}
    RED, GREY, BLUE = "#d62728", "#8c8c8c", "#1f77b4"

    # ---- data source ------------------------------------------------------
    csv_path = find_csv()
    have_main = SCORED_PATH.exists() and META_PATH.exists()
    if csv_path is None and not have_main:
        st.title("🛡️ Mobile-Money Fraud Triage")
        st.error("No PaySim file found.")
        st.markdown(f"Copy the PaySim CSV (for example `PS_20174392719_1491204439457_log.csv`) into\n\n"
                    f"`{APP_DIR}`\n\nor its `data` subfolder, then reload this page. Download: "
                    "kaggle.com/datasets/ealaxi/paysim1")
        return
    if csv_path is not None:
        with st.spinner("Scoring the dataset (first run only, about a minute) ..."):
            @st.cache_resource(show_spinner=False)
            def _prep(path: str, size: int, mtime: float):
                return prepare(Path(path))
            meta_main = _prep(str(csv_path), csv_path.stat().st_size, csv_path.stat().st_mtime)
    else:
        meta_main = json.loads(META_PATH.read_text(encoding="utf-8"))

    source = "Your XGBoost model"
    if HARD_PATH.exists():
        source = st.sidebar.radio("Model", ["Your XGBoost model", "Hard-mode LightGBM (no shortcut features)"],
                                  help="The second option appears after running optional/train_hard_mode.py")
    use_hard = source.startswith("Hard")

    @st.cache_data(show_spinner=False)
    def _load(path: str, mtime: float) -> pd.DataFrame:
        raw = load_scored(Path(path))
        d = raw.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)
        d["rank"] = np.arange(1, len(d) + 1)
        d["cum_fraud"] = d["isFraud"].cumsum()
        d["cum_fraud_amt"] = (d["amount"] * d["isFraud"]).cumsum()
        return d

    path = HARD_PATH if use_hard else SCORED_PATH
    d = _load(str(path), path.stat().st_mtime)
    n = len(d)
    total_fraud = int(d["isFraud"].sum())
    total_fraud_amt = float(d.loc[d["isFraud"] == 1, "amount"].sum())
    if total_fraud == 0:
        st.error("No fraud cases in the scored window, so metrics cannot be computed.")
        return

    st.session_state.setdefault("budget", 1.0)
    st.session_state.setdefault("decisions", [])

    # ---- sidebar ------------------------------------------------------------
    with st.sidebar:
        st.header("Controls")
        review_cost = st.number_input("Analyst cost per review ($)", 1.0, 5000.0, 25.0, 1.0)
        miss_mult = st.number_input("Value of stopping fraud (x amount)", 0.1, 5.0, 1.0, 0.1,
                                    help="1.0 means stopping a fraud saves exactly the transaction amount.")
        ks_all = np.arange(1, n + 1)
        net_curve = d["cum_fraud_amt"].to_numpy() * miss_mult - ks_all * review_cost
        k_best = int(net_curve.argmax()) + 1
        best_pct = round(min(max(k_best / n * 100, 0.05), 5.0), 2)

        def jump():
            st.session_state.budget = best_pct

        st.slider("Review budget (% of transactions)", 0.05, 5.0, step=0.05, key="budget",
                  help="Analyst capacity: review the riskiest X% of transactions.")
        st.button("🎯 Jump to recommended operating point", on_click=jump, **S)
        st.caption(f"Recommended: top {k_best / n * 100:.2f}% ({k_best:,} reviews)")
        k = max(1, int(n * st.session_state.budget / 100))
        st.caption(f"Equivalent score threshold: **{d['score'].iloc[k - 1]:.3f}**")
        st.divider()
        st.subheader("Decision log")
        if st.session_state.decisions:
            log_df = pd.DataFrame(st.session_state.decisions)
            st.dataframe(log_df, hide_index=True, **S)
            st.download_button("Download decision log (CSV)", log_df.to_csv(index=False).encode("utf-8"),
                               "decision_log.csv", "text/csv", **S)
        else:
            st.caption("No decisions yet.")

    # ---- header + banner ---------------------------------------------------------
    st.title("🛡️ Mobile-Money Fraud Triage")
    if use_hard:
        window = f"steps {int(d['step'].min())}-{int(d['step'].max())} (its own hold-out window)"
    else:
        window = (f"steps after {meta_main['train_cutoff']} (the model was trained on earlier steps)")
    st.warning(f"Synthetic data (PaySim); results will not transfer one-to-one to a real bank. "
               f"Scored window: {window}, {n:,} transactions, {total_fraud:,} fraud cases.", icon="⚠️")

    caught = int(d["cum_fraud"].iloc[k - 1])
    caught_amt = float(d["cum_fraud_amt"].iloc[k - 1])
    precision = caught / k
    count_recall = caught / total_fraud
    dollar_recall = caught_amt / total_fraud_amt
    net = caught_amt * miss_mult - k * review_cost
    base = d[d["isFlaggedFraud"] == 1]
    base_dollar = float(base.loc[base["isFraud"] == 1, "amount"].sum()) / total_fraud_amt
    base_count = int(base["isFraud"].sum()) / total_fraud

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Transactions reviewed", f"{k:,}", f"{st.session_state.budget:.2f}% of volume", delta_color="off")
    m2.metric("Fraud cases caught", f"{caught:,} / {total_fraud:,}", f"{count_recall:.0%} of cases", delta_color="off")
    m3.metric("Fraud $ stopped", f"{dollar_recall:.1%}", f"existing rule {base_dollar:.1%}", delta_color="off")
    m4.metric("Precision", f"{precision:.1%}")
    m5.metric("Net savings", f"${net:,.0f}")

    tab_queue, tab_perf = st.tabs(["📋 Review queue", "📈 Model performance"])

    # ---- review queue + detail ----------------------------------------------------
    with tab_queue:
        left, right = st.columns([3, 2], gap="large")
        queue = d.head(min(k, QUEUE_MAX_ROWS)).copy()
        view = pd.DataFrame({"rank": queue["rank"], "risk": queue["score"].map(risk_label),
                             "score": queue["score"], "amount": queue["amount"], "type": queue["type"],
                             "top reason": queue["shap_json"].map(top_reason)})
        with left:
            st.subheader(f"Queue (top {len(queue):,} of {k:,})")
            if k > QUEUE_MAX_ROWS:
                st.caption(f"Showing the {QUEUE_MAX_ROWS} riskiest; the metrics above cover all {k:,}.")
            event = st.dataframe(
                view, on_select="rerun", selection_mode="single-row", hide_index=True, height=520, **S,
                column_config={
                    "score": st.column_config.ProgressColumn("score", min_value=0.0, max_value=1.0, format="%.2f"),
                    "amount": st.column_config.NumberColumn("amount", format="$%.0f")})
            st.download_button("Download this queue (CSV)", queue.drop(columns=["cum_fraud", "cum_fraud_amt"])
                               .to_csv(index=False).encode("utf-8"), "review_queue.csv", "text/csv")

        with right:
            st.subheader("Transaction detail")
            rows = event.selection.rows
            if not rows:
                st.info("Select a row in the queue to see details.")
            else:
                tx = queue.iloc[rows[0]]
                c1, c2 = st.columns(2)
                c1.metric("Risk score", f"{tx['score']:.2f}")
                c2.metric("Amount", f"${tx['amount']:,.2f}")
                st.caption(f"{tx['type']} - step {int(tx['step'])} - {tx['nameOrig']} -> {tx['nameDest']}")
                st.dataframe(pd.DataFrame({"": ["Sender", "Receiver"],
                                           "Before": [tx["oldbalanceOrg"], tx["oldbalanceDest"]],
                                           "After": [tx["newbalanceOrig"], tx["newbalanceDest"]]}),
                             hide_index=True, **S)
                factors = json.loads(tx["shap_json"]) if tx["shap_json"] else []
                if factors:
                    f = pd.DataFrame(factors)
                    f["label"] = f["feature"].map(pretty)
                    f["direction"] = np.where(f["shap"] >= 0, "raises risk", "lowers risk")
                    st.altair_chart(
                        alt.Chart(f).mark_bar().encode(
                            x=alt.X("shap:Q", title="Contribution to risk"),
                            y=alt.Y("label:N", sort="-x", title=None),
                            color=alt.Color("direction:N", legend=None,
                                            scale=alt.Scale(domain=["raises risk", "lowers risk"], range=[RED, BLUE])),
                            tooltip=["label", "value", "shap"]).properties(height=150), **S)
                facts = build_facts(tx, int(tx["rank"]), k, factors)
                texts = st.session_state.get(f"ai_{tx['tx_id']}") or template_texts(facts)
                source_txt = "Claude (checked)" if st.session_state.get(f"ai_{tx['tx_id']}") else "template"
                ok_s, bad_s = verify_text(texts["summary"], facts)
                ok_a, bad_a = verify_text(texts["alert"], facts)
                st.markdown(f"**Analyst summary** ({source_txt})")
                st.info(texts["summary"])
                st.markdown("**Customer alert (draft)**")
                st.success(texts["alert"])
                if ok_s and ok_a:
                    st.caption("✅ Every number in the text appears in the model facts.")
                else:
                    st.caption(f"⚠️ Unverified numbers: {bad_s + bad_a}. Do not send.")
                if llm_ready():
                    if st.button("✨ Rewrite with Claude", key=f"llm_{tx['tx_id']}"):
                        try:
                            new = llm_texts(facts)
                            if all(verify_text(new[x], facts)[0] for x in ("summary", "alert")):
                                st.session_state[f"ai_{tx['tx_id']}"] = new
                                st.rerun()
                            else:
                                st.warning("Claude's text failed the number check, so it was discarded.")
                        except Exception as exc:
                            st.warning(f"Claude call failed: {exc}")
                else:
                    st.caption("Optional: pip install anthropic and set ANTHROPIC_API_KEY to enable the "
                               "Claude rewrite button.")
                b1, b2, b3 = st.columns(3)
                for col, label in zip((b1, b2, b3), ("Approve", "Block", "Escalate")):
                    if col.button(label, key=f"{label}_{tx['tx_id']}", **S):
                        st.session_state.decisions.append({"tx": int(tx["tx_id"]), "amount": float(tx["amount"]),
                                                           "decision": label})
                        st.rerun()
                with st.expander("Reveal ground truth (demo only)"):
                    st.write("Actual fraud" if tx["isFraud"] == 1 else "Legitimate")

    # ---- performance ---------------------------------------------------------------------
    with tab_perf:
        prec_all = d["cum_fraud"].to_numpy() / np.arange(1, n + 1)
        rec_all = d["cum_fraud"].to_numpy() / total_fraud
        pts_idx = np.unique(np.concatenate([np.geomspace(1, n, 1500).astype(int) - 1, [k - 1, k_best - 1]]))
        pr = pd.DataFrame({"recall": rec_all[pts_idx], "precision": prec_all[pts_idx]})
        dots = pd.DataFrame({"recall": [rec_all[k - 1], rec_all[k_best - 1]],
                             "precision": [prec_all[k - 1], prec_all[k_best - 1]],
                             "point": ["Current setting", "Recommended"]})
        line = alt.Chart(pr).mark_line(color=BLUE).encode(
            x=alt.X("recall:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("precision:Q", scale=alt.Scale(domain=[0, 1.02])))
        pts = alt.Chart(dots).mark_point(size=160, filled=True).encode(
            x="recall:Q", y="precision:Q", tooltip=["point", "recall", "precision"],
            color=alt.Color("point:N", scale=alt.Scale(range=[RED, "#2ca02c"]), title=None))
        p1, p2 = st.columns(2, gap="large")
        with p1:
            st.subheader("Precision-recall curve")
            st.altair_chart((line + pts).properties(height=340), **S)
            st.caption(f"PR-AUC on the scored window: {average_precision(d['isFraud'].to_numpy(), d['score'].to_numpy()):.3f} "
                       f"(random guessing: {total_fraud / n:.3f})")
        with p2:
            st.subheader("Fraud $ stopped at fixed budgets")
            bars = []
            for pct in (0.5, 1.0, 2.0):
                kk = max(1, int(n * pct / 100))
                bars.append({"budget": f"Top {pct}%", "share": d["cum_fraud_amt"].iloc[kk - 1] / total_fraud_amt})
            bars.append({"budget": "Existing rule", "share": base_dollar})
            st.altair_chart(
                alt.Chart(pd.DataFrame(bars)).mark_bar().encode(
                    x=alt.X("budget:N", sort=None, title=None),
                    y=alt.Y("share:Q", axis=alt.Axis(format="%"), title="Share of fraud $ stopped"),
                    color=alt.condition(alt.datum.budget == "Existing rule", alt.value(GREY), alt.value(BLUE)),
                    tooltip=["budget", alt.Tooltip("share:Q", format=".1%")]).properties(height=340), **S)

        if not use_hard:
            imp = pd.Series(meta_main["importance"]).head(8)
            top3 = imp.head(3)
            st.subheader("Shortcut check: what the model relies on")
            st.altair_chart(
                alt.Chart(pd.DataFrame({"feature": [pretty(i) for i in imp.index], "share": imp.values})).mark_bar(
                    color=BLUE).encode(x=alt.X("share:Q", axis=alt.Axis(format="%"), title="Share of model importance"),
                                       y=alt.Y("feature:N", sort="-x", title=None)).properties(height=220), **S)
            msg = (f"The top 3 features ({', '.join(pretty(i) for i in top3.index)}) carry {top3.sum():.0%} of the "
                   "model's importance. In PaySim, fraud is simulated by emptying the sender's account, so these "
                   "signals are almost a label leak. Read the scores above as an upper bound, not as expected "
                   "real-world performance.")
            (st.warning if top3.sum() > 0.7 else st.info)(msg)

        st.subheader("Limitations")
        st.markdown(
            "- Trained and scored on synthetic PaySim data. Real-world performance would differ.\n"
            "- The existing-rule baseline is PaySim's own `isFlaggedFraud` column.\n"
            "- Scores come from a class-weighted model, so they rank risk but are not calibrated probabilities.\n"
            "- The cost model is a simplification: a stopped fraud saves its full amount.\n"
            "- Summaries are built from the model's own facts and number-checked, but still need analyst review.")


if __name__ == "__main__":
    if "--prepare" in sys.argv:
        csv = find_csv()
        if csv is None:
            log(f"No PaySim CSV found. Copy it into {APP_DIR} or {APP_DIR / 'data'} and run again.")
            sys.exit(1)
        try:
            prepare(csv, force="--force" in sys.argv)
        except Exception as exc:
            log(f"ERROR: {exc}")
            sys.exit(1)
    else:
        main()
