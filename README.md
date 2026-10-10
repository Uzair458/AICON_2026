# Mobile-Money Fraud Triage (PaySim)

A review-queue tool for a bank or wallet fraud team. It scores transactions with your XGBoost
model, ranks the riskiest into a queue, explains each one, and lets an analyst decide.
You choose how much of the volume to review and see, live, how much fraud money that stops.

## Quick start (Windows)

1. Unzip this folder anywhere.
2. Copy the PaySim CSV into it (the main folder or the `data` folder).
   Usually named `PS_20174392719_1491204439457_log.csv`
   (download: https://www.kaggle.com/datasets/ealaxi/paysim1).
3. Double-click **`run.bat`**.

The first run creates a private Python environment, installs packages (a few minutes), scores the
dataset (about a minute) and opens the app at http://localhost:8501. Later runs start in seconds.
Mac / Linux: run `./run.sh` instead. Stop the app with Ctrl+C in the console window.

You need Python 3.10+ installed (python.org, tick "Add Python to PATH").

## What you see

| Part | What it does |
|---|---|
| Sidebar | Review-budget slider (riskiest X% of transactions), analyst cost per review, value of stopping fraud, a button that jumps to the most profitable setting, and the decision log (downloadable). |
| Five cards | Transactions reviewed, fraud cases caught, fraud dollars stopped (vs the dataset's existing rule), precision, net savings. They update as you move the slider. |
| Review queue tab | Riskiest transactions first. Click a row for balances, a chart of why it was flagged, an analyst summary, a draft customer message, and Approve / Block / Escalate buttons. |
| Model performance tab | Precision-recall curve with your setting and the recommended point, fraud dollars stopped at fixed budgets vs the existing rule, a **shortcut check**, and limitations. |

## 90-second demo

1. Start at the recommended setting. Say what share of fraud dollars is stopped and how little of the volume is reviewed.
2. Open the top case: read the reasons, then the summary. Point out the "numbers verified" line.
3. Drag the slider down and up; the cards and net savings move.
4. Press "Jump to recommended operating point".
5. Show the existing rule's result next to the model's.
6. Finish on the **Shortcut check** and **Limitations** (see below). Saying this yourself is a strength.

## How it works

- **Model.** `fraud_model.py` holds your XGBoost classifier. The app rebuilds its 16 input features from the CSV
  (only TRANSFER and CASH_OUT rows, the only types with fraud in PaySim).
- **Fair evaluation window.** The largest `step` any tree in the model splits on is 500, so the model was
  trained on steps up to 500. The app scores only **steps after 500**. This is inferred from the model file;
  if you know your training split differs, say so in your presentation rather than relying on this.
- **Reasons.** Each of the 1,000 riskiest transactions gets per-feature contributions computed by XGBoost itself
  (they add up exactly to the model's score), shown as the bar chart and the "top reason" column.
- **Text with a number check.** The summary and customer message are written from the model's facts. Every
  number in any text is checked against those facts; unverified numbers are flagged "Do not send".
  Optional: `pip install anthropic`, set `ANTHROPIC_API_KEY` (and optionally `ANTHROPIC_MODEL`), and a
  "Rewrite with Claude" button appears. Claude's text is used only if it passes the same number check.
- **Feature-definition check.** The model file does not record exactly how four yes/no features were defined,
  so on first run the app tests the plausible definitions on the model's own training rows and uses the best.
  Details are printed in the console and saved in `cache/scored_eval.json`.

## Be upfront about these (judges will ask)

- **The data is synthetic.** PaySim is simulated, so results will not transfer one-to-one to a real bank.
- **There is a shortcut in the data.** PaySim simulates fraud by emptying the sender's account. Most of
  your model's importance sits on features that detect exactly that (see the Shortcut check chart), which is
  why scores on PaySim can look near-perfect. Treat them as an upper bound.
- **Scores are not calibrated probabilities.** The model is class-weighted (`scale_pos_weight` about 458), so
  scores rank risk well but should not be read as "97% likely to be fraud".
- **The baseline is PaySim's own rule** (`isFlaggedFraud`), which flags almost nothing. That makes the
  comparison favourable; say what the baseline is.
- **The cost model is a simplification** (a stopped fraud saves its full amount).

## Optional: a stricter second model

To show what happens without the shortcut, train a LightGBM model that uses no balance-based features:

```
.venv\Scripts\activate
pip install -r optional\requirements-optional.txt
python optional\train_hard_mode.py
```

Restart the app and a **Model** switch appears in the sidebar. Expect clearly lower numbers; that is the
honest picture, and showing both is a stronger story than showing only the perfect one.

## Troubleshooting

- **"No PaySim file found"**: the CSV must be in this folder or `data`, with the original column names.
- **Package install fails (very new Python)**: install Python 3.12 or 3.13, delete the `.venv` folder, run `run.bat` again.
- **"Could not load fraud_model.py"**: needs `xgboost>=2.0` and `numpy>=2.0`. Run `pip install -U xgboost numpy` in the `.venv`.
- **If the numbers look wrong**: during the first run the console prints the model's PR-AUC on its own training
  rows. Below about 0.9 means your model's features are defined differently from `build_features()` in `app.py`;
  edit that function to match your training notebook, then delete the `cache` folder.
- **Rescore from scratch** (new CSV or changed code): delete the `cache` folder, or run `python app.py --prepare --force`.
- **Port already in use**: close the other Streamlit window, or run `python -m streamlit run app.py --server.port 8502`.

## Files

| File | Purpose |
|---|---|
| `run.bat` / `run.sh` | One-click setup and launch |
| `app.py` | Everything: data prep, scoring, evaluation, explanations, text check, UI |
| `fraud_model.py` | Your XGBoost model (embedded) |
| `requirements.txt` | Packages |
| `optional/` | Optional stricter LightGBM model |
| `cache/` | Created on first run (scored data and metadata) |
