#!/usr/bin/env bash
cd "$(dirname "$0")" || exit 1
PY=python3; command -v python3 >/dev/null || PY=python
[ -d .venv ] || "$PY" -m venv .venv || { echo "Could not create venv"; exit 1; }
source .venv/bin/activate
if [ ! -f .venv/.deps_ok ]; then
  python -m pip install --quiet --upgrade pip && python -m pip install --quiet -r requirements.txt \
    && touch .venv/.deps_ok || { echo "Package install failed - see README Troubleshooting"; exit 1; }
fi
mkdir -p ~/.streamlit; [ -f ~/.streamlit/credentials.toml ] || printf '[general]\nemail = ""\n' > ~/.streamlit/credentials.toml
python app.py --prepare || exit 1
python -m streamlit run app.py
