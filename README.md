# MatchrAI (Phase-2 PoC) — Streamlit Resume Analyzer & Job Matcher

## Setup
python -m venv .venv
source .venv/bin/activate  # (Windows) .venv\Scripts\activate
pip install -r requirements.txt

## Run
streamlit run app.py

## Notes
- If sentence-transformers is installed, semantic matching uses embeddings.
- Otherwise it falls back to TF-IDF cosine similarity.