# MatchrAI (Capstone Project Version) — Streamlit Resume Analyzer & Job Matcher

## Setup
python -m venv .venv
source .venv/bin/activate  # (Windows) .venv\Scripts\activate
pip install -r requirements.txt

## Run
streamlit run app.py

## Notes
- If sentence-transformers is installed, semantic matching uses embeddings.
- Otherwise it falls back to TF-IDF cosine similarity.
- PoC version used generic sentence-transformer model for semantic similarity
- Capstone version customizes it by fine-tuning the embedding model on resume–job-description pairs labeled as weak, medium, or strong matches.
- Vector DB adds storage layer that saves resume/JD embeddings and retrieves similar past matches.
