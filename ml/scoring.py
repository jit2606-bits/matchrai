from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from .utils import tokenize_simple


# adding function for phase3 to determine ATS skill overlap
def ats_skill_overlap(resume_skills: set[str], jd_skills: set[str]) -> float:
    if not jd_skills:
        return 0.0
    return len(resume_skills & jd_skills) / len(jd_skills)

# --- Adding LoRA in addition to the custom model for comparison ---
def load_sentence_transformer(model_type="full"):
    project_root = Path(__file__).resolve().parent.parent

    if model_type == "lora":
        model_path = project_root / "models" / "resume_jd_matcher_lora"
    else:
        model_path = project_root / "models" / "resume_jd_matcher"

    print("Looking for model at:", model_path)

    if model_path.exists() and (model_path / "modules.json").exists():
        return SentenceTransformer(str(model_path))

    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

'''
# --- Adding custom model instead of the generic SLM ---
def _try_load_sentence_transformer():
    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / "models" / "resume_jd_matcher"

    print("Looking for model at:", model_path)

    if model_path.exists() and (model_path / "modules.json").exists():
        print("Loading fine-tuned model...")
        return SentenceTransformer(str(model_path))

    print("Fine-tuned model not found. Loading base model.")
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
'''
# --- Optional semantic model ---
#def _try_load_sentence_transformer(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):

'''
def _try_load_sentence_transformer(model_name: str = "models/resume_jd_matcher"):
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)
    except Exception:
        return None
'''

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def semantic_similarity(resume_text: str, jd_text: str, model_type="full"):
    model = load_sentence_transformer(model_type)

    emb = model.encode([resume_text, jd_text], normalize_embeddings=True)
    score = float(emb[0] @ emb[1])

    return max(0.0, min(1.0, score)), model_type


# --- Commenting out for choice between custom model and LoRA --
'''
def semantic_similarity(resume_text: str, jd_text: str, model=None) -> Tuple[float, str]:
    """
    Returns (score 0..1, method)
    """
    if model is None:
        model = _try_load_sentence_transformer()

    if model is not None:
        emb = model.encode([resume_text, jd_text], normalize_embeddings=True)
        score = float(np.dot(emb[0], emb[1]))
        return max(0.0, min(1.0, score)), "sentence-transformers"

    # Fallback: TF-IDF cosine
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vec.fit_transform([resume_text, jd_text]).toarray()
    score = cosine(X[0], X[1])
    return max(0.0, min(1.0, score)), "tfidf"
'''

def ats_keyword_score(resume_text: str, jd_text: str) -> float:
    """
    Simple ATS-like keyword overlap based on token sets.
    """
    r = set(tokenize_simple(resume_text))
    j = set(tokenize_simple(jd_text))
    if not j:
        return 0.0
    overlap = len(r & j) / max(1, len(j))
    return float(max(0.0, min(1.0, overlap)))

@dataclass
class MatchScores:
    semantic_score: float
    ats_score: float
    final_score: float
    method: str
    breakdown: Dict[str, Any]

def fresher_weighting(years_exp: Optional[float], fresher_mode: bool) -> Dict[str, float]:
    """
    Weights for combining semantic and ATS scores.
    - Fresher mode: slightly favors semantic match (projects/education alignment)
    - Experienced: balance or slightly favors ATS keywords (role-specific terms)
    """
    if fresher_mode or (years_exp is not None and years_exp < 2.0):
        return {"semantic": 0.70, "ats": 0.30}
    return {"semantic": 0.55, "ats": 0.45}

def compute_match(
    resume_text,
    jd_text,
    years_exp=None,
    fresher_mode=False,
    ats_override=None,
    model_type="full"
):
    sem, method = semantic_similarity(resume_text, jd_text, model_type=model_type)

    if ats_override is not None:
        ats = ats_override
        ats_source = "skill_overlap"
    else:
        ats = ats_keyword_score(resume_text, jd_text)
        ats_source = "keyword"

    w = fresher_weighting(years_exp, fresher_mode)
    final = w["semantic"] * sem + w["ats"] * ats

    return MatchScores(
        semantic_score=sem,
        ats_score=ats,
        final_score=final,
        method=method,
        breakdown={
            "weights": w,
            "ats_source": ats_source,
            "model_type": model_type,
        },
    )

# --- Commenting out for LoRA implementation ---
'''
def compute_match(
    resume_text: str,
    jd_text: str,
    years_exp: Optional[float] = None,
    fresher_mode: bool = False,
    ats_override: Optional[float] = None
) -> MatchScores:

    sem, method = semantic_similarity(resume_text, jd_text)

    if ats_override is not None:
        ats = ats_override
        ats_source = "skill_overlap"
    else:
        ats = ats_keyword_score(resume_text, jd_text)
        ats_source = "keyword"

    w = fresher_weighting(years_exp, fresher_mode)

    final = w["semantic"] * sem + w["ats"] * ats
    final = float(max(0.0, min(1.0, final)))

    return MatchScores(
    semantic_score=sem,
    ats_score=ats,
    final_score=final,
    method=method,
    breakdown={
        "weights": w,
        "ats_source": ats_source,
        "years_experience_estimate": years_exp,   
        "fresher_mode": fresher_mode              
    },
)
'''

def score_to_percent(x: float) -> int:
    return int(round(100 * max(0.0, min(1.0, x))))