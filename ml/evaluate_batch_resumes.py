from pathlib import Path
import time
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from parsing import parse_resume, parse_job_description
from skills import build_skill_taxonomy, extract_skills, keyword_gaps
from scoring import ats_skill_overlap

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESUME_DIR = PROJECT_ROOT / "batch_resumes"
JD_DIR = PROJECT_ROOT / "batch_jobs"
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)

SKILLS_PATH = PROJECT_ROOT / "data" / "skills_taxonomy.txt"
taxonomy = build_skill_taxonomy(SKILLS_PATH)

MODELS = {
    "generic": "sentence-transformers/all-MiniLM-L6-v2",
    "full_fine_tuned": PROJECT_ROOT / "models" / "resume_jd_matcher",
    "lora_peft": PROJECT_ROOT / "models" / "resume_jd_matcher_lora",
}

def cosine_score(model, resume_text, jd_text):
    emb = model.encode([resume_text, jd_text], normalize_embeddings=True)
    return float(np.dot(emb[0], emb[1]))

def final_score(semantic, ats, fresher_mode=True):
    if fresher_mode:
        return 0.70 * semantic + 0.30 * ats
    return 0.55 * semantic + 0.45 * ats

rows = []

resumes = list(RESUME_DIR.glob("*.pdf")) + list(RESUME_DIR.glob("*.docx"))
jobs = list(JD_DIR.glob("*.txt"))

loaded_models = {
    name: SentenceTransformer(str(path))
    for name, path in MODELS.items()
}

for resume_path in resumes:
    parsed = parse_resume(str(resume_path))

    for jd_path in jobs:
        jd_clean = parse_job_description(jd_path.read_text(encoding="utf-8"))

        resume_sk = set(extract_skills(parsed.raw_text, taxonomy))
        jd_sk = set(extract_skills(jd_clean, taxonomy))
        ats = ats_skill_overlap(resume_sk, jd_sk)

        gaps = keyword_gaps(parsed.raw_text, jd_clean, taxonomy)

        for model_name, model in loaded_models.items():
            start = time.time()

            semantic = cosine_score(model, parsed.raw_text, jd_clean)
            final = final_score(semantic, ats)

            elapsed = time.time() - start

            rows.append({
                "resume_file": resume_path.name,
                "job_file": jd_path.name,
                "model": model_name,
                "semantic_score": round(semantic, 4),
                "ats_skill_overlap": round(ats, 4),
                "final_score": round(final, 4),
                "matched_skills": ", ".join(gaps["matched"]),
                "missing_skills": ", ".join(gaps["missing"]),
                "cgpa": parsed.cgpa or "Not detected",
                "years_experience": parsed.years_experience_estimate,
                "inference_time_sec": round(elapsed, 4),
            })

df = pd.DataFrame(rows)

csv_path = REPORT_DIR / "batch_resume_model_comparison.csv"
excel_path = REPORT_DIR / "batch_resume_model_comparison.xlsx"

df.to_csv(csv_path, index=False)

with pd.ExcelWriter(excel_path) as writer:
    df.to_excel(writer, sheet_name="All Runs", index=False)

    summary = (
        df.groupby("model")
        .agg(
            avg_semantic_score=("semantic_score", "mean"),
            avg_ats_score=("ats_skill_overlap", "mean"),
            avg_final_score=("final_score", "mean"),
            avg_inference_time=("inference_time_sec", "mean"),
        )
        .reset_index()
    )

    summary.to_excel(writer, sheet_name="Model Summary", index=False)

print("Batch evaluation complete.")
print("CSV:", csv_path)
print("Excel:", excel_path)