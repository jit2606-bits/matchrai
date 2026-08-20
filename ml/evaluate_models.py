from pathlib import Path
import time
import math
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "resume_jd_pairs_expanded.csv"

MODELS = {
    "generic_all_MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "full_fine_tuned": PROJECT_ROOT / "models" / "resume_jd_matcher",
    "lora_peft": PROJECT_ROOT / "models" / "resume_jd_matcher_lora",
}

OUTPUT_DIR = PROJECT_ROOT / "reports"
OUTPUT_DIR.mkdir(exist_ok=True)


def folder_size_mb(path: Path) -> float:
    if not path.exists() or not path.is_dir():
        return 0.0
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total / (1024 * 1024), 2)


def cosine_score(model, resume_text, jd_text):
    emb = model.encode([resume_text, jd_text], normalize_embeddings=True)
    return float(np.dot(emb[0], emb[1]))


df = pd.read_csv(DATA_PATH)

df = df[["pair_id", "resume_text", "jd_text", "label", "match_category", "target_role", "domain"]].dropna()
df["label"] = df["label"].astype(float)

all_predictions = []
summary_rows = []

for model_name, model_path in MODELS.items():
    print(f"\nEvaluating: {model_name}")
    print(f"Model path: {model_path}")

    start_load = time.time()
    model = SentenceTransformer(str(model_path))
    load_time = time.time() - start_load

    predictions = []
    start_infer = time.time()

    for _, row in df.iterrows():
        pred = cosine_score(model, row["resume_text"], row["jd_text"])
        predictions.append(pred)

        all_predictions.append({
            "pair_id": row["pair_id"],
            "model": model_name,
            "label": row["label"],
            "prediction": pred,
            "absolute_error": abs(row["label"] - pred),
            "match_category": row["match_category"],
            "target_role": row["target_role"],
            "domain": row["domain"],
        })

    infer_time = time.time() - start_infer

    y_true = df["label"].values
    y_pred = np.array(predictions)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))

    pearson = pearsonr(y_true, y_pred)[0]
    spearman = spearmanr(y_true, y_pred)[0]

    if isinstance(model_path, Path):
        size_mb = folder_size_mb(model_path)
    else:
        size_mb = 0.0

    summary_rows.append({
        "model": model_name,
        "num_examples": len(df),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "pearson_corr": round(pearson, 4),
        "spearman_corr": round(spearman, 4),
        "avg_label": round(float(np.mean(y_true)), 4),
        "avg_prediction": round(float(np.mean(y_pred)), 4),
        "load_time_sec": round(load_time, 4),
        "inference_time_sec": round(infer_time, 4),
        "avg_inference_per_pair_sec": round(infer_time / len(df), 6),
        "model_size_mb": size_mb,
    })


predictions_df = pd.DataFrame(all_predictions)
summary_df = pd.DataFrame(summary_rows)

predictions_path = OUTPUT_DIR / "model_predictions.csv"
summary_path = OUTPUT_DIR / "model_comparison_summary.csv"
excel_path = OUTPUT_DIR / "model_comparison_report.xlsx"

predictions_df.to_csv(predictions_path, index=False)
summary_df.to_csv(summary_path, index=False)

with pd.ExcelWriter(excel_path) as writer:
    summary_df.to_excel(writer, sheet_name="Summary", index=False)
    predictions_df.to_excel(writer, sheet_name="Predictions", index=False)

print("\nEvaluation complete.")
print("Summary:", summary_path)
print("Predictions:", predictions_path)
print("Excel report:", excel_path)
print(summary_df)