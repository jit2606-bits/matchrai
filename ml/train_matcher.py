from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

DATA_PATH = PROJECT_ROOT / "data" / "resume_jd_pairs_expanded.csv"
MODEL_OUTPUT = PROJECT_ROOT / "models" / "resume_jd_matcher"

print("Data path:", DATA_PATH)
print("Model output path:", MODEL_OUTPUT)
#df = pd.read_csv("training_data/resume_jd_pairs.csv")
#df = pd.read_csv("data/resume_jd_pairs.csv")

#Debug
print("entered train_matcher.py")
#df = pd.read_csv("../data/resume_jd_pairs.csv")
#df = pd.read_csv("../data/resume_jd_pairs_expanded.csv")
df = pd.read_csv(DATA_PATH)

#Debug
print(df.columns.tolist())

train_examples = [
    InputExample(
        texts=[row["resume_text"], row["jd_text"]],
        label=float(row["label"])
    )
    for _, row in df.iterrows()
]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

train_dataloader = DataLoader(
    train_examples,
    shuffle=True,
    batch_size=8
)

train_loss = losses.CosineSimilarityLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=20,
    output_path=str(MODEL_OUTPUT)
)