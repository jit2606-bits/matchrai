from pathlib import Path
import pandas as pd

from datasets import Dataset
from peft import LoraConfig, TaskType
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.losses import CosineSimilarityLoss
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

DATA_PATH = PROJECT_ROOT / "data" / "resume_jd_pairs_expanded.csv"
MODEL_OUTPUT = PROJECT_ROOT / "models" / "resume_jd_matcher_lora"

df = pd.read_csv(DATA_PATH)

df = df[["resume_text", "jd_text", "label"]].dropna()
df["label"] = df["label"].astype(float)

train_dataset = Dataset.from_pandas(df)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

peft_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "value"]
)

model.add_adapter(peft_config)

loss = CosineSimilarityLoss(model)

args = SentenceTransformerTrainingArguments(
    output_dir=str(MODEL_OUTPUT),
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    logging_steps=10,
    save_strategy="epoch"
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=loss
)

trainer.train()

model.save_pretrained(str(MODEL_OUTPUT))

print("LoRA model saved to:", MODEL_OUTPUT)