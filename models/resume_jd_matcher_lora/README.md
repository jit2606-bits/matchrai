---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:150
- loss:CosineSimilarityLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: Human resources resume with recruiting coordination, onboarding
    documents, scheduling, and employee records. Strong communication, teamwork, documentation,
    and problem-solving skills.
  sentences:
  - Entry-Level AI Application Developer role in GenAI Apps requiring FastAPI, APIs,
    SQL, Docker, LLMs. Preferred qualifications include ChromaDB, Hugging Face. Candidate
    should communicate results clearly, work with Git, and document model behavior.
  - Entry-Level AI Application Developer role in GenAI Apps requiring Docker, evaluation,
    LLMs, RAG, Git, SQL. Preferred qualifications include AWS, Streamlit, ChromaDB,
    authentication. Candidate should communicate results clearly, work with Git, and
    document model behavior.
  - Junior NLP Engineer role in LLM/NLP requiring RAG, semantic search, Python, transformers,
    text classification. Preferred qualifications include FAISS, ChromaDB. Candidate
    should communicate results clearly, work with Git, and document model behavior.
- source_sentence: Used Python, SQL, pandas, and visualization for academic analytics
    projects with some machine learning exposure. Some exposure to SQL, regression,
    NumPy.
  sentences:
  - Entry-Level MLOps Engineer role in MLOps requiring GitHub Actions, FastAPI, AWS,
    model deployment, MLflow, Python. Preferred qualifications include SageMaker,
    ECS, Terraform, Kubernetes. Candidate should communicate results clearly, work
    with Git, and document model behavior.
  - Entry-Level Data Scientist role in Analytics requiring regression, NumPy, SQL,
    data visualization, statistics, pandas. Preferred qualifications include Git,
    ETL, Power BI, XGBoost. Candidate should communicate results clearly, work with
    Git, and document model behavior.
  - Junior Computer Vision Engineer role in Computer Vision requiring image classification,
    NumPy, OpenCV, CNNs, object detection. Preferred qualifications include deployment,
    data augmentation, ResNet. Candidate should communicate results clearly, work
    with Git, and document model behavior.
- source_sentence: Used Python, SQL, pandas, and visualization for academic analytics
    projects with some machine learning exposure. Some exposure to regression, pandas,
    scikit-learn.
  sentences:
  - Junior NLP Engineer role in LLM/NLP requiring transformers, Python, FastAPI, NLP,
    embeddings. Preferred qualifications include LangChain, FAISS, ChromaDB. Candidate
    should communicate results clearly, work with Git, and document model behavior.
  - Entry-Level Data Scientist role in Analytics requiring regression, SQL, NumPy,
    scikit-learn, data visualization, Python, pandas. Preferred qualifications include
    ETL, XGBoost, experiment design, Git. Candidate should communicate results clearly,
    work with Git, and document model behavior.
  - Junior Computer Vision Engineer role in Computer Vision requiring object detection,
    Python, CNNs, model evaluation, PyTorch, NumPy, OpenCV, image classification.
    Preferred qualifications include data augmentation, deployment, ResNet. Candidate
    should communicate results clearly, work with Git, and document model behavior.
- source_sentence: Created NLP and LLM applications with Hugging Face transformers,
    sentence-transformers, embeddings, RAG, ChromaDB, FastAPI, and semantic search.
    Additional experience includes semantic search, NLP, LangChain, Hugging Face.
  sentences:
  - Entry-Level Data Scientist role in Analytics requiring NumPy, Python, data visualization,
    statistics, A/B testing, SQL. Preferred qualifications include Tableau, XGBoost.
    Candidate should communicate results clearly, work with Git, and document model
    behavior.
  - Entry-Level Machine Learning Engineer role in General AI requiring FastAPI, feature
    engineering, Docker, TensorFlow, ML pipelines, model evaluation. Preferred qualifications
    include MLflow, Airflow, CI/CD. Candidate should communicate results clearly,
    work with Git, and document model behavior.
  - Junior NLP Engineer role in LLM/NLP requiring FastAPI, embeddings, semantic search,
    NLP, Hugging Face. Preferred qualifications include LangChain, prompt engineering.
    Candidate should communicate results clearly, work with Git, and document model
    behavior.
- source_sentence: Built healthcare AI prototype using Python, FastAPI, RAG, FHIR
    concepts, clinical care gap logic, LLM evaluation, and medical terminology. Additional
    experience includes FastAPI, semantic search, HL7, FHIR.
  sentences:
  - Junior Healthcare AI Engineer role in Healthcare AI requiring FHIR, Python, clinical
    data, HL7, LLMs, FastAPI. Preferred qualifications include EHR integration, semantic
    search. Candidate should communicate results clearly, work with Git, and document
    model behavior.
  - Junior Computer Vision Engineer role in Computer Vision requiring Python, OpenCV,
    image classification, NumPy, CNNs, PyTorch, model evaluation, object detection.
    Preferred qualifications include data augmentation, YOLO, ResNet, MONAI. Candidate
    should communicate results clearly, work with Git, and document model behavior.
  - Junior NLP Engineer role in LLM/NLP requiring Python, RAG, Hugging Face, semantic
    search, embeddings, text classification. Preferred qualifications include LangChain,
    FAISS, ChromaDB. Candidate should communicate results clearly, work with Git,
    and document model behavior.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for retrieval.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'transformer_task': 'feature-extraction', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'last_hidden_state'}}, 'module_output_name': 'token_embeddings', 'architecture': 'BertModel'})
  (1): Pooling({'embedding_dimension': 384, 'pooling_mode': 'mean', 'include_prompt': True})
  (2): Normalize({})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```
Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Built healthcare AI prototype using Python, FastAPI, RAG, FHIR concepts, clinical care gap logic, LLM evaluation, and medical terminology. Additional experience includes FastAPI, semantic search, HL7, FHIR.',
    'Junior Healthcare AI Engineer role in Healthcare AI requiring FHIR, Python, clinical data, HL7, LLMs, FastAPI. Preferred qualifications include EHR integration, semantic search. Candidate should communicate results clearly, work with Git, and document model behavior.',
    'Junior NLP Engineer role in LLM/NLP requiring Python, RAG, Hugging Face, semantic search, embeddings, text classification. Preferred qualifications include LangChain, FAISS, ChromaDB. Candidate should communicate results clearly, work with Git, and document model behavior.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.8705, 0.4491],
#         [0.8705, 1.0000, 0.5427],
#         [0.4491, 0.5427, 1.0000]])
```
<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 150 training samples
* Columns: <code>resume_text</code>, <code>jd_text</code>, and <code>label</code>
* Approximate statistics based on the first 100 samples:
  |          | resume_text                                                                        | jd_text                                                                            | label                                                            |
  |:---------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------|
  | type     | string                                                                             | string                                                                             | float                                                            |
  | modality | text                                                                               | text                                                                               |                                                                  |
  | details  | <ul><li>min: 32 tokens</li><li>mean: 42.32 tokens</li><li>max: 56 tokens</li></ul> | <ul><li>min: 51 tokens</li><li>mean: 61.83 tokens</li><li>max: 76 tokens</li></ul> | <ul><li>min: 0.16</li><li>mean: 0.69</li><li>max: 0.96</li></ul> |
* Samples:
  | resume_text                                                                                                                                                                                                                           | jd_text                                                                                                                                                                                                                                                                                                                             | label             |
  |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------|
  | <code>Developed data science projects using Python, SQL, pandas, scikit-learn, regression, classification, A/B testing, statistics, and Tableau dashboards. Additional experience includes statistics, ETL, NumPy, Python.</code>     | <code>Entry-Level Data Scientist role in Analytics requiring NumPy, regression, SQL, statistics, Python, scikit-learn, pandas. Preferred qualifications include Tableau, Git, ETL. Candidate should communicate results clearly, work with Git, and document model behavior.</code>                                                 | <code>0.85</code> |
  | <code>Built finance analytics projects using Python, SQL, time-series forecasting, financial news NLP, sentiment features, statistics, and scikit-learn. Additional experience includes statistics, scikit-learn, Python, SQL.</code> | <code>Entry-Level Financial Data Scientist role in Finance AI requiring Python, statistics, SQL, forecasting, time series, financial data, scikit-learn. Preferred qualifications include risk modeling, PyTorch. Candidate should communicate results clearly, work with Git, and document model behavior.</code>                  | <code>0.94</code> |
  | <code>Retail management resume with scheduling, inventory, team leadership, and customer service experience. Strong communication, teamwork, documentation, and problem-solving skills.</code>                                        | <code>Junior Healthcare AI Engineer role in Healthcare AI requiring evaluation, LLMs, FHIR, Python, HL7, RAG, FastAPI, HIPAA-aware design. Preferred qualifications include FHIR resources, EHR integration, care gaps, LangGraph. Candidate should communicate results clearly, work with Git, and document model behavior.</code> | <code>0.22</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss",
      "cos_score_transformation": "torch.nn.modules.linear.Identity"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `learning_rate`: 0.0002
- `warmup_steps`: 0.1

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 8
- `num_train_epochs`: 3
- `max_steps`: -1
- `learning_rate`: 0.0002
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0.1
- `optim`: adamw_torch
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1.0
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: None
- `trackio_bucket_id`: None
- `trackio_static_space_id`: None
- `per_device_eval_batch_size`: 8
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_static_graph`: None
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: None
- `fsdp_config`: None
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `warmup_ratio`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.5263 | 10   | 0.0360        |
| 1.0526 | 20   | 0.0302        |
| 1.5789 | 30   | 0.0163        |
| 2.1053 | 40   | 0.0138        |
| 2.6316 | 50   | 0.0112        |


### Training Time
- **Training**: 16.7 seconds

### Framework Versions
- Python: 3.13.14
- Sentence Transformers: 5.6.0
- Transformers: 5.13.0
- PyTorch: 2.6.0+cpu
- Accelerate: 1.14.0
- Datasets: 5.0.0
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->