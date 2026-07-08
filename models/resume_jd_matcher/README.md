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
- source_sentence: Used Python, SQL, pandas, and visualization for academic analytics
    projects with some machine learning exposure. Some exposure to regression, pandas,
    scikit-learn.
  sentences:
  - Entry-Level Data Scientist role in Analytics requiring scikit-learn, NumPy, statistics,
    A/B testing, pandas, SQL, data visualization. Preferred qualifications include
    Power BI, XGBoost, experiment design. Candidate should communicate results clearly,
    work with Git, and document model behavior.
  - Entry-Level Data Scientist role in Analytics requiring regression, SQL, NumPy,
    scikit-learn, data visualization, Python, pandas. Preferred qualifications include
    ETL, XGBoost, experiment design, Git. Candidate should communicate results clearly,
    work with Git, and document model behavior.
  - Junior Healthcare AI Engineer role in Healthcare AI requiring HIPAA-aware design,
    LLMs, Python, FastAPI, clinical data, FHIR. Preferred qualifications include medical
    terminology, LangGraph. Candidate should communicate results clearly, work with
    Git, and document model behavior.
- source_sentence: Implemented computer vision projects using PyTorch, OpenCV, CNNs,
    ResNet, YOLO, data augmentation, and object detection evaluation. Additional experience
    includes TensorFlow, MONAI, Python, ResNet.
  sentences:
  - Entry-Level Data Scientist role in Analytics requiring NumPy, regression, SQL,
    statistics, Python, scikit-learn, pandas. Preferred qualifications include Tableau,
    Git, ETL. Candidate should communicate results clearly, work with Git, and document
    model behavior.
  - Junior Computer Vision Engineer role in Computer Vision requiring Python, image
    classification, PyTorch, NumPy, object detection, OpenCV, CNNs. Preferred qualifications
    include TensorFlow, ResNet, MONAI. Candidate should communicate results clearly,
    work with Git, and document model behavior.
  - Entry-Level Machine Learning Engineer role in General AI requiring scikit-learn,
    FastAPI, TensorFlow, model evaluation, Python, ML pipelines, feature engineering,
    PyTorch. Preferred qualifications include CI/CD, AWS SageMaker, MLflow, Hugging
    Face. Candidate should communicate results clearly, work with Git, and document
    model behavior.
- source_sentence: Used Python, SQL, pandas, and visualization for academic analytics
    projects with some machine learning exposure. Some exposure to data visualization,
    statistics, SQL.
  sentences:
  - Entry-Level MLOps Engineer role in MLOps requiring GitHub Actions, FastAPI, AWS,
    model deployment, MLflow, Python. Preferred qualifications include SageMaker,
    ECS, Terraform, Kubernetes. Candidate should communicate results clearly, work
    with Git, and document model behavior.
  - Entry-Level Data Scientist role in Analytics requiring A/B testing, data visualization,
    NumPy, SQL, statistics, Python. Preferred qualifications include experiment design,
    Power BI. Candidate should communicate results clearly, work with Git, and document
    model behavior.
  - Junior NLP Engineer role in LLM/NLP requiring RAG, text classification, Hugging
    Face, semantic search, NLP, transformers. Preferred qualifications include ChromaDB,
    LangChain, prompt engineering. Candidate should communicate results clearly, work
    with Git, and document model behavior.
- source_sentence: Academic research resume focused on literature review, presentation
    writing, and laboratory documentation without deployment experience. Strong communication,
    teamwork, documentation, and problem-solving skills.
  sentences:
  - Junior NLP Engineer role in LLM/NLP requiring RAG, Python, transformers, FastAPI,
    NLP, Hugging Face, semantic search. Preferred qualifications include Docker, ChromaDB,
    sentence-transformers, FAISS. Candidate should communicate results clearly, work
    with Git, and document model behavior.
  - Junior Healthcare AI Engineer role in Healthcare AI requiring clinical data, FastAPI,
    FHIR, HL7, LLMs. Preferred qualifications include FHIR resources, LangGraph, EHR
    integration, care gaps. Candidate should communicate results clearly, work with
    Git, and document model behavior.
  - Entry-Level MLOps Engineer role in MLOps requiring CI/CD, model deployment, Docker,
    GitHub Actions, FastAPI, AWS, monitoring. Preferred qualifications include Terraform,
    Airflow, CloudWatch, SageMaker. Candidate should communicate results clearly,
    work with Git, and document model behavior.
- source_sentence: Built Python ML projects and containerized simple APIs with Docker
    and GitHub version control. Some exposure to model deployment, Python, Docker.
  sentences:
  - Entry-Level MLOps Engineer role in MLOps requiring GitHub Actions, CI/CD, Python,
    model deployment, MLflow, Docker. Preferred qualifications include ECS, SageMaker,
    CloudWatch. Candidate should communicate results clearly, work with Git, and document
    model behavior.
  - Entry-Level MLOps Engineer role in MLOps requiring MLflow, monitoring, AWS, CI/CD,
    Docker, Python, FastAPI, model deployment. Preferred qualifications include Terraform,
    CloudWatch, ECS. Candidate should communicate results clearly, work with Git,
    and document model behavior.
  - Junior NLP Engineer role in LLM/NLP requiring FastAPI, semantic search, Python,
    NLP, transformers, text classification, RAG. Preferred qualifications include
    LangChain, Docker. Candidate should communicate results clearly, work with Git,
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
    'Built Python ML projects and containerized simple APIs with Docker and GitHub version control. Some exposure to model deployment, Python, Docker.',
    'Entry-Level MLOps Engineer role in MLOps requiring GitHub Actions, CI/CD, Python, model deployment, MLflow, Docker. Preferred qualifications include ECS, SageMaker, CloudWatch. Candidate should communicate results clearly, work with Git, and document model behavior.',
    'Junior NLP Engineer role in LLM/NLP requiring FastAPI, semantic search, Python, NLP, transformers, text classification, RAG. Preferred qualifications include LangChain, Docker. Candidate should communicate results clearly, work with Git, and document model behavior.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.7514, 0.4395],
#         [0.7514, 1.0000, 0.4907],
#         [0.4395, 0.4907, 1.0000]])
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
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 100 samples:
  |          | sentence_0                                                                         | sentence_1                                                                         | label                                                           |
  |:---------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:----------------------------------------------------------------|
  | type     | string                                                                             | string                                                                             | float                                                           |
  | modality | text                                                                               | text                                                                               |                                                                 |
  | details  | <ul><li>min: 32 tokens</li><li>mean: 42.24 tokens</li><li>max: 59 tokens</li></ul> | <ul><li>min: 50 tokens</li><li>mean: 61.78 tokens</li><li>max: 73 tokens</li></ul> | <ul><li>min: 0.16</li><li>mean: 0.7</li><li>max: 0.95</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                  | sentence_1                                                                                                                                                                                                                                                                                                                                       | label             |
  |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------|
  | <code>Completed healthcare analytics coursework using Python, SQL, dashboards, and some clinical data exploration. Some exposure to FHIR, evaluation, HIPAA-aware design.</code>                            | <code>Junior Healthcare AI Engineer role in Healthcare AI requiring clinical data, evaluation, HIPAA-aware design, Python, FHIR. Preferred qualifications include LangGraph, EHR integration, semantic search, FHIR resources. Candidate should communicate results clearly, work with Git, and document model behavior.</code>                  | <code>0.56</code> |
  | <code>Developed GenAI applications with Python, FastAPI, Streamlit, RAG, ChromaDB, Hugging Face models, Docker, APIs, and evaluation. Additional experience includes SQL, authentication, RAG, LLMs.</code> | <code>Entry-Level AI Application Developer role in GenAI Apps requiring LLMs, FastAPI, RAG, SQL, evaluation. Preferred qualifications include Streamlit, AWS, authentication, Hugging Face. Candidate should communicate results clearly, work with Git, and document model behavior.</code>                                                     | <code>0.95</code> |
  | <code>Completed healthcare analytics coursework using Python, SQL, dashboards, and some clinical data exploration. Some exposure to RAG, HIPAA-aware design, clinical data.</code>                          | <code>Junior Healthcare AI Engineer role in Healthcare AI requiring HIPAA-aware design, LLMs, FHIR, clinical data, RAG, FastAPI, evaluation. Preferred qualifications include care gaps, semantic search, medical terminology, EHR integration. Candidate should communicate results clearly, work with Git, and document model behavior.</code> | <code>0.72</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss",
      "cos_score_transformation": "torch.nn.modules.linear.Identity"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 8
- `num_train_epochs`: 3
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1
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
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Time
- **Training**: 22.8 seconds

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