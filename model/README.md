# Model Pipeline

Documentation for model selection, training, evaluation, registry, and inference for IMDB sentiment classification.

**Training notebooks:** [`notebooks/training_v1.ipynb`](../notebooks/training_v1.ipynb), [`notebooks/training_v2.ipynb`](../notebooks/training_v2.ipynb)  
**Training script (stub):** [`src/train.py`](../src/train.py)  
**Inference entry point:** [`src/inference.py`](../src/inference.py)  
**Deployed model:** [rakeshlrng/imdb-sentiment](https://huggingface.co/rakeshlrng/imdb-sentiment)

---

## Purpose in the MLOps workflow

The fine-tuned classifier is treated as a **black box**: this repository owns everything around it — how it is trained, versioned, evaluated, published, and served. Final accuracy is reported for transparency, but marks come from pipeline completeness, not leaderboard position.

---

## Model selection

| Decision | Choice | Justification |
|----------|--------|---------------|
| Architecture | [DistilBERT](https://huggingface.co/distilbert-base-uncased) (`distilbert-base-uncased`) | 40% fewer parameters than BERT-base; ~60% faster inference; strong sentiment baseline; fits Kaggle T4 memory with batch size 16 |
| Alternatives considered | BERT-base, RoBERTa-base | Higher accuracy potential but slower training/inference and larger Docker images; poor fit for a pipeline-focused assignment |
| Task head | `AutoModelForSequenceClassification` (`num_labels=2`) | Standard Hugging Face pattern; minimal custom code |
| Pretrained weights | Uncased English DistilBERT | IMDB reviews are informal English; uncased vocabulary matches the text domain |

---

## Training platform

| Decision | Choice | Justification |
|----------|--------|---------------|
| Environment | [Kaggle Notebooks](https://www.kaggle.com/) | Free Tesla T4 GPU; integrated secrets (`HF_TOKEN`, `WANDB_API_KEY`); reproducible notebook exports in `notebooks/` |
| Local script | `src/train.py` (placeholder) | Mirrors notebook logic for future CI/local runs; training currently executed in Kaggle |
| Experiment tracking | [Weights & Biases](https://wandb.ai) | Logs hyperparameters, per-epoch metrics, and run lineage; project: `mlops-imdb-sentiment` |
| Model registry | [Hugging Face Hub](https://huggingface.co/rakeshlrng/imdb-sentiment) | Versioned weights + tokenizer; no large binaries in git; one-line load in inference |

---

## Hyperparameter runs

Two training runs were executed with identical data and architecture. The only deliberate change was learning rate.

### Run v1 (`training_v1.ipynb`)

| Hyperparameter | Value |
|----------------|-------|
| `num_train_epochs` | 3 |
| `per_device_train_batch_size` | 16 |
| `per_device_eval_batch_size` | 16 |
| `learning_rate` | 3e-5 |
| `eval_strategy` | epoch |
| `save_strategy` | epoch |
| `load_best_model_at_end` | True |
| W&B run name | `run-v1` |

**Results:** accuracy 88.9%, F1 0.889, eval loss 0.525

### Run v2 (`training_v2.ipynb`) — **deployed**

| Hyperparameter | Value |
|----------------|-------|
| `num_train_epochs` | 3 |
| `per_device_train_batch_size` | 16 |
| `per_device_eval_batch_size` | 16 |
| `learning_rate` | **5e-5** |
| `eval_strategy` | epoch |
| `save_strategy` | epoch |
| `load_best_model_at_end` | True |
| W&B run name | `run-v2` |

**Results:** accuracy **89.7%**, F1 **0.897**, eval loss 0.495

### Why v2 was selected

- Higher validation accuracy (+0.8 pp) and lower eval loss on the same held-out test split
- Same training budget (3 epochs, full 25k train set) — improvement comes from learning rate only, isolating the variable
- `5e-5` is the commonly recommended fine-tuning rate for DistilBERT; v1's `3e-5` was slightly conservative

No further hyperparameter search was performed — diminishing returns vs. pipeline documentation effort, aligned with assignment goals.

---

## Training pipeline (Kaggle notebook)

```
Load tokenised IMDB dataset
        │
        ▼
AutoModelForSequenceClassification.from_pretrained(distilbert-base-uncased)
        │
        ▼
Trainer(train + test, compute_metrics=accuracy/f1)
        │
        ▼
trainer.train()  →  W&B logs per epoch
        │
        ▼
trainer.evaluate()
        │
        ▼
Set id2label / label2id on model.config
        │
        ▼
model.push_to_hub("rakeshlrng/imdb-sentiment")
tokenizer.push_to_hub("rakeshlrng/imdb-sentiment")
```

### Metrics

```python
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }
```

Accuracy is intuitive for balanced binary classification; weighted F1 accounts for any minor class skew in predictions.

### Label configuration (Hub model)

```python
model.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
model.config.label2id = {"NEGATIVE": 0, "POSITIVE": 1}
```

---

## Inference

Inference does **not** reload training code. It pulls the published checkpoint and runs the Hugging Face `pipeline` API.

**Script:** [`src/inference.py`](../src/inference.py)

```python
classifier = pipeline("text-classification", model=os.getenv("HF_MODEL_NAME", "rakeshlrng/imdb-sentiment"))
result = classifier(os.getenv("INPUT_TEXT", "This movie is fantastic"))
```

The pipeline returns `LABEL_0` / `LABEL_1`; the script maps these to human-readable `NEGATIVE` / `POSITIVE`.

### Inference surfaces

| Surface | How to run |
|---------|------------|
| Local | `INPUT_TEXT="..." python src/inference.py` |
| Docker | `docker run -e INPUT_TEXT="..." imdb-sentiment` (see [Dockerfile](../Dockerfile)) |
| GitHub Actions | Manual dispatch on [inference.yml](https://github.com/rakeshlrng/mlops-imdb-sentiment/actions/workflows/inference.yml) |

Docker and Actions prove the model is consumable as an external artefact — no notebook or GPU required at serve time.

---

## Container image

[`Dockerfile`](../Dockerfile) builds a minimal Python 3.11-slim image:

- Installs `requirements.txt`
- Copies `src/`
- Sets `HF_MODEL_NAME` (build-arg, default `rakeshlrng/imdb-sentiment`)
- Runs `python src/inference.py` on container start

Model weights are downloaded from Hugging Face Hub on first run (cached in the container layer if built with network access).

```bash
docker build -t imdb-sentiment .
docker run -e INPUT_TEXT="A wonderful film with great performances" imdb-sentiment
```

---

## CI integration

| Check | Scope | Rationale |
|-------|-------|-----------|
| `flake8 src/` | Lint only | Fast feedback on code quality; does not retrain (GPU cost, time) |
| `inference.yml` | Manual smoke test | Validates end-to-end inference in a clean Ubuntu runner |

Training is intentionally outside CI — Kaggle provides the GPU; CI validates the serving path.

---

## Model inputs and outputs (black-box contract)

### Inputs

| Input | Type | Constraints |
|-------|------|-------------|
| `text` | `string` | Raw review text; English; tokenizer handles up to 512 tokens (model trained with max 256) |

### Outputs

| Field | Type | Values |
|-------|------|--------|
| `label` | `string` | `NEGATIVE` or `POSITIVE` |
| `score` | `float` | Confidence in [0, 1] |

Example:

```json
[{"label": "POSITIVE", "score": 0.9987}]
```

---

## Artefacts

| Artefact | Location | In git? |
|----------|----------|---------|
| Fine-tuned weights | [Hugging Face Hub](https://huggingface.co/rakeshlrng/imdb-sentiment) | No (too large) |
| Tokenizer | Same Hub repo | No |
| Label mapping (data) | `artifacts/id2label.json` | Yes |
| Training checkpoints | Kaggle output (`./results`, `./results_v2`) | No |
| W&B run history | wandb.ai project `mlops-imdb-sentiment` | External |

---

## Reproducing training

1. Open a Kaggle notebook with GPU accelerator
2. Add secrets: `HF_TOKEN`, `WANDB_API_KEY`
3. Copy logic from [`notebooks/training_v2.ipynb`](../notebooks/training_v2.ipynb)
4. Run all cells → model pushes to Hub (requires write access to `rakeshlrng/imdb-sentiment`)

For local reproduction, complete the stub in `src/train.py` and run with a CUDA-capable machine.

---

## Related links

- [Deployed model — rakeshlrng/imdb-sentiment](https://huggingface.co/rakeshlrng/imdb-sentiment)
- [Base model — distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)
- [Transformers Trainer docs](https://huggingface.co/docs/transformers/main_classes/trainer)
- [Data pipeline docs](../data/README.md)
- [Project root README](../README.md)
