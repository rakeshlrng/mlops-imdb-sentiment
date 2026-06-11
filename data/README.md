# Data Pipeline

Documentation for how raw IMDB review data is sourced, inspected, and prepared for the sentiment classification pipeline.

**Script:** [`src/data_prep.py`](../src/data_prep.py)  
**Source dataset:** [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb)  
**Output artefact:** [`artifacts/id2label.json`](../artifacts/id2label.json)

---

## Purpose in the MLOps workflow

Data preparation is the first reproducible step in the pipeline. This stage does not aim to maximise model accuracy through aggressive feature engineering. Instead it:

1. Loads a well-known public benchmark dataset programmatically
2. Validates structure and class balance before training spends GPU time
3. Produces a version-controlled label mapping consumed downstream by training and inference

The model treats tokenised inputs as a black box; this document explains what enters that box and why.

---

## Dataset choice

| Decision | Choice | Justification |
|----------|--------|---------------|
| Dataset | [Stanford IMDB](https://huggingface.co/datasets/stanfordnlp/imdb) | Industry-standard NLP benchmark; 50k labelled movie reviews; publicly hosted on Hugging Face with a stable API |
| Access method | `datasets.load_dataset()` | Reproducible, scriptable, no manual CSV downloads; same call works locally, in Kaggle, and in CI |
| Splits used | `train` (25k) + `test` (25k) | Official splits avoid data leakage; `unsupervised` (50k) is available but unused to keep the pipeline simple |
| Authentication | Optional `HF_TOKEN` | Dataset is public; token supports rate limits and future private assets |

---

## Dataset schema

| Field | Type | Description |
|-------|------|-------------|
| `text` | `string` | Full movie review (raw text) |
| `label` | `int` | `0` = negative, `1` = positive |

Splits after load:

| Split | Rows | Used in training |
|-------|------|------------------|
| `train` | 25,000 | Yes |
| `test` | 25,000 | Yes (evaluation) |
| `unsupervised` | 50,000 | No (labels are `-1`; reserved for semi-supervised work) |

---

## Cleaning and preprocessing decisions

| Step | Action | Justification |
|------|--------|---------------|
| Text cleaning | **None** | IMDB reviews are pre-tokenised at the word level by the dataset authors; additional regex/HTML stripping risks removing signal and is unnecessary for transformer models |
| Lowercasing | **Deferred to tokenizer** | `distilbert-base-uncased` handles case normalisation internally |
| Class balancing | **None** | Class distribution is approximately 50/50 (verified via `Counter` in `data_prep.py`); synthetic oversampling would add complexity without clear benefit |
| Train/test split | **Use official splits** | Custom splits would break comparability with published benchmarks |
| Subsampling | **Rejected** | Notebooks include commented-out subsample code for quick experiments; production runs use the full 25k/25k splits for stable metrics |
| Tokenisation | **Done in training notebook** | `max_length=256`, truncation + padding — 256 covers ~95% of reviews while keeping GPU memory predictable on Kaggle T4 |

---

## Pipeline steps (`src/data_prep.py`)

```
load_dataset("stanfordnlp/imdb")
        │
        ▼
Inspect splits (train/test sizes)
        │
        ▼
Check class distribution (Counter)
        │
        ▼
Write artifacts/id2label.json
```

### 1. Load dataset

```python
dataset = load_dataset("stanfordnlp/imdb", token=hf_token)
```

Loads all three splits into a `DatasetDict`. The token is read from `HF_TOKEN` in `.env` (see `.env` in `.gitignore` — never commit secrets).

### 2. Dataset inspection

Prints split sizes and structure so any schema drift (e.g. Hugging Face revision change) is visible before training.

### 3. Class distribution

```python
Counter(dataset["train"]["label"])
```

Confirms balanced labels. An imbalanced distribution would trigger resampling or class-weight adjustments — not needed here.

### 4. Label mapping artefact

```json
{
    "0": "negative",
    "1": "positive"
}
```

Saved to `artifacts/id2label.json`. This file is committed to git as a lightweight, human-readable contract between data, training, and inference. The deployed Hugging Face model uses uppercase labels (`NEGATIVE` / `POSITIVE`); `inference.py` maps pipeline output accordingly.

---

## How to run

From the repository root:

```bash
# Optional — only needed if you hit rate limits or use private data
export HF_TOKEN=<your_huggingface_token>

python src/data_prep.py
```

Expected console output (approximate):

```
DatasetDict({ train: 25000 rows, test: 25000 rows, unsupervised: 50000 rows })
Train size: 25000
Test size: 25000
Counter({0: 12500, 1: 12500})
id2label saved
```

---

## Tokenisation (training stage)

Tokenisation is not in `data_prep.py` — it runs in the Kaggle training notebooks before the `Trainer` loop. Parameters:

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `max_length` | 256 | Balances context (IMDB reviews can be long) vs. memory on 16 GB T4 |
| `truncation` | `True` | Drops overflow tokens rather than failing on long reviews |
| `padding` | `max_length` | Fixed tensor shapes for efficient batching |
| `batched` map | `True` | Faster preprocessing on 25k rows |

After tokenisation, the raw `text` column is removed; the trainer receives `input_ids`, `attention_mask`, `token_type_ids`, and `label`.

---

## Versioning and reproducibility

| Approach | Status | Notes |
|----------|--------|-------|
| Hugging Face dataset revision | Implicit latest | Pin `revision=` in `load_dataset()` for stricter reproducibility |
| DVC | Planned | Commented in `requirements.txt`; would track local snapshots if dataset moves off Hub |
| Git-tracked artefacts | `id2label.json` only | Raw reviews are not stored in git (too large); fetched on demand |

---

## Outputs and downstream consumers

| Artefact / output | Consumer |
|-------------------|----------|
| `artifacts/id2label.json` | Training notebooks, documentation, inference label mapping |
| In-memory `DatasetDict` | `notebooks/training_v1.ipynb`, `training_v2.ipynb` |
| Tokenised tensors | Hugging Face `Trainer` in Kaggle |

---

## Related links

- [Stanford IMDB dataset card](https://huggingface.co/datasets/stanfordnlp/imdb)
- [Hugging Face `datasets` library](https://huggingface.co/docs/datasets)
- [Project root README](../README.md)
- [Model training docs](../model/README.md)
