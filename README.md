# IMDB Sentiment — MLOps Pipeline

End-to-end MLOps project for binary sentiment classification on the [Stanford IMDB dataset](https://huggingface.co/datasets/stanfordnlp/imdb). The goal is not peak accuracy — it is to demonstrate a complete, reproducible pipeline: data preparation, training, experiment tracking, model registry, containerised inference, and CI/CD automation.

**Author:** Rakesh Kumar  
**Repository:** [github.com/rakeshlrng/mlops-imdb-sentiment](https://github.com/rakeshlrng/mlops-imdb-sentiment)

---

## Assignment focus

| Principle | How this project addresses it |
|-----------|-------------------------------|
| Pipeline over model score | Accuracy is reported but not optimised; effort goes into automation, documentation, and reproducibility |
| Model as black box | Training lives in notebooks/Kaggle; production consumes a versioned artefact from Hugging Face Hub |
| Justified decisions | Every major choice (dataset, model, hyperparameters, tooling) is documented in [`data/README.md`](data/README.md) and [`model/README.md`](model/README.md) |
| Live links | All external references below are publicly accessible at submission time |

---

## Live links

| Resource | URL |
|----------|-----|
| GitHub repository | https://github.com/rakeshlrng/mlops-imdb-sentiment |
| Kaggle Training Notebook | https://www.kaggle.com/code/rathodishag25ait2084/ml-opsgroupassignment |
| Trained model (Hugging Face Hub) | https://huggingface.co/IshaIIT/g25ait2084-imdb-sentiment |
| Source dataset | https://huggingface.co/datasets/stanfordnlp/imdb |
| Base model | https://huggingface.co/distilbert-base-uncased |
| Experiment tracking (W&B) | https://wandb.ai/g25ait2084-iit/imdb-distilbert/overview |
| CI workflow | https://github.com/rakeshlrng/mlops-imdb-sentiment/actions/workflows/ci.yml |
| Inference workflow | https://github.com/rakeshlrng/mlops-imdb-sentiment/actions/workflows/inference.yml |

---

## Pipeline overview

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────────┐
│  Raw data   │───▶│  data_prep   │───▶│   Training  │───▶│  HF Model Hub    │
│  (IMDB)     │    │  (src/)      │    │  (Kaggle)   │    │  rakeshlrng/...  │
└─────────────┘    └──────────────┘    └─────────────┘    └────────┬─────────┘
                                                                    │
                    ┌──────────────┐    ┌─────────────┐             │
                    │  GitHub CI   │    │   Docker    │◀────────────┘
                    │  (flake8)    │    │  inference  │
                    └──────────────┘    └─────────────┘
```

1. **Data** — Load and inspect IMDB; persist label mapping → see [`data/README.md`](data/README.md)
2. **Train** — Fine-tune DistilBERT on Kaggle GPU; log to W&B; push to Hub → see [`model/README.md`](model/README.md)
3. **Serve** — Run predictions via `src/inference.py`, Docker, or GitHub Actions
4. **Automate** — Lint on every PR; on-demand inference via workflow dispatch

---

## Project structure

```
mlops-imdb-sentiment/
├── data/                  # Data documentation (see data/README.md)
├── model/                 # Model documentation (see model/README.md)
├── src/
│   ├── data_prep.py       # Dataset load, inspection, label mapping
│   ├── train.py           # Training script (mirrors Kaggle notebook logic)
│   └── inference.py       # Production inference entry point
├── notebooks/
│   ├── training_v1.ipynb  # First training run (lr=3e-5)
│   └── training_v2.ipynb  # Second run (lr=5e-5) — deployed model
├── artifacts/
│   └── id2label.json      # Label mapping artefact
├── .github/workflows/
│   ├── ci.yml             # Lint on push/PR
│   └── inference.yml      # Manual inference trigger
├── Dockerfile             # Containerised inference
└── requirements.txt
```

---

## Quick start

### Prerequisites

- Python 3.11+
- Hugging Face token (`HF_TOKEN`) for dataset/model access — optional for public models

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data preparation

```bash
export HF_TOKEN=<your_token>   # optional for public dataset
python src/data_prep.py
```

Writes `artifacts/id2label.json` and prints dataset statistics.

### Inference (local)

```bash
export HF_MODEL_NAME=rakeshlrng/imdb-sentiment
export INPUT_TEXT="This movie is fantastic"
python src/inference.py
```

Expected output:

```json
[{'label': 'POSITIVE', 'score': <confidence>}]
```

### Inference (Docker)

```bash
docker build -t imdb-sentiment .
docker run -e INPUT_TEXT="Terrible acting and boring plot" imdb-sentiment
```

Override the model at build time:

```bash
docker build --build-arg HF_MODEL_NAME=rakeshlrng/imdb-sentiment -t imdb-sentiment .
```

### Inference (GitHub Actions)

1. Open [Actions → Inference](https://github.com/rakeshlrng/mlops-imdb-sentiment/actions/workflows/inference.yml)
2. Click **Run workflow**
3. Enter review text in `input_text`
4. Inspect the job log for the prediction

---

## CI/CD

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `ci.yml` | Push to `develop`; PR to `main` | `flake8` lint on `src/` (max line length 120) |
| `inference.yml` | Manual `workflow_dispatch` | Run `src/inference.py` with user-supplied text |

CI treats the model as an external dependency — it does not retrain. It validates code quality and proves the inference path works in a clean environment.

---

## Model summary

| Item | Value |
|------|-------|
| Task | Binary text classification (positive / negative) |
| Base model | `distilbert-base-uncased` |
| Deployed model | `IshaIIT/g25ait2084-imdb-sentiment` |
| Eval accuracy (v3) | 91.0% |
| Eval F1 (v3) | 0.9102 |

v1 (lr 3e-5) achieved 88.9% accuracy, v2 (lr 5e-5) achieved 89.7% accuracy, and the final v3 run achieved 91.0% accuracy with an F1 score of 0.9102. Therefore, v3 was selected for deployment and published to Hugging Face Hub.
---

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | No | — | Hugging Face API token for private assets |
| `HF_MODEL_NAME` | No | `rakeshlrng/imdb-sentiment` | Model repo for inference |
| `INPUT_TEXT` | No | `"This movie is fantastic"` | Text to classify |
| `WANDB_API_KEY` | Training only | — | Weights & Biases API key |

---

## Design decisions (summary)

Full justifications live in the component READMEs. Highlights:

- **Dataset:** Stanford IMDB — standard benchmark, balanced classes, no extra cleaning needed
- **Model:** DistilBERT — 40% smaller than BERT, fast enough for CI/Docker, strong baseline for sentiment
- **Training platform:** Kaggle — free GPU (Tesla T4), secrets management, reproducible notebook exports
- **Model registry:** Hugging Face Hub — versioned weights, one-line inference via `transformers` pipeline
- **Experiment tracking:** W&B — logs hyperparameters, metrics, and run lineage per assignment requirement
- **No local model weights in git:** Keeps the repo small; inference pulls from Hub at runtime
- **Docker:** Immutable inference environment matching production constraints

---

## Future improvements

- Complete `src/train.py` to mirror the Kaggle notebook for local/CI training
- Add unit tests for `data_prep.py` and `inference.py`
- Integrate DVC for dataset versioning (stubbed in `requirements.txt`)
- Add a REST API layer (e.g. FastAPI) in front of `inference.py`
- Pin dependency versions in `requirements.txt` for full reproducibility

---

## License

See [LICENSE](LICENSE).
