# Load model, tokenize, train, evaluate, push to HF

"""
Training was executed in Kaggle.

This script contains the same logic
used in the Kaggle notebook.
"""
import os
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "distilbert-base-uncased"

hf_token = os.getenv("HF_TOKEN")
dataset = load_dataset("stanfordnlp/imdb", token=hf_token)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Training pipeline placeholder")
