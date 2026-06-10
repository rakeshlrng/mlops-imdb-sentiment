# src/data_prep.py
# Data cleaning & label mapping for the IMDB dataset
# 1. Load Dataset
import os
import json
from dotenv import load_dotenv
from datasets import load_dataset
from collections import Counter

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

dataset = load_dataset("stanfordnlp/imdb", token=hf_token)

# 2. Dataset Inspection
print(dataset)

print("Train size:", len(dataset["train"]))
print("Test size:", len(dataset["test"]))

# 3. Class Distribution

print(
    Counter(dataset["train"]["label"])
)

# 4. Create Label Mapping id2label.json

id2label = {
    0: "negative",
    1: "positive"
}

with open("artifacts/id2label.json", "w") as f:
    json.dump(id2label, f, indent=4)

print("id2label saved")
print(dataset)
