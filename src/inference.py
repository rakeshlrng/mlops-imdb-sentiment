# Prediction code used by Docker and GitHub Actions
import os

from transformers import pipeline

model_name = os.getenv(
    "HF_MODEL_NAME",
    "rakeshlrng/imdb-sentiment"
)

classifier = pipeline(
    "text-classification",
    model=model_name
)

text = os.getenv(
    "INPUT_TEXT",
    "This movie is fantastic"
)

result = classifier(text)

label_map = {
    "LABEL_0": "NEGATIVE",
    "LABEL_1": "POSITIVE"
}

if result and result[0]["label"] in label_map:
    result[0]["label"] = label_map[result[0]["label"]]

print(result)
