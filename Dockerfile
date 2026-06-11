FROM python:3.11-slim

ARG HF_MODEL_NAME=rakeshlrng/imdb-sentiment

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/

ENV HF_MODEL_NAME=$HF_MODEL_NAME

ENV INPUT_TEXT="This movie is fantastic"

CMD ["python","src/inference.py"]
