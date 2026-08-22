FROM python:3.11-slim

COPY experiments/000_text_baseline/requirements-train.txt /tmp/requirements-train.txt
RUN pip install --no-cache-dir -r /tmp/requirements-train.txt

WORKDIR /work
