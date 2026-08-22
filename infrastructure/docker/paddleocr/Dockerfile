FROM odsai/ecup26-quality-baseline:1.0

# PaddleOCR-VL-1.5 custom code is compatible with the Transformers 4.55 API.
# Install everything at build time because the competition evaluator is offline.
COPY requirements.txt /tmp/paddleocr-requirements.txt
RUN python -m pip install --no-cache-dir --no-deps \
      -r /tmp/paddleocr-requirements.txt \
    && python - <<'PY'
import einops
import google.protobuf
import huggingface_hub
import sentencepiece
import tokenizers
import transformers

assert transformers.__version__ == "4.55.0"
assert tokenizers.__version__ == "0.21.4"
print("PaddleOCR runtime verified", transformers.__version__)
PY
