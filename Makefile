TRAIN_DATA ?= /data/data.csv
IMAGE ?= ecup-quality-train:latest
SUBMIT_ZIP ?= experiments/000_text_baseline/artifacts/submissions/quality-text-strong-submit.zip

.PHONY: audit docker-train experiments package safety smoke test train validation

PYTHON ?= python3
IMAGES ?= /data/images

docker-train:
	docker build -f infrastructure/docker/text-baseline.Dockerfile -t $(IMAGE) .
	docker run --rm -v "$(CURDIR):/work" -v "$(TRAIN_DATA):/data/data.csv:ro" \
		$(IMAGE) python -u experiments/000_text_baseline/train.py --train-data /data/data.csv \
		--output /work/experiments/000_text_baseline/submission/strong_text.joblib

train:
	$(PYTHON) experiments/000_text_baseline/train.py --train-data "$(TRAIN_DATA)" \
		--output experiments/000_text_baseline/submission/strong_text.joblib

package:
	cd experiments/000_text_baseline/submission && zip -q -r "$(CURDIR)/$(SUBMIT_ZIP)" \
		metadata.json run.py strong_text.joblib src \
		-x 'src/__pycache__/*' '*.DS_Store'

smoke:
	$(PYTHON) experiments/000_text_baseline/submission/run.py -i "$(TRAIN_DATA)" -o /tmp/ecup-smoke-submit.csv

audit:
	PYTHONPATH=src $(PYTHON) tools/audit_dataset.py \
		--data "$(TRAIN_DATA)" --images "$(IMAGES)" --output-dir /tmp/ecup-data-audit

validation:
	PYTHONPATH=src $(PYTHON) validation/build_folds.py --data "$(TRAIN_DATA)" \
		--output validation/grouped_text_v1/folds.csv \
		--basket-output validation/grouped_text_v1/basket.csv \
		--manifest-output validation/grouped_text_v1/manifest.json

experiments:
	$(PYTHON) tools/list_experiments.py

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q

safety:
	$(PYTHON) tools/check_publish_safety.py
