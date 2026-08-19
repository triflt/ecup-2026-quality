TRAIN_DATA ?= /data/data.csv
IMAGE ?= ecup-quality-train:latest
SUBMIT_ZIP ?= quality-text-strong-submit.zip

.PHONY: docker-train train package smoke

docker-train:
	docker build -f Dockerfile.train -t $(IMAGE) .
	docker run --rm -v "$(CURDIR):/work" -v "$(TRAIN_DATA):/data/data.csv:ro" \
		$(IMAGE) python -u train.py --train-data /data/data.csv \
		--output /work/submission/strong_text.joblib

train:
	python train.py --train-data "$(TRAIN_DATA)" \
		--output submission/strong_text.joblib

package:
	cd submission && zip -q -r "../$(SUBMIT_ZIP)" \
		metadata.json run.py strong_text.joblib src \
		-x 'src/__pycache__/*' '*.DS_Store'

smoke:
	python submission/run.py -i "$(TRAIN_DATA)" -o /tmp/ecup-smoke-submit.csv
