.PHONY: install test lint demo run api dashboard docker mlflow-ui clean
export PYTHONPATH := src:.
ROOT ?= workspace

install:           ## install runtime + dev dependencies
	pip install -r requirements-dev.txt

test:              ## unit + integration tests
	python -m pytest

lint:
	ruff check src apps tests

demo:              ## whole pipeline on synthetic data (no Kaggle account needed)
	python -m olympics.cli run --demo --root $(ROOT)_demo --set ml.profile=fast

run:               ## real data (needs Kaggle credentials); resumes from checkpoints
	python -m olympics.cli run --root $(ROOT)

retrain:           ## champion/challenger retraining with collected human feedback
	python -m olympics.cli retrain --root $(ROOT)

api:
	OLYMPICS_ROOT=$(ROOT) uvicorn apps.api.main:app --reload --port 8000

dashboard:
	OLYMPICS_ROOT=$(ROOT) python -m apps.dashboard.app

mlflow-ui:
	mlflow ui --backend-store-uri file:$(ROOT)/mlruns

docker:
	docker compose up --build

clean:
	rm -rf $(ROOT)_demo .pytest_cache
