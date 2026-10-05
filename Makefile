.PHONY: install train test lint format api mlflow docker-build docker-run clean

install:
	pip install -r requirements.txt

train:
	cd src && python train.py

test:
	pytest tests/ -v

lint:
	flake8 src/ api/ tests/ --max-line-length=100

format:
	black src/ api/ tests/

api:
	uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

mlflow:
	mlflow ui --host 0.0.0.0 --port 5000

docker-build:
	docker build -t iris-mlops:latest .

docker-run:
	docker-compose up -d

clean:
	rm -rf models/*.joblib models/*.json mlruns/ __pycache__/ .pytest_cache/