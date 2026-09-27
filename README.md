# 🌸 Iris MLOps Pipeline

End-to-end MLOps: Train · Track · Serve · Test · Deploy

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/iris-mlops.git
cd iris-mlops
make install   # install dependencies
make train     # train model + log to MLflow
make test      # run all tests
make api       # start FastAPI (http://localhost:8000/docs)
make mlflow    # view experiments (http://localhost:5000)
```

## API Usage

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

## Docker

```bash
make train          # train first
make docker-build   # build image
make docker-run     # start API + MLflow
```

## CI/CD

Every push to `main` or `develop` automatically:
1. Lints code (flake8 + black)
2. Trains the model
3. Runs all tests
4. Builds and smoke-tests the Docker image