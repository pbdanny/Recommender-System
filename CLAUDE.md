# CLAUDE.md

## Project Overview

Hybrid Recommendation System using collaborative filtering (SVD via scikit-surprise), served via BentoML REST API with Redis caching. MLflow tracks experiments; DVC manages data pipelines.

## Tech Stack

- **ML**: scikit-surprise (SVD), scikit-learn, pandas, numpy
- **Serving**: BentoML, FastAPI, Uvicorn
- **Tracking**: MLflow
- **Pipeline**: DVC
- **Cache**: Redis
- **Infra**: Docker Compose, Prometheus, Grafana
- **Package manager**: UV (Python 3.11)

## Commands

### Setup

```bash
uv sync                        # Install all dependencies
uv pip install -e ".[dev]"     # Install with dev extras
```

### Run Pipeline

```bash
uv run dvc repro               # Run full DVC pipeline (preprocess + train)
uv run python src/data/preprocessing.py    # Preprocess only
uv run python src/models/batch_train.py   # Train only
```

### Start Services

```bash
docker-compose up -d           # Start MLflow, Redis, Prometheus, Grafana
uv run bentoml serve src.service:HybridRecSys  # Serve recommendation API
```

### Tests & Linting

```bash
uv run pytest                  # Run tests
black src/                     # Format code
ruff check src/                # Lint code
uv run locust                  # Load testing
```

### Service Ports

| Service    | Port |
|------------|------|
| MLflow     | 5000 |
| BentoML    | 3000 |
| Redis      | 6379 |
| Prometheus | 9090 |
| Grafana    | 3001 |

## Project Structure

```
src/
  service.py              # BentoML API service
  data/preprocessing.py   # Data loading and train/test split
  models/batch_train.py   # SVD training, MLflow logging, batch prediction
data/raw/                 # Raw dataset (DVC-tracked, MovieLens format)
params.yaml               # ML hyperparameters
dvc.yaml                  # DVC pipeline definition
bentofile.yaml            # BentoML service config
docker-compose.yml        # Infrastructure services
```

## ML Parameters (params.yaml)

```yaml
train_size: 0.8
n_factors: 50
n_epochs: 20
lr_all: 0.005
reg_all: 0.02
```

## Architecture

1. **Batch**: `batch_train.py` trains SVD on MovieLens data, generates top-500 recommendations per user, stores in Redis via MLflow-tracked runs.
2. **Serve**: `service.py` exposes a BentoML endpoint that reads pre-computed recommendations from Redis.
3. **Monitor**: Prometheus scrapes BentoML metrics; Grafana dashboards visualize them.

## Environment

The code detects Docker via `AM_I_IN_A_DOCKER_CONTAINER` env var and switches connection strings accordingly (localhost vs. Docker service names).

## Coding Conventions

- Python 3.11+, type hints encouraged
- Formatting: `black`, linting: `ruff`
- All hyperparameters go in `params.yaml`, not hardcoded
- Data files tracked via DVC, never committed to git
