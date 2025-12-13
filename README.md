Product recommender separate MLflow and Redis server use BentoML for serving

# ML Project Setup: uv + MLflow + DVC + Podman

## Project Structure

```
ml-project/
├── .git/
├── .dvc/
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── train.py
│   │   └── predict.py
│   └── utils/
│       └── config.py
├── tests/
├── container/
│   ├── Containerfile
│   └── .containerignore
├── .dvcignore
├── .gitignore
├── pyproject.toml
├── dvc.yaml
├── params.yaml
└── README.md
```

## Step 1: Initialize Project with uv

### Install uv
```bash
# Install uv (fast Python package installer)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Create pyproject.toml
```toml
[project]
name = "ml-project"
version = "0.1.0"
description = "ML project with MLflow, DVC, and Podman"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.3.0",
    "mlflow>=2.9.0",
    "dvc>=3.30.0",
    "dvc-s3>=3.0.0",  # or dvc-gdrive, dvc-azure
    "python-dotenv>=1.0.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "jupyter>=1.0.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### Initialize Project
```bash
# Create project
mkdir ml-project && cd ml-project

# Initialize uv project
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e ".[dev]"
```

## Step 2: Setup DVC for Data Versioning

### Initialize DVC
```bash
# Initialize git first
git init
git branch -M main

# Initialize DVC
dvc init

# Configure remote storage (example with S3)
dvc remote add -d myremote s3://my-bucket/dvc-storage

# Or use local remote for testing
dvc remote add -d local /tmp/dvc-storage
```

### Create dvc.yaml (Pipeline Definition)
```yaml
stages:
  preprocess:
    cmd: uv run python src/data/preprocessing.py
    deps:
      - src/data/preprocessing.py
      - data/raw
    params:
      - preprocess.train_size
      - preprocess.random_state
    outs:
      - data/processed/train.csv
      - data/processed/test.csv

  train:
    cmd: uv run python src/models/train.py
    deps:
      - src/models/train.py
      - data/processed/train.csv
    params:
      - train.n_estimators
      - train.max_depth
      - train.learning_rate
    outs:
      - models/model.pkl
    metrics:
      - metrics.json:
          cache: false

  evaluate:
    cmd: uv run python src/models/evaluate.py
    deps:
      - src/models/evaluate.py
      - models/model.pkl
      - data/processed/test.csv
    metrics:
      - results/metrics.json:
          cache: false
    plots:
      - results/confusion_matrix.json:
          cache: false
```

### Create params.yaml
```yaml
preprocess:
  train_size: 0.8
  random_state: 42

train:
  n_estimators: 100
  max_depth: 5
  learning_rate: 0.1
  random_state: 42

mlflow:
  tracking_uri: "http://localhost:5000"
  experiment_name: "ml-project-experiment"
```

### Track Data with DVC
```bash
# Add data to DVC tracking
dvc add data/raw/dataset.csv

# Git commit the .dvc file (not the actual data)
git add data/raw/dataset.csv.dvc .gitignore
git commit -m "Add raw dataset to DVC"

# Push data to remote storage
dvc push
```

## Step 3: MLflow for Experiment Tracking

### Start MLflow Server
```bash
# In a separate terminal
mlflow server --host 0.0.0.0 --port 5000

# Or use Podman (see Step 5)
```

## Step 4: Git Workflow with Branches and Tags

### Branch Strategy
```bash
# Create develop branch
git checkout -b develop

# Create feature branch for new experiment
git checkout -b experiment/xgboost-model

# Make changes, run experiments
dvc repro  # Run DVC pipeline
git add .
git commit -m "experiment: XGBoost model implementation"

# Push to GitHub
git push origin experiment/xgboost-model

# Merge to develop after review
git checkout develop
git merge experiment/xgboost-model

# Tag model version
git tag -a model-v1.0.0 -m "Random Forest baseline model"
git push origin model-v1.0.0

# Tag data version
git tag -a data-v1.0.0 -m "Initial dataset"
git push origin data-v1.0.0
```

### .gitignore
```
# Python
__pycache__/
*.py[cod]
.venv/
*.egg-info/

# DVC
/data/raw/*
/data/processed/*
/models/*.pkl
!/data/.gitkeep

# MLflow
mlruns/

# Jupyter
.ipynb_checkpoints/

# Environment
.env
```

## Step 5: Containerization with Podman

### container/Containerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/
COPY models/ ./models/

# Install dependencies
RUN uv pip install --system -e .

# Expose port for API
EXPOSE 8000

# Run inference service
CMD ["python", "-m", "src.models.serve"]
```

### container/.containerignore
```
.git
.dvc
.venv
__pycache__
*.pyc
.pytest_cache
data/raw
data/processed
mlruns
notebooks
.env
```

### Build and Run with Podman
```bash
# Build container
podman build -t ml-project:v1.0.0 -f container/Containerfile .

# Run container
podman run -d -p 8000:8000 --name ml-app ml-project:v1.0.0

# Run MLflow in container
podman run -d -p 5000:5000 \
  -v $(pwd)/mlruns:/mlruns \
  --name mlflow-server \
  ghcr.io/mlflow/mlflow:latest \
  mlflow server --host 0.0.0.0 --port 5000

# Create a pod for multiple containers
podman pod create --name ml-pod -p 8000:8000 -p 5000:5000

# Run containers in pod
podman run -d --pod ml-pod --name mlflow mlflow-server
podman run -d --pod ml-pod --name ml-app ml-project:v1.0.0
```

### docker-compose.yml (Podman compatible)
```yaml
version: '3.8'

services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlruns
    command: mlflow server --host 0.0.0.0 --port 5000

  ml-app:
    build:
      context: .
      dockerfile: container/Containerfile
    ports:
      - "8000:8000"
    depends_on:
      - mlflow
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
```

```bash
# Run with podman-compose
podman-compose up -d
```

## Step 6: Complete Workflow Example

### 1. Setup Project
```bash
# Clone repo
git clone https://github.com/yourteam/ml-project.git
cd ml-project

# Create environment with uv
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -e ".[dev]"

# Pull data from DVC
dvc pull
```

### 2. Create New Experiment
```bash
# Create experiment branch
git checkout -b experiment/new-features

# Modify params.yaml
# Change hyperparameters

# Run pipeline
dvc repro

# Check MLflow UI
# Open http://localhost:5000
```

### 3. Compare Experiments
```bash
# Compare metrics across branches
git checkout experiment/new-features
dvc metrics show

git checkout develop
dvc metrics show

# Use MLflow UI to compare runs
```

### 4. Deploy Best Model
```bash
# Merge to main
git checkout main
git merge develop

# Tag release
git tag -a model-v2.0.0 -m "Improved model with new features"
git push origin model-v2.0.0

# Build and push container
podman build -t ml-project:v2.0.0 -f container/Containerfile .
podman tag ml-project:v2.0.0 registry.example.com/ml-project:v2.0.0
podman push registry.example.com/ml-project:v2.0.0
```

## Step 7: CI/CD with GitHub Actions

### .github/workflows/ci.yml
```yaml
name: ML Pipeline CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
      
      - name: Setup Python
        run: uv venv && source .venv/bin/activate
      
      - name: Install dependencies
        run: uv pip install -e ".[dev]"
      
      - name: Run tests
        run: pytest tests/
      
      - name: Pull DVC data
        run: dvc pull
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
      
      - name: Run DVC pipeline
        run: dvc repro
      
      - name: Check metrics
        run: dvc metrics show

  build-container:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      
      - name: Build with Podman
        run: |
          podman build -t ml-project:${{ github.sha }} -f container/Containerfile .
          podman tag ml-project:${{ github.sha }} ml-project:latest
```

## Useful Commands Cheat Sheet

```bash
# uv
uv pip install <package>          # Install package
uv pip list                        # List packages
uv run python script.py            # Run script in venv

# DVC
dvc add <file>                     # Track file
dvc push                           # Push to remote
dvc pull                           # Pull from remote
dvc repro                          # Reproduce pipeline
dvc metrics show                   # Show metrics
dvc plots show                     # Show plots
dvc dag                            # Show pipeline DAG

# MLflow
mlflow ui                          # Start UI server
mlflow runs                        # List runs
mlflow models serve -m <model>    # Serve model

# Podman
podman build -t <name> .           # Build image
podman run -p 8000:8000 <name>    # Run container
podman ps                          # List containers
podman logs <container>            # View logs
podman exec -it <container> bash   # Enter container

# Git + Tagging
git tag -a v1.0.0 -m "message"    # Create tag
git push origin v1.0.0             # Push tag
git tag -l                         # List tags
```

This setup provides a production-ready ML engineering workflow that scales from 1 to 100+ team members!