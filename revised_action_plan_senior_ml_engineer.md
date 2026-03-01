# Revised Action Plan: Senior ML Engineer
## Cloud-Agnostic Focus + GCP as Deployment Target + CI/CD Priority

**For:** Danny Thanakrit Boonquarmdee (@pbdanny)  
**Date:** March 1, 2026  
**Confidence Level:** 90% — Aligned with your specific goals and market data  

---

## Strategic Direction: Why NOT GCP-Specific

My previous analysis over-indexed on GCP. Given your answers, here's the revised strategy:

**Your goal** = Senior ML Engineer (company-agnostic)  
**Your cloud** = GCP (at work, but not the core of your identity)  
**Your urgent gap** = CI/CD & testing  

This means: **invest 80% in cloud-agnostic engineering skills, 20% in GCP deployment knowledge.**

Why? Because what makes someone "senior" is not knowing Vertex AI buttons — it's the ability to:
- Design reliable ML systems end-to-end
- Write production-quality, tested code
- Automate everything (CI/CD)
- Debug and monitor systems in production
- Make sound trade-off decisions

Your repo already has the right tools (MLflow, DVC, BentoML, Docker, Prometheus). What's missing is the **engineering discipline layer** — tests, CI/CD, and deployment automation.

---

## Your Repo: What's Already Strong

```
✅ MLflow          → Experiment tracking (cloud-agnostic, works on GCP too)
✅ DVC             → Data versioning (can use GCS as remote storage)
✅ BentoML         → Model serving (can deploy to Cloud Run/GKE)
✅ Docker Compose  → Local orchestration
✅ Prometheus      → Monitoring
✅ pyproject.toml  → Modern Python packaging
✅ params.yaml     → Parameterized experiments
✅ metrics.json    → Metric tracking
```

These are all **portable** — they work on GCP, AWS, Alibaba Cloud, or bare metal.

---

## Phase 1: CI/CD & Testing (Your #1 Priority)
**Timeline: 2-3 weeks | Impact: HUGE for Senior-level credibility**

This is the single biggest gap between your current repo and a senior-level project. Here's exactly what to add:

### 1.1 — Add `tests/` Directory with pytest

```
tests/
├── unit/
│   ├── test_data_processing.py    # Test your data transforms
│   ├── test_feature_engineering.py # Test feature pipelines
│   └── test_model.py              # Test model input/output shapes
├── integration/
│   └── test_pipeline.py           # Test DVC pipeline stages end-to-end
└── conftest.py                    # Shared fixtures
```

**What to test (Senior-level thinking):**
- Data validation: schema checks, null handling, type enforcement
- Feature engineering: input → output correctness
- Model: expected output shape, prediction range sanity
- Pipeline: DVC stages run without error on sample data

**Tools:**
- `pytest` — test runner
- `pytest-cov` — coverage reports
- `great_expectations` or `pandera` — data validation (optional but impressive)

**Reference:** https://mlops-guide.github.io/MLOps/CICDML/

### 1.2 — Add GitHub Actions CI Pipeline

Create `.github/workflows/ci.yml`:

```yaml
name: ML CI Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version-file: '.python-version'
      - name: Install uv
        run: pip install uv
      - name: Install dependencies
        run: uv sync
      - name: Lint (ruff)
        run: uv run ruff check src/ tests/
      - name: Type check (mypy)
        run: uv run mypy src/ --ignore-missing-imports
      - name: Run tests
        run: uv run pytest tests/ -v --cov=src --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v4

  build-docker:
    needs: lint-and-test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build BentoML container
        run: docker compose build
```

**Why this matters for Senior level:** CI/CD in ML means every code or data change automatically triggers validation. This is what separates notebook-only data scientists from ML engineers.

**Reference:** https://github.blog/enterprise-software/ci-cd/streamlining-your-mlops-pipeline-with-github-actions-and-arm64-runners/

### 1.3 — Add CD Pipeline (Deploy to GCP)

Create `.github/workflows/cd.yml` (triggered on release/tag):

```yaml
name: ML CD Pipeline

on:
  push:
    tags: ['v*']  # Triggered when you create a release tag

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Authenticate to GCP
        uses: google-github-actions/auth@v2
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}
      - name: Build & Push to Artifact Registry
        run: |
          gcloud builds submit --tag gcr.io/${{ secrets.GCP_PROJECT }}/recommender:${{ github.ref_name }}
      - name: Deploy to Cloud Run
        run: |
          gcloud run deploy recommender \
            --image gcr.io/${{ secrets.GCP_PROJECT }}/recommender:${{ github.ref_name }} \
            --region asia-southeast1 \
            --allow-unauthenticated
```

**Why Cloud Run (not Vertex AI Endpoints):** Cloud Run is simpler, cheaper, and your BentoML container already works with it. This is the pragmatic Senior choice — use the simplest tool that works.

### 1.4 — Fix the `.env` Security Issue

```bash
# Remove .env from tracking (keep it locally)
git rm --cached .env
echo ".env" >> .gitignore

# Create a template instead
cp .env .env.example
# Remove actual secrets from .env.example, leave only key names
```

---

## Phase 2: Engineering Quality (Weeks 3-6)
**What separates "mid" from "senior"**

### 2.1 — Add Type Hints + Docstrings

```python
# Before (mid-level)
def process_data(df, min_interactions):
    filtered = df[df['count'] >= min_interactions]
    return filtered

# After (senior-level)
def process_data(
    df: pd.DataFrame,
    min_interactions: int = 5,
) -> pd.DataFrame:
    """Filter users with fewer than min_interactions.

    Args:
        df: Raw interaction dataframe with 'user_id', 'item_id', 'count' columns.
        min_interactions: Minimum interaction threshold.

    Returns:
        Filtered dataframe.

    Raises:
        ValueError: If required columns are missing.
    """
    required_cols = {"user_id", "item_id", "count"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing columns: {required_cols - set(df.columns)}")
    return df[df["count"] >= min_interactions]
```

### 2.2 — Add Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: check-yaml
      - id: end-of-file-fixer
      - id: trailing-whitespace
      - id: detect-private-key  # Catches accidentally committed secrets
```

### 2.3 — Add Architecture Documentation

Create `docs/architecture.md` with a Mermaid diagram:

```mermaid
graph LR
    A[Raw Data] -->|DVC| B[Feature Engineering]
    B -->|DVC Pipeline| C[Model Training]
    C -->|MLflow| D[Model Registry]
    D -->|BentoML| E[Docker Container]
    E -->|GitHub Actions CD| F[GCP Cloud Run]
    F -->|Prometheus| G[Monitoring]
```

Senior engineers document their design decisions, not just their code.

---

## Phase 3: Light GCP Integration (Weeks 6-10)
**Just enough GCP to be effective at work — NOT certification-driven**

Since GCP is your work platform, learn these practical skills (not exam topics):

| What | Why | How |
|---|---|---|
| GCS as DVC remote | You already use DVC — just point it to GCS | `dvc remote add -d gcs gs://your-bucket/dvc` |
| Cloud Run for serving | Your BentoML Docker already works — just deploy | `gcloud run deploy` |
| Artifact Registry | Store your Docker images | Replace Docker Hub with `gcr.io` |
| Cloud Build (optional) | Alternative to GitHub Actions, native to GCP | `cloudbuild.yaml` |
| BigQuery for data | If your work data lives in BQ | `pandas-gbq` or `google-cloud-bigquery` |

**What to SKIP (for now):**
- Vertex AI Pipelines (your DVC pipeline is fine)
- Vertex AI Feature Store (overkill unless your team uses it)
- AutoML (learn it only if your work needs it)
- GCP Professional ML Engineer cert (nice-to-have, not urgent)

---

## Phase 4: Senior-Level Differentiators (Ongoing)

These are what hiring managers look for in Senior ML Engineers:

### 4.1 — Add a Model Validation Gate

In your CI pipeline, add an automated check that compares the new model against the current production model:

```python
# tests/integration/test_model_quality.py
def test_model_beats_baseline():
    """New model must beat current production metrics."""
    new_metrics = json.load(open("metrics.json"))
    baseline_metrics = {"precision@10": 0.15, "recall@10": 0.08}  # from production

    assert new_metrics["precision@10"] >= baseline_metrics["precision@10"] * 0.95, \
        "New model precision dropped >5% vs production"
```

This is the "Continuous Training" pattern — a hallmark of senior MLOps thinking.

**Reference:** https://www.thirstysprout.com/post/mlops-best-practices

### 4.2 — Add Data Validation

```python
# src/validate_data.py
import pandera as pa

schema = pa.DataFrameSchema({
    "user_id": pa.Column(int, nullable=False),
    "item_id": pa.Column(int, nullable=False),
    "rating": pa.Column(float, pa.Check.in_range(0.0, 5.0)),
    "timestamp": pa.Column(int, pa.Check.gt(0)),
})
```

### 4.3 — Write a Design Doc / ADR

Create `docs/adr/001-serving-framework.md`:

```markdown
# ADR-001: Why BentoML over Vertex AI Endpoints

## Status: Accepted

## Context
We need a model serving framework for our recommender system.

## Decision
We chose BentoML because:
- Cloud-agnostic (works on GCP, AWS, bare metal)
- Native support for batching and async serving
- Easy containerization with `bentofile.yaml`
- Lower cost than Vertex AI Endpoints for our scale

## Trade-offs
- No built-in A/B testing (would need to add via Cloud Run traffic splitting)
- No automatic model monitoring (we use Prometheus instead)
```

Senior engineers make AND document design decisions.

---

## Revised Skills Radar

```
                    Current → Target (Senior ML Engineer, cloud-agnostic)

ML Algorithms          ████████░░  (8/10) → ████████░░ (8/10)  ✅ On track
Python Engineering     ███████░░░  (7/10) → █████████░ (9/10)  🟡 Add types, tests
CI/CD & Automation     ██░░░░░░░░  (2/10) → █████████░ (9/10)  🔴 YOUR #1 PRIORITY
Testing                ██░░░░░░░░  (2/10) → ████████░░ (8/10)  🔴 Part of CI/CD
MLOps (Open Source)    ████████░░  (8/10) → █████████░ (9/10)  ✅ Almost there
Model Serving          ███████░░░  (7/10) → ████████░░ (8/10)  🟡 Add Cloud Run deploy
Monitoring             █████░░░░░  (5/10) → ███████░░░ (7/10)  🟡 Add data validation
GCP (practical)        ███░░░░░░░  (3/10) → ██████░░░░ (6/10)  🟡 GCS, Cloud Run, BQ
System Design          █████░░░░░  (5/10) → ████████░░ (8/10)  🟡 Add docs & ADRs
```

---

## Summary: What Changed from Previous Analysis

| Previous Advice | Revised Advice | Reason |
|---|---|---|
| 🔴 "Learn Vertex AI Pipelines" | 🟢 Keep DVC, add GitHub Actions CI/CD | DVC is cloud-agnostic and you already know it |
| 🔴 "Learn BigQuery ML" | 🟡 Learn BigQuery (data only), skip BQML | BQML is niche; your custom model is better |
| 🔴 "Get GCP ML Engineer cert" | 🟡 Optional later, not urgent | Your goal is Senior role, not certification |
| 🔴 "Learn Vertex AI Feature Store" | 🟢 Add pandera/great_expectations | Cloud-agnostic data validation is more portable |
| 🔴 "Deploy on Vertex AI Endpoints" | 🟢 Deploy BentoML on Cloud Run | Simpler, cheaper, your container already works |
| Not mentioned | 🔴 **Add pytest + GitHub Actions NOW** | This is your #1 gap for Senior level |
| Not mentioned | 🔴 **Add type hints + pre-commit** | Engineering quality signals seniority |
| Not mentioned | 🟡 **Add architecture docs + ADRs** | Senior engineers document decisions |

---

## Quick-Start Checklist (This Week)

- [ ] `git rm --cached .env` and add `.env.example`
- [ ] `mkdir -p tests/unit tests/integration`
- [ ] Write 3 unit tests with pytest for your core `src/` functions
- [ ] Create `.github/workflows/ci.yml` with lint + test
- [ ] Add `ruff` and `mypy` to `pyproject.toml` dev dependencies
- [ ] Set up `.pre-commit-config.yaml`
- [ ] Push and watch your first green CI check ✅

**This alone will level up your repo from "good personal project" to "senior-level production codebase."**

---

## Key References

| Resource | Link |
|---|---|
| GitHub Actions for MLOps | https://github.blog/enterprise-software/ci-cd/streamlining-your-mlops-pipeline-with-github-actions-and-arm64-runners/ |
| MLOps CI/CD Guide | https://mlops-guide.github.io/MLOps/CICDML/ |
| MLOps Best Practices 2025 | https://www.thirstysprout.com/post/mlops-best-practices |
| ML CI/CD with GitHub Actions (tutorial) | https://lo-victoria.com/implementing-cicd-pipelines-with-github-actions-for-mlops |
| pytest Documentation | https://docs.pytest.org/ |
| Ruff Linter | https://docs.astral.sh/ruff/ |
| Pandera (Data Validation) | https://pandera.readthedocs.io/ |
| Cloud Run Deployment | https://cloud.google.com/run/docs |
