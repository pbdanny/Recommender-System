#!/bin/bash
# Setup script for new team members

set -e

echo "🚀 Setting up ML Project..."

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Create virtual environment
echo "Creating virtual environment..."
uv venv

# Activate environment
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
uv pip install -e ".[dev]"

# Pull data from DVC
echo "Pulling data from DVC..."
dvc pull

# Setup pre-commit hooks (optional)
if [ -f .pre-commit-config.yaml ]; then
    echo "Setting up pre-commit hooks..."
    uv pip install pre-commit
    pre-commit install
fi

echo "✓ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Activate environment: source .venv/bin/activate"
echo "  2. Start MLflow: mlflow server --host 0.0.0.0 --port 5000"
echo "  3. Run pipeline: dvc repro"