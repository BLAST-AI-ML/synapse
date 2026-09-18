# Developer notes

## Code style

Python code is linted and formatted with Ruff through the pre-commit hooks in {repo}`.pre-commit-config.yaml`:

```bash
pre-commit run --files <modified files>
```

Ruff runs with its default rule set; there is no `pyproject.toml` or `ruff.toml` that overrides it.

## Conda environments

- Dashboard dependencies live in {repo}`dashboard/environment.yml`.
- ML dependencies live in {repo}`ml/environment.yml`.
- Regenerate the corresponding `environment-lock.yml` after dependency changes.

## Build the documentation

Create the documentation conda environment once, from {repo-dir}`docs/`, using {repo}`docs.yml <docs/docs.yml>`:

```bash
conda env create -f docs.yml
```

Build the documentation locally with:

```bash
conda activate synapse-docs
cd docs
make html
```

The generated HTML is written to `docs/build/html/`.

## Testing

The project does not have a full pytest suite.
The main integration check is {repo}`tests/test_ml_pipeline.py`:

```bash
python tests/test_ml_pipeline.py
```

It requires a local MLflow server.

## Code organization

- Dashboard features use manager classes in `dashboard/*_manager.py`.
- Experiment-specific behavior belongs under `experiments/synapse-*`.
- Shared dashboard helpers live in {repo}`dashboard/utils.py`.
