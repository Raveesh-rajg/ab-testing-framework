# Experiment Decision System

Design and evaluate a controlled experiment, including power, sample-ratio mismatch, frequentist and Bayesian inference, and sequential monitoring.

## Implementation and validation

Local Python framework and Streamlit app. No external service is required.

Automated checks: **38 tests**. The GitHub Actions run linked above the file browser is the current CI result. Local checks and external integrations are separate claims.

## Reproduce locally

Use Python 3.12. Run from this repository’s root in a fresh virtual environment.

```sh
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install -e ".[dev,dashboard]"
```

For repositories using `src/`, set the import path before running commands:

```powershell
# PowerShell
$env:PYTHONPATH="src"
```
```sh
# macOS/Linux
export PYTHONPATH=src
```

```sh
python examples/01_checkout_case_study.py
python -m pytest tests -q
```

## Open the local application

```sh
python -m streamlit run dashboard/app.py
```

## Data and interpretation

Seeded synthetic experiments; simulated false-positive and coverage checks do not establish performance on a live product.

## Inspect the work

- [`tests/`](tests/) — executable checks and examples.
- [`docs/`](docs/) — methodology, integration specifications and the historical design.
- [Portfolio](https://raveesh-rajg.github.io/) — project directory.

## Completion boundary

Passing local tests establishes the checks listed in this repository. It does not establish cloud deployment, real-data quality, production security, or native BI rendering unless an explicit verification record says so.
