# Repository Guidelines

## Project Structure & Module Organization
Core code lives under `src/teamredbench/`. Use `cli.py` for command entrypoints, `runner.py` for suite execution, `config.py` for YAML loading, and `registry.py` for benchmark and metric registration. Built-in benchmarks live in `src/teamredbench/benchmarks/`; derived calculations live in `src/teamredbench/metrics/`. Configuration files belong under `configs/` with hardware and ROCm profiles in `configs/profiles/` and runnable suites in `configs/suites/`. Keep tests in `tests/`, extension notes in `docs/`, and cluster launch examples in `examples/slurm/`. Generated outputs go to `results/`, which is ignored by Git.

## Build, Test, and Development Commands
Create an environment and install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,torch]"
```

Useful day-to-day commands:

```bash
teamredbench discover
teamredbench run configs/suites/smoke.yaml
teamredbench run configs/suites/full.yaml
pytest
pytest --cov=teamredbench --cov-report=term-missing
```

`discover` prints the detected ROCm/PyTorch environment. `run` executes a suite and writes JSON/CSV results. `pytest` runs the unit tests; add coverage output for nontrivial changes.

## Coding Style & Naming Conventions
Target Python 3.10+. Follow the existing style: 4-space indentation, type hints on public functions, and grouped imports (`stdlib`, third-party, local). Use `snake_case` for modules, functions, variables, and tests; use `PascalCase` for classes. Keep YAML filenames descriptive and hardware-specific, for example `amd_instinct_mi350x.yaml`. Add new benchmarks and metrics through the registry decorators instead of wiring them directly into the runner.

## Testing Guidelines
Tests use `pytest` with `src` on the Python path. Name files `test_<area>.py` and name tests `test_<behavior>_<scenario>`. Prefer `tmp_path` fixtures for config-driven flows, and assert both result records and generated output files when touching the runner or CLI behavior.

## Commit & Pull Request Guidelines
This branch has no commit history yet, so no repo-specific convention is established. Use short imperative commit subjects such as `Add MI350X hardware profile` and keep each commit focused. Pull requests should describe the behavior change, list any updated config paths, note ROCm/PyTorch assumptions, and include the test commands you ran. Attach sample CLI output when command behavior or result formatting changes.
