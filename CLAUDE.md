# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TeamRedBench is a config-driven benchmark suite for AMD RDNA and CDNA GPUs on ROCm. It measures HBM bandwidth, intra/inter-node collective communication bandwidth (RCCL), and GEMM-based MFU. Hardware profiles, benchmarks, and metrics are all registry-driven — adding new ones never requires modifying core runner code.

## Setup & Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,torch]"
```

Requires ROCm 6.x and `hipcc` for native HIP kernel compilation.

## Common Commands

```bash
# Check detected ROCm/PyTorch environment
teamredbench discover

# Run a benchmark suite
teamredbench run configs/suites/smoke.yaml
teamredbench run configs/suites/full.yaml

# Inspect registered components
teamredbench list-benchmarks
teamredbench list-metrics
teamredbench list-native-kernels

# Run all tests
pytest

# Run a specific test file
pytest tests/test_config.py

# With coverage
pytest --cov=teamredbench --cov-report=term-missing

# Clear compiled native kernel cache
teamredbench clean-native-cache
```

## Architecture

**Execution flow:** CLI → load suite YAML → build context (device detection, hardware/runtime profiles) → load plugins → for each benchmark invocation: instantiate, run, apply metrics, collect records → write JSON/CSV/metadata results.

**Core abstractions:**

- **Benchmark** (`benchmarks/base.py`): Abstract base. Subclass and decorate with `@register_benchmark(name)`. `run()` returns `list[BenchmarkRecord]` with `raw_metrics` populated.
- **Metric** (`metrics/`): Functions decorated with `@register_metric(name, description)`. Take `raw_metrics` dict, return a derived float or None.
- **Registry** (`registry.py`): Central lookup for benchmarks and metrics. Auto-populated at import via decorators. External plugins loaded via `plugins:` list in suite YAML.
- **Config** (`config.py`): Loads `HardwareProfile` (peak BW/compute specs), `RuntimeProfile` (ROCm version, env vars), and `SuiteConfig` (benchmark invocations, output format, profiling) from YAML.
- **Native kernels** (`native/`): Optional HIP/C++ kernels compiled on demand with `hipcc`. Contract: accept CLI args, emit JSON to stdout. Compiled binaries cached in `~/.cache/teamredbench/native/`.
- **Profiling** (`profiling/`): Optional rocprof wrapper. Extensible via `register_profile_engine()`.

**Key files:**
- `src/teamredbench/cli.py` — Typer CLI entry point
- `src/teamredbench/runner.py` — `run_suite()` orchestration
- `src/teamredbench/config.py` — YAML loading and dataclasses
- `src/teamredbench/registry.py` — Benchmark/metric registration
- `src/teamredbench/builtin.py` — Triggers import of all built-ins
- `src/teamredbench/benchmarks/` — `hbm.py`, `collective.py`, `mfu.py`
- `configs/profiles/hardware/` — Per-SKU peak specs (e.g., `amd_instinct_mi300x.yaml`)
- `configs/suites/` — Runnable suite definitions

## Extending the Benchmark Suite

- **New hardware profile**: Add a YAML to `configs/profiles/hardware/` following the existing schema (peak HBM BW, compute TOPS per dtype, link BW).
- **New benchmark**: Subclass `Benchmark`, decorate with `@register_benchmark`, place in `benchmarks/` or an external plugin module.
- **New metric**: Write a function, decorate with `@register_metric`, place in `metrics/` or a plugin.
- **Native kernel**: Register via `register_native_kernel()` or reference source/binary in suite config.
- **External plugin**: Import via `plugins:` list in suite YAML — the runner imports the module, triggering decorator registration.

## Coding Style

- Python 3.10+, 4-space indentation, type hints on public functions
- Grouped imports: stdlib → third-party → local
- `snake_case` for modules/functions/variables, `PascalCase` for classes
- `test_<area>.py` files, `test_<behavior>_<scenario>` test names
- Use `tmp_path` pytest fixture for config-driven test flows; assert both result records and generated output files when touching runner/CLI behavior
- YAML filenames should be descriptive and hardware-specific (e.g., `amd_instinct_mi350x.yaml`)

## Commit & PR Guidelines

Short imperative subjects: `Add MI350X hardware profile`. PRs should describe the behavior change, list updated config paths, note ROCm/PyTorch assumptions, and include test commands run. Attach sample CLI output when command behavior or result formatting changes.
