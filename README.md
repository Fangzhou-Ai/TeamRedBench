# TeamRedBench

TeamRedBench is a config-driven benchmark repo for AMD RDNA and CDNA GPUs on ROCm. It focuses on three benchmark families out of the box:

- HBM bandwidth
- Intra-node and inter-node communication bandwidth
- MFU-style compute utilization from GEMM throughput

The repo is built to stay adaptable when hardware, ROCm, or metrics change:

- Hardware profiles live in YAML instead of code.
- Runtime profiles capture ROCm/library assumptions separately from hardware.
- Benchmarks and metrics are both registry-driven, so new modules can be added without editing the runner.
- Dtype support is discovered dynamically from the installed `torch` build, including optional float8 types when present.

## Repo Layout

```text
.
├── configs/
│   ├── profiles/
│   │   ├── hardware/
│   │   └── rocm/
│   └── suites/
├── docs/
├── examples/
├── src/teamredbench/
│   ├── benchmarks/
│   └── metrics/
└── tests/
```

## Quick Start

1. Install the package and runtime dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,torch]"
```

2. Inspect the detected ROCm/PyTorch environment:

```bash
teamredbench discover
```

3. Copy or edit a hardware profile under `configs/profiles/hardware/` and fill in the peak numbers for the target GPU.
   Concrete published profiles are included for `AMD Instinct MI300X`, `MI325X`, `MI350X`, and `MI355X`.
   If a suite still points at a generic profile, the runner will auto-select a matching published profile when it recognizes the local GPU SKU.

4. Run a suite:

```bash
teamredbench run configs/suites/smoke.yaml
```

Results are written to `results/` in JSON and CSV, plus a `*.metadata.json` sidecar with the run command, config contents, environment snapshot, git state, and software/runtime versions needed to reproduce the run.
When the hardware profile defines peak bandwidth or compute values, the live run output also prints the percentage of theoretical peak. If a peak is not configured, the percentage is shown as `n/a`.

## Benchmark Coverage

### HBM bandwidth

The `hbm` benchmark uses large tensor kernels to stress device memory traffic. It supports:

- `copy`
- `scale`
- `triad`

Each result reports raw counters and derived metrics such as:

- `hbm_bandwidth_gbps`
- `hbm_efficiency_pct`
- `latency_us`

### Communication bandwidth

The `collective` benchmark uses `torch.distributed` with RCCL. It supports:

- `all_reduce`
- `all_gather`
- `broadcast`

The benchmark classifies the run as `intra-node` or `inter-node` by gathering hostnames after process-group init. It reports:

- `payload_bandwidth_gbps`
- `bus_bandwidth_gbps`
- `link_efficiency_pct`
- `latency_us`

Launch collectives with `torchrun`, `srun`, or another distributed launcher that sets the standard environment variables.
Use `backend: rccl` in suite params when you need to override the default.

### MFU

The `mfu` benchmark runs GEMM sweeps and compares achieved throughput against per-dtype theoretical peaks from the selected hardware profile. It reports:

- `achieved_tops`
- `mfu_pct`
- `latency_us`

For integer and complex dtypes, the repo uses dtype-specific operation-count factors so MFU remains tied to the configured theoretical peak.

## Dtype Strategy

`teamredbench list-dtypes` shows every dtype the local `torch` build exposes. The repo tries to cover:

- `bool`
- integer types
- `float16`, `bfloat16`, `float32`, `float64`
- `complex64`, `complex128`
- float8 variants when the installed `torch` exposes them

Some dtype and benchmark combinations are not valid on every ROCm stack. Those cases are recorded as `skipped` with an error message instead of aborting the whole suite.

## Adapting to New Hardware

Hardware-specific numbers are isolated in YAML:

- peak HBM bandwidth
- peak communication link bandwidths
- peak per-dtype compute throughput

Published SKU profiles are provided for:

- `configs/profiles/hardware/amd_instinct_mi300x.yaml`
- `configs/profiles/hardware/amd_instinct_mi325x.yaml`
- `configs/profiles/hardware/amd_instinct_mi350x.yaml`
- `configs/profiles/hardware/amd_instinct_mi355x.yaml`

To bring up a new accelerator:

1. Run `teamredbench discover`.
2. Copy the closest profile from `configs/profiles/hardware/`.
3. Fill in the target GPU's peak numbers.
4. Point the suite at the new profile.

Nothing in the benchmark runner is hard-coded to MI2xx, MI3xx, or RDNA SKUs.
For inter-node communication, the `network` peak remains system-specific because it depends on the installed NIC and fabric rather than the GPU alone.

## Adapting to New ROCm Versions

ROCm assumptions live under `configs/profiles/rocm/`. Keep runtime-specific items there:

- expected ROCm version
- library versions or notes
- RCCL-specific environment overrides when the target stack needs them

This keeps runtime drift separate from hardware drift.

## Adding a New Benchmark or Metric

Benchmarks and metrics register themselves at import time.

New benchmark:

1. Add a module under `src/teamredbench/benchmarks/`.
2. Decorate the class with `@register_benchmark("name")`.
3. Return `BenchmarkRecord` objects with raw counters.

New metric:

1. Add a function under `src/teamredbench/metrics/`.
2. Decorate it with `@register_metric("metric_name", "...")`.
3. Compute from the benchmark's raw counters.

External modules can also be loaded via the suite `plugins:` field.

More detail is in [docs/extending.md](/root/TeamRedBench/docs/extending.md).

## Example Commands

List built-ins:

```bash
teamredbench list-benchmarks
teamredbench list-metrics
teamredbench list-dtypes
```

Run the full suite:

```bash
teamredbench run configs/suites/full.yaml
```

Multi-node collective example:

```bash
sbatch examples/slurm/multi_node_collective.sh
```
