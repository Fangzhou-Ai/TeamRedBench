# Extending TeamRedBench

## Hardware Profiles

Hardware profiles are plain YAML files. The runner consumes only a small stable schema:

```yaml
name: example-gpu
family: cdna
peak_hbm_bandwidth_gbps: 0
peak_link_bandwidth_gbps:
  xgmi: 0
  network: 0
peak_compute_tops:
  float16: 0
  bfloat16: 0
  float32: 0
```

That keeps profile maintenance cheap when a new GPU arrives.
Concrete profiles for `MI300X`, `MI325X`, `MI350X`, and `MI355X` are provided under `configs/profiles/hardware/`.
The current runner uses `peak_compute_tops` for GEMM-based MFU and `peak_link_bandwidth_gbps.xgmi` for intra-node collective efficiency.
If you want to track richer published data such as sparse peaks, TF32, or alternate interconnect ceilings, add extra keys to the YAML and consume them from `HardwareProfile.raw` in new metrics.


## ROCm Profiles

ROCm profiles capture runtime assumptions that should not be baked into benchmark logic:

```yaml
name: rocm-generic
rocm_version: "6.x"
libraries:
  pytorch: "2.x"
  rccl: "system"
env: {}
```

Use one profile per runtime stack you care about tracking. Add `RCCL_*` overrides only when the target cluster needs them.

## Metric Plugins

Metrics consume raw counters from a benchmark result. The built-in registration pattern looks like this:

```python
from teamredbench.registry import register_metric

@register_metric("example_metric", "Short description.")
def example_metric(raw: dict[str, object]) -> float | None:
    elapsed = raw.get("elapsed_s")
    payload = raw.get("payload_bytes")
    if not elapsed or not payload:
        return None
    return float(payload) / float(elapsed)
```

Keep benchmark modules focused on collecting counters and benchmark-specific metadata. Put formulas in metrics when possible.

## Benchmark Plugins

Benchmarks subclass `Benchmark` and return a list of `BenchmarkRecord` values:

```python
@register_benchmark("example")
class ExampleBenchmark(Benchmark):
    default_metrics = ["latency_us"]

    def run(self) -> list[BenchmarkRecord]:
        ...
```

That contract is intentionally narrow:

- input: `context` + `params`
- output: `BenchmarkRecord` list

As long as that stays stable, the CLI and result writers do not need to change.

## External Plugin Modules

Suites can import external Python modules before execution:

```yaml
plugins:
  - my_company.redbench_plugins
```

Any benchmarks or metrics registered by that module become available to the runner.

## Native Kernel Plugins

HBM and MFU also support `backend: native`. The built-in kernels are:

- `hbm_hip`
- `mfu_hipblas`

Register your own native kernel from a plugin module:

```python
from pathlib import Path

from teamredbench.native.registry import register_native_kernel

register_native_kernel(
    name="my_hbm_kernel",
    benchmark="hbm",
    description="Custom HIP bandwidth kernel.",
    source_path=Path(__file__).with_name("my_hbm_kernel.hip.cpp"),
    compile_args=(),
)
```

Then select it in a suite:

```yaml
plugins:
  - my_company.teamredbench_native

benchmarks:
  - benchmark: hbm
    params:
      backend: native
      dtypes: [float32]
      modes: [copy]
      size_mib: 4096
      native:
        kernel: my_hbm_kernel
```

You can also skip Python registration and point directly at a source or binary:

```yaml
native:
  source: ./my_hbm_kernel.hip.cpp
```

or:

```yaml
native:
  binary: ./my_hbm_kernel
```

Supported native config keys:

- `kernel`: registered native kernel name
- `source`: HIP/C++ source path, resolved relative to the suite file
- `binary`: prebuilt executable path, resolved relative to the suite file
- `compiler`: compiler override
- `compile_args`: additional compile flags
- `run_args`: extra runtime arguments appended after the standard TeamRedBench arguments
- `env`: environment overrides for the native process
- `cache_dir`: override the native binary cache location

Runtime contract for custom executables:

- HBM receives:
  `--dtype --mode --size-mib --warmup --iterations --device-id --scale`
- MFU receives:
  `--dtype --m --n --k --warmup --iterations --device-id`
- The last non-empty line on `stdout` must be JSON shaped like:

```json
{
  "status": "ok",
  "raw_metrics": {
    "elapsed_s": 0.001
  },
  "metadata": {
    "implementation": "my-kernel"
  }
}
```

Use `status: "skipped"` with an `error` field when a dtype or mode is intentionally unsupported.
