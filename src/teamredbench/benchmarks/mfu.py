from __future__ import annotations

from teamredbench.benchmarks.base import Benchmark, BenchmarkRecord
from teamredbench.device import rocm_device_available
from teamredbench.registry import register_benchmark
from teamredbench.utils import make_tensor, timed_iterations


DEFAULT_SHAPES = [
    [4096, 4096, 4096],
    [8192, 8192, 8192],
]


@register_benchmark("mfu")
class MfuBenchmark(Benchmark):
    default_metrics = ["achieved_tops", "mfu_pct", "latency_us"]

    def run(self) -> list[BenchmarkRecord]:
        if self.context.torch_module is None:
            return [self.emit_record(self.skipped_record(case="global", dtype="n/a", error="PyTorch is not installed."))]

        torch = self.torch
        if not rocm_device_available(torch):
            return [
                self.emit_record(
                    self.skipped_record(case="global", dtype="n/a", error="No ROCm device visible to PyTorch.")
                )
            ]
        self.set_device()

        warmup = int(self.params.get("warmup", 10))
        iterations = int(self.params.get("iterations", 50))
        shapes = list(self.params.get("shapes", DEFAULT_SHAPES))

        records: list[BenchmarkRecord] = []
        for spec in self.selected_dtype_specs():
            peak_tops = self.context.hardware_profile.peak_compute_tops.get(spec.name)
            for shape in shapes:
                m, n, k = [int(dim) for dim in shape]
                case = f"gemm_{m}x{n}x{k}"
                if spec.matmul_ops_factor == 0:
                    records.append(
                        self.emit_record(
                            self.skipped_record(
                                case=case,
                                dtype=spec.name,
                                error="MFU is not defined for this dtype.",
                                metadata={"m": m, "n": n, "k": k},
                            )
                        )
                    )
                    continue
                if peak_tops is None:
                    records.append(
                        self.emit_record(
                            self.skipped_record(
                                case=case,
                                dtype=spec.name,
                                error="Missing peak_compute_tops entry for dtype in hardware profile.",
                                metadata={"m": m, "n": n, "k": k},
                            )
                        )
                    )
                    continue

                try:
                    a = make_tensor(torch, spec, (m, k), self.device)
                    b = make_tensor(torch, spec, (k, n), self.device)
                    out = torch.empty((m, n), device=self.device, dtype=getattr(torch, spec.torch_attr))
                    elapsed_s = timed_iterations(
                        lambda: torch.mm(a, b, out=out),
                        torch_module=torch,
                        warmup=warmup,
                        iterations=iterations,
                    )
                except Exception as exc:
                    records.append(
                        self.emit_record(
                            self.skipped_record(
                                case=case,
                                dtype=spec.name,
                                error=str(exc),
                                metadata={"m": m, "n": n, "k": k},
                            )
                        )
                    )
                    continue

                achieved_ops = spec.matmul_ops_factor * m * n * k
                records.append(
                    self.emit_record(
                        BenchmarkRecord(
                            benchmark=self.name,
                            case=case,
                            dtype=spec.name,
                            raw_metrics={
                                "elapsed_s": elapsed_s,
                                "achieved_ops": achieved_ops,
                                "peak_tops": peak_tops,
                            },
                            metadata={"m": m, "n": n, "k": k},
                        )
                    )
                )
        return records
