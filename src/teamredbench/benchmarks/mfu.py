from __future__ import annotations

from dataclasses import replace

from teamredbench.benchmarks.base import Benchmark, BenchmarkRecord
from teamredbench.device import rocm_device_available
from teamredbench.native.runtime import (
    NativeKernelError,
    prepare_native_kernel_binary,
    resolve_native_kernel,
    run_native_kernel,
)
from teamredbench.registry import register_benchmark
from teamredbench.utils import make_tensor, timed_iterations


DEFAULT_SHAPES = [
    [4096, 4096, 4096],
    [8192, 8192, 8192],
]


@register_benchmark("mfu")
class MfuBenchmark(Benchmark):
    default_metrics = ["achieved_tops", "mfu_pct", "latency_us"]

    def _common_params(self) -> tuple[int, int, list[list[int]]]:
        warmup = int(self.params.get("warmup", 10))
        iterations = int(self.params.get("iterations", 50))
        shapes = list(self.params.get("shapes", DEFAULT_SHAPES))
        return warmup, iterations, shapes

    def _run_native(self) -> list[BenchmarkRecord]:
        warmup, iterations, shapes = self._common_params()
        kernel = resolve_native_kernel(
            benchmark=self.name,
            params=self.params,
            suite_path=self.context.suite_path,
            repo_root=self.context.repo_root,
            default_kernel="mfu_hipblas",
        )
        try:
            kernel = replace(kernel, binary_path=prepare_native_kernel_binary(kernel))
        except NativeKernelError as exc:
            error = str(exc)
            records: list[BenchmarkRecord] = []
            for spec in self.selected_dtype_specs():
                for shape in shapes:
                    m, n, k = [int(dim) for dim in shape]
                    records.append(
                        self.emit_record(
                            self.skipped_record(
                                case=f"gemm_{m}x{n}x{k}",
                                dtype=spec.name,
                                error=error,
                                metadata={
                                    "backend": "native",
                                    "kernel": kernel.name,
                                    "m": m,
                                    "n": n,
                                    "k": k,
                                },
                            )
                        )
                    )
            return records

        records: list[BenchmarkRecord] = []
        for spec in self.selected_dtype_specs():
            peak_tops = self.context.hardware_profile.peak_compute_tops.get(spec.name)
            for shape in shapes:
                m, n, k = [int(dim) for dim in shape]
                case = f"gemm_{m}x{n}x{k}"
                metadata = {
                    "backend": "native",
                    "kernel": kernel.name,
                    "m": m,
                    "n": n,
                    "k": k,
                }
                if spec.matmul_ops_factor == 0:
                    records.append(
                        self.emit_record(
                            self.skipped_record(
                                case=case,
                                dtype=spec.name,
                                error="MFU is not defined for this dtype.",
                                metadata=metadata,
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
                                metadata=metadata,
                            )
                        )
                    )
                    continue

                try:
                    result = run_native_kernel(
                        kernel,
                        [
                            "--dtype",
                            spec.name,
                            "--m",
                            str(m),
                            "--n",
                            str(n),
                            "--k",
                            str(k),
                            "--warmup",
                            str(warmup),
                            "--iterations",
                            str(iterations),
                            "--device-id",
                            str(self.context.device_id),
                        ],
                    )
                except NativeKernelError as exc:
                    records.append(
                        self.emit_record(
                            self.skipped_record(
                                case=case,
                                dtype=spec.name,
                                error=str(exc),
                                metadata=metadata,
                            )
                        )
                    )
                    continue

                metadata.update(result.metadata)
                if result.status != "ok":
                    records.append(
                        self.emit_record(
                            self.skipped_record(
                                case=case,
                                dtype=spec.name,
                                error=result.error or "Native kernel did not report success.",
                                metadata=metadata,
                            )
                        )
                    )
                    continue

                raw_metrics = dict(result.raw_metrics)
                raw_metrics.setdefault("elapsed_s", raw_metrics.get("elapsed_seconds"))
                if raw_metrics.get("elapsed_s") is None:
                    records.append(
                        self.emit_record(
                            self.skipped_record(
                                case=case,
                                dtype=spec.name,
                                error="Native kernel did not report elapsed_s.",
                                metadata=metadata,
                            )
                        )
                    )
                    continue

                raw_metrics.setdefault("achieved_ops", spec.matmul_ops_factor * m * n * k)
                raw_metrics.setdefault("peak_tops", peak_tops)
                records.append(
                    self.emit_record(
                        BenchmarkRecord(
                            benchmark=self.name,
                            case=case,
                            dtype=spec.name,
                            raw_metrics=raw_metrics,
                            metadata=metadata,
                        )
                    )
                )
        return records

    def _run_torch(self) -> list[BenchmarkRecord]:
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

        warmup, iterations, shapes = self._common_params()

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
                                metadata={"backend": "torch", "m": m, "n": n, "k": k},
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
                                metadata={"backend": "torch", "m": m, "n": n, "k": k},
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
                                metadata={"backend": "torch", "m": m, "n": n, "k": k},
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
                            metadata={"backend": "torch", "m": m, "n": n, "k": k},
                        )
                    )
                )
        return records

    def run(self) -> list[BenchmarkRecord]:
        backend = str(self.params.get("backend", "torch")).lower()
        if backend == "torch":
            return self._run_torch()
        if backend == "native":
            return self._run_native()
        raise KeyError(f"Unknown MFU backend: {backend}")
