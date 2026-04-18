from __future__ import annotations

from dataclasses import replace
from typing import Any

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


MODE_BYTE_FACTORS = {
    "copy": 2,
    "scale": 2,
    "triad": 3,
}


@register_benchmark("hbm")
class HbmBandwidthBenchmark(Benchmark):
    default_metrics = ["hbm_bandwidth_gbps", "hbm_efficiency_pct", "latency_us"]

    def _run_mode(self, mode: str, src: Any, dst: Any, aux: Any, scale: float) -> None:
        if mode == "copy":
            dst.copy_(src)
            return
        if mode == "scale":
            self.torch.mul(src, scale, out=dst)
            return
        if mode == "triad":
            self.torch.add(src, aux, alpha=scale, out=dst)
            return
        raise KeyError(f"Unknown HBM mode: {mode}")

    def _common_params(self) -> tuple[int, int, int, list[str], float]:
        size_mib = int(self.params.get("size_mib", 1024))
        warmup = int(self.params.get("warmup", 10))
        iterations = int(self.params.get("iterations", 50))
        modes = list(self.params.get("modes", ["copy"]))
        scale = float(self.params.get("scale", 1.0))
        return size_mib, warmup, iterations, modes, scale

    def _run_native(self) -> list[BenchmarkRecord]:
        size_mib, warmup, iterations, modes, scale = self._common_params()
        total_bytes = size_mib * 1024 * 1024
        kernel = resolve_native_kernel(
            benchmark=self.name,
            params=self.params,
            suite_path=self.context.suite_path,
            repo_root=self.context.repo_root,
            default_kernel="hbm_hip",
        )
        try:
            kernel = replace(kernel, binary_path=prepare_native_kernel_binary(kernel))
        except NativeKernelError as exc:
            error = str(exc)
            records: list[BenchmarkRecord] = []
            for spec in self.selected_dtype_specs():
                for mode in modes:
                    records.append(
                        self.emit_record(
                            self.skipped_record(
                                case=f"{mode}_{size_mib}MiB",
                                dtype=spec.name,
                                error=error,
                                metadata={
                                    "backend": "native",
                                    "kernel": kernel.name,
                                    "mode": mode,
                                    "size_mib": size_mib,
                                },
                            )
                        )
                    )
            return records

        records: list[BenchmarkRecord] = []
        for spec in self.selected_dtype_specs():
            numel = max(total_bytes // spec.bytes_per_element, 1)
            for mode in modes:
                case = f"{mode}_{size_mib}MiB"
                metadata = {
                    "backend": "native",
                    "kernel": kernel.name,
                    "mode": mode,
                    "size_mib": size_mib,
                    "elements": numel,
                }
                kernel_args = [
                    "--dtype",
                    spec.name,
                    "--mode",
                    mode,
                    "--size-mib",
                    str(size_mib),
                    "--warmup",
                    str(warmup),
                    "--iterations",
                    str(iterations),
                    "--device-id",
                    str(self.context.device_id),
                    "--scale",
                    str(scale),
                ]
                blocks_per_cu = self.params.get("blocks_per_cu")
                if blocks_per_cu is not None:
                    kernel_args += ["--blocks-per-cu", str(int(blocks_per_cu))]
                    metadata["blocks_per_cu"] = int(blocks_per_cu)
                try:
                    result = run_native_kernel(kernel, kernel_args)
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

                raw_metrics.setdefault("moved_bytes", MODE_BYTE_FACTORS[mode] * numel * spec.bytes_per_element)
                raw_metrics.setdefault("peak_bandwidth_gbps", self.context.hardware_profile.peak_hbm_bandwidth_gbps)
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

        size_mib, warmup, iterations, modes, scale = self._common_params()
        total_bytes = size_mib * 1024 * 1024

        records: list[BenchmarkRecord] = []
        for spec in self.selected_dtype_specs():
            numel = max(total_bytes // spec.bytes_per_element, 1)
            shape = (numel,)
            for mode in modes:
                case = f"{mode}_{size_mib}MiB"
                try:
                    src = make_tensor(torch, spec, shape, self.device)
                    dst = torch.empty_like(src)
                    aux = make_tensor(torch, spec, shape, self.device) if mode == "triad" else None
                    elapsed_s = timed_iterations(
                        lambda: self._run_mode(mode, src, dst, aux, scale),
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
                                metadata={"backend": "torch", "mode": mode, "size_mib": size_mib},
                            )
                        )
                    )
                    continue

                moved_bytes = MODE_BYTE_FACTORS[mode] * numel * spec.bytes_per_element
                records.append(
                    self.emit_record(
                        BenchmarkRecord(
                            benchmark=self.name,
                            case=case,
                            dtype=spec.name,
                            raw_metrics={
                                "elapsed_s": elapsed_s,
                                "moved_bytes": moved_bytes,
                                "peak_bandwidth_gbps": self.context.hardware_profile.peak_hbm_bandwidth_gbps,
                            },
                            metadata={
                                "backend": "torch",
                                "mode": mode,
                                "size_mib": size_mib,
                                "elements": numel,
                            },
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
        raise KeyError(f"Unknown HBM backend: {backend}")
