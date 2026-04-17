from __future__ import annotations

from typing import Any

from teamredbench.benchmarks.base import Benchmark, BenchmarkRecord
from teamredbench.device import rocm_device_available
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

        size_mib = int(self.params.get("size_mib", 1024))
        warmup = int(self.params.get("warmup", 10))
        iterations = int(self.params.get("iterations", 50))
        modes = list(self.params.get("modes", ["copy"]))
        scale = float(self.params.get("scale", 1.0))
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
                                metadata={"mode": mode, "size_mib": size_mib},
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
                                "mode": mode,
                                "size_mib": size_mib,
                                "elements": numel,
                            },
                        )
                    )
                )
        return records
