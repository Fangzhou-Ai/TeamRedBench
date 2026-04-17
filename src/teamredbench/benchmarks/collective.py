from __future__ import annotations

import socket
from typing import Any

from teamredbench.benchmarks.base import Benchmark, BenchmarkRecord
from teamredbench.device import rocm_device_available
from teamredbench.registry import register_benchmark
from teamredbench.utils import make_tensor, timed_iterations


BUS_FACTORS = {
    "all_reduce": lambda world_size: 2 * (world_size - 1) / world_size,
    "all_gather": lambda world_size: (world_size - 1) / world_size,
    "broadcast": lambda world_size: 1.0,
}


@register_benchmark("collective")
class CollectiveBandwidthBenchmark(Benchmark):
    default_metrics = ["payload_bandwidth_gbps", "bus_bandwidth_gbps", "link_efficiency_pct", "latency_us"]

    @staticmethod
    def _normalize_backend(backend: str) -> str:
        normalized = backend.strip().lower()
        if normalized == "rccl":
            return "nccl"
        return normalized

    def _ensure_process_group(self, backend: str):
        dist = self.torch.distributed
        if not dist.is_available():
            raise RuntimeError("torch.distributed is not available in this build.")
        if dist.is_initialized():
            return dist
        dist.init_process_group(backend=backend, init_method="env://")
        return dist

    def _scope(self, dist: Any) -> str:
        hosts = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(hosts, socket.gethostname())
        return "inter-node" if len(set(hosts)) > 1 else "intra-node"

    def _run_collective(self, op_name: str, dist: Any, tensor: Any) -> None:
        if op_name == "all_reduce":
            dist.all_reduce(tensor)
            return
        if op_name == "all_gather":
            gather_list = [self.torch.empty_like(tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(gather_list, tensor)
            return
        if op_name == "broadcast":
            dist.broadcast(tensor, src=0)
            return
        raise KeyError(f"Unknown collective operation: {op_name}")

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

        backend = self._normalize_backend(str(self.params.get("backend", "rccl")))
        try:
            dist = self._ensure_process_group(backend=backend)
        except Exception as exc:
            return [self.emit_record(self.skipped_record(case="global", dtype="n/a", error=str(exc)))]

        world_size = dist.get_world_size()
        if world_size < 2:
            return [
                self.emit_record(
                    self.skipped_record(case="global", dtype="n/a", error="Collective benchmark requires WORLD_SIZE >= 2.")
                )
            ]

        operations = list(self.params.get("operations", ["all_reduce"]))
        sizes_mib = list(self.params.get("sizes_mib", [8, 64, 256]))
        warmup = int(self.params.get("warmup", 20))
        iterations = int(self.params.get("iterations", 100))
        scope = self._scope(dist)

        records: list[BenchmarkRecord] = []
        for spec in self.selected_dtype_specs():
            for size_mib in sizes_mib:
                size_bytes = int(size_mib) * 1024 * 1024
                numel = max(size_bytes // spec.bytes_per_element, 1)
                shape = (numel,)
                for op_name in operations:
                    case = f"{op_name}_{size_mib}MiB"
                    try:
                        tensor = make_tensor(torch, spec, shape, self.device)
                        dist.barrier()
                        elapsed_s = timed_iterations(
                            lambda: self._run_collective(op_name, dist, tensor),
                            torch_module=torch,
                            warmup=warmup,
                            iterations=iterations,
                        )
                        dist.barrier()
                    except Exception as exc:
                        records.append(
                            self.emit_record(
                                self.skipped_record(
                                    case=case,
                                    dtype=spec.name,
                                    error=str(exc),
                                    metadata={"operation": op_name, "scope": scope, "size_mib": size_mib},
                                )
                            )
                        )
                        continue

                    bus_factor = BUS_FACTORS[op_name](world_size)
                    records.append(
                        self.emit_record(
                            BenchmarkRecord(
                                benchmark=self.name,
                                case=case,
                                dtype=spec.name,
                                raw_metrics={
                                    "elapsed_s": elapsed_s,
                                    "payload_bytes": numel * spec.bytes_per_element,
                                    "bus_factor": bus_factor,
                                    "peak_link_bandwidth_gbps": self.context.hardware_profile.peak_link_bandwidth_gbps.get(
                                        "network" if scope == "inter-node" else "xgmi"
                                    ),
                                },
                                metadata={
                                    "operation": op_name,
                                    "scope": scope,
                                    "size_mib": size_mib,
                                    "world_size": world_size,
                                },
                            )
                        )
                    )
        return records
