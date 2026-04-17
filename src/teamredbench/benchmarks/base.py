from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from teamredbench.config import HardwareProfile, RuntimeProfile
from teamredbench.device import DeviceInfo, rocm_device, rocm_device_available, set_rocm_device
from teamredbench.dtypes import DTypeSpec, resolve_dtype_names


@dataclass
class BenchmarkContext:
    hardware_profile: HardwareProfile
    runtime_profile: RuntimeProfile
    device_info: DeviceInfo | None
    torch_module: Any
    dtype_specs: dict[str, DTypeSpec]
    device_id: int = 0
    repo_root: Path | None = None
    suite_path: Path | None = None


@dataclass
class BenchmarkRecord:
    benchmark: str
    case: str
    dtype: str
    raw_metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float | None] = field(default_factory=dict)
    status: str = "ok"
    error: str | None = None


class Benchmark(ABC):
    name = "benchmark"
    default_metrics: list[str] = []

    def __init__(
        self,
        context: BenchmarkContext,
        params: dict[str, Any],
        record_sink: Callable[[BenchmarkRecord], None] | None = None,
    ):
        self.context = context
        self.params = params
        self.record_sink = record_sink

    @property
    def torch(self) -> Any:
        if self.context.torch_module is None:
            raise RuntimeError("PyTorch is required for GPU benchmarks.")
        return self.context.torch_module

    @property
    def device(self) -> Any:
        return rocm_device(self.torch, self.context.device_id)

    def set_device(self) -> None:
        if rocm_device_available(self.context.torch_module):
            set_rocm_device(self.context.torch_module, self.context.device_id)

    def selected_dtype_specs(self) -> list[DTypeSpec]:
        names = resolve_dtype_names(self.params.get("dtypes"), self.context.dtype_specs)
        return [self.context.dtype_specs[name] for name in names]

    def emit_record(self, record: BenchmarkRecord) -> BenchmarkRecord:
        if self.record_sink is not None:
            self.record_sink(record)
        return record

    def skipped_record(self, *, case: str, dtype: str, error: str, metadata: dict[str, Any] | None = None) -> BenchmarkRecord:
        return BenchmarkRecord(
            benchmark=self.name,
            case=case,
            dtype=dtype,
            metadata=metadata or {},
            status="skipped",
            error=error,
        )

    @abstractmethod
    def run(self) -> list[BenchmarkRecord]:
        raise NotImplementedError
