from types import SimpleNamespace

from teamredbench.benchmarks.base import BenchmarkContext
from teamredbench.benchmarks.collective import CollectiveBandwidthBenchmark
from teamredbench.config import HardwareProfile, RuntimeProfile


class _FakeCuda:
    def is_available(self) -> bool:
        return True

    def set_device(self, device_id: int) -> None:
        self.device_id = device_id


class _FakeTorch:
    def __init__(self):
        self.cuda = _FakeCuda()

    def device(self, spec: str) -> str:
        return spec


class _FakeDist:
    def get_world_size(self) -> int:
        return 2


def test_collective_accepts_rccl_backend_alias(monkeypatch):
    context = BenchmarkContext(
        hardware_profile=HardwareProfile(
            name="test-hw",
            family="cdna",
            peak_link_bandwidth_gbps={"xgmi": 1.0, "network": 1.0},
        ),
        runtime_profile=RuntimeProfile(name="test-runtime"),
        device_info=None,
        torch_module=_FakeTorch(),
        dtype_specs={},
    )
    benchmark = CollectiveBandwidthBenchmark(context=context, params={}, record_sink=None)

    seen: list[str] = []
    monkeypatch.setattr(benchmark, "_ensure_process_group", lambda backend: seen.append(backend) or _FakeDist())
    monkeypatch.setattr(benchmark, "_scope", lambda dist: "intra-node")
    monkeypatch.setattr(benchmark, "selected_dtype_specs", lambda: [])

    records = benchmark.run()

    assert records == []
    assert seen == ["nccl"]
