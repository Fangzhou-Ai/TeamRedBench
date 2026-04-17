import json
from pathlib import Path

import yaml

from teamredbench.benchmarks.base import Benchmark, BenchmarkRecord
from teamredbench.registry import register_benchmark
from teamredbench.runner import run_suite


@register_benchmark("test_dummy_callback")
class DummyCallbackBenchmark(Benchmark):
    default_metrics = ["latency_us", "mfu_pct"]

    def run(self) -> list[BenchmarkRecord]:
        return [
            self.emit_record(
                BenchmarkRecord(
                    benchmark=self.name,
                    case="single",
                    dtype="float32",
                    raw_metrics={
                        "elapsed_s": 0.001,
                        "achieved_ops": 1_000_000_000.0,
                    },
                )
            )
        ]


def test_run_suite_invokes_record_callback_with_metrics(tmp_path: Path):
    hardware = tmp_path / "hardware.yaml"
    runtime = tmp_path / "runtime.yaml"
    suite = tmp_path / "suite.yaml"
    outputs = tmp_path / "results"

    hardware.write_text(
        yaml.safe_dump(
            {
                "name": "test-hw",
                "family": "cdna",
                "peak_hbm_bandwidth_gbps": 1,
                "peak_link_bandwidth_gbps": {"xgmi": 1, "network": 1},
                "peak_compute_tops": {"float32": 1},
            }
        ),
        encoding="utf-8",
    )
    runtime.write_text(
        yaml.safe_dump(
            {
                "name": "test-runtime",
                "rocm_version": "test",
                "libraries": {},
                "env": {},
            }
        ),
        encoding="utf-8",
    )
    suite.write_text(
        yaml.safe_dump(
            {
                "name": "callback-suite",
                "device": {"profile": str(hardware)},
                "runtime": {"profile": str(runtime)},
                "outputs": {"directory": str(outputs), "formats": ["json", "csv"]},
                "benchmarks": [{"benchmark": "test_dummy_callback"}],
            }
        ),
        encoding="utf-8",
    )

    seen: list[BenchmarkRecord] = []
    summary = run_suite(suite, record_callback=seen.append)

    assert len(summary.records) == 1
    assert len(seen) == 1
    assert seen[0].metrics["latency_us"] == 1000.0
    assert seen[0].metrics["mfu_pct"] is None
    assert summary.outputs["json"].exists()
    assert summary.outputs["csv"].exists()
    assert summary.outputs["metadata"].exists()

    records_payload = json.loads(summary.outputs["json"].read_text(encoding="utf-8"))
    metadata_payload = json.loads(summary.outputs["metadata"].read_text(encoding="utf-8"))

    assert records_payload[0]["benchmark"] == "test_dummy_callback"
    assert metadata_payload["record_count"] == 1
    assert metadata_payload["outputs"]["json"] == str(summary.outputs["json"])
    assert metadata_payload["outputs"]["metadata"] == str(summary.outputs["metadata"])
    assert metadata_payload["run"]["configs"]["suite"]["raw"]["name"] == "callback-suite"
    assert metadata_payload["run"]["configs"]["hardware_profile"]["raw"]["name"] == "test-hw"
    assert metadata_payload["run"]["configs"]["runtime_profile"]["raw"]["name"] == "test-runtime"
    assert metadata_payload["run"]["python"]["version"]
    assert metadata_payload["run"]["reproduce"]["run_command"]
