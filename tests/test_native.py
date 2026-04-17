import json
from pathlib import Path

import pytest
from typer.testing import CliRunner
import yaml

import teamredbench.builtin  # noqa: F401
from teamredbench.cli import app
from teamredbench.native.registry import list_native_kernels
from teamredbench.runner import run_suite


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")
    path.chmod(0o755)


def test_builtin_native_kernels_are_registered():
    names = {definition.name for definition in list_native_kernels()}
    assert {"hbm_hip", "mfu_hipblas"} <= names


def test_clean_native_cache_command_removes_entries(tmp_path: Path):
    cache_dir = tmp_path / "native-cache"
    cache_dir.mkdir()
    (cache_dir / "kernel-a").write_text("a", encoding="utf-8")
    nested = cache_dir / "kernel-b"
    nested.mkdir()
    (nested / "payload").write_text("b", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(app, ["clean-native-cache", "--cache-dir", str(cache_dir)])

    assert result.exit_code == 0
    assert f"Native cache: {cache_dir}" in result.stdout
    assert "Removed entries: 2" in result.stdout
    assert list(cache_dir.iterdir()) == []


def test_run_suite_hbm_native_backend_uses_relative_binary(tmp_path: Path):
    hardware = tmp_path / "hardware.yaml"
    runtime = tmp_path / "runtime.yaml"
    suite = tmp_path / "suite.yaml"
    outputs = tmp_path / "results"
    executable = tmp_path / "fake_hbm.py"
    seen_args = tmp_path / "hbm_args.json"

    _write_executable(
        executable,
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json",
                "import sys",
                f"open({str(seen_args)!r}, 'w', encoding='utf-8').write(json.dumps(sys.argv[1:]))",
                "print(json.dumps({'status': 'ok', 'raw_metrics': {'elapsed_s': 0.002}, 'metadata': {'executor': 'fake'}}))",
            ]
        ),
    )

    hardware.write_text(
        yaml.safe_dump(
            {
                "name": "test-hw",
                "family": "cdna",
                "peak_hbm_bandwidth_gbps": 4000,
                "peak_link_bandwidth_gbps": {"xgmi": 1, "network": 1},
                "peak_compute_tops": {"float32": 100},
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
                "name": "native-hbm-suite",
                "device": {"profile": str(hardware)},
                "runtime": {"profile": str(runtime)},
                "outputs": {"directory": str(outputs), "formats": ["json"]},
                "benchmarks": [
                    {
                        "benchmark": "hbm",
                        "params": {
                            "backend": "native",
                            "dtypes": ["float32"],
                            "modes": ["copy"],
                            "size_mib": 1024,
                            "warmup": 2,
                            "iterations": 5,
                            "native": {"binary": "./fake_hbm.py"},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = run_suite(suite)

    record = summary.records[0]
    assert record.status == "ok"
    assert record.metadata["backend"] == "native"
    assert record.metadata["executor"] == "fake"
    assert record.metrics["hbm_bandwidth_gbps"] == pytest.approx(1073.741824)
    assert record.metrics["hbm_efficiency_pct"] == pytest.approx(26.8435456)

    argv = json.loads(seen_args.read_text(encoding="utf-8"))
    assert "--dtype" in argv
    assert "float32" in argv
    assert "--mode" in argv
    assert "copy" in argv
    assert "--size-mib" in argv
    assert "1024" in argv


def test_run_suite_mfu_native_backend_uses_relative_binary(tmp_path: Path):
    hardware = tmp_path / "hardware.yaml"
    runtime = tmp_path / "runtime.yaml"
    suite = tmp_path / "suite.yaml"
    outputs = tmp_path / "results"
    executable = tmp_path / "fake_mfu.py"
    seen_args = tmp_path / "mfu_args.json"

    _write_executable(
        executable,
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json",
                "import sys",
                f"open({str(seen_args)!r}, 'w', encoding='utf-8').write(json.dumps(sys.argv[1:]))",
                "print(json.dumps({'status': 'ok', 'raw_metrics': {'elapsed_s': 0.001}, 'metadata': {'executor': 'fake'}}))",
            ]
        ),
    )

    hardware.write_text(
        yaml.safe_dump(
            {
                "name": "test-hw",
                "family": "cdna",
                "peak_hbm_bandwidth_gbps": 4000,
                "peak_link_bandwidth_gbps": {"xgmi": 1, "network": 1},
                "peak_compute_tops": {"float32": 20},
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
                "name": "native-mfu-suite",
                "device": {"profile": str(hardware)},
                "runtime": {"profile": str(runtime)},
                "outputs": {"directory": str(outputs), "formats": ["json"]},
                "benchmarks": [
                    {
                        "benchmark": "mfu",
                        "params": {
                            "backend": "native",
                            "dtypes": ["float32"],
                            "shapes": [[1024, 1024, 1024]],
                            "warmup": 2,
                            "iterations": 5,
                            "native": {"binary": "./fake_mfu.py"},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = run_suite(suite)

    record = summary.records[0]
    assert record.status == "ok"
    assert record.metadata["backend"] == "native"
    assert record.metadata["executor"] == "fake"
    assert record.metrics["achieved_tops"] == pytest.approx(2.147483648)
    assert record.metrics["mfu_pct"] == pytest.approx(10.73741824)

    argv = json.loads(seen_args.read_text(encoding="utf-8"))
    assert "--dtype" in argv
    assert "float32" in argv
    assert "--m" in argv and "1024" in argv
    assert "--n" in argv and "1024" in argv
    assert "--k" in argv and "1024" in argv
