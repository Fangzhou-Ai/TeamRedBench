import json
import os
from pathlib import Path

import yaml

from teamredbench.runner import run_suite


def test_run_suite_with_profile_engine_plugin(tmp_path: Path, monkeypatch):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    plugin_module = plugin_dir / "fake_profile_plugin.py"
    plugin_module.write_text(
        "\n".join(
            [
                "import sys",
                "from teamredbench.profiling.registry import ProfilingLaunch, register_profile_engine",
                "",
                "def build_command(params, target_command, artifact_dir, base_dir):",
                "    code = (",
                "        \"from pathlib import Path; import subprocess, sys; \"",
                "        \"artifact = Path(sys.argv[1]); artifact.mkdir(parents=True, exist_ok=True); \"",
                "        \"(artifact / 'marker.txt').write_text('profile', encoding='utf-8'); \"",
                "        \"raise SystemExit(subprocess.run(sys.argv[2:]).returncode)\"",
                "    )",
                "    return ProfilingLaunch(",
                "        command=(sys.executable, '-c', code, str(artifact_dir), *target_command),",
                "        artifact_dir=artifact_dir,",
                "        metadata={'wrapper': 'fake'},",
                "    )",
                "",
                "register_profile_engine(",
                "    name='fake-profiler',",
                "    description='Fake profiler for tests.',",
                "    build_command=build_command,",
                ")",
            ]
        ),
        encoding="utf-8",
    )

    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    monkeypatch.syspath_prepend(str(plugin_dir))
    monkeypatch.setenv(
        "PYTHONPATH",
        str(plugin_dir) if not existing_pythonpath else f"{plugin_dir}{os.pathsep}{existing_pythonpath}",
    )

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
                "name": "profile-suite",
                "device": {"profile": str(hardware)},
                "runtime": {"profile": str(runtime)},
                "plugins": ["fake_profile_plugin"],
                "profiling": {
                    "enabled": True,
                    "engine": "fake-profiler",
                },
                "outputs": {"directory": str(outputs), "formats": ["json", "csv"]},
                "benchmarks": [
                    {
                        "benchmark": "hbm",
                        "params": {
                            "dtypes": ["float32"],
                            "modes": ["copy"],
                            "size_mib": 1,
                            "warmup": 0,
                            "iterations": 1,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = run_suite(suite)

    assert "profiling" in summary.outputs
    assert summary.outputs["profiling"].exists()
    assert (summary.outputs["profiling"] / "marker.txt").exists()
    assert summary.outputs["metadata"].exists()

    metadata_payload = json.loads(summary.outputs["metadata"].read_text(encoding="utf-8"))
    assert metadata_payload["outputs"]["profiling"] == str(summary.outputs["profiling"])
    assert metadata_payload["run"]["profiling"]["engine"] == "fake-profiler"
    assert metadata_payload["run"]["profiling"]["engine_metadata"]["wrapper"] == "fake"
