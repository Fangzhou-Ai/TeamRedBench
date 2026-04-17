from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from teamredbench.benchmarks.base import BenchmarkContext, BenchmarkRecord
from teamredbench.config import ProfilingConfig, SuiteConfig
from teamredbench.profiling.registry import get_profile_engine


INTERNAL_RUN_ENV = "TEAMREDBENCH_INTERNAL_RUN"
FORCE_JSON_OUTPUT_ENV = "TEAMREDBENCH_FORCE_JSON_OUTPUT"


class ProfilingError(RuntimeError):
    pass


def profiling_active() -> bool:
    return os.environ.get(INTERNAL_RUN_ENV) == "1"


def profiling_enabled(config: ProfilingConfig) -> bool:
    return bool(config.enabled and config.engine and not profiling_active())


def force_json_output_enabled() -> bool:
    return os.environ.get(FORCE_JSON_OUTPUT_ENV) == "1"


def effective_output_formats(formats: list[str]) -> list[str]:
    resolved = list(formats)
    if force_json_output_enabled() and "json" not in resolved:
        resolved.append("json")
    return resolved


def _artifact_dir(output_dir: Path, suite_name: str, engine_name: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return output_dir / "profiling" / f"{suite_name}-{engine_name}-{timestamp}"


def _metadata_paths(output_dir: Path, suite_name: str) -> set[Path]:
    if not output_dir.exists():
        return set()
    return set(output_dir.glob(f"{suite_name}-*.metadata.json"))


def _deserialize_record(payload: dict[str, Any]) -> BenchmarkRecord:
    return BenchmarkRecord(
        benchmark=str(payload["benchmark"]),
        case=str(payload["case"]),
        dtype=str(payload["dtype"]),
        status=str(payload.get("status", "ok")),
        error=str(payload["error"]) if payload.get("error") is not None else None,
        metrics=dict(payload.get("metrics", {})),
        raw_metrics=dict(payload.get("raw_metrics", {})),
        metadata=dict(payload.get("metadata", {})),
    )


def _load_summary_from_metadata(
    metadata_path: Path,
    *,
    suite: SuiteConfig,
    context: BenchmarkContext,
):
    from teamredbench.runner import RunSummary

    metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    outputs = {name: Path(path) for name, path in metadata_payload.get("outputs", {}).items()}
    json_path = outputs.get("json")
    if json_path is None or not json_path.exists():
        raise ProfilingError("Profiled run did not produce a JSON results file.")

    records_payload = json.loads(json_path.read_text(encoding="utf-8"))
    records = [_deserialize_record(item) for item in records_payload]
    return RunSummary(
        suite=suite,
        context=context,
        records=records,
        outputs=outputs,
        run_metadata=dict(metadata_payload.get("run", {})),
        profile_note=metadata_payload.get("run", {}).get("configs", {}).get("profile_note"),
    ), metadata_payload


def run_profiled_suite(
    *,
    suite: SuiteConfig,
    context: BenchmarkContext,
    output_dir: Path,
    device_id: int,
    record_callback,
):
    from teamredbench.runner import RunSummary

    if suite.path is None:
        raise ProfilingError("Suite path is required for profiled runs.")
    if not suite.profiling.engine:
        raise ProfilingError("Profiling is enabled but no engine is configured.")

    engine = get_profile_engine(suite.profiling.engine)
    artifact_dir = _artifact_dir(output_dir, suite.name, engine.name)
    command = [
        sys.executable,
        "-m",
        "teamredbench",
        "run",
        str(suite.path),
        "--device-id",
        str(device_id),
        "--output-dir",
        str(output_dir),
        "--quiet-records",
    ]
    launch = engine.build_command(
        dict(suite.profiling.params),
        command,
        artifact_dir,
        suite.path.parent,
    )

    env = os.environ.copy()
    env[INTERNAL_RUN_ENV] = "1"
    env[FORCE_JSON_OUTPUT_ENV] = "1"
    env.update(launch.env)

    previous_metadata = _metadata_paths(output_dir, suite.name)
    completed = subprocess.run(
        list(launch.command),
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if completed.returncode != 0:
        error = completed.stderr.strip() or completed.stdout.strip() or "unknown profiling failure"
        raise ProfilingError(f"Profile engine {engine.name} failed: {error}")

    current_metadata = _metadata_paths(output_dir, suite.name)
    new_metadata = sorted(current_metadata - previous_metadata)
    if not new_metadata:
        raise ProfilingError("Profiled run completed but no metadata file was produced.")
    metadata_path = new_metadata[-1]

    summary, metadata_payload = _load_summary_from_metadata(
        metadata_path,
        suite=suite,
        context=context,
    )

    profiling_metadata = {
        "enabled": True,
        "engine": engine.name,
        "artifact_dir": str(launch.artifact_dir),
        "command": list(launch.command),
        "engine_metadata": dict(launch.metadata),
    }
    metadata_payload.setdefault("outputs", {})["profiling"] = str(launch.artifact_dir)
    metadata_payload.setdefault("run", {})["profiling"] = profiling_metadata
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")

    summary.outputs["profiling"] = launch.artifact_dir
    summary.run_metadata = dict(metadata_payload.get("run", {}))
    summary.profile_note = summary.run_metadata.get("configs", {}).get("profile_note")

    if record_callback is not None:
        for record in summary.records:
            record_callback(record)

    return RunSummary(
        suite=summary.suite,
        context=summary.context,
        records=summary.records,
        outputs=summary.outputs,
        run_metadata=summary.run_metadata,
        profile_note=summary.profile_note,
    )
