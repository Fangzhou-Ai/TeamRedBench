from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
import yaml

from teamredbench import builtin as _builtin  # noqa: F401
from teamredbench.benchmarks.base import BenchmarkRecord
from teamredbench.device import discover_environment, load_torch
from teamredbench.dtypes import discover_dtypes
from teamredbench.registry import list_benchmarks, list_metrics
from teamredbench.runner import run_suite

app = typer.Typer(no_args_is_help=True)


def _format_metric_value(metric_name: str, value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        formatted = f"{value:.6g}"
        if metric_name.endswith("_pct"):
            return f"{formatted}%"
        return formatted
    return str(value)


def _format_record(record: BenchmarkRecord) -> str:
    prefix = f"[{record.status.upper()}] {record.benchmark} case={record.case} dtype={record.dtype}"
    if record.status != "ok":
        return f"{prefix} error={record.error}"

    if record.metrics:
        metrics = ", ".join(
            f"{name}={_format_metric_value(name, value)}"
            for name, value in sorted(record.metrics.items())
        )
        return f"{prefix} {metrics}"

    raw_metrics = ", ".join(
        f"{name}={_format_metric_value(name, value)}"
        for name, value in sorted(record.raw_metrics.items())
    )
    if raw_metrics:
        return f"{prefix} {raw_metrics}"
    return prefix


def _print_run_metadata_summary(run_metadata: dict[str, object]) -> None:
    system = run_metadata.get("system", {})
    python = run_metadata.get("python", {})
    torch_info = run_metadata.get("torch", {})
    rocm = run_metadata.get("rocm", {})

    if isinstance(system, dict):
        hostname = system.get("hostname")
        platform_name = system.get("platform")
        if hostname or platform_name:
            typer.echo(f"System: host={hostname or 'n/a'} platform={platform_name or 'n/a'}")

    summary_parts: list[str] = []
    if isinstance(python, dict) and python.get("version"):
        summary_parts.append(f"python={python['version']}")
    if isinstance(torch_info, dict) and torch_info.get("version"):
        summary_parts.append(f"torch={torch_info['version']}")
    if isinstance(torch_info, dict) and torch_info.get("hip_version"):
        summary_parts.append(f"hip={torch_info['hip_version']}")
    if isinstance(rocm, dict) and rocm.get("profile_version"):
        summary_parts.append(f"rocm_profile={rocm['profile_version']}")
    if isinstance(rocm, dict) and rocm.get("driver_version"):
        first_line = str(rocm["driver_version"]).splitlines()[0]
        summary_parts.append(f"driver={first_line}")
    if summary_parts:
        typer.echo(f"Runtime: {' '.join(summary_parts)}")


@app.command()
def run(
    suite: Path = typer.Argument(..., help="Path to suite YAML."),
    output_dir: Path | None = typer.Option(None, help="Override result directory."),
    device_id: int = typer.Option(0, help="GPU index."),
    print_records: bool = typer.Option(True, "--print-records/--quiet-records", help="Print each result record."),
) -> None:
    summary = run_suite(
        suite_path=suite,
        output_dir=output_dir,
        device_id=device_id,
        record_callback=(lambda record: typer.echo(_format_record(record))) if print_records else None,
    )
    typer.echo(f"Suite: {summary.suite.name}")
    typer.echo(f"Hardware Profile: {summary.context.hardware_profile.name}")
    if summary.profile_note:
        typer.echo(summary.profile_note)
    _print_run_metadata_summary(summary.run_metadata)
    typer.echo(f"Records: {len(summary.records)}")
    for fmt, path in summary.outputs.items():
        typer.echo(f"{fmt.upper()}: {path}")


@app.command("list-benchmarks")
def list_benchmarks_command() -> None:
    for name in list_benchmarks():
        typer.echo(name)


@app.command("list-metrics")
def list_metrics_command() -> None:
    for name in list_metrics():
        typer.echo(name)


@app.command("list-dtypes")
def list_dtypes_command() -> None:
    torch_module = load_torch()
    dtypes = discover_dtypes(torch_module)
    for name in sorted(dtypes):
        typer.echo(name)


@app.command()
def discover(
    device_id: int = typer.Option(0, help="GPU index."),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON instead of YAML."),
) -> None:
    payload = discover_environment(load_torch(), device_id=device_id)
    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return
    typer.echo(yaml.safe_dump(payload, sort_keys=False))
