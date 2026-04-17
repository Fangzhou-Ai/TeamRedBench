from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import re
from typing import Callable

from teamredbench import builtin as _builtin  # noqa: F401
from teamredbench.benchmarks.base import BenchmarkContext, BenchmarkRecord
from teamredbench.config import HardwareProfile, SuiteConfig, load_hardware_profile, load_suite
from teamredbench.device import detect_device, load_torch
from teamredbench.dtypes import discover_dtypes
from teamredbench.io import write_results
from teamredbench.provenance import build_run_metadata
from teamredbench.registry import get_benchmark, get_metric, load_plugins


@dataclass
class RunSummary:
    suite: SuiteConfig
    context: BenchmarkContext
    records: list[BenchmarkRecord]
    outputs: dict[str, Path]
    run_metadata: dict[str, object]
    profile_note: str | None = None


def _apply_metrics(record: BenchmarkRecord, metric_names: list[str]) -> None:
    if record.status != "ok":
        return
    for metric_name in metric_names:
        metric = get_metric(metric_name)
        record.metrics[metric_name] = metric.compute(record.raw_metrics)


def _extract_sku_tokens(text: str | None) -> set[str]:
    if not text:
        return set()
    return {token.lower() for token in re.findall(r"mi\d{3,4}x", text, flags=re.IGNORECASE)}


def _has_positive_peak(values: dict[str, float | None]) -> bool:
    return any(value is not None and float(value) > 0 for value in values.values())


def _profile_has_theoretical_limits(profile: HardwareProfile) -> bool:
    return (
        profile.peak_hbm_bandwidth_gbps is not None
        and float(profile.peak_hbm_bandwidth_gbps) > 0
        and _has_positive_peak(profile.peak_compute_tops)
    )


def _resolve_hardware_profile(
    profile: HardwareProfile,
    *,
    device_name: str | None,
    repo_root: Path | None,
) -> tuple[HardwareProfile, str | None]:
    if _profile_has_theoretical_limits(profile):
        return profile, None
    if repo_root is None:
        return profile, None

    device_tokens = _extract_sku_tokens(device_name)
    if not device_tokens:
        return profile, None

    hardware_dir = repo_root / "configs" / "profiles" / "hardware"
    if not hardware_dir.exists():
        return profile, None

    for candidate_path in sorted(hardware_dir.glob("*.yaml")):
        candidate = load_hardware_profile(candidate_path)
        candidate_tokens = _extract_sku_tokens(f"{candidate.name} {candidate_path.stem}")
        if not candidate_tokens.intersection(device_tokens):
            continue
        if not _profile_has_theoretical_limits(candidate):
            continue
        note = (
            f"Auto-selected hardware profile {candidate.name} from detected device "
            f"{device_name!r} because {profile.name} has no theoretical peak data."
        )
        return candidate, note

    return profile, None


def build_context(suite: SuiteConfig, device_id: int = 0) -> tuple[BenchmarkContext, str | None]:
    resolved_device_id = int(os.environ.get("LOCAL_RANK", device_id))
    torch_module = load_torch()
    device_info = detect_device(torch_module, device_id=resolved_device_id)
    repo_root = suite.path.parent.parent.parent if suite.path else None
    hardware_profile, profile_note = _resolve_hardware_profile(
        suite.hardware_profile,
        device_name=device_info.name if device_info else None,
        repo_root=repo_root,
    )
    return BenchmarkContext(
        hardware_profile=hardware_profile,
        runtime_profile=suite.runtime_profile,
        device_info=device_info,
        torch_module=torch_module,
        dtype_specs=discover_dtypes(torch_module),
        device_id=resolved_device_id,
        repo_root=repo_root,
    ), profile_note


def run_suite(
    suite_path: str | Path,
    output_dir: str | Path | None = None,
    device_id: int = 0,
    record_callback: Callable[[BenchmarkRecord], None] | None = None,
) -> RunSummary:
    started_at = datetime.now(timezone.utc)
    suite = load_suite(suite_path)
    load_plugins(suite.plugins)
    context, profile_note = build_context(suite, device_id=device_id)
    suite.hardware_profile = context.hardware_profile
    target_dir = Path(output_dir).resolve() if output_dir else suite.outputs.directory

    records: list[BenchmarkRecord] = []
    for invocation in suite.benchmarks:
        benchmark_cls = get_benchmark(invocation.benchmark)
        metric_names = invocation.metrics or list(benchmark_cls.default_metrics)

        def record_sink(record: BenchmarkRecord) -> None:
            _apply_metrics(record, metric_names)
            if record_callback is not None:
                record_callback(record)

        benchmark = benchmark_cls(
            context=context,
            params=invocation.params,
            record_sink=record_sink,
        )
        benchmark_records = benchmark.run()
        records.extend(benchmark_records)

    finished_at = datetime.now(timezone.utc)
    run_metadata = build_run_metadata(
        suite=suite,
        context=context,
        requested_device_id=device_id,
        output_dir=target_dir.resolve(),
        profile_note=profile_note,
        started_at=started_at,
        finished_at=finished_at,
    )
    outputs = write_results(
        records=records,
        suite_name=suite.name,
        output_dir=target_dir.resolve(),
        formats=suite.outputs.formats,
        run_metadata=run_metadata,
    )
    return RunSummary(
        suite=suite,
        context=context,
        records=records,
        outputs=outputs,
        run_metadata=run_metadata,
        profile_note=profile_note,
    )
