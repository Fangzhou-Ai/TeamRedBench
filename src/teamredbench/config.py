from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class HardwareProfile:
    name: str
    family: str
    peak_hbm_bandwidth_gbps: float | None = None
    peak_link_bandwidth_gbps: dict[str, float] = field(default_factory=dict)
    peak_compute_tops: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    path: Path | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeProfile:
    name: str
    rocm_version: str = "unknown"
    libraries: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    path: Path | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class OutputConfig:
    directory: Path
    formats: list[str] = field(default_factory=lambda: ["json", "csv"])


@dataclass
class ProfilingConfig:
    enabled: bool = False
    engine: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkInvocation:
    benchmark: str
    params: dict[str, Any] = field(default_factory=dict)
    metrics: list[str] = field(default_factory=list)


@dataclass
class SuiteConfig:
    name: str
    benchmarks: list[BenchmarkInvocation]
    hardware_profile: HardwareProfile
    runtime_profile: RuntimeProfile
    outputs: OutputConfig
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    plugins: list[str] = field(default_factory=list)
    path: Path | None = None
    raw: dict[str, Any] = field(default_factory=dict)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def _resolve(base_dir: Path, target: str) -> Path:
    path = Path(target)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_hardware_profile(path: Path) -> HardwareProfile:
    data = _load_yaml(path)
    return HardwareProfile(
        name=str(data["name"]),
        family=str(data["family"]),
        peak_hbm_bandwidth_gbps=data.get("peak_hbm_bandwidth_gbps"),
        peak_link_bandwidth_gbps=dict(data.get("peak_link_bandwidth_gbps", {})),
        peak_compute_tops={k: float(v) for k, v in data.get("peak_compute_tops", {}).items()},
        notes=list(data.get("notes", [])),
        path=path,
        raw=data,
    )


def load_runtime_profile(path: Path) -> RuntimeProfile:
    data = _load_yaml(path)
    return RuntimeProfile(
        name=str(data["name"]),
        rocm_version=str(data.get("rocm_version", "unknown")),
        libraries={k: str(v) for k, v in data.get("libraries", {}).items()},
        env={k: str(v) for k, v in data.get("env", {}).items()},
        notes=list(data.get("notes", [])),
        path=path,
        raw=data,
    )


def load_suite(path: str | Path) -> SuiteConfig:
    suite_path = Path(path).resolve()
    data = _load_yaml(suite_path)
    suite_dir = suite_path.parent

    hardware_path = _resolve(suite_dir, str(data["device"]["profile"]))
    runtime_path = _resolve(suite_dir, str(data["runtime"]["profile"]))
    outputs_data = data.get("outputs", {})
    profiling_data = data.get("profiling", {})
    if profiling_data is None:
        profiling_data = {}
    if not isinstance(profiling_data, dict):
        raise ValueError(f"Expected mapping for profiling config in {suite_path}")

    benchmarks = [
        BenchmarkInvocation(
            benchmark=str(item["benchmark"]),
            params=dict(item.get("params", {})),
            metrics=list(item.get("metrics", [])),
        )
        for item in data.get("benchmarks", [])
    ]

    return SuiteConfig(
        name=str(data.get("name", suite_path.stem)),
        benchmarks=benchmarks,
        hardware_profile=load_hardware_profile(hardware_path),
        runtime_profile=load_runtime_profile(runtime_path),
        outputs=OutputConfig(
            directory=_resolve(suite_dir, str(outputs_data.get("directory", "../../results"))),
            formats=list(outputs_data.get("formats", ["json", "csv"])),
        ),
        profiling=ProfilingConfig(
            enabled=bool(profiling_data.get("enabled", bool(profiling_data.get("engine")))),
            engine=str(profiling_data["engine"]) if profiling_data.get("engine") is not None else None,
            params=dict(profiling_data.get("params", {})),
            raw=dict(profiling_data),
        ),
        plugins=list(data.get("plugins", [])),
        path=suite_path,
        raw=data,
    )
