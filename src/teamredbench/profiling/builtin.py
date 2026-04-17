from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from teamredbench.profiling.registry import ProfilingLaunch, register_profile_engine


def _resolve_path(base_dir: Path | None, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if base_dir is None:
        return path.resolve()
    return (base_dir / path).resolve()


def _toggle(value: bool) -> str:
    return "on" if value else "off"


def _merge_env_path(extra_dirs: list[Path]) -> dict[str, str]:
    current_path = os.environ.get("PATH", "")
    ordered_dirs: list[str] = []
    for directory in extra_dirs:
        if directory.exists():
            value = str(directory)
            if value not in ordered_dirs:
                ordered_dirs.append(value)
    if current_path:
        ordered_dirs.append(current_path)
    return {"PATH": os.pathsep.join(ordered_dirs)}


def _build_rocprof_command(
    params: dict[str, Any],
    target_command: list[str],
    artifact_dir: Path,
    base_dir: Path | None,
) -> ProfilingLaunch:
    binary_value = str(params.get("binary", "rocprof"))
    if Path(binary_value).is_absolute():
        profiler_binary = str(Path(binary_value))
    else:
        profiler_binary = shutil.which(binary_value) or binary_value

    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_path = artifact_dir / "rocprof.csv"
    data_dir = artifact_dir / "data"
    command: list[str] = [profiler_binary]

    if params.get("tool_version") is not None:
        command.extend(["--tool-version", str(params["tool_version"])])

    if params.get("input"):
        input_path = _resolve_path(base_dir, str(params["input"]))
        command.extend(["-i", str(input_path)])
    if params.get("metric_file"):
        metric_path = _resolve_path(base_dir, str(params["metric_file"]))
        command.extend(["-m", str(metric_path)])

    if params.get("output"):
        output_path = _resolve_path(base_dir, str(params["output"]))
    command.extend(["-o", str(output_path)])

    if params.get("data_dir"):
        data_dir = _resolve_path(base_dir, str(params["data_dir"]))
    command.extend(["-d", str(data_dir)])

    if params.get("temporary_dir"):
        temp_dir = _resolve_path(base_dir, str(params["temporary_dir"]))
        command.extend(["-t", str(temp_dir)])

    on_off_flags = {
        "cmd_qts": "--cmd-qts",
        "basenames": "--basenames",
        "timestamp": "--timestamp",
        "ctx_wait": "--ctx-wait",
        "obj_tracking": "--obj-tracking",
        "trace_start": "--trace-start",
    }
    for key, flag in on_off_flags.items():
        if key in params:
            command.extend([flag, _toggle(bool(params[key]))])

    value_flags = {
        "ctx_limit": "--ctx-limit",
        "heartbeat": "--heartbeat",
        "trace_period": "--trace-period",
        "flush_rate": "--flush-rate",
    }
    for key, flag in value_flags.items():
        if params.get(key) is not None:
            command.extend([flag, str(params[key])])

    bare_flags = {
        "stats": "--stats",
        "roctx_trace": "--roctx-trace",
        "hip_trace": "--hip-trace",
        "hsa_trace": "--hsa-trace",
        "sys_trace": "--sys-trace",
        "roctx_rename": "--roctx-rename",
        "parallel_kernels": "--parallel-kernels",
    }
    for key, flag in bare_flags.items():
        if params.get(key):
            command.append(flag)

    extra_args = params.get("extra_args", [])
    if extra_args:
        if not isinstance(extra_args, list):
            raise ValueError("profiling.params.extra_args must be a list of strings.")
        command.extend(str(item) for item in extra_args)

    resolved_binary = Path(profiler_binary)
    resolved_real_binary = Path(profiler_binary).resolve()
    rocm_path = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
    env = _merge_env_path(
        [
            resolved_binary.parent,
            resolved_real_binary.parent,
            rocm_path / "bin",
        ]
    )
    if params.get("env"):
        if not isinstance(params["env"], dict):
            raise ValueError("profiling.params.env must be a mapping.")
        for key, value in params["env"].items():
            env[str(key)] = str(value)

    command.extend(target_command)
    return ProfilingLaunch(
        command=tuple(command),
        artifact_dir=artifact_dir,
        env=env,
        metadata={
            "binary": profiler_binary,
            "output": str(output_path),
            "data_dir": str(data_dir),
        },
    )


register_profile_engine(
    name="rocprof",
    description="Profile a TeamRedBench run with ROCm rocprof.",
    build_command=_build_rocprof_command,
)
