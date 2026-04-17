from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import shlex
import shutil
import socket
import subprocess
import sys
from typing import Any

from teamredbench import __version__
from teamredbench.benchmarks.base import BenchmarkContext
from teamredbench.config import SuiteConfig
from teamredbench.device import detect_runtime


_ENV_NAMES = {
    "CONDA_DEFAULT_ENV",
    "CONDA_PREFIX",
    "HIP_VISIBLE_DEVICES",
    "LD_LIBRARY_PATH",
    "LOCAL_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
    "OMP_NUM_THREADS",
    "PATH",
    "PYTHONPATH",
    "RANK",
    "ROCM_PATH",
    "ROCR_VISIBLE_DEVICES",
    "VIRTUAL_ENV",
    "WORLD_SIZE",
}
_ENV_PREFIXES = (
    "FI_",
    "HIP_",
    "HSA_",
    "MPI_",
    "OMPI_",
    "PMI_",
    "PYTORCH_",
    "RCCL_",
    "ROCM_",
    "SLURM_",
    "TORCH_",
    "UCX_",
)


def _run_command(command: list[str], *, max_lines: int | None = None) -> str | None:
    executable = command[0]
    if not os.path.isabs(executable) and shutil.which(executable) is None:
        return None
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    output = completed.stdout.strip()
    if not output:
        return None
    if max_lines is not None:
        output = "\n".join(output.splitlines()[:max_lines])
    return output


def _run_first_command(commands: list[list[str]], *, max_lines: int | None = None) -> str | None:
    for command in commands:
        output = _run_command(command, max_lines=max_lines)
        if output:
            return output
    return None


def _read_text(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _parse_os_release() -> dict[str, str]:
    path = Path("/etc/os-release")
    text = _read_text(path)
    if text is None:
        return {}
    parsed: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        parsed[key] = value.strip().strip('"')
    return parsed


def _system_metadata() -> dict[str, Any]:
    uname = platform.uname()
    libc_name, libc_version = platform.libc_ver()
    return {
        "hostname": socket.gethostname(),
        "fqdn": socket.getfqdn(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "architecture": list(platform.architecture()),
        "libc": {
            "name": libc_name or None,
            "version": libc_version or None,
        },
        "os_release": _parse_os_release(),
        "uname": {
            "system": uname.system,
            "node": uname.node,
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine,
            "processor": uname.processor,
        },
    }


def _python_metadata() -> dict[str, Any]:
    return {
        "executable": sys.executable,
        "version": platform.python_version(),
        "version_full": sys.version,
        "implementation": platform.python_implementation(),
        "build": list(platform.python_build()),
    }


def _torch_metadata(torch_module: Any) -> dict[str, Any]:
    if torch_module is None:
        return {}

    config = getattr(torch_module, "__config__", None)
    config_output = None
    if config is not None and hasattr(config, "show"):
        try:
            config_output = config.show()
        except Exception:
            config_output = None

    distributed_available = None
    distributed = getattr(torch_module, "distributed", None)
    if distributed is not None and hasattr(distributed, "is_available"):
        try:
            distributed_available = bool(distributed.is_available())
        except Exception:
            distributed_available = None

    version = getattr(torch_module, "version", None)
    return {
        "version": getattr(torch_module, "__version__", None),
        "hip_version": getattr(version, "hip", None),
        "git_version": getattr(version, "git_version", None),
        "file": getattr(torch_module, "__file__", None),
        "distributed_available": distributed_available,
        "config": config_output,
    }


def _rocm_metadata(torch_module: Any, context: BenchmarkContext) -> dict[str, Any]:
    runtime = detect_runtime(torch_module)
    return {
        "profile_name": context.runtime_profile.name,
        "profile_version": context.runtime_profile.rocm_version,
        "profile_libraries": dict(context.runtime_profile.libraries),
        "hipcc_version": runtime.get("hipcc_version"),
        "rocminfo_head": runtime.get("rocminfo_head"),
        "rocm_smi_product": runtime.get("rocm_smi_product"),
        "driver_version": _run_first_command(
            [
                ["rocm-smi", "--showdriverversion"],
                ["amd-smi", "static"],
            ],
            max_lines=40,
        ),
        "amdgpu_module_version": _read_text(Path("/sys/module/amdgpu/version")),
    }


def _environment_metadata() -> dict[str, str]:
    selected: dict[str, str] = {}
    for key in sorted(os.environ):
        if key in _ENV_NAMES or key.startswith(_ENV_PREFIXES):
            selected[key] = os.environ[key]
    return selected


def _git_metadata(repo_root: Path | None) -> dict[str, Any]:
    if repo_root is None or not (repo_root / ".git").exists():
        return {}

    branch = _run_command(["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"])
    commit = _run_command(["git", "-C", str(repo_root), "rev-parse", "HEAD"])
    describe = _run_command(["git", "-C", str(repo_root), "describe", "--always", "--dirty", "--tags"])
    status_short = _run_command(["git", "-C", str(repo_root), "status", "--short"])
    return {
        "branch": branch,
        "commit": commit,
        "describe": describe,
        "dirty": bool(status_short),
        "status_short": status_short,
    }


def _config_metadata(path: Path | None, raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path) if path is not None else None,
        "raw": raw,
        "content": _read_text(path),
    }


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _build_reproduce_command(
    suite_path: Path | None,
    *,
    device_id: int,
    output_dir: Path,
) -> str:
    command = ["teamredbench", "run"]
    if suite_path is not None:
        command.append(str(suite_path))
    command.extend(["--device-id", str(device_id), "--output-dir", str(output_dir)])
    return shlex.join(command)


def build_run_metadata(
    *,
    suite: SuiteConfig,
    context: BenchmarkContext,
    requested_device_id: int,
    output_dir: Path,
    profile_note: str | None,
    started_at: datetime,
    finished_at: datetime,
) -> dict[str, Any]:
    return {
        "tool": {
            "name": "teamredbench",
            "version": __version__,
        },
        "timestamps": {
            "started_at_utc": _format_timestamp(started_at),
            "finished_at_utc": _format_timestamp(finished_at),
            "duration_s": max((finished_at - started_at).total_seconds(), 0.0),
        },
        "invocation": {
            "argv": list(sys.argv),
            "shell_command": shlex.join(sys.argv),
            "working_directory": os.getcwd(),
            "requested_device_id": requested_device_id,
            "resolved_device_id": context.device_id,
            "output_dir": str(output_dir),
        },
        "reproduce": {
            "discover_command": shlex.join(
                ["teamredbench", "discover", "--json", "--device-id", str(context.device_id)]
            ),
            "run_command": _build_reproduce_command(
                suite.path,
                device_id=requested_device_id,
                output_dir=output_dir,
            ),
        },
        "system": _system_metadata(),
        "python": _python_metadata(),
        "torch": _torch_metadata(context.torch_module),
        "rocm": _rocm_metadata(context.torch_module, context),
        "device": asdict(context.device_info) if context.device_info is not None else None,
        "environment": {
            "variables": _environment_metadata(),
        },
        "git": _git_metadata(context.repo_root),
        "configs": {
            "suite": _config_metadata(suite.path, suite.raw),
            "hardware_profile": _config_metadata(context.hardware_profile.path, context.hardware_profile.raw),
            "runtime_profile": _config_metadata(context.runtime_profile.path, context.runtime_profile.raw),
            "profile_note": profile_note,
            "plugins": list(suite.plugins),
            "output_formats": list(suite.outputs.formats),
        },
    }
