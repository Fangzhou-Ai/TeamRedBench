from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

from teamredbench.native.registry import get_native_kernel


class NativeKernelError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResolvedNativeKernel:
    benchmark: str
    name: str
    description: str
    source_path: Path | None = None
    binary_path: Path | None = None
    compiler: str | None = None
    compile_args: tuple[str, ...] = ()
    run_args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    cache_dir: Path | None = None


@dataclass(frozen=True)
class NativeKernelResult:
    status: str
    raw_metrics: dict[str, Any]
    metadata: dict[str, Any]
    error: str | None = None


def _string_list(value: Any, *, label: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list of strings.")
    return [str(item) for item in value]


def _string_mapping(value: Any, *, label: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping.")
    return {str(key): str(item) for key, item in value.items()}


def _resolve_user_path(base_dir: Path, target: str) -> Path:
    path = Path(target)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def resolve_native_kernel(
    *,
    benchmark: str,
    params: dict[str, Any],
    suite_path: Path | None,
    repo_root: Path | None,
    default_kernel: str | None = None,
) -> ResolvedNativeKernel:
    native_params = params.get("native", {})
    if native_params is None:
        native_params = {}
    if not isinstance(native_params, dict):
        raise ValueError("benchmark params.native must be a mapping.")

    base_dir = suite_path.parent if suite_path else (repo_root or Path.cwd())
    kernel_name = str(native_params.get("kernel", default_kernel or "")).strip() or None
    definition = get_native_kernel(kernel_name) if kernel_name else None
    if definition is not None and definition.benchmark != benchmark:
        raise ValueError(f"Native kernel {definition.name} is registered for {definition.benchmark}, not {benchmark}.")

    source_path = definition.source_path if definition is not None else None
    if native_params.get("source"):
        source_path = _resolve_user_path(base_dir, str(native_params["source"]))

    binary_path = None
    if native_params.get("binary"):
        binary_path = _resolve_user_path(base_dir, str(native_params["binary"]))

    compiler = str(native_params["compiler"]) if native_params.get("compiler") else None
    cache_dir = None
    if native_params.get("cache_dir"):
        cache_dir = _resolve_user_path(base_dir, str(native_params["cache_dir"]))

    compile_args: list[str] = list(definition.compile_args if definition is not None else ())
    compile_args.extend(_string_list(native_params.get("compile_args"), label="native.compile_args"))
    run_args = _string_list(native_params.get("run_args"), label="native.run_args")
    env = _string_mapping(native_params.get("env"), label="native.env")

    if source_path is None and binary_path is None:
        if kernel_name is None:
            raise ValueError("No native kernel configured. Set params.native.kernel, params.native.source, or params.native.binary.")
        raise ValueError(f"Native kernel {kernel_name} does not define a source or binary path.")

    return ResolvedNativeKernel(
        benchmark=benchmark,
        name=kernel_name or (source_path.stem if source_path else binary_path.stem),
        description=definition.description if definition is not None else "external native kernel",
        source_path=source_path,
        binary_path=binary_path,
        compiler=compiler,
        compile_args=tuple(compile_args),
        run_args=tuple(run_args),
        env=env,
        cache_dir=cache_dir,
    )


def _resolve_native_cache_dir(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return Path(xdg_cache_home).expanduser().resolve() / "teamredbench" / "native"
    return Path.home().resolve() / ".cache" / "teamredbench" / "native"


def native_cache_dir(explicit: Path | None = None) -> Path:
    return _resolve_native_cache_dir(explicit)


def clear_native_cache(explicit: Path | None = None) -> tuple[Path, int]:
    cache_dir = _resolve_native_cache_dir(explicit)
    if not cache_dir.exists():
        return cache_dir, 0

    removed = 0
    for entry in cache_dir.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry)
            removed += 1
        else:
            entry.unlink()
            removed += 1
    return cache_dir, removed


def _find_hip_compiler(preferred: str | None = None) -> str:
    candidates: list[str] = []
    for candidate in [
        preferred,
        os.environ.get("TEAMREDBENCH_HIPCC"),
        shutil.which("hipcc"),
        "/opt/rocm/bin/hipcc",
        shutil.which("amdclang++"),
        "/opt/rocm/bin/amdclang++",
    ]:
        if candidate:
            candidates.append(candidate)

    for candidate in candidates:
        path = Path(candidate)
        if path.is_absolute():
            if path.exists():
                return str(path)
            continue
        resolved = shutil.which(candidate)
        if resolved:
            return resolved

    raise NativeKernelError(
        "No HIP compiler found. Set TEAMREDBENCH_HIPCC, params.native.compiler, or params.native.binary."
    )


def prepare_native_kernel_binary(kernel: ResolvedNativeKernel) -> Path:
    if kernel.binary_path is not None:
        if not kernel.binary_path.exists():
            raise NativeKernelError(f"Native kernel binary does not exist: {kernel.binary_path}")
        return kernel.binary_path

    if kernel.source_path is None:
        raise NativeKernelError(f"Native kernel {kernel.name} is missing both source_path and binary_path.")
    if not kernel.source_path.exists():
        raise NativeKernelError(f"Native kernel source does not exist: {kernel.source_path}")

    compiler = _find_hip_compiler(kernel.compiler)
    cache_dir = _resolve_native_cache_dir(kernel.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha256()
    digest.update(kernel.source_path.read_bytes())
    digest.update(kernel.benchmark.encode("utf-8"))
    digest.update(kernel.name.encode("utf-8"))
    digest.update(compiler.encode("utf-8"))
    for item in kernel.compile_args:
        digest.update(item.encode("utf-8"))

    binary_path = cache_dir / f"{kernel.name}-{digest.hexdigest()[:16]}"
    if binary_path.exists():
        return binary_path

    compiler_name = Path(compiler).name
    command = [compiler, "-O3", "-std=c++17"]
    if compiler_name.endswith("clang++"):
        command.extend(["-x", "hip"])
    command.extend([str(kernel.source_path), "-o", str(binary_path)])
    command.extend(kernel.compile_args)

    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        error = completed.stderr.strip() or completed.stdout.strip() or "unknown compiler failure"
        raise NativeKernelError(f"Failed to compile native kernel {kernel.name}: {error}")

    binary_path.chmod(0o755)
    return binary_path


def _parse_native_result(stdout: str) -> NativeKernelResult:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise NativeKernelError("Native kernel produced no JSON output.")
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise NativeKernelError(f"Native kernel output was not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise NativeKernelError("Native kernel output JSON must be an object.")

    raw_metrics = payload.get("raw_metrics", {})
    metadata = payload.get("metadata", {})
    if not isinstance(raw_metrics, dict):
        raise NativeKernelError("Native kernel raw_metrics must be a JSON object.")
    if not isinstance(metadata, dict):
        raise NativeKernelError("Native kernel metadata must be a JSON object.")

    return NativeKernelResult(
        status=str(payload.get("status", "ok")),
        raw_metrics=dict(raw_metrics),
        metadata=dict(metadata),
        error=str(payload["error"]) if payload.get("error") is not None else None,
    )


def run_native_kernel(kernel: ResolvedNativeKernel, args: list[str]) -> NativeKernelResult:
    binary_path = prepare_native_kernel_binary(kernel)
    command = [str(binary_path), *args, *kernel.run_args]
    env = os.environ.copy()
    env.update(kernel.env)
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if completed.returncode != 0:
        error = completed.stderr.strip() or completed.stdout.strip() or "native kernel process failed"
        raise NativeKernelError(f"Native kernel {kernel.name} failed: {error}")
    return _parse_native_result(completed.stdout)
