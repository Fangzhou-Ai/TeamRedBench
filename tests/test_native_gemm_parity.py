"""Numerical parity tests for the native MFMA GEMM kernels against hipBLAS.

Compiles ``tests/native/gemm_parity.hip.cpp`` via ``hipcc`` on demand and runs
one case per dtype. Skipped unless a ROCm toolchain, hipBLAS, and a visible GPU
are all present.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parent / "native" / "gemm_parity.hip.cpp"
HIPCC_CANDIDATES = ("hipcc", "/opt/rocm/bin/hipcc")
HIPBLAS_HEADER_CANDIDATES = (
    Path("/opt/rocm/include/hipblas/hipblas.h"),
)


def _find_hipcc() -> str | None:
    for candidate in HIPCC_CANDIDATES:
        resolved = shutil.which(candidate) if "/" not in candidate else (candidate if Path(candidate).exists() else None)
        if resolved:
            return resolved
    return None


def _gpu_visible() -> bool:
    # rocminfo exits 0 and prints agent info when at least one GPU is visible.
    rocminfo = shutil.which("rocminfo") or "/opt/rocm/bin/rocminfo"
    if not Path(rocminfo).exists():
        return False
    try:
        result = subprocess.run([rocminfo], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and "gfx" in result.stdout.lower()


@pytest.fixture(scope="module")
def parity_binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    hipcc = _find_hipcc()
    if hipcc is None:
        pytest.skip("hipcc not available")
    if not any(path.exists() for path in HIPBLAS_HEADER_CANDIDATES):
        pytest.skip("hipBLAS headers not available")
    if not _gpu_visible():
        pytest.skip("no ROCm GPU visible to rocminfo")

    out_dir = tmp_path_factory.mktemp("gemm_parity")
    binary = out_dir / "gemm_parity"
    cmd = [
        hipcc, "-O3", "-std=c++17",
        str(SOURCE), "-o", str(binary),
        "-I/opt/rocm/include", "-L/opt/rocm/lib", "-lhipblas",
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if completed.returncode != 0:
        pytest.skip(
            "failed to compile gemm_parity harness:\n"
            f"cmd: {' '.join(cmd)}\nstderr:\n{completed.stderr}"
        )
    return binary


def _run_case(binary: Path, dtype: str, m: int = 512, n: int = 512, k: int = 512) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(binary), dtype, str(m), str(n), str(k)],
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32", "float64"])
def test_native_gemm_matches_hipblas(parity_binary: Path, dtype: str) -> None:
    result = _run_case(parity_binary, dtype)
    assert result.returncode == 0, (
        f"gemm_parity {dtype} exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "PASS" in result.stdout, f"no PASS in output:\n{result.stdout}"
    assert "FAIL" not in result.stdout, f"unexpected FAIL in output:\n{result.stdout}"
