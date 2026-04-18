from __future__ import annotations

from pathlib import Path

from teamredbench.native.registry import register_native_kernel


KERNEL_DIR = Path(__file__).resolve().parent / "kernels"


register_native_kernel(
    name="hbm_hip",
    benchmark="hbm",
    description="HIP STREAM-style HBM bandwidth kernel.",
    source_path=KERNEL_DIR / "hbm_bandwidth.hip.cpp",
)

register_native_kernel(
    name="mfu_hipblas",
    benchmark="mfu",
    description="rocWMMA/MFMA GEMM kernel for MFU measurements with scalar fallback.",
    source_path=KERNEL_DIR / "mfu_gemm.hip.cpp",
)
