from __future__ import annotations

from dataclasses import dataclass, asdict
import shutil
import subprocess
from typing import Any


@dataclass
class DeviceInfo:
    index: int
    name: str
    total_memory_bytes: int | None = None
    multiprocessor_count: int | None = None
    clock_rate_khz: int | None = None
    gcn_arch_name: str | None = None


def load_torch():
    try:
        import torch

        return torch
    except ImportError:
        return None


def rocm_device_available(torch_module: Any) -> bool:
    if torch_module is None:
        return False
    return bool(torch_module.cuda.is_available())


def rocm_device(torch_module: Any, device_id: int):
    # PyTorch exposes ROCm accelerators through the torch.cuda device API.
    return torch_module.device(f"cuda:{device_id}")


def set_rocm_device(torch_module: Any, device_id: int) -> None:
    if rocm_device_available(torch_module):
        torch_module.cuda.set_device(device_id)


def synchronize_rocm_device(torch_module: Any) -> None:
    if rocm_device_available(torch_module):
        torch_module.cuda.synchronize()


def _run_command(command: list[str]) -> str | None:
    if shutil.which(command[0]) is None:
        return None
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def detect_device(torch_module: Any, device_id: int = 0) -> DeviceInfo | None:
    if not rocm_device_available(torch_module):
        return None

    properties = torch_module.cuda.get_device_properties(device_id)

    gcn_arch_name = None
    for attr_name in ("gcnArchName", "gcn_arch_name", "arch"):
        value = getattr(properties, attr_name, None)
        if value:
            gcn_arch_name = str(value)
            break

    return DeviceInfo(
        index=device_id,
        name=str(properties.name),
        total_memory_bytes=getattr(properties, "total_memory", None),
        multiprocessor_count=getattr(properties, "multi_processor_count", None),
        clock_rate_khz=getattr(properties, "clock_rate", None),
        gcn_arch_name=gcn_arch_name,
    )


def detect_runtime(torch_module: Any) -> dict[str, Any]:
    torch_info: dict[str, Any] = {}
    if torch_module is not None:
        torch_info = {
            "torch_version": getattr(torch_module, "__version__", "unknown"),
            "hip_version": getattr(getattr(torch_module, "version", None), "hip", None),
        }

    rocminfo_output = _run_command(["rocminfo"])
    if rocminfo_output:
        rocminfo_output = "\n".join(rocminfo_output.splitlines()[:40])

    return {
        "torch": torch_info,
        "hipcc_version": _run_command(["hipcc", "--version"]),
        "rocminfo_head": rocminfo_output,
        "rocm_smi_product": _run_command(["rocm-smi", "--showproductname"]),
    }


def discover_environment(torch_module: Any, device_id: int = 0) -> dict[str, Any]:
    device = detect_device(torch_module, device_id=device_id)
    return {
        "device": asdict(device) if device else None,
        "runtime": detect_runtime(torch_module),
        "suggested_hardware_profile": {
            "name": device.name if device else "fill-me",
            "family": "cdna-or-rdna",
            "peak_hbm_bandwidth_gbps": None,
            "peak_link_bandwidth_gbps": {
                "xgmi": None,
                "network": None,
            },
            "peak_compute_tops": {},
            "notes": [
                "Fill in theoretical peak values before using MFU or efficiency metrics.",
            ],
        },
    }
