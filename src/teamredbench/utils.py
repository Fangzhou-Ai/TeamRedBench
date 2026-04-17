from __future__ import annotations

import time
from typing import Any, Callable

from teamredbench.device import synchronize_rocm_device
from teamredbench.dtypes import DTypeSpec


def synchronize(torch_module: Any) -> None:
    synchronize_rocm_device(torch_module)


def timed_iterations(
    fn: Callable[[], None],
    *,
    torch_module: Any,
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        fn()
    synchronize(torch_module)
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    synchronize(torch_module)
    elapsed = time.perf_counter() - start
    return elapsed / max(iterations, 1)


def make_tensor(torch_module: Any, spec: DTypeSpec, shape: tuple[int, ...], device: Any):
    dtype = getattr(torch_module, spec.torch_attr)

    if spec.kind == "bool":
        return torch_module.randint(0, 2, shape, device=device, dtype=torch_module.uint8).to(dtype=dtype)

    if spec.kind == "integer":
        return torch_module.randint(0, 17, shape, device=device, dtype=dtype)

    if spec.kind == "complex":
        real_dtype = torch_module.float32 if spec.name == "complex64" else torch_module.float64
        real = torch_module.randn(shape, device=device, dtype=real_dtype)
        imag = torch_module.randn(shape, device=device, dtype=real_dtype)
        return torch_module.complex(real, imag).to(dtype=dtype)

    tensor = torch_module.randn(shape, device=device, dtype=torch_module.float32)
    return tensor.to(dtype=dtype)


def tensor_nbytes(tensor: Any) -> int:
    return int(tensor.numel() * tensor.element_size())
