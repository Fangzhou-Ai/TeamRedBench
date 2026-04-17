from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NativeKernelDefinition:
    name: str
    benchmark: str
    description: str
    source_path: Path | None = None
    compile_args: tuple[str, ...] = ()


NATIVE_KERNEL_REGISTRY: dict[str, NativeKernelDefinition] = {}


def register_native_kernel(
    *,
    name: str,
    benchmark: str,
    description: str,
    source_path: Path | None = None,
    compile_args: tuple[str, ...] = (),
) -> NativeKernelDefinition:
    definition = NativeKernelDefinition(
        name=name,
        benchmark=benchmark,
        description=description,
        source_path=source_path,
        compile_args=tuple(compile_args),
    )
    NATIVE_KERNEL_REGISTRY[name] = definition
    return definition


def get_native_kernel(name: str) -> NativeKernelDefinition:
    if name not in NATIVE_KERNEL_REGISTRY:
        raise KeyError(f"Unknown native kernel: {name}")
    return NATIVE_KERNEL_REGISTRY[name]


def list_native_kernels(benchmark: str | None = None) -> list[NativeKernelDefinition]:
    definitions = sorted(NATIVE_KERNEL_REGISTRY.values(), key=lambda item: item.name)
    if benchmark is None:
        return definitions
    return [definition for definition in definitions if definition.benchmark == benchmark]
