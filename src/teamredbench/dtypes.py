from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DTypeSpec:
    name: str
    torch_attr: str
    bytes_per_element: int
    kind: str
    matmul_ops_factor: int


DTYPE_CANDIDATES = [
    DTypeSpec("bool", "bool", 1, "bool", 0),
    DTypeSpec("uint8", "uint8", 1, "integer", 2),
    DTypeSpec("int8", "int8", 1, "integer", 2),
    DTypeSpec("int16", "int16", 2, "integer", 2),
    DTypeSpec("int32", "int32", 4, "integer", 2),
    DTypeSpec("int64", "int64", 8, "integer", 2),
    DTypeSpec("float8_e4m3fn", "float8_e4m3fn", 1, "float", 2),
    DTypeSpec("float8_e5m2", "float8_e5m2", 1, "float", 2),
    DTypeSpec("float16", "float16", 2, "float", 2),
    DTypeSpec("bfloat16", "bfloat16", 2, "float", 2),
    DTypeSpec("float32", "float32", 4, "float", 2),
    DTypeSpec("float64", "float64", 8, "float", 2),
    DTypeSpec("complex64", "complex64", 8, "complex", 8),
    DTypeSpec("complex128", "complex128", 16, "complex", 8),
]


def discover_dtypes(torch_module: Any) -> dict[str, DTypeSpec]:
    if torch_module is None:
        return {spec.name: spec for spec in DTYPE_CANDIDATES}
    return {
        spec.name: spec
        for spec in DTYPE_CANDIDATES
        if hasattr(torch_module, spec.torch_attr)
    }


def resolve_dtype_names(requested: list[str] | None, available: dict[str, DTypeSpec]) -> list[str]:
    if not requested:
        return list(available)
    if "all" in requested:
        return list(available)
    unknown = [item for item in requested if item not in available]
    if unknown:
        raise KeyError(f"Unknown dtypes requested: {', '.join(unknown)}")
    return requested

