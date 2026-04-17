from __future__ import annotations

import importlib
from typing import Any, Callable

from teamredbench.metrics.base import MetricDefinition


BENCHMARK_REGISTRY: dict[str, type] = {}
METRIC_REGISTRY: dict[str, MetricDefinition] = {}


def register_benchmark(name: str):
    def decorator(cls: type) -> type:
        BENCHMARK_REGISTRY[name] = cls
        cls.name = name
        return cls

    return decorator


def register_metric(name: str, description: str):
    def decorator(function: Callable[[dict[str, Any]], float | None]):
        METRIC_REGISTRY[name] = MetricDefinition(
            name=name,
            description=description,
            compute=function,
        )
        return function

    return decorator


def get_benchmark(name: str) -> type:
    if name not in BENCHMARK_REGISTRY:
        raise KeyError(f"Unknown benchmark: {name}")
    return BENCHMARK_REGISTRY[name]


def get_metric(name: str) -> MetricDefinition:
    if name not in METRIC_REGISTRY:
        raise KeyError(f"Unknown metric: {name}")
    return METRIC_REGISTRY[name]


def list_benchmarks() -> list[str]:
    return sorted(BENCHMARK_REGISTRY)


def list_metrics() -> list[str]:
    return sorted(METRIC_REGISTRY)


def load_plugins(module_names: list[str]) -> None:
    for module_name in module_names:
        importlib.import_module(module_name)

