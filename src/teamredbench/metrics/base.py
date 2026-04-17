from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    description: str
    compute: Callable[[dict[str, Any]], float | None]

