from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class ProfilingLaunch:
    command: tuple[str, ...]
    artifact_dir: Path
    env: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


BuildProfileCommand = Callable[[dict[str, Any], list[str], Path, Path | None], ProfilingLaunch]


@dataclass(frozen=True)
class ProfilingEngineDefinition:
    name: str
    description: str
    build_command: BuildProfileCommand


PROFILE_ENGINE_REGISTRY: dict[str, ProfilingEngineDefinition] = {}


def register_profile_engine(
    *,
    name: str,
    description: str,
    build_command: BuildProfileCommand,
) -> ProfilingEngineDefinition:
    definition = ProfilingEngineDefinition(
        name=name,
        description=description,
        build_command=build_command,
    )
    PROFILE_ENGINE_REGISTRY[name] = definition
    return definition


def get_profile_engine(name: str) -> ProfilingEngineDefinition:
    if name not in PROFILE_ENGINE_REGISTRY:
        raise KeyError(f"Unknown profile engine: {name}")
    return PROFILE_ENGINE_REGISTRY[name]


def list_profile_engines() -> list[ProfilingEngineDefinition]:
    return sorted(PROFILE_ENGINE_REGISTRY.values(), key=lambda item: item.name)
