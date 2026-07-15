from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    title: str
    mode: str
    root: Path | None = None
    required_paths: tuple[str, ...] = ()
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetStatus:
    name: str
    title: str
    mode: str
    ready: bool
    root: Path | None
    missing_paths: tuple[Path, ...]
    message: str


def load_dataset_specs(path: str | Path = "configs/datasets.yaml") -> dict[str, DatasetSpec]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    specs: dict[str, DatasetSpec] = {}
    for name, raw_spec in (payload.get("datasets") or {}).items():
        root = raw_spec.get("root")
        specs[name] = DatasetSpec(
            name=name,
            title=raw_spec.get("title", name),
            mode=raw_spec.get("mode", "user_managed"),
            root=Path(root) if root else None,
            required_paths=tuple(raw_spec.get("required_paths") or ()),
            description=raw_spec.get("description", ""),
            metadata={
                key: value
                for key, value in raw_spec.items()
                if key not in {"title", "mode", "root", "required_paths", "description"}
            },
        )
    return specs


def dataset_status(spec: DatasetSpec, *, project_root: str | Path = ".") -> DatasetStatus:
    if spec.mode in {"generated", "builtin"}:
        return DatasetStatus(
            name=spec.name,
            title=spec.title,
            mode=spec.mode,
            ready=True,
            root=spec.root,
            missing_paths=(),
            message="Ready without local dataset files.",
        )

    if spec.root is None:
        return DatasetStatus(
            name=spec.name,
            title=spec.title,
            mode=spec.mode,
            ready=False,
            root=None,
            missing_paths=(),
            message="No dataset root is configured.",
        )

    root = Path(project_root) / spec.root
    missing = tuple(root / path for path in spec.required_paths if not (root / path).exists())
    ready = root.exists() and not missing
    message = "Ready." if ready else "User needs to place dataset files under the configured root."
    return DatasetStatus(
        name=spec.name,
        title=spec.title,
        mode=spec.mode,
        ready=ready,
        root=root,
        missing_paths=missing,
        message=message,
    )
