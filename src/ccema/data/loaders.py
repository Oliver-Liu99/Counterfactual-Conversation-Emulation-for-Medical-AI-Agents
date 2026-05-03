"""Unified entry point for dataset loading."""

from __future__ import annotations

from pathlib import Path

from ccema.data.aci_bench import load_aci_bench
from ccema.data.mts_dialog import load_mts_dialog
from ccema.data.schema import Encounter
from ccema.data.synthetic import SyntheticConfig, make_synthetic_dataset


def load_dataset(name: str, root: str | Path | None = None, **kwargs) -> list[Encounter]:
    """Load a named dataset by string identifier.

    Supported:
        - "aci_bench" (requires root)
        - "mts_dialog" (requires root)
        - "synthetic" (no root)
    """
    name = name.lower().strip()

    if name == "aci_bench":
        if root is None:
            raise ValueError("aci_bench requires `root` pointing to the data directory")
        return load_aci_bench(root, **kwargs)

    if name == "mts_dialog":
        if root is None:
            raise ValueError("mts_dialog requires `root` pointing to the repo")
        return load_mts_dialog(root, **kwargs)

    if name == "synthetic":
        cfg = SyntheticConfig(**kwargs) if kwargs else SyntheticConfig()
        return make_synthetic_dataset(cfg)

    raise ValueError(f"Unknown dataset {name!r}")
