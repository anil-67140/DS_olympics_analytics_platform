"""Configuration loading and workspace path management."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "params.yaml"


def _deep_update(base: dict, new: dict) -> dict:
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _set_dotted(cfg: dict, dotted: str, raw_value: str) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = yaml.safe_load(raw_value)


def load_config(path: str | os.PathLike | None = None, overrides: list[str] | dict | None = None) -> dict:
    """Load params.yaml and apply overrides (dict, or a list of 'a.b.c=value' strings)."""
    cfg_path = Path(path) if path else (DEFAULT_CONFIG_PATH if DEFAULT_CONFIG_PATH.exists() else Path("params.yaml"))
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if isinstance(overrides, dict):
        _deep_update(cfg, copy.deepcopy(overrides))
    elif overrides:
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"Override must look like key.sub=value, got: {item!r}")
            key, value = item.split("=", 1)
            _set_dotted(cfg, key.strip(), value.strip())
    return cfg


def config_fingerprint(cfg: dict, sections: list[str] | None = None) -> str:
    """Stable hash of (a subset of) the config, used for checkpoint invalidation and model versioning."""
    subset = cfg if not sections else {s: cfg.get(s) for s in sections}
    blob = json.dumps(subset, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


@dataclass(frozen=True)
class Paths:
    """All outputs live under one root so that a Google Drive folder can hold the whole workspace."""

    root: Path

    @classmethod
    def from_env(cls, root: str | os.PathLike | None = None) -> "Paths":
        chosen = root or os.environ.get("OLYMPICS_ROOT") or "workspace"
        return cls(Path(chosen).expanduser().resolve())

    def _sub(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    @property
    def raw(self) -> Path: return self._sub("data", "raw")
    @property
    def interim(self) -> Path: return self._sub("data", "interim")
    @property
    def processed(self) -> Path: return self._sub("data", "processed")
    @property
    def incoming(self) -> Path: return self._sub("data", "incoming")
    @property
    def db(self) -> Path: return self._sub("db")
    @property
    def db_file(self) -> Path: return self.db / "olympics.sqlite"
    @property
    def figures(self) -> Path: return self._sub("reports", "figures")
    @property
    def tables(self) -> Path: return self._sub("reports", "tables")
    @property
    def metrics(self) -> Path: return self._sub("reports", "metrics")
    @property
    def models(self) -> Path: return self._sub("models")
    @property
    def mlruns(self) -> Path: return self._sub("mlruns")
    @property
    def checkpoints(self) -> Path: return self._sub("checkpoints")
    @property
    def feedback(self) -> Path: return self._sub("feedback")
    @property
    def logs(self) -> Path: return self._sub("logs")

    def ensure(self) -> "Paths":
        for p in (self.raw, self.interim, self.processed, self.incoming, self.db, self.figures,
                  self.tables, self.metrics, self.models, self.mlruns, self.checkpoints,
                  self.feedback, self.logs):
            p.mkdir(parents=True, exist_ok=True)
        return self


def to_jsonable(obj: Any) -> Any:
    """Convert numpy/pandas scalars and containers to plain JSON types."""
    import numpy as np
    import pandas as pd

    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if value != value else value
    if isinstance(obj, float):
        return None if obj != obj else obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if obj is pd.NA or obj is pd.NaT:
        return None
    return obj
