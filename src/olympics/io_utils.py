"""Atomic, crash-safe file helpers. Partial writes never leave a corrupt file behind (Colab disconnect safe)."""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from .config import to_jsonable

try:  # parquet is preferred but optional
    import pyarrow  # noqa: F401
    HAS_PARQUET = True
except Exception:  # pragma: no cover - depends on environment
    HAS_PARQUET = False


def get_logger(name: str = "olympics", log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    if log_file is not None and not any(getattr(h, "_olympics_file", False) for h in logger.handlers):
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        fh._olympics_file = True  # type: ignore[attr-defined]
        logger.addHandler(fh)
    return logger


def _atomic_write_bytes(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        writer(tmp)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def write_text(path: Path, text: str) -> Path:
    path = Path(path)
    _atomic_write_bytes(path, lambda tmp: Path(tmp).write_text(text, encoding="utf-8"))
    return path


def write_json(path: Path, obj: Any) -> Path:
    return write_text(Path(path), json.dumps(to_jsonable(obj), indent=2, ensure_ascii=False))


def read_json(path: Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def save_frame(df: pd.DataFrame, stem: Path) -> Path:
    """Save a DataFrame as parquet when pyarrow exists, otherwise gzip-CSV. `stem` has no extension."""
    stem = Path(stem)
    if HAS_PARQUET:
        out = stem.with_suffix(".parquet")
        _atomic_write_bytes(out, lambda tmp: df.to_parquet(tmp, index=False))
    else:
        out = stem.with_suffix(".csv.gz")
        _atomic_write_bytes(out, lambda tmp: df.to_csv(tmp, index=False, compression="gzip"))
    return out


def frame_path(stem: Path) -> Path | None:
    stem = Path(stem)
    for suffix in (".parquet", ".csv.gz"):
        candidate = stem.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def load_frame(stem: Path, **read_kwargs) -> pd.DataFrame:
    found = frame_path(stem)
    if found is None:
        raise FileNotFoundError(f"No saved frame for {stem} (looked for .parquet and .csv.gz)")
    if found.suffix == ".parquet":
        return pd.read_parquet(found)
    return pd.read_csv(found, **read_kwargs)


def save_table(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    path = Path(path)
    _atomic_write_bytes(path, lambda tmp: df.to_csv(tmp, index=index))
    return path
