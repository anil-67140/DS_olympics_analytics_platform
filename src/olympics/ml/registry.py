"""Model versioning: every saved model gets an immutable folder with the model, schema, metrics and lineage.

models/
  v20261031-0712-ab12cd34/   model.joblib, metadata.json, MODEL_CARD.md, history_lookup.json
  latest.json                pointer to the champion (atomic update; rollback = edit the pointer)
"""
from __future__ import annotations

import hashlib
import platform
import subprocess
import time
from pathlib import Path

import joblib
import sklearn

from ..data.schema import ALL_FEATURES
from ..io_utils import read_json, write_json, write_text
from .model_wrapper import MedalModel


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5,
                              check=True).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def schema_hash(features: list[str] | None = None) -> str:
    return hashlib.sha256(",".join(features or ALL_FEATURES).encode()).hexdigest()[:10]


def new_version(data_fingerprint: str, params_fingerprint: str) -> str:
    h = hashlib.sha256(f"{data_fingerprint}|{params_fingerprint}|{schema_hash()}".encode()).hexdigest()[:8]
    return f"v{time.strftime('%Y%m%d-%H%M')}-{h}"


def model_card(meta: dict) -> str:
    t = meta["test_metrics"]
    return "\n".join([
        f"# Model card — {meta['version']}", "",
        f"**Task:** predict whether an athlete's entry in an event results in a medal (binary, ~15% positive).",
        f"**Algorithm:** {meta['algorithm']} · calibration: {meta['calibration']} · decision threshold: {meta['threshold']:.4f} "
        f"({meta['threshold_strategy']})", "",
        "## Test-set performance (evaluated once, after all choices were frozen)",
        f"| PR-AUC | ROC-AUC | F1 | Precision | Recall | Brier | ECE | MCC |", "|---|---|---|---|---|---|---|---|",
        f"| {t['pr_auc']:.3f} | {t['roc_auc']:.3f} | {t['f1']:.3f} | {t['precision']:.3f} | {t['recall']:.3f} | {t['brier']:.3f} | {t['ece']:.3f} | {t['mcc']:.3f} |", "",
        f"Baseline (always predict prevalence): PR-AUC = {t['prevalence']:.3f}.", "",
        "## Intended use", "Benchmarking, scenario analysis and education. Not for selection or funding decisions about individual athletes.", "",
        "## Data & lineage",
        f"- Data fingerprint: `{meta['data_fingerprint']}` · feature schema hash: `{meta['schema_hash']}` · git: `{meta.get('git_sha')}`",
        f"- Split: {meta['split_mode']} (groups: {meta['split_groups']}); train/val/test = {meta['split_sizes']}",
        f"- Library versions: scikit-learn {meta['sklearn_version']}, python {meta['python_version']}", "",
        "## Known limitations",
        "- Historical data only; medal rules, events and team composition change over time.",
        "- Team events share one outcome across members, so entries are not independent (hence delegation-grouped splitting).",
        "- Height/weight are missing for many early athletes (imputed + indicator).",
        "- History features reflect past results of the athlete/country; new nations or athletes fall back to the global prior.",
        "- Performance differs by slice (sex, era, sport) — see `slice_metrics.csv` before relying on a subgroup.", "",
    ])


def save_model_version(models_dir: Path, model: MedalModel, meta: dict, history_lookup: dict | None = None) -> Path:
    models_dir = Path(models_dir)
    version = meta["version"]
    vdir = models_dir / version
    vdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, vdir / "model.joblib", compress=3)
    meta = {**meta, "sklearn_version": sklearn.__version__, "python_version": platform.python_version(),
            "schema_hash": schema_hash(), "features": ALL_FEATURES, "git_sha": _git_sha(),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    write_json(vdir / "metadata.json", meta)
    write_text(vdir / "MODEL_CARD.md", model_card(meta))
    if history_lookup is not None:
        write_json(vdir / "history_lookup.json", history_lookup)
    return vdir


def promote(models_dir: Path, version: str) -> None:
    """Atomically point ``latest.json`` at ``version`` (use again with an older version to roll back)."""
    if not (Path(models_dir) / version / "model.joblib").exists():
        raise FileNotFoundError(f"{version} has no model.joblib")
    write_json(Path(models_dir) / "latest.json", {"version": version, "promoted_at": time.strftime("%Y-%m-%d %H:%M:%S")})


def list_versions(models_dir: Path) -> list[str]:
    return sorted(p.name for p in Path(models_dir).glob("v*") if (p / "model.joblib").exists())


def latest_version(models_dir: Path) -> str | None:
    ptr = read_json(Path(models_dir) / "latest.json")
    return ptr["version"] if ptr else None


def load_model(models_dir: Path, version: str | None = None) -> tuple[MedalModel, dict]:
    version = version or latest_version(models_dir)
    if not version:
        raise FileNotFoundError(f"No promoted model in {models_dir}. Run the pipeline (or `olympics.cli stage ml_final`) first.")
    vdir = Path(models_dir) / version
    meta = read_json(vdir / "metadata.json", {})
    if meta.get("schema_hash") and meta["schema_hash"] != schema_hash():
        raise RuntimeError(f"Feature schema changed since {version} was trained ({meta['schema_hash']} != {schema_hash()}).")
    if meta.get("sklearn_version") and meta["sklearn_version"] != sklearn.__version__:
        import warnings
        warnings.warn(f"Model trained with scikit-learn {meta['sklearn_version']}, running {sklearn.__version__}.")
    return joblib.load(vdir / "model.joblib"), meta
