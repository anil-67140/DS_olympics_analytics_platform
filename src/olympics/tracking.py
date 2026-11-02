"""Experiment tracking: MLflow when installed, otherwise an append-only JSONL log with the same information."""
from __future__ import annotations

import json
import time
from pathlib import Path

from .config import to_jsonable
from .io_utils import get_logger

log = get_logger()

try:
    import mlflow  # type: ignore
    HAS_MLFLOW = True
except Exception:  # noqa: BLE001
    HAS_MLFLOW = False


class Tracker:
    def __init__(self, mlruns_dir: Path, experiment: str = "olympics-medal-model"):
        self.dir = Path(mlruns_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.experiment = experiment
        self.jsonl = self.dir / "runs.jsonl"
        if HAS_MLFLOW:
            mlflow.set_tracking_uri(f"file:{self.dir.resolve()}")
            mlflow.set_experiment(experiment)

    def log_run(self, run_name: str, params: dict, metrics: dict, artifacts: list[Path] | None = None,
                tags: dict | None = None) -> None:
        params, metrics = to_jsonable(params), to_jsonable(metrics)
        flat_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and v is not None}
        with open(self.jsonl, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "experiment": self.experiment, "run": run_name,
                                 "params": params, "metrics": flat_metrics, "tags": tags or {}}) + "\n")
        if not HAS_MLFLOW:
            return
        try:
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
                mlflow.log_metrics({k: v for k, v in flat_metrics.items() if v == v})
                if tags:
                    mlflow.set_tags({k: str(v) for k, v in tags.items()})
                for a in artifacts or []:
                    if Path(a).exists():
                        mlflow.log_artifact(str(a))
        except Exception as exc:  # noqa: BLE001 - tracking must never break training
            log.warning("MLflow logging failed (%s); JSONL log kept", exc)

    def runs(self) -> list[dict]:
        if not self.jsonl.exists():
            return []
        return [json.loads(line) for line in self.jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
