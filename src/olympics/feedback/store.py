"""Human feedback loop: store predictions made by the API/dashboard, and the human-verified outcomes that follow.

Append-only JSONL (crash safe, trivially mergeable, easy to version with DVC). A prediction and its feedback
are joined by ``prediction_id``.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import pandas as pd

from ..data.schema import ALL_FEATURES


class FeedbackStore:
    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.pred_path = self.dir / "predictions.jsonl"
        self.fb_path = self.dir / "feedback.jsonl"

    @staticmethod
    def _append(path: Path, record: dict) -> None:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    @staticmethod
    def _read(path: Path) -> pd.DataFrame:
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a torn last line from a crash must not break reading
        return pd.DataFrame(rows)

    def log_prediction(self, features: dict, probability: float, label: int, model_version: str, threshold: float) -> str:
        pid = uuid.uuid4().hex[:12]
        self._append(self.pred_path, {"prediction_id": pid, "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "model_version": model_version,
                                      "probability": float(probability), "predicted": int(label), "threshold": float(threshold),
                                      "features": {k: features.get(k) for k in ALL_FEATURES}})
        return pid

    def log_feedback(self, prediction_id: str, actual_medal: bool, reviewer: str = "anonymous", note: str = "") -> None:
        if prediction_id not in set(self.predictions()["prediction_id"] if len(self.predictions()) else []):
            raise KeyError(f"Unknown prediction_id {prediction_id!r}")
        self._append(self.fb_path, {"prediction_id": prediction_id, "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "actual": int(bool(actual_medal)), "reviewer": reviewer, "note": note})

    def predictions(self) -> pd.DataFrame:
        return self._read(self.pred_path)

    def feedback(self) -> pd.DataFrame:
        return self._read(self.fb_path)

    def labelled(self) -> pd.DataFrame:
        """Predictions joined with verified outcomes (latest feedback per prediction wins)."""
        p, f = self.predictions(), self.feedback()
        if p.empty or f.empty:
            return pd.DataFrame()
        f = f.sort_values("ts").drop_duplicates("prediction_id", keep="last")
        j = p.merge(f[["prediction_id", "actual", "reviewer", "note"]], on="prediction_id", how="inner")
        feats = pd.json_normalize(j["features"]).set_index(j.index)
        return pd.concat([j.drop(columns=["features"]), feats], axis=1)

    def summary(self) -> dict:
        p, lab = self.predictions(), self.labelled()
        out = {"predictions": int(len(p)), "labelled": int(len(lab))}
        if len(lab):
            out["accuracy_on_feedback"] = float((lab["predicted"] == lab["actual"]).mean())
            out["false_positives"] = int(((lab["predicted"] == 1) & (lab["actual"] == 0)).sum())
            out["false_negatives"] = int(((lab["predicted"] == 0) & (lab["actual"] == 1)).sum())
        return out
