"""Deployable model object: preprocessing + classifier + probability calibrator + decision threshold in ONE artifact.

Bundling these means the API can never apply a different preprocessing, calibration or threshold than the
one that was validated (the "model and vectorizer versioning" requirement: one object, one version).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class ProbabilityCalibrator:
    def __init__(self, method: str = "none"):
        if method not in {"none", "sigmoid", "isotonic"}:
            raise ValueError(method)
        self.method = method
        self._model = None

    def fit(self, p: np.ndarray, y: np.ndarray) -> "ProbabilityCalibrator":
        if self.method == "isotonic":
            self._model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
        elif self.method == "sigmoid":
            self._model = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), y)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, float)
        if self.method == "isotonic":
            return np.clip(self._model.predict(p), 1e-6, 1 - 1e-6)
        if self.method == "sigmoid":
            return self._model.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
        return p


class MedalModel:
    """pipeline -> raw probability -> calibrator -> threshold."""

    def __init__(self, pipeline, calibrator: ProbabilityCalibrator, threshold: float, feature_names: list[str],
                 metadata: dict | None = None):
        self.pipeline = pipeline
        self.calibrator = calibrator
        self.threshold = float(threshold)
        self.feature_names = list(feature_names)
        self.metadata = metadata or {}

    def _check(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.feature_names if c not in X.columns]
        if missing:
            raise ValueError(f"Missing model inputs: {missing}")
        return X[self.feature_names]

    def raw_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(self._check(X))[:, 1]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.calibrator.transform(self.raw_proba(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold).astype(int)
