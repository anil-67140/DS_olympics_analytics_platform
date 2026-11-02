"""Explainability: global (permutation importance, optional SHAP) and local (per-prediction contributions).

Permutation importance is computed on *raw* input columns through the whole pipeline, so it answers
"how much does the model depend on Age / NOC / Sport?" rather than on opaque one-hot columns.
The local explanation is model-agnostic: replace one feature at a time by a typical (baseline) value and
measure how the predicted probability moves.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.schema import ALL_FEATURES, HIGH_CARD_CATEGORICAL, LOW_CARD_CATEGORICAL


def global_importance(model, X: pd.DataFrame, y: np.ndarray, n_repeats: int = 5, max_rows: int = 20000,
                      seed: int = 42, scoring: str = "average_precision") -> pd.DataFrame:
    """Permutation importance on RAW input columns, measured as the drop in PR-AUC (average precision)."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    score_fn = average_precision_score if scoring == "average_precision" else roc_auc_score
    rng = np.random.default_rng(seed)
    if len(X) > max_rows:
        idx = rng.choice(len(X), max_rows, replace=False)
        X, y = X.iloc[idx], np.asarray(y)[idx]
    X = X[model.feature_names].reset_index(drop=True)
    y = np.asarray(y)
    base = score_fn(y, model.predict_proba(X))
    rows = []
    for f in model.feature_names:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[f] = Xp[f].to_numpy()[rng.permutation(len(Xp))]
            drops.append(base - score_fn(y, model.predict_proba(Xp)))
        rows.append({"feature": f, "importance_mean": float(np.mean(drops)), "importance_std": float(np.std(drops))})
    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def baseline_row(X_train: pd.DataFrame) -> pd.Series:
    """'Typical' athlete: median numeric, most common category."""
    base = {}
    for c in ALL_FEATURES:
        base[c] = X_train[c].median() if c not in LOW_CARD_CATEGORICAL + HIGH_CARD_CATEGORICAL else X_train[c].mode().iat[0]
    return pd.Series(base)


def local_explanation(model, row: pd.DataFrame, baseline: pd.Series, top: int = 6) -> pd.DataFrame:
    """Effect of each feature on this prediction: p(actual) - p(feature replaced by its baseline value)."""
    assert len(row) == 1
    p0 = float(model.predict_proba(row)[0])
    variants = []
    for f in model.feature_names:
        v = row.copy()
        v[f] = baseline[f]
        variants.append(v)
    p_alt = model.predict_proba(pd.concat(variants, ignore_index=True))
    out = pd.DataFrame({"feature": model.feature_names, "value": [row.iloc[0][f] for f in model.feature_names],
                        "baseline_value": [baseline[f] for f in model.feature_names], "effect_on_probability": p0 - p_alt})
    out["abs_effect"] = out["effect_on_probability"].abs()
    out = out.sort_values("abs_effect", ascending=False).head(top).drop(columns="abs_effect")
    out.attrs["probability"] = p0
    return out.reset_index(drop=True)


def shap_summary(pipeline_model, X: pd.DataFrame, max_rows: int = 2000):
    """Optional SHAP (tree models). Returns (shap_values, transformed_X) or None if shap is unavailable."""
    try:
        import shap  # type: ignore
    except ImportError:
        return None
    pipe = pipeline_model.pipeline
    Xs = X[pipeline_model.feature_names].head(max_rows)
    Xt = pipe[:-1].transform(Xs)
    try:
        explainer = shap.TreeExplainer(pipe[-1])
        return explainer.shap_values(Xt), Xt
    except Exception:  # noqa: BLE001 - not a tree model
        return None
