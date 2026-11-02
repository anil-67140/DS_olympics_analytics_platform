"""Every metric that matters for an imbalanced classification problem, in one place.

Accuracy alone is misleading here (~85% of entries win no medal, so "never predict a medal" scores 85%).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import metrics as skm


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10, strategy: str = "quantile") -> pd.DataFrame:
    y, p = np.asarray(y, float), np.asarray(p, float)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    if len(edges) < 2:  # constant predictions (e.g. the baseline) -> one single bin
        return pd.DataFrame([{"bin": 0, "n": int(len(p)), "mean_predicted": float(p.mean()), "observed_rate": float(y.mean())}])
    edges[0], edges[-1] = -np.inf, np.inf
    bins = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        mask = bins == b
        if mask.any():
            rows.append({"bin": b, "n": int(mask.sum()), "mean_predicted": p[mask].mean(), "observed_rate": y[mask].mean()})
    return pd.DataFrame(rows)


def expected_calibration_error(y, p, n_bins: int = 10, strategy: str = "quantile") -> tuple[float, float]:
    t = calibration_table(y, p, n_bins, strategy)
    gap = (t["mean_predicted"] - t["observed_rate"]).abs()
    return float((t["n"] * gap).sum() / t["n"].sum()), float(gap.max())


def precision_lift_at_k(y: np.ndarray, p: np.ndarray, frac: float = 0.10) -> tuple[float, float]:
    k = max(1, int(len(y) * frac))
    top = np.argsort(-p)[:k]
    prec = float(np.asarray(y)[top].mean())
    return prec, prec / float(np.mean(y)) if np.mean(y) else np.nan


def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    y = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(proba, float), 1e-7, 1 - 1e-7)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = skm.confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prev = y.mean()
    ece, mce = expected_calibration_error(y, p)
    brier = skm.brier_score_loss(y, p)
    base_brier = prev * (1 - prev)
    p_at, lift_at = precision_lift_at_k(y, p, 0.10)
    safe = lambda a, b: float(a / b) if b else 0.0  # noqa: E731
    return {
        "threshold": float(threshold),
        "accuracy": skm.accuracy_score(y, pred), "balanced_accuracy": skm.balanced_accuracy_score(y, pred),
        "precision": safe(tp, tp + fp), "recall": safe(tp, tp + fn), "specificity": safe(tn, tn + fp),
        "npv": safe(tn, tn + fn), "f1": skm.f1_score(y, pred, zero_division=0),
        "f2": skm.fbeta_score(y, pred, beta=2, zero_division=0), "f0_5": skm.fbeta_score(y, pred, beta=0.5, zero_division=0),
        "mcc": skm.matthews_corrcoef(y, pred), "cohen_kappa": skm.cohen_kappa_score(y, pred),
        "roc_auc": skm.roc_auc_score(y, p), "pr_auc": skm.average_precision_score(y, p),
        "log_loss": skm.log_loss(y, p), "brier": brier,
        "brier_skill_score": 1 - brier / base_brier if base_brier else np.nan,
        "ece": ece, "mce": mce, "precision_at_top10pct": p_at, "lift_at_top10pct": lift_at,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "prevalence": float(prev), "predicted_positive_rate": float(pred.mean()), "n": int(len(y)),
    }


def curves(y, p, max_points: int = 400) -> dict:
    y = np.asarray(y).astype(int)
    pr, rc, _ = skm.precision_recall_curve(y, p)
    fpr, tpr, _ = skm.roc_curve(y, p)
    thin = lambda a: a[np.linspace(0, len(a) - 1, min(len(a), max_points)).astype(int)]  # noqa: E731
    return {"precision": thin(pr).tolist(), "recall": thin(rc).tolist(), "fpr": thin(fpr).tolist(), "tpr": thin(tpr).tolist()}


def cluster_bootstrap_ci(y, p, groups, metric: str = "pr_auc", n_boot: int = 200, seed: int = 42, threshold: float = 0.5) -> tuple[float, float]:
    """Bootstrap by resampling whole groups (delegations), which respects the dependence between teammates."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    codes, uniq = pd.factorize(pd.Series(groups))
    members = [np.flatnonzero(codes == i) for i in range(len(uniq))]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[i] for i in pick])
        if y[idx].min() == y[idx].max():
            continue
        vals.append(compute_metrics(y[idx], p[idx], threshold)[metric])
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))
