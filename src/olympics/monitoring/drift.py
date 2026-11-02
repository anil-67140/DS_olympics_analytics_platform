"""Drift detection and performance monitoring.

Self-contained (PSI, KS, chi-square, Jensen-Shannon) so it works anywhere; if the ``evidently`` package is
installed, ``evidently_report`` additionally writes its interactive HTML report.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

from ..data.schema import HIGH_CARD_CATEGORICAL, LOW_CARD_CATEGORICAL, NUMERIC_FEATURES
from ..ml.metrics import compute_metrics


def psi_numeric(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_categorical(ref: pd.Series, cur: pd.Series, top: int = 30) -> float:
    cats = ref.value_counts().head(top).index
    r = ref.where(ref.isin(cats), "__other__").value_counts(normalize=True)
    c = cur.where(cur.isin(cats), "__other__").value_counts(normalize=True)
    idx = r.index.union(c.index)
    r, c = r.reindex(idx, fill_value=0).clip(lower=1e-4), c.reindex(idx, fill_value=0).clip(lower=1e-4)
    return float(np.sum((c - r) * np.log(c / r)))


def _status(psi: float, warn: float, alert: float) -> str:
    if psi != psi:
        return "unknown"
    return "alert" if psi >= alert else "warn" if psi >= warn else "ok"


def detect_drift(reference: pd.DataFrame, current: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    m = cfg["monitoring"]
    ignore = set(m.get("ignore_features") or [])
    rows = []
    for col in [c for c in NUMERIC_FEATURES if c not in ignore]:
        if col not in reference or col not in current:
            continue
        r, c = reference[col].astype(float), current[col].astype(float)
        psi = psi_numeric(r.to_numpy(), c.to_numpy())
        ks = stats.ks_2samp(r.dropna(), c.dropna()) if r.notna().any() and c.notna().any() else None
        rows.append({"feature": col, "type": "numeric", "psi": psi, "ks_stat": ks.statistic if ks else np.nan,
                     "p_value": ks.pvalue if ks else np.nan, "ref_mean": r.mean(), "cur_mean": c.mean(),
                     "missing_ref_pct": 100 * r.isna().mean(), "missing_cur_pct": 100 * c.isna().mean(),
                     "status": _status(psi, m["psi_warn"], m["psi_alert"])})
    for col in [c for c in LOW_CARD_CATEGORICAL + HIGH_CARD_CATEGORICAL if c not in ignore]:
        if col not in reference or col not in current:
            continue
        r, c = reference[col].astype(str), current[col].astype(str)
        psi = psi_categorical(r, c)
        cats = r.value_counts().head(30).index
        rc = r.where(r.isin(cats), "__other__").value_counts()
        cc = c.where(c.isin(cats), "__other__").value_counts().reindex(rc.index, fill_value=0)
        p = stats.chi2_contingency(np.vstack([rc.values, cc.values]))[1] if cc.sum() > 0 and len(rc) > 1 else np.nan
        js = float(jensenshannon(rc.values / rc.sum(), (cc.values + 1e-9) / (cc.sum() + 1e-9 * len(cc)))) if cc.sum() > 0 else np.nan
        rows.append({"feature": col, "type": "categorical", "psi": psi, "ks_stat": js, "p_value": p,
                     "ref_mean": np.nan, "cur_mean": np.nan, "missing_ref_pct": 0.0, "missing_cur_pct": 0.0,
                     "status": _status(psi, m["psi_warn"], m["psi_alert"])})
    out = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    return out


def drift_summary(table: pd.DataFrame) -> dict:
    return {"features_checked": int(len(table)), "alerts": int((table["status"] == "alert").sum()),
            "warnings": int((table["status"] == "warn").sum()),
            "overall": "alert" if (table["status"] == "alert").any() else "warn" if (table["status"] == "warn").any() else "ok",
            "worst_feature": table.iloc[0]["feature"] if len(table) else None,
            "worst_psi": float(table.iloc[0]["psi"]) if len(table) else None}


def prediction_drift(ref_proba: np.ndarray, cur_proba: np.ndarray) -> dict:
    ks = stats.ks_2samp(ref_proba, cur_proba)
    return {"psi": psi_numeric(ref_proba, cur_proba), "ks_stat": float(ks.statistic), "p_value": float(ks.pvalue),
            "ref_mean": float(np.mean(ref_proba)), "cur_mean": float(np.mean(cur_proba))}


def performance_monitor(y_ref, p_ref, y_cur, p_cur, threshold: float, cfg: dict) -> dict:
    """Compare live (labelled) performance with the reference; flags degradation beyond the tolerance."""
    ref, cur = compute_metrics(y_ref, p_ref, threshold), compute_metrics(y_cur, p_cur, threshold)
    tol = cfg["monitoring"]["performance_drop_tolerance"]
    drops = {k: ref[k] - cur[k] for k in ("pr_auc", "roc_auc", "f1", "recall", "precision")}
    degraded = [k for k, d in drops.items() if d > tol]
    return {"reference": {k: ref[k] for k in drops}, "current": {k: cur[k] for k in drops}, "drops": drops,
            "calibration_gap": {"ref_ece": ref["ece"], "cur_ece": cur["ece"]},
            "degraded_metrics": degraded, "status": "degraded" if degraded else "healthy", "n_current": int(len(y_cur))}


def evidently_report(reference: pd.DataFrame, current: pd.DataFrame, out_html: Path) -> Path | None:
    """Optional interactive Evidently data-drift report (works with evidently 0.4.x - 0.7.x APIs)."""
    try:
        try:
            from evidently import Report  # type: ignore  # >= 0.7
            from evidently.presets import DataDriftPreset  # type: ignore
            snap = Report([DataDriftPreset()]).run(current, reference)
            snap.save_html(str(out_html))
        except ImportError:
            from evidently.metric_preset import DataDriftPreset  # type: ignore  # < 0.7
            from evidently.report import Report  # type: ignore
            rep = Report(metrics=[DataDriftPreset()])
            rep.run(reference_data=reference, current_data=current)
            rep.save_html(str(out_html))
        return out_html
    except Exception:  # noqa: BLE001 - evidently missing or API changed: built-in report is still produced
        return None
