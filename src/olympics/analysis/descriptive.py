"""Descriptive statistics: central tendency, spread, shape, distributions."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def describe_numeric(df: pd.DataFrame, cols: list[str], by: str | None = None) -> pd.DataFrame:
    """mean / median / std / IQR / skew / kurtosis / missing% for each numeric column (optionally per group)."""
    def _one(frame: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for c in cols:
            x = frame[c].dropna().astype(float)
            if len(x) == 0:
                continue
            q1, q3 = x.quantile([0.25, 0.75])
            rows.append({
                "variable": c, "n": len(x), "missing_pct": round(100 * (1 - len(x) / len(frame)), 2),
                "mean": x.mean(), "median": x.median(), "std": x.std(ddof=1), "min": x.min(), "p05": x.quantile(0.05),
                "q1": q1, "q3": q3, "iqr": q3 - q1, "p95": x.quantile(0.95), "max": x.max(),
                "skew": stats.skew(x), "excess_kurtosis": stats.kurtosis(x),
                "cv": x.std(ddof=1) / x.mean() if x.mean() else np.nan,
            })
        return pd.DataFrame(rows)

    if by is None:
        return _one(df).round(3)
    parts = []
    for key, g in df.groupby(by, observed=True):
        out = _one(g)
        out.insert(0, by, key)
        parts.append(out)
    return pd.concat(parts, ignore_index=True).round(3) if parts else pd.DataFrame()


def describe_categorical(df: pd.DataFrame, cols: list[str], top: int = 15) -> pd.DataFrame:
    rows = []
    for c in cols:
        vc = df[c].value_counts(dropna=False)
        total = len(df)
        for value, n in vc.head(top).items():
            rows.append({"variable": c, "value": value, "count": int(n), "pct": round(100 * n / total, 2)})
    return pd.DataFrame(rows)


def medal_rate_by(df: pd.DataFrame, col: str, min_n: int = 100) -> pd.DataFrame:
    g = df.groupby(col, observed=True)["won_medal"].agg(entries="size", medals="sum")
    g["medal_rate"] = g["medals"] / g["entries"]
    return g[g["entries"] >= min_n].sort_values("medal_rate", ascending=False).reset_index()


def sport_physique_summary(df: pd.DataFrame, top_n: int = 15, min_n: int = 200) -> pd.DataFrame:
    g = df.groupby("Sport", observed=True).agg(
        entries=("ID", "size"), mean_age=("Age", "mean"), median_age=("Age", "median"), std_age=("Age", "std"),
        mean_height=("Height", "mean"), mean_weight=("Weight", "mean"), mean_bmi=("BMI", "mean"),
        medal_rate=("won_medal", "mean"))
    return g[g["entries"] >= min_n].sort_values("entries", ascending=False).head(top_n).round(3).reset_index()
