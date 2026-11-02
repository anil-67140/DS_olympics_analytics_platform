"""Time-series analysis of participation and medals, plus the host-advantage test."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def mann_kendall(y: np.ndarray, alpha: float = 0.05) -> dict:
    """Mann-Kendall monotonic trend test + Theil-Sen slope (robust to outliers and non-normality)."""
    y = np.asarray(y, float)
    y = y[~np.isnan(y)]
    n = len(y)
    if n < 4:
        return {"n": n, "trend": "insufficient data"}
    s = sum(np.sign(y[j] - y[i]) for i in range(n - 1) for j in range(i + 1, n))
    _, counts = np.unique(y, return_counts=True)
    var_s = (n * (n - 1) * (2 * n + 5) - sum(c * (c - 1) * (2 * c + 5) for c in counts)) / 18
    z = (s - np.sign(s)) / np.sqrt(var_s) if s != 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    slope, intercept, lo, hi = stats.theilslopes(y, np.arange(n))
    return {"n": n, "S": float(s), "z": float(z), "p_value": float(p), "sen_slope_per_edition": float(slope),
            "sen_slope_ci_low": float(lo), "sen_slope_ci_high": float(hi),
            "kendall_tau": float(stats.kendalltau(np.arange(n), y).statistic),
            "trend": ("increasing" if z > 0 else "decreasing") if p < alpha else "no significant trend"}


def lag1_autocorrelation(y: np.ndarray) -> float:
    y = np.asarray(y, float)
    y = y - y.mean()
    return float((y[:-1] * y[1:]).sum() / (y**2).sum()) if len(y) > 2 else np.nan


def participation_trends(df: pd.DataFrame, season: str = "Summer", alpha: float = 0.05) -> dict:
    sub = df[df["Season"] == season]
    ed = sub.groupby("Year").agg(athletes=("ID", "nunique"), nations=("NOC", "nunique"), events=("Event", "nunique"),
                                 sports=("Sport", "nunique"), mean_age=("Age", "mean")).reset_index()
    women = sub.drop_duplicates(["ID", "Year"]).groupby("Year")["Sex"].apply(lambda s: (s == "F").mean() * 100)
    ed["women_pct"] = ed["Year"].map(women)
    tests = {c: mann_kendall(ed[c].to_numpy(), alpha) for c in ("athletes", "nations", "events", "women_pct", "mean_age")}
    for c, t in tests.items():
        t["lag1_autocorr_of_levels"] = lag1_autocorrelation(ed[c].to_numpy())
    growth = ed[["Year", "athletes", "nations", "events"]].set_index("Year").pct_change().add_suffix("_growth").reset_index()
    out = {"season": season, "editions": ed, "tests": tests, "growth": growth}
    try:  # augmented Dickey-Fuller only if statsmodels is installed (Colab has it)
        from statsmodels.tsa.stattools import adfuller
        out["adf_athletes_p"] = float(adfuller(ed["athletes"].to_numpy(), maxlag=2)[1]) if len(ed) > 8 else None
    except Exception:  # noqa: BLE001
        out["adf_athletes_p"] = None
    return out


def host_advantage(df: pd.DataFrame, medals: pd.DataFrame, season: str = "Summer", alpha: float = 0.05) -> dict:
    """Do hosts win a larger share of all medals at home than at their other Games? (paired Wilcoxon)."""
    m = medals[medals["Season"] == season]
    share = (m.groupby(["Games", "NOC"]).size() / m.groupby("Games").size()).rename("share").reset_index()
    hosts = df[(df["Season"] == season) & (df["is_host"] == 1)][["Games", "NOC"]].drop_duplicates()
    rows = []
    for noc, h in hosts.groupby("NOC"):
        s = share[share["NOC"] == noc]
        home = s[s["Games"].isin(h["Games"])]["share"]
        other = s[~s["Games"].isin(h["Games"])]["share"]
        if len(home) and len(other):
            rows.append({"NOC": noc, "home_share_pct": 100 * home.mean(), "other_share_pct": 100 * other.mean(),
                         "n_home_games": len(home), "n_other_games": len(other)})
    tbl = pd.DataFrame(rows)
    if len(tbl) < 5:
        return {"table": tbl, "error": "too few host countries for a test"}
    diff = tbl["home_share_pct"] - tbl["other_share_pct"]
    w = stats.wilcoxon(tbl["home_share_pct"], tbl["other_share_pct"], alternative="greater")
    lo, hi = np.quantile(np.random.default_rng(42).choice(diff, (4000, len(diff))).mean(axis=1), [alpha / 2, 1 - alpha / 2])
    return {"table": tbl.sort_values("home_share_pct", ascending=False), "hosts": int(len(tbl)),
            "mean_uplift_pp": float(diff.mean()), "uplift_ci_low": float(lo), "uplift_ci_high": float(hi),
            "wilcoxon_stat": float(w.statistic), "wilcoxon_p_one_sided": float(w.pvalue),
            "interpretation": (f"Hosts win on average {diff.mean():+.2f} percentage points more of all medals at home "
                               f"(bootstrap 95% CI {lo:+.2f} to {hi:+.2f}; one-sided Wilcoxon p={w.pvalue:.3g}).")}
