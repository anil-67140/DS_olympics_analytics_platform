"""Statistical inference: confidence intervals, hypothesis tests, effect sizes, multiple-testing control.

Every test returns the statistic, p-value, an effect size and a plain-English reading, because a p-value
alone says nothing about whether a difference *matters* (with 270k rows almost everything is "significant").
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats


# ------------------------------------------------------------------------------------ intervals
def wilson_ci(k: int | np.ndarray, n: int | np.ndarray, alpha: float = 0.05):
    """Wilson score interval for a proportion (far better than the normal approx for small n / extreme p)."""
    k, n = np.asarray(k, dtype=float), np.asarray(n, dtype=float)
    z = stats.norm.ppf(1 - alpha / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = k / n
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def bootstrap_ci(x: np.ndarray, stat=np.mean, n_boot: int = 2000, alpha: float = 0.05, seed: int = 42):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan, np.nan, np.nan
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = np.apply_along_axis(stat, 1, x[idx]) if stat not in (np.mean, np.median) else stat(x[idx], axis=1)
    return float(stat(x)), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def mean_ci(x: np.ndarray, alpha: float = 0.05):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    m, se = x.mean(), stats.sem(x)
    lo, hi = stats.t.interval(1 - alpha, len(x) - 1, loc=m, scale=se)
    return float(m), float(lo), float(hi)


def medal_rate_ci(df: pd.DataFrame, by: str = "NOC", min_entries: int = 200, alpha: float = 0.05) -> pd.DataFrame:
    """Medal rate (medals per entry) with Wilson CI per group — quantifies uncertainty in 'who is best'."""
    g = df.groupby(by, observed=True)["won_medal"].agg(entries="size", medals="sum").reset_index()
    g = g[g["entries"] >= min_entries].copy()
    g["rate"] = g["medals"] / g["entries"]
    g["ci_low"], g["ci_high"] = wilson_ci(g["medals"].values, g["entries"].values, alpha)
    return g.sort_values("rate", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------------------------ multiple testing
def adjust_pvalues(p: np.ndarray, method: str = "holm") -> np.ndarray:
    """Holm (FWER) or Benjamini-Hochberg (FDR) adjustment."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    if method == "holm":
        adj = np.maximum.accumulate((n - np.arange(n)) * ranked)
    elif method == "bh":
        adj = np.minimum.accumulate((n / (np.arange(n) + 1) * ranked)[::-1])[::-1]
    else:
        raise ValueError("method must be 'holm' or 'bh'")
    out = np.empty(n)
    out[order] = np.clip(adj, 0, 1)
    return out


# ------------------------------------------------------------------------------------ effect sizes
def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled else np.nan


def rank_biserial(u_stat: float, n1: int, n2: int) -> float:
    return float(1 - 2 * u_stat / (n1 * n2))


def cramers_v(chi2: float, n: int, shape: tuple[int, int]) -> float:
    """Bias-corrected Cramér's V (Bergsma 2013)."""
    r, k = shape
    phi2 = chi2 / n
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2c / denom)) if denom > 0 else np.nan


def _magnitude(value: float, small: float, medium: float, large: float) -> str:
    v = abs(value)
    return "negligible" if v < small else "small" if v < medium else "medium" if v < large else "large"


# ------------------------------------------------------------------------------------ categorical association
def chi_square_independence(df: pd.DataFrame, a: str, b: str, min_expected: float = 5.0, top_a: int | None = None) -> dict:
    sub = df[[a, b]].dropna()
    if top_a:
        sub = sub[sub[a].isin(sub[a].value_counts().head(top_a).index)]
    table = pd.crosstab(sub[a], sub[b])
    chi2, p, dof, expected = stats.chi2_contingency(table)
    v = cramers_v(chi2, int(table.values.sum()), table.shape)
    resid = (table - expected) / np.sqrt(expected)
    return {
        "test": "chi-square test of independence", "variables": f"{a} x {b}", "n": int(table.values.sum()),
        "chi2": float(chi2), "dof": int(dof), "p_value": float(p), "cramers_v": v,
        "effect_magnitude": _magnitude(v, 0.1, 0.3, 0.5),
        "cells_below_min_expected_pct": float((expected < min_expected).mean() * 100),
        "interpretation": (f"{a} and {b} are {'associated' if p < 0.05 else 'not clearly associated'} "
                           f"(Cramér's V={v:.3f}, {_magnitude(v, 0.1, 0.3, 0.5)} effect)."),
        "_table": table, "_std_residuals": resid,
    }


# ------------------------------------------------------------------------------------ k-group comparisons
def welch_anova(groups: list[np.ndarray]) -> tuple[float, float]:
    k = len(groups)
    n = np.array([len(g) for g in groups], float)
    m = np.array([g.mean() for g in groups])
    v = np.array([g.var(ddof=1) for g in groups])
    w = n / v
    grand = (w * m).sum() / w.sum()
    a = ((w * (m - grand) ** 2).sum()) / (k - 1)
    tmp = ((1 - w / w.sum()) ** 2 / (n - 1)).sum()
    b = 1 + 2 * (k - 2) / (k**2 - 1) * tmp
    f = a / b
    df2 = (k**2 - 1) / (3 * tmp)
    return float(f), float(stats.f.sf(f, k - 1, df2))


def compare_groups(df: pd.DataFrame, value: str, group: str, top_k: int = 12, min_n: int = 30,
                   alpha: float = 0.05, pairwise: bool = True, seed: int = 42) -> dict:
    """Compare a numeric variable across groups (e.g. age across sports) with the right battery of tests.

    Runs Levene (equal variances?), Shapiro on a sample (normality?), classic and Welch ANOVA, Kruskal-Wallis
    (rank-based; robust to the non-normal, skewed ages seen here) with eta-squared / epsilon-squared effect
    sizes, then Holm-adjusted pairwise Mann-Whitney tests with rank-biserial effect sizes.
    """
    sub = df[[value, group]].dropna()
    keep = sub[group].value_counts()
    keep = keep[keep >= min_n].head(top_k).index
    sub = sub[sub[group].isin(keep)]
    samples = {g: s[value].to_numpy(float) for g, s in sub.groupby(group, observed=True)}
    arrays = list(samples.values())
    n_total, k = sum(len(a) for a in arrays), len(arrays)
    if k < 2:
        return {"error": "need at least two groups with enough observations"}

    lev = stats.levene(*arrays)
    rng = np.random.default_rng(seed)
    sh_sample = rng.choice(sub[value].to_numpy(float), size=min(5000, len(sub)), replace=False)
    shapiro = stats.shapiro(sh_sample)
    f_stat, f_p = stats.f_oneway(*arrays)
    grand = np.concatenate(arrays).mean()
    eta2 = sum(len(a) * (a.mean() - grand) ** 2 for a in arrays) / ((np.concatenate(arrays) - grand) ** 2).sum()
    wf, wp = welch_anova(arrays)
    h_stat, h_p = stats.kruskal(*arrays)
    eps2 = h_stat * (n_total + 1) / (n_total**2 - 1)  # epsilon-squared for Kruskal-Wallis

    result = {
        "variable": value, "grouped_by": group, "groups": k, "n": int(n_total),
        "levene_p": float(lev.pvalue), "shapiro_p_sample": float(shapiro.pvalue),
        "anova_F": float(f_stat), "anova_p": float(f_p), "eta_squared": float(eta2),
        "welch_F": wf, "welch_p": wp,
        "kruskal_H": float(h_stat), "kruskal_p": float(h_p), "epsilon_squared": float(eps2),
        "effect_magnitude": _magnitude(eta2, 0.01, 0.06, 0.14),
        "assumption_note": ("Variances unequal / data non-normal -> prefer Welch ANOVA and Kruskal-Wallis."
                            if lev.pvalue < alpha or shapiro.pvalue < alpha else
                            "Assumptions look acceptable; classic ANOVA is reliable."),
        "interpretation": (f"{value} differs across {group} (Kruskal-Wallis p={h_p:.2e}, "
                           f"eta²={eta2:.3f} → {_magnitude(eta2, 0.01, 0.06, 0.14)} effect)."
                           if h_p < alpha else f"No significant difference in {value} across {group}."),
        "group_summary": pd.DataFrame({
            "group": list(samples), "n": [len(a) for a in arrays], "mean": [a.mean() for a in arrays],
            "median": [np.median(a) for a in arrays], "std": [a.std(ddof=1) for a in arrays]}).sort_values("median"),
    }
    if pairwise:
        rows = []
        for (ga, a), (gb, b) in itertools.combinations(samples.items(), 2):
            u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            rows.append({"group_a": ga, "group_b": gb, "median_a": np.median(a), "median_b": np.median(b),
                         "U": u, "p_raw": p, "rank_biserial": rank_biserial(u, len(a), len(b))})
        pw = pd.DataFrame(rows)
        pw["p_holm"] = adjust_pvalues(pw["p_raw"].values, "holm")
        pw["significant"] = pw["p_holm"] < alpha
        result["pairwise"] = pw.sort_values("p_holm")
    return result


# ------------------------------------------------------------------------------------ two-group tests
def two_group_test(a: np.ndarray, b: np.ndarray, label_a: str, label_b: str, variable: str, alpha: float = 0.05) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    t = stats.ttest_ind(a, b, equal_var=False)
    u, p_u = stats.mannwhitneyu(a, b, alternative="two-sided")
    d = cohens_d(a, b)
    diff = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    dof = se**4 / ((a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1) + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1))
    tcrit = stats.t.ppf(1 - alpha / 2, dof)
    return {
        "variable": variable, "group_a": label_a, "group_b": label_b, "n_a": len(a), "n_b": len(b),
        "mean_a": a.mean(), "mean_b": b.mean(), "mean_diff": diff,
        "diff_ci_low": diff - tcrit * se, "diff_ci_high": diff + tcrit * se,
        "welch_t": float(t.statistic), "welch_p": float(t.pvalue), "mannwhitney_p": float(p_u),
        "cohens_d": d, "effect_magnitude": _magnitude(d, 0.2, 0.5, 0.8),
        "interpretation": (f"{label_a} vs {label_b}: mean {variable} differs by {diff:+.2f} "
                           f"(95% CI {diff - tcrit * se:+.2f} to {diff + tcrit * se:+.2f}; d={d:.2f}, "
                           f"{_magnitude(d, 0.2, 0.5, 0.8)})."),
    }


def medalist_vs_rest_tests(df: pd.DataFrame, cols=("Age", "Height", "Weight", "BMI"), alpha: float = 0.05) -> pd.DataFrame:
    rows = []
    for c in cols:
        sub = df[[c, "won_medal"]].dropna()
        r = two_group_test(sub.loc[sub.won_medal == 1, c], sub.loc[sub.won_medal == 0, c], "medallists", "non-medallists", c, alpha)
        rows.append(r)
    out = pd.DataFrame(rows)
    out["welch_p_holm"] = adjust_pvalues(out["welch_p"].values, "holm")
    return out


# ------------------------------------------------------------------------------------ correlation
def correlation_with_ci(x: np.ndarray, y: np.ndarray, method: str = "spearman", alpha: float = 0.05) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    r, p = (stats.spearmanr(x, y) if method == "spearman" else stats.pearsonr(x, y))
    n = len(x)
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = 1 / np.sqrt(max(n - 3, 1))
    zc = stats.norm.ppf(1 - alpha / 2)
    return {"method": method, "n": int(n), "r": float(r), "p_value": float(p),
            "ci_low": float(np.tanh(z - zc * se)), "ci_high": float(np.tanh(z + zc * se))}


def participation_vs_medals(df: pd.DataFrame, medals: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Correlation between delegation size and medals won (per country per edition)."""
    part = df.groupby(["Games", "NOC"], observed=True)["ID"].nunique().rename("athletes")
    won = medals.groupby(["Games", "NOC"], observed=True).size().rename("medals_won")
    panel = pd.concat([part, won], axis=1).fillna({"medals_won": 0}).reset_index()
    panel = panel.dropna(subset=["athletes"])
    sp = correlation_with_ci(panel["athletes"], panel["medals_won"], "spearman", alpha)
    pe_log = correlation_with_ci(np.log1p(panel["athletes"]), np.log1p(panel["medals_won"]), "pearson", alpha)
    return {"spearman": sp, "pearson_log_log": pe_log, "panel_rows": int(len(panel)), "panel": panel,
            "interpretation": (f"Larger delegations win more medals (Spearman ρ={sp['r']:.2f}, "
                               f"95% CI {sp['ci_low']:.2f}–{sp['ci_high']:.2f}); correlation, not causation: "
                               "strong nations also send more athletes.")}
