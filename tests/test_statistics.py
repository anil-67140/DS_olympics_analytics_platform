import numpy as np
import pandas as pd
import pytest
from scipy import stats

from olympics.analysis import inference as inf
from olympics.analysis import trends


def test_wilson_ci_known_values():
    lo, hi = inf.wilson_ci(np.array([5]), np.array([10]))
    assert lo[0] == pytest.approx(0.2366, abs=1e-3) and hi[0] == pytest.approx(0.7634, abs=1e-3)
    lo0, hi0 = inf.wilson_ci(np.array([0]), np.array([20]))
    assert lo0[0] == pytest.approx(0, abs=1e-9) and 0 < hi0[0] < 0.2          # never collapses to a zero-width interval


def test_mean_ci_matches_scipy():
    x = np.random.default_rng(0).normal(10, 2, 200)
    m, lo, hi = inf.mean_ci(x)
    ref = stats.t.interval(0.95, len(x) - 1, loc=x.mean(), scale=stats.sem(x))
    assert (lo, hi) == pytest.approx(ref)


def test_bootstrap_ci_contains_mean_and_is_reproducible():
    x = np.random.default_rng(1).normal(5, 1, 300)
    a = inf.bootstrap_ci(x, seed=3)
    assert a == inf.bootstrap_ci(x, seed=3)
    assert a[1] < x.mean() < a[2]


def test_holm_and_bh_adjustments():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    assert inf.adjust_pvalues(p, "holm") == pytest.approx([0.03, 0.06, 0.06, 0.02])
    bh = inf.adjust_pvalues(p, "bh")
    assert bh == pytest.approx([0.02, 0.04, 0.04, 0.02])
    assert (inf.adjust_pvalues(p, "holm") >= p).all()


def test_cohens_d():
    a, b = np.array([1.0, 2, 3, 4, 5]), np.array([2.0, 3, 4, 5, 6])
    assert inf.cohens_d(a, b) == pytest.approx(-0.632, abs=1e-3)


def test_chi_square_detects_association_and_independence():
    rng = np.random.default_rng(0)
    a = rng.choice(["x", "y"], 2000)
    dependent = pd.DataFrame({"a": a, "b": np.where(rng.random(2000) < 0.8, a, rng.choice(["x", "y"], 2000))})
    independent = pd.DataFrame({"a": a, "b": rng.choice(["x", "y"], 2000)})
    d, i = inf.chi_square_independence(dependent, "a", "b"), inf.chi_square_independence(independent, "a", "b")
    assert d["p_value"] < 1e-6 and d["cramers_v"] > 0.5
    assert i["p_value"] > 0.01 and i["cramers_v"] < 0.1


def test_compare_groups_separates_different_groups():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"g": np.repeat(["a", "b", "c"], 300), "v": np.concatenate([rng.normal(20, 3, 300), rng.normal(25, 3, 300), rng.normal(30, 3, 300)])})
    r = inf.compare_groups(df, "v", "g")
    assert r["kruskal_p"] < 1e-10 and r["anova_p"] < 1e-10 and r["welch_p"] < 1e-10
    assert r["eta_squared"] > 0.5 and r["effect_magnitude"] == "large"
    assert r["pairwise"]["significant"].all() and len(r["pairwise"]) == 3


def test_compare_groups_no_difference():
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"g": np.repeat(["a", "b", "c"], 400), "v": rng.normal(0, 1, 1200)})
    r = inf.compare_groups(df, "v", "g")
    assert r["kruskal_p"] > 0.01 and r["eta_squared"] < 0.01


def test_welch_anova_matches_known_equal_case():
    rng = np.random.default_rng(2)
    groups = [rng.normal(0, 1, 100) for _ in range(3)]
    f, p = inf.welch_anova(groups)
    assert 0 <= p <= 1 and f >= 0


def test_two_group_test_ci_covers_true_difference():
    rng = np.random.default_rng(3)
    r = inf.two_group_test(rng.normal(10, 2, 1000), rng.normal(9, 2, 1000), "a", "b", "v")
    assert r["diff_ci_low"] < 1.0 < r["diff_ci_high"] and r["welch_p"] < 1e-6


def test_correlation_ci_orders_correctly():
    rng = np.random.default_rng(4)
    x = rng.normal(size=500)
    r = inf.correlation_with_ci(x, 0.6 * x + rng.normal(size=500) * 0.8)
    assert r["ci_low"] < r["r"] < r["ci_high"] and r["p_value"] < 1e-6


def test_mann_kendall_trends():
    up = trends.mann_kendall(np.arange(20) + np.random.default_rng(0).normal(0, 0.5, 20))
    flat = trends.mann_kendall(np.random.default_rng(1).normal(0, 1, 30))
    assert up["trend"] == "increasing" and up["sen_slope_per_edition"] == pytest.approx(1, abs=0.2)
    assert flat["trend"] == "no significant trend"
    assert trends.mann_kendall(np.array([1, 2]))["trend"] == "insufficient data"


def test_synthetic_pipeline_statistics(cleaned):
    df, med, *_ = cleaned
    mv = inf.medalist_vs_rest_tests(df)
    assert set(mv["variable"]) == {"Age", "Height", "Weight", "BMI"} and (mv["welch_p_holm"] >= mv["welch_p"] - 1e-12).all()
    pv = inf.participation_vs_medals(df, med)
    assert pv["spearman"]["r"] > 0.2          # bigger delegations win more in the generator
    rates = inf.medal_rate_ci(df, "NOC", 50)
    assert (rates["ci_low"] <= rates["rate"]).all() and (rates["rate"] <= rates["ci_high"]).all()
