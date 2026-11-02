import numpy as np
import pandas as pd
import pytest

from olympics.forecasting import medals as fc


@pytest.fixture(scope="module")
def panel(cleaned):
    _, med, *_ = cleaned
    return fc.build_panel(med, "Summer", 10)


def test_panel_shape_and_host_flags(panel):
    Y = panel["Y"]
    assert Y.shape[0] <= 10 and Y.shape[1] == len(panel["years"]) and (Y.values >= 0).all()
    assert panel["host"].shape == (Y.shape[0], Y.shape[1] + 1)       # one extra column: the edition being forecast


def test_backtest_never_trains_on_the_edition_it_predicts(panel, monkeypatch):
    seen = []
    orig = fc._naive

    def spy(Y, host_train, host_next):
        seen.append(Y.shape[1])
        return orig(Y, host_train, host_next)
    monkeypatch.setitem(fc.available_models.__globals__, "_naive", spy)
    fc.rolling_backtest(panel, holdout=3)
    T = panel["Y"].shape[1]
    assert seen == [T - 3, T - 2, T - 1]                              # training history strictly shorter than the target edition


def test_model_zoo_and_scores(panel):
    bt, store = fc.rolling_backtest(panel, 3)
    summ = fc.summarize_backtest(bt)
    assert len(summ) >= 8 and {"naive_last", "ensemble_mean", "poisson_glm", "hist_gbm_poisson"} <= set(summ["model"])
    assert summ["mae"].is_monotonic_increasing
    assert summ.loc[summ["model"] == "naive_last", "rel_mae_vs_naive"].iloc[0] == pytest.approx(1.0)


def test_forecast_is_valid(panel, cfg):
    bt, store = fc.rolling_backtest(panel, 3)
    best = fc.summarize_backtest(bt).iloc[0]["model"]
    q = fc.residual_interval_quantiles(store, best)
    out = fc.forecast_next(panel, best, q)
    assert (out["forecast"] >= 0).all() and (out["lower"] <= out["forecast"]).all() and (out["forecast"] <= out["upper"]).all()
    assert (out["edition"] == panel["years"][-1] + 4).all() and len(out) == panel["Y"].shape[0]


def test_baselines_on_a_trivial_series():
    Y = np.tile(np.array([[10.0, 10, 10, 10, 10]]), (2, 1))
    host = np.zeros((2, 6))
    assert fc._naive(Y, host[:, :5], host[:, 5]) == pytest.approx([10, 10])
    assert fc._mean3(Y, host[:, :5], host[:, 5]) == pytest.approx([10, 10])
    assert fc._ewma(Y, host[:, :5], host[:, 5]) == pytest.approx([10, 10])
    assert fc._theil_sen(Y, host[:, :5], host[:, 5]) == pytest.approx([10, 10])


def test_run_forecast_end_to_end(cleaned, cfg):
    res = fc.run_forecast(cleaned[1], cfg)
    assert res["best_model"] in set(res["summary"]["model"]) and 0 <= res["interval_coverage_in_backtest"] <= 1
