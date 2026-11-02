"""Medal-count forecasting with honest, leakage-free validation.

Validation is a *rolling-origin* backtest: for each of the last H editions the models see only earlier
editions, predict that edition, and are scored against what really happened. Random k-fold would leak the
future into the past, so it is deliberately not used.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import PoissonRegressor, Ridge

from ..data.schema import HOST_NOC
from ..io_utils import get_logger

log = get_logger()
L = 3  # number of lags used by the supervised models


# ---------------------------------------------------------------------------------------------- panel
def build_panel(medals: pd.DataFrame, season: str, top_k: int, exclude_years: list[int] | None = None) -> dict:
    m = medals[(medals["Season"] == season) & ~medals["Year"].isin(exclude_years or [])]
    years = sorted(m["Year"].unique())
    recent = years[-5:]
    candidates = (m[m["Year"].isin(years[-3:])].groupby("region").size().index)  # still active recently
    strength = m[m["Year"].isin(recent) & m["region"].isin(candidates)].groupby("region").size().sort_values(ascending=False)
    countries = list(strength.head(top_k).index)
    Y = (m[m["region"].isin(countries)].groupby(["region", "Year"]).size().unstack(fill_value=0)
         .reindex(index=countries, columns=years, fill_value=0))
    noc_region = m.drop_duplicates("NOC").set_index("NOC")["region"].to_dict()
    host = pd.DataFrame(0, index=countries, columns=years + [max(years) + 4])
    for y in host.columns:
        h = HOST_NOC.get((y, season))
        if h and noc_region.get(h) in host.index:
            host.loc[noc_region[h], y] = 1
    return {"Y": Y, "host": host, "years": years, "season": season}


# ---------------------------------------------------------------------------------------------- models
def _supervised(Y: np.ndarray, host: np.ndarray):
    """Rows: every (country, edition j>=L). Features: lag1..lag3, mean3, long-run mean, host flag."""
    X, y = [], []
    for c in range(Y.shape[0]):
        for j in range(L, Y.shape[1]):
            lags = Y[c, j - L:j][::-1]
            X.append([*np.log1p(lags), np.log1p(lags.mean()), np.log1p(Y[c, :j].mean()), host[c, j]])
            y.append(Y[c, j])
    return np.array(X), np.array(y, dtype=float)


def _next_features(Y: np.ndarray, host_next: np.ndarray) -> np.ndarray:
    lags = Y[:, -L:][:, ::-1]
    return np.column_stack([np.log1p(lags), np.log1p(lags.mean(1)), np.log1p(Y.mean(1)), host_next])


def _naive(Y, host_train, host_next): return Y[:, -1].astype(float)
def _mean3(Y, host_train, host_next): return Y[:, -3:].mean(axis=1)


def _ewma(Y, host_train, host_next, alpha: float = 0.5):
    w = (1 - alpha) ** np.arange(Y.shape[1])[::-1]
    return (Y * w).sum(axis=1) / w.sum()


def _theil_sen(Y, host_train, host_next, window: int = 6):
    out = []
    for row in Y:
        r = row[-window:]
        if len(r) < 3:
            out.append(r.mean())
            continue
        slope, intercept, *_ = stats.theilslopes(r, np.arange(len(r)))
        out.append(max(0.0, intercept + slope * len(r)))
    return np.array(out)


def _poisson(Y, host_train, host_next):
    X, y = _supervised(Y, host_train)
    model = PoissonRegressor(alpha=1e-3, max_iter=500).fit(X, y)
    return model.predict(_next_features(Y, host_next))


def _ridge(Y, host_train, host_next):
    X, y = _supervised(Y, host_train)
    model = Ridge(alpha=1.0).fit(X, np.log1p(y))
    return np.expm1(model.predict(_next_features(Y, host_next))).clip(min=0)


def _rf(Y, host_train, host_next):
    X, y = _supervised(Y, host_train)
    model = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=42, n_jobs=-1).fit(X, y)
    return model.predict(_next_features(Y, host_next))


def _hgb(Y, host_train, host_next):
    X, y = _supervised(Y, host_train)
    model = HistGradientBoostingRegressor(loss="poisson", max_iter=150, learning_rate=0.06, max_depth=3,
                                          min_samples_leaf=10, random_state=42).fit(X, y)
    return model.predict(_next_features(Y, host_next))


def _ets(Y, host_train, host_next):  # optional: needs statsmodels
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    preds = []
    for row in Y:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = ExponentialSmoothing(row.astype(float) + 0.5, trend="add", damped_trend=True).fit()
            preds.append(max(0.0, float(fit.forecast(1)[0]) - 0.5))
        except Exception:  # noqa: BLE001
            preds.append(row[-3:].mean())
    return np.array(preds)


def _arima(Y, host_train, host_next):  # optional: needs statsmodels
    from statsmodels.tsa.arima.model import ARIMA
    preds = []
    for row in Y:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                preds.append(max(0.0, float(ARIMA(row.astype(float), order=(1, 0, 0)).fit().forecast(1)[0])))
        except Exception:  # noqa: BLE001
            preds.append(row[-3:].mean())
    return np.array(preds)


def available_models() -> dict:
    models = {"naive_last": _naive, "mean_last3": _mean3, "ewma": _ewma, "theil_sen_trend": _theil_sen,
              "poisson_glm": _poisson, "ridge_lag": _ridge, "random_forest_lag": _rf, "hist_gbm_poisson": _hgb}
    try:
        import statsmodels  # noqa: F401
        models["ets_damped"] = _ets
        models["arima_1_0_0"] = _arima
    except ImportError:
        log.info("statsmodels not installed: ETS/ARIMA skipped (installed automatically on Colab)")
    return models


ENSEMBLE_MEMBERS = ("poisson_glm", "ridge_lag", "random_forest_lag", "hist_gbm_poisson")


# ---------------------------------------------------------------------------------------------- metrics
def _poisson_deviance(y, mu):
    mu = np.clip(mu, 1e-6, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0)
    return float(2 * np.mean(term - (y - mu)))


def _scores(y: np.ndarray, pred: np.ndarray, naive_mae: float) -> dict:
    err = y - pred
    mae = float(np.mean(np.abs(err)))
    return {"mae": mae, "rmse": float(np.sqrt(np.mean(err**2))),
            "smape": float(100 * np.mean(2 * np.abs(err) / (np.abs(y) + np.abs(pred) + 1e-9))),
            "rel_mae_vs_naive": mae / naive_mae if naive_mae else np.nan, "bias": float(-err.mean()),
            "spearman": float(stats.spearmanr(y, pred).statistic) if np.std(pred) > 0 and np.std(y) > 0 else np.nan,
            "poisson_deviance": _poisson_deviance(y, pred)}


# ---------------------------------------------------------------------------------------------- backtest
def rolling_backtest(panel: dict, holdout: int) -> tuple[pd.DataFrame, dict]:
    Y_all, host_all = panel["Y"].to_numpy(float), panel["host"].to_numpy(float)
    years = panel["years"]
    T = Y_all.shape[1]
    models = available_models()
    rows, preds_store = [], {name: [] for name in [*models, "ensemble_mean"]}
    actual_store = []
    for t in range(T - holdout, T):
        Ytr, y_true = Y_all[:, :t], Y_all[:, t]
        pred_t = {}
        for name, fn in models.items():
            try:
                pred_t[name] = np.clip(fn(Ytr, host_all[:, :t], host_all[:, t]), 0, None)
            except Exception as exc:  # noqa: BLE001
                log.warning("model %s failed on %s: %s", name, years[t], exc)
                pred_t[name] = np.full(len(y_true), np.nan)
        members = [pred_t[m] for m in ENSEMBLE_MEMBERS if m in pred_t]
        pred_t["ensemble_mean"] = np.mean(members, axis=0)
        naive_mae = float(np.mean(np.abs(y_true - pred_t["naive_last"])))
        for name, p in pred_t.items():
            if np.isnan(p).any():
                continue
            rows.append({"edition": years[t], "model": name, **_scores(y_true, p, naive_mae)})
            preds_store[name].append(p)
        actual_store.append(y_true)
    return pd.DataFrame(rows), {"preds": preds_store, "actual": actual_store}


def summarize_backtest(bt: pd.DataFrame) -> pd.DataFrame:
    agg = bt.groupby("model").agg(mae=("mae", "mean"), rmse=("rmse", "mean"), smape=("smape", "mean"),
                                  rel_mae_vs_naive=("rel_mae_vs_naive", "mean"), bias=("bias", "mean"),
                                  spearman=("spearman", "mean"), poisson_deviance=("poisson_deviance", "mean"),
                                  mae_std_across_editions=("mae", "std"), editions=("edition", "nunique"))
    return agg.sort_values("mae").round(4).reset_index()


def residual_interval_quantiles(store: dict, model: str, interval=(0.05, 0.95)) -> tuple[float, float]:
    pred = np.concatenate(store["preds"][model])
    act = np.concatenate(store["actual"])
    z = (act - pred) / np.sqrt(pred + 1.0)  # variance-stabilised residuals (counts are heteroscedastic)
    return float(np.quantile(z, interval[0])), float(np.quantile(z, interval[1]))


def forecast_next(panel: dict, model_name: str, q: tuple[float, float]) -> pd.DataFrame:
    Y, host = panel["Y"].to_numpy(float), panel["host"].to_numpy(float)
    next_year = panel["years"][-1] + 4
    models = available_models()
    if model_name == "ensemble_mean":
        pred = np.mean([models[m](Y, host[:, :-1], host[:, -1]) for m in ENSEMBLE_MEMBERS if m in models], axis=0)
    else:
        pred = models[model_name](Y, host[:, :-1], host[:, -1])
    pred = np.clip(pred, 0, None)
    out = pd.DataFrame({"country": panel["Y"].index, "edition": next_year, "forecast": pred,
                        "lower": np.clip(pred + q[0] * np.sqrt(pred + 1), 0, None),
                        "upper": pred + q[1] * np.sqrt(pred + 1), "last_actual": Y[:, -1],
                        "host": host[:, -1].astype(int), "model": model_name})
    return out.sort_values("forecast", ascending=False).reset_index(drop=True)


def run_forecast(medals: pd.DataFrame, cfg: dict) -> dict:
    fc = cfg["forecast"]
    panel = build_panel(medals, fc["season"], fc["top_k_countries"], cfg["data"].get("exclude_years"))
    holdout = min(fc["holdout_editions"], len(panel["years"]) - (L + 4))
    bt, store = rolling_backtest(panel, holdout)
    summary = summarize_backtest(bt)
    best = summary.iloc[0]["model"]
    q = residual_interval_quantiles(store, best, tuple(fc["interval"]))
    forecast = forecast_next(panel, best, q)
    # empirical coverage of the interval on the backtest (optimistic: same residuals used to set the width)
    pred = np.concatenate(store["preds"][best])
    act = np.concatenate(store["actual"])
    cover = float(np.mean((act >= np.clip(pred + q[0] * np.sqrt(pred + 1), 0, None)) & (act <= pred + q[1] * np.sqrt(pred + 1))))
    return {"panel": panel, "backtest": bt, "summary": summary, "best_model": best, "forecast": forecast,
            "interval_coverage_in_backtest": cover, "holdout_editions": holdout,
            "backtest_years": panel["years"][-holdout:]}


# ---------------------------------------------------------------------------------------------- plots
def plot_forecast(book, result: dict, n_countries: int = 6):
    import matplotlib.pyplot as plt
    Y, years = result["panel"]["Y"], result["panel"]["years"]
    fc = result["forecast"].head(n_countries)
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True)
    for ax, (_, row) in zip(axes.ravel(), fc.iterrows()):
        ax.plot(years, Y.loc[row["country"]].values, marker="o", ms=3)
        ax.errorbar([row["edition"]], [row["forecast"]], yerr=[[row["forecast"] - row["lower"]], [row["upper"] - row["forecast"]]],
                    fmt="s", color="#E4572E", capsize=3)
        ax.set_title(row["country"])
    fig.suptitle(f"Next-edition medal forecast ({result['best_model']}, 90% interval)", fontweight="bold")
    return book.save(fig, "12_forecast_fan", "Medal forecast with uncertainty",
                     "History of medals per edition (blue) and the forecast for the next Games with a prediction interval (red).",
                     "Planning and benchmarking need a number *and* its uncertainty; the interval width comes from errors the model actually made when forecasting past editions.")


def plot_model_comparison(book, summary: pd.DataFrame):
    import matplotlib.pyplot as plt
    s = summary.sort_values("mae", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#E4572E" if m == "naive_last" else "#2E5EAA" for m in s["model"]]
    ax.barh(s["model"], s["mae"], color=colors)
    ax.set_xlabel("Mean absolute error (medals) over backtest editions")
    ax.set_title("Forecast model comparison (lower is better)")
    return book.save(fig, "13_forecast_model_comparison", "Which forecasting method wins?",
                     "Average error of each method when predicting held-out editions; the red bar is the 'same as last time' baseline.",
                     "A model that cannot beat the naive baseline adds complexity without value - this chart keeps every method honest.")
