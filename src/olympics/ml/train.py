"""Leakage-safe training pipeline: compare many models -> calibrate -> tune threshold -> evaluate once -> version."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline

from ..analysis.viz import FigureBook
from ..checkpoint import cached_json
from ..config import Paths, config_fingerprint, to_jsonable
from ..io_utils import get_logger, read_json, save_table, write_json
from ..tracking import Tracker
from . import plots
from .explain import global_importance
from .features import build_history_lookup, make_preprocessor, to_model_frame
from .metrics import cluster_bootstrap_ci, compute_metrics, curves
from .model_wrapper import MedalModel, ProbabilityCalibrator
from .models import ModelSpec, build_models
from .registry import new_version, promote, save_model_version
from .split import make_splits, split_report
from .threshold import optimize_threshold

log = get_logger()


# ---------------------------------------------------------------------------------------------- helpers
def _pipeline(spec: ModelSpec, seed: int) -> Pipeline:
    return Pipeline([("prep", make_preprocessor(seed)), ("clf", clone(spec.estimator))])


def _cap(idx: np.ndarray, y: np.ndarray, max_rows: int | None, seed: int) -> np.ndarray:
    if max_rows is None or len(idx) <= max_rows:
        return idx
    keep, _ = train_test_split(idx, train_size=max_rows, stratify=y[idx], random_state=seed)
    return np.sort(keep)


def _selected_specs(cfg: dict) -> list[ModelSpec]:
    specs = build_models(cfg["ml"]["seed"], cfg["ml"]["profile"], cfg["ml"].get("n_jobs", -1))
    wanted = cfg["ml"].get("models")
    return [s for s in specs if s.is_baseline or not wanted or s.name in wanted]


def save_splits(paths: Paths, splits: dict[str, np.ndarray]) -> Path:
    return write_json(paths.processed / "splits.json", {k: v.tolist() for k, v in splits.items()})


def load_splits(paths: Paths) -> dict[str, np.ndarray]:
    raw = read_json(paths.processed / "splits.json")
    return {k: np.asarray(v, dtype=int) for k, v in raw.items()}


# ---------------------------------------------------------------------------------------------- 1. compare
def _evaluate_candidate(spec: ModelSpec, X, y, splits, cfg, preds_dir: Path) -> dict:
    seed = cfg["ml"]["seed"]
    cap = min(c for c in (spec.max_rows, cfg["ml"].get("compare_max_rows")) if c) if (spec.max_rows or cfg["ml"].get("compare_max_rows")) else None
    tr = _cap(splits["train"], y, cap, seed)
    va = splits["val"]
    try:
        pipe = _pipeline(spec, seed)
        t0 = time.time()
        pipe.fit(X.iloc[tr], y[tr])
        fit_s = time.time() - t0
        t1 = time.time()
        proba = pipe.predict_proba(X.iloc[va])[:, 1]
        pred_s = time.time() - t1
    except Exception as exc:  # noqa: BLE001 - one broken candidate must not stop the comparison
        log.warning("%s failed: %s", spec.name, exc)
        return {"model": spec.name, "family": spec.family, "is_baseline": spec.is_baseline, "error": str(exc)}
    np.save(preds_dir / f"{spec.name}.npy", proba)
    yv = y[va]
    best = optimize_threshold(yv, proba, "fbeta", beta=1.0)
    m = compute_metrics(yv, proba, 0.5)
    return {"model": spec.name, "family": spec.family, "is_baseline": spec.is_baseline, "train_rows": int(len(tr)),
            "fit_seconds": round(fit_s, 2), "predict_seconds": round(pred_s, 2), "best_threshold": best["threshold"],
            "f1_best_threshold": best["objective_value"], **m, "curves": curves(yv, proba), "error": None}


def compare_models(cfg: dict, paths: Paths, ml_df: pd.DataFrame, splits: dict, tracker: Tracker, force: bool = False) -> pd.DataFrame:
    X, y = to_model_frame(ml_df), ml_df[cfg["ml"]["target"]].to_numpy()
    cache, preds_dir = paths.metrics / "candidates", paths.metrics / "val_predictions"
    cache.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)
    results = []
    specs = _selected_specs(cfg)
    for i, spec in enumerate(specs, 1):
        log.info("[%d/%d] model: %s", i, len(specs), spec.name)
        path = cache / f"{spec.name}.json"
        res = cached_json(path, lambda s=spec: _evaluate_candidate(s, X, y, splits, cfg, preds_dir), force=force)
        if res.get("error"):
            path.unlink(missing_ok=True)  # allow a retry on the next run
        else:
            tracker.log_run(f"compare-{spec.name}", {"model": spec.name, "family": spec.family, "profile": cfg["ml"]["profile"], "train_rows": res["train_rows"]},
                            {k: v for k, v in res.items() if isinstance(v, (int, float)) and k not in ("n",)}, tags={"stage": "comparison"})
        results.append(res)
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "curves"} for r in results])
    df.to_pickle(paths.metrics / "comparison_raw.pkl")
    write_json(paths.metrics / "comparison_curves.json", {r["model"]: r["curves"] for r in results if r.get("curves")})
    ok = df[df["error"].isna()] if "error" in df else df
    ranked = ok.sort_values(cfg["ml"]["primary_metric"], ascending=False).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    save_table(ranked.drop(columns=["error"], errors="ignore"), paths.tables / "model_comparison_validation.csv")
    failed = df[df["error"].notna()] if "error" in df else pd.DataFrame()
    if len(failed):
        save_table(failed[["model", "error"]], paths.tables / "model_comparison_failed.csv")
    return ranked


# ---------------------------------------------------------------------------------------------- 2. stability (CV)
def cross_validate_top(cfg: dict, paths: Paths, ml_df: pd.DataFrame, splits: dict, ranked: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    k = cfg["ml"]["cv_top_k"]
    if k <= 0:
        return pd.DataFrame()
    X, y = to_model_frame(ml_df), ml_df[cfg["ml"]["target"]].to_numpy()
    groups = ml_df[cfg["ml"]["split"]["group_cols"]].astype(str).agg("|".join, axis=1).to_numpy()
    specs = {s.name: s for s in _selected_specs(cfg)}
    top = [m for m in ranked.loc[~ranked["is_baseline"], "model"].head(k)]
    rows = []
    for name in top:
        spec = specs[name]

        def _run(spec=spec):
            tr = _cap(splits["train"], y, min(spec.max_rows or 10**9, cfg["ml"]["compare_max_rows"]), cfg["ml"]["seed"])
            sg = StratifiedGroupKFold(n_splits=cfg["ml"]["cv_folds"], shuffle=True, random_state=cfg["ml"]["seed"])
            folds = []
            for a, b in sg.split(np.zeros(len(tr)), y[tr], groups[tr]):
                pipe = _pipeline(spec, cfg["ml"]["seed"]).fit(X.iloc[tr[a]], y[tr[a]])
                p = pipe.predict_proba(X.iloc[tr[b]])[:, 1]
                m = compute_metrics(y[tr[b]], p, 0.5)
                folds.append({"pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"], "f1": m["f1"], "brier": m["brier"]})
            f = pd.DataFrame(folds)
            return {"model": spec.name, "folds": len(f), **{f"{c}_mean": float(f[c].mean()) for c in f},
                    **{f"{c}_std": float(f[c].std(ddof=1)) for c in f}}

        rows.append(cached_json(paths.metrics / "cv" / f"{name}.json", _run, force=force))
    out = pd.DataFrame(rows)
    save_table(out, paths.tables / "cross_validation_top_models.csv")
    return out


# ---------------------------------------------------------------------------------------------- 3. final model
def slice_metrics(frame: pd.DataFrame, y: np.ndarray, p: np.ndarray, thr: float, top_sports: int = 12) -> pd.DataFrame:
    f = frame.assign(_y=y, _p=p).copy()
    f["Era"] = (f["Year"] // 20 * 20).astype(str) + "s"
    slices = {"Season": f["Season"], "Sex": f["Sex"], "Era": f["Era"],
              "Sport": f["Sport"].where(f["Sport"].isin(f["Sport"].value_counts().head(top_sports).index))}
    rows = [{"slice": "ALL", "value": "all", **_slice_row(f["_y"], f["_p"], thr)}]
    for name, col in slices.items():
        for val, idx in f.groupby(col, observed=True).groups.items():
            rows.append({"slice": name, "value": val, **_slice_row(f.loc[idx, "_y"], f.loc[idx, "_p"], thr)})
    return pd.DataFrame(rows)


def _slice_row(y: pd.Series, p: pd.Series, thr: float) -> dict:
    y, p = y.to_numpy(), p.to_numpy()
    out = {"n": len(y), "positives": int(y.sum()), "prevalence": float(y.mean()), "pr_auc": np.nan, "roc_auc": np.nan}
    pred = p >= thr
    out["recall"] = float(pred[y == 1].mean()) if y.sum() else np.nan
    out["precision"] = float(y[pred].mean()) if pred.any() else np.nan
    if 0 < y.sum() < len(y):
        m = compute_metrics(y, p, thr)
        out["pr_auc"], out["roc_auc"] = m["pr_auc"], m["roc_auc"]
    return out


def _split_val(cfg: dict, ml_df: pd.DataFrame, va: np.ndarray, y: np.ndarray):
    groups = ml_df.iloc[va][cfg["ml"]["split"]["group_cols"]].astype(str).agg("|".join, axis=1).to_numpy()
    sg = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=cfg["ml"]["seed"] + 7)
    a, b = next(iter(sg.split(np.zeros(len(va)), y[va], groups)))
    return va[a], va[b]


def finalize_model(cfg: dict, paths: Paths, ml_df: pd.DataFrame, splits: dict, ranked: pd.DataFrame, cv: pd.DataFrame,
                   clean_df: pd.DataFrame, tracker: Tracker, data_fp: str, do_promote: bool = True) -> dict:
    seed = cfg["ml"]["seed"]
    X, y = to_model_frame(ml_df), ml_df[cfg["ml"]["target"]].to_numpy()
    champion = ranked.loc[~ranked["is_baseline"]].iloc[0]
    spec = {s.name: s for s in _selected_specs(cfg)}[champion["model"]]
    log.info("Champion on validation: %s (PR-AUC %.4f)", spec.name, champion["pr_auc"])

    tr = _cap(splits["train"], y, spec.max_rows, seed)
    pipe = _pipeline(spec, seed).fit(X.iloc[tr], y[tr])

    # -- calibration: fit on one half of validation, choose method + threshold on the other half
    cal_idx, thr_idx = _split_val(cfg, ml_df, splits["val"], y)
    p_cal_fit, p_thr = pipe.predict_proba(X.iloc[cal_idx])[:, 1], pipe.predict_proba(X.iloc[thr_idx])[:, 1]
    rows, calibrators = [], {}
    for method in cfg["ml"]["calibration_methods"]:
        cal = ProbabilityCalibrator(method).fit(p_cal_fit, y[cal_idx])
        calibrators[method] = cal
        m = compute_metrics(y[thr_idx], cal.transform(p_thr), 0.5)
        rows.append({"method": method, "log_loss": m["log_loss"], "brier": m["brier"], "ece": m["ece"], "pr_auc": m["pr_auc"]})
    cal_cmp = pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
    save_table(cal_cmp, paths.tables / "calibration_comparison.csv")
    best_cal = cal_cmp.loc[0, "method"]
    calibrator = calibrators[best_cal]

    # -- threshold on calibrated probabilities of the held-out half
    tcfg = cfg["ml"]["threshold"]
    chosen = optimize_threshold(y[thr_idx], calibrator.transform(p_thr), tcfg["strategy"], tcfg["beta"], tcfg["cost_fp"], tcfg["cost_fn"], tcfg["target_recall"])
    strat_rows = []
    for strat in ("fbeta", "cost", "precision_at_recall", "youden"):
        o = optimize_threshold(y[thr_idx], calibrator.transform(p_thr), strat, tcfg["beta"], tcfg["cost_fp"], tcfg["cost_fn"], tcfg["target_recall"])
        strat_rows.append({"strategy": strat, "objective": o["objective"], "threshold": o["threshold"], "precision": o["precision"], "recall": o["recall"]})
    save_table(pd.DataFrame(strat_rows), paths.tables / "threshold_strategies.csv")
    thr = chosen["threshold"]

    model = MedalModel(pipe, calibrator, thr, list(X.columns))

    # -- test set: touched exactly once, after every choice above is frozen
    te = splits["test"]
    p_raw, p_final = model.raw_proba(X.iloc[te]), model.predict_proba(X.iloc[te])
    groups_te = ml_df.iloc[te][cfg["ml"]["split"]["group_cols"]].astype(str).agg("|".join, axis=1).to_numpy()
    test_m = compute_metrics(y[te], p_final, thr)
    test_m_at_half = compute_metrics(y[te], p_final, 0.5)
    test_raw_m = compute_metrics(y[te], p_raw, 0.5)
    ci = {m: cluster_bootstrap_ci(y[te], p_final, groups_te, m, n_boot=150 if cfg["ml"]["profile"] == "fast" else 300, threshold=thr)
          for m in ("pr_auc", "roc_auc", "f1", "precision", "recall")}
    base_m = compute_metrics(y[te], np.full(len(te), y[splits["train"]].mean()), 0.5)

    # -- slices, importance, figures
    sl = slice_metrics(ml_df.iloc[te], y[te], p_final, thr, cfg["stats"]["top_sports"])
    save_table(sl, paths.tables / "slice_metrics.csv")
    imp = global_importance(model, X.iloc[splits["val"]], y[splits["val"]], n_repeats=3 if cfg["ml"]["profile"] == "fast" else 5, seed=seed)
    save_table(imp, paths.tables / "permutation_importance.csv")

    book = FigureBook(paths.figures)
    curve_store = read_json(paths.metrics / "comparison_curves.json", {})
    top_names = [m for m in ranked.loc[~ranked["is_baseline"], "model"].head(5) if m in curve_store]
    plots.plot_model_comparison(book, ranked)
    plots.plot_pr_roc(book, curve_store, top_names, float(y[splits["val"]].mean()))
    plots.plot_calibration(book, y[te], p_raw, p_final, "raw model", f"calibrated ({best_cal})")
    plots.plot_confusion(book, test_m)
    plots.plot_threshold(book, chosen["table"], thr)
    plots.plot_importance(book, imp)
    plots.plot_slices(book, sl)

    # -- forward-looking sanity check: train on old editions, test on the newest ones
    temporal = temporal_backtest(cfg, ml_df, X, y, spec, model.threshold)
    write_json(paths.metrics / "temporal_backtest.json", temporal)

    version = new_version(data_fp, config_fingerprint(cfg, ["ml", "data"]))
    meta = {
        "version": version, "algorithm": spec.name, "family": spec.family, "calibration": best_cal, "threshold": thr,
        "threshold_strategy": f"{chosen['strategy']}: {chosen['objective']}", "test_metrics": test_m,
        "test_metrics_at_0_5": test_m_at_half, "test_metrics_uncalibrated": test_raw_m, "baseline_test_metrics": base_m,
        "test_confidence_intervals_95": ci, "validation_pr_auc": float(champion["pr_auc"]), "temporal_backtest": temporal,
        "data_fingerprint": data_fp, "split_mode": cfg["ml"]["split"]["mode"], "split_groups": cfg["ml"]["split"]["group_cols"],
        "split_sizes": {k: int(len(v)) for k, v in splits.items()}, "trained_rows": int(len(tr)),
        "cv_stability": cv[cv["model"] == spec.name].to_dict("records") if len(cv) else [],
        "params": {"ml": cfg["ml"]},
    }
    vdir = save_model_version(paths.models, model, meta, build_history_lookup(clean_df))
    if do_promote:
        promote(paths.models, version)
    write_json(paths.metrics / "final_model_metrics.json", {k: v for k, v in meta.items() if k != "params"})
    tracker.log_run(f"champion-{spec.name}", {"model": spec.name, "version": version, "calibration": best_cal, "threshold": thr},
                    {f"test_{k}": v for k, v in test_m.items() if isinstance(v, (int, float))}, tags={"stage": "champion"})
    log.info("Saved %s | test PR-AUC %.4f (baseline %.4f) | F1 %.4f | threshold %.3f", version, test_m["pr_auc"], base_m["pr_auc"], test_m["f1"], thr)
    return {"version": version, "model_dir": str(vdir), "champion": spec.name, "calibration": best_cal, "threshold": thr,
            "test_pr_auc": test_m["pr_auc"], "test_roc_auc": test_m["roc_auc"], "test_f1": test_m["f1"], "baseline_pr_auc": base_m["pr_auc"]}


def temporal_backtest(cfg: dict, ml_df: pd.DataFrame, X: pd.DataFrame, y: np.ndarray, spec: ModelSpec, threshold: float) -> dict:
    """Train on every edition before the last K, evaluate on the last K editions (the honest 'future' test)."""
    k = cfg["ml"]["temporal_backtest_editions"]
    eds = ml_df[["Year", "Season"]].drop_duplicates().sort_values(["Year", "Season"]).reset_index(drop=True)
    last = eds.tail(k)
    key = ml_df["Year"].astype(str) + "|" + ml_df["Season"].astype(str)
    test_mask = key.isin(set(last["Year"].astype(str) + "|" + last["Season"].astype(str))).to_numpy()
    tr, te = np.flatnonzero(~test_mask), np.flatnonzero(test_mask)
    tr = _cap(tr, y, spec.max_rows, cfg["ml"]["seed"])
    pipe = _pipeline(spec, cfg["ml"]["seed"]).fit(X.iloc[tr], y[tr])
    p = pipe.predict_proba(X.iloc[te])[:, 1]
    m = compute_metrics(y[te], p, threshold)
    base = compute_metrics(y[te], np.full(len(te), y[tr].mean()), 0.5)
    return {"test_editions": [f"{r.Year} {r.Season}" for r in last.itertuples()], "train_rows": int(len(tr)), "test_rows": int(len(te)),
            "pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"], "f1_at_champion_threshold": m["f1"], "recall": m["recall"], "precision": m["precision"],
            "baseline_pr_auc": base["pr_auc"], "note": "Uncalibrated probabilities; threshold carried over from the main run, so F1/precision/recall are indicative only."}
