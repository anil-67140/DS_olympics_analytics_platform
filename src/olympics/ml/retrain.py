"""Champion/challenger retraining that folds in human-verified feedback.

Flow: load champion -> build a training set = original training split + labelled feedback rows (up-weighted by
duplication is avoided; they are simply appended) -> train challenger with the champion's algorithm -> compare
both on the SAME untouched test split -> promote only if the challenger is not worse by more than
``retrain.min_gain`` PR-AUC. The old version always stays on disk, so rollback is a pointer change.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone

from ..config import Paths, config_fingerprint
from ..feedback.store import FeedbackStore
from ..io_utils import get_logger, write_json
from ..tracking import Tracker
from .features import make_preprocessor, to_model_frame
from .metrics import compute_metrics
from .model_wrapper import MedalModel, ProbabilityCalibrator
from .models import build_models
from .registry import load_model, new_version, promote, save_model_version
from .threshold import optimize_threshold
from .train import _cap, load_splits

log = get_logger()


def retrain_with_feedback(cfg: dict, paths: Paths, ml_df: pd.DataFrame, tracker: Tracker, data_fp: str,
                          force: bool = False) -> dict:
    store = FeedbackStore(paths.feedback)
    labelled = store.labelled()
    min_rows = cfg["retrain"]["min_feedback_rows"]
    if len(labelled) < min_rows and not force:
        return {"status": "skipped", "reason": f"only {len(labelled)} labelled feedback rows (< {min_rows})"}

    champion, meta = load_model(paths.models)
    splits = load_splits(paths)
    seed = cfg["ml"]["seed"]
    X_all, y_all = to_model_frame(ml_df), ml_df[cfg["ml"]["target"]].to_numpy()
    spec = {s.name: s for s in build_models(seed, cfg["ml"]["profile"], cfg["ml"].get("n_jobs", -1))}[meta["algorithm"]]

    tr = _cap(splits["train"], y_all, spec.max_rows, seed)
    X_tr, y_tr = X_all.iloc[tr], y_all[tr]
    if len(labelled):
        X_fb = to_model_frame(labelled)
        X_tr = pd.concat([X_tr, X_fb], ignore_index=True)
        y_tr = np.concatenate([y_tr, labelled["actual"].to_numpy(int)])

    from sklearn.pipeline import Pipeline
    pipe = Pipeline([("prep", make_preprocessor(seed)), ("clf", clone(spec.estimator))]).fit(X_tr, y_tr)

    va, te = splits["val"], splits["test"]
    p_val = pipe.predict_proba(X_all.iloc[va])[:, 1]
    cal = ProbabilityCalibrator(meta["calibration"]).fit(p_val, y_all[va])
    t = cfg["ml"]["threshold"]
    thr = optimize_threshold(y_all[va], cal.transform(p_val), t["strategy"], t["beta"], t["cost_fp"], t["cost_fn"], t["target_recall"])["threshold"]
    challenger = MedalModel(pipe, cal, thr, list(X_all.columns))

    m_champ = compute_metrics(y_all[te], champion.predict_proba(X_all.iloc[te]), champion.threshold)
    m_chal = compute_metrics(y_all[te], challenger.predict_proba(X_all.iloc[te]), thr)
    gain = m_chal["pr_auc"] - m_champ["pr_auc"]
    promote_it = gain >= cfg["retrain"]["min_gain"]

    result = {"status": "promoted" if promote_it else "rejected", "champion": meta["version"], "feedback_rows": int(len(labelled)),
              "champion_pr_auc": m_champ["pr_auc"], "challenger_pr_auc": m_chal["pr_auc"], "pr_auc_gain": gain,
              "champion_f1": m_champ["f1"], "challenger_f1": m_chal["f1"]}
    version = new_version(data_fp + f"-fb{len(labelled)}", config_fingerprint(cfg, ["ml", "data"]))
    new_meta = {**{k: v for k, v in meta.items() if k not in ("created_at", "git_sha")}, "version": version, "threshold": thr,
                "test_metrics": m_chal, "parent_version": meta["version"], "feedback_rows_used": int(len(labelled)),
                "trained_rows": int(len(y_tr))}
    vdir = save_model_version(paths.models, challenger, new_meta)
    if promote_it:
        promote(paths.models, version)
        result["new_version"] = version
    else:
        result["rejected_version_dir"] = str(vdir)
    write_json(paths.metrics / "last_retrain.json", result)
    tracker.log_run(f"retrain-{version}", {"parent": meta["version"], "feedback_rows": len(labelled)},
                    {"champion_pr_auc": m_champ["pr_auc"], "challenger_pr_auc": m_chal["pr_auc"], "gain": gain}, tags={"stage": "retrain", "decision": result["status"]})
    log.info("Retrain: %s (gain %+.4f PR-AUC)", result["status"], gain)
    return result
