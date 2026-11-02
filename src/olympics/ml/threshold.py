"""Decision-threshold optimisation (0.5 is almost never the right cut-off on imbalanced data).

Strategies
  fbeta               maximise F-beta (beta>1 favours recall; beta<1 favours precision)
  cost                minimise cost_fp * false positives + cost_fn * false negatives
  precision_at_recall highest precision subject to recall >= target_recall
  youden              maximise TPR - FPR
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import metrics as skm


def threshold_table(y, p) -> pd.DataFrame:
    y = np.asarray(y).astype(int)
    precision, recall, thr = skm.precision_recall_curve(y, p)
    precision, recall = precision[:-1], recall[:-1]
    pos = y.sum()
    tp = recall * pos
    with np.errstate(divide="ignore", invalid="ignore"):
        fp = np.where(precision > 0, tp / precision - tp, 0.0)
    fn = pos - tp
    tpr = recall
    fpr = fp / max(len(y) - pos, 1)
    return pd.DataFrame({"threshold": thr, "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn,
                         "fpr": fpr, "youden_j": tpr - fpr})


def optimize_threshold(y, p, strategy: str = "fbeta", beta: float = 1.0, cost_fp: float = 1.0, cost_fn: float = 3.0,
                       target_recall: float = 0.8) -> dict:
    t = threshold_table(y, p)
    if strategy == "fbeta":
        b2 = beta**2
        with np.errstate(divide="ignore", invalid="ignore"):
            t["score"] = np.where(t["precision"] + t["recall"] > 0,
                                  (1 + b2) * t["precision"] * t["recall"] / (b2 * t["precision"] + t["recall"]), 0.0)
        best = t.loc[t["score"].idxmax()]
        label = f"F{beta:g}"
    elif strategy == "cost":
        t["score"] = -(cost_fp * t["fp"] + cost_fn * t["fn"])
        best = t.loc[t["score"].idxmax()]
        label = f"-cost (FP={cost_fp:g}, FN={cost_fn:g})"
    elif strategy == "precision_at_recall":
        ok = t[t["recall"] >= target_recall]
        t["score"] = np.where(t["recall"] >= target_recall, t["precision"], -1.0)
        best = ok.loc[ok["precision"].idxmax()] if len(ok) else t.loc[t["recall"].idxmax()]
        label = f"precision @ recall>={target_recall:g}"
    elif strategy == "youden":
        t["score"] = t["youden_j"]
        best = t.loc[t["score"].idxmax()]
        label = "Youden J"
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    return {"threshold": float(best["threshold"]), "strategy": strategy, "objective": label,
            "objective_value": float(best["score"]) if "score" in best else float("nan"),
            "precision": float(best["precision"]), "recall": float(best["recall"]), "table": t}
