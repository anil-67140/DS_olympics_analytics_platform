"""Evaluation figures for the model report (all registered with captions in the FigureBook)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..analysis.viz import PALETTE, FigureBook  # noqa: E402
from .metrics import calibration_table  # noqa: E402


def plot_model_comparison(book: FigureBook, results: pd.DataFrame):
    r = results[~results["is_baseline"]].sort_values("pr_auc")
    fig, axes = plt.subplots(1, 3, figsize=(14, max(4, 0.38 * len(r) + 1.5)), sharey=True)
    for ax, col, title in zip(axes, ["pr_auc", "roc_auc", "f1_best_threshold"], ["PR-AUC", "ROC-AUC", "F1 (val-tuned threshold)"]):
        ax.barh(r["model"], r[col], color=PALETTE[0])
        ax.set_title(title)
        ax.set_xlim(max(0, r[col].min() - 0.05), min(1.0, r[col].max() + 0.03))
    base = results.loc[results["is_baseline"], "pr_auc"]
    if len(base):
        axes[0].axvline(float(base.iloc[0]), ls="--", color=PALETTE[1], label="baseline")
        axes[0].legend(frameon=False)
    fig.suptitle("Model comparison on the validation set", fontweight="bold")
    return book.save(fig, "20_model_comparison", "Model comparison", "PR-AUC, ROC-AUC and F1 for every candidate model on held-out validation data.",
                     "Models are judged on PR-AUC (robust to the ~15% positive rate) rather than accuracy; the dashed line is what a model with no skill would score.")


def plot_pr_roc(book: FigureBook, curves: dict[str, dict], top: list[str], prevalence: float):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    for i, name in enumerate(top):
        c = curves[name]
        a1.plot(c["recall"], c["precision"], label=name, color=PALETTE[i % len(PALETTE)])
        a2.plot(c["fpr"], c["tpr"], label=name, color=PALETTE[i % len(PALETTE)])
    a1.axhline(prevalence, ls="--", color="grey")
    a1.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves")
    a2.plot([0, 1], [0, 1], ls="--", color="grey")
    a2.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves")
    a1.legend(frameon=False, fontsize=8)
    return book.save(fig, "21_pr_roc_curves", "Precision-recall and ROC curves (top models)", "Trade-off between catching medallists (recall) and being right when flagging one (precision).",
                     "On imbalanced data the PR curve exposes weaknesses ROC hides; the grey dashed PR line is the no-skill level.")


def plot_calibration(book: FigureBook, y, p_raw, p_cal, label_raw="raw", label_cal="calibrated"):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    for p, lab, col in ((p_raw, label_raw, PALETTE[5]), (p_cal, label_cal, PALETTE[0])):
        t = calibration_table(y, p, 10)
        ax.plot(t["mean_predicted"], t["observed_rate"], marker="o", label=lab, color=col)
    ax.plot([0, 1], [0, 1], ls="--", color="grey")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed medal rate", title="Reliability diagram")
    ax.legend(frameon=False)
    ax2.hist(p_cal, bins=40, color=PALETTE[0])
    ax2.set(title="Distribution of predicted probabilities", xlabel="Predicted probability")
    return book.save(fig, "22_calibration", "Calibration (are the probabilities trustworthy?)", "Predicted probability vs the medal rate actually observed, before and after calibration.",
                     "If the model says 30% the medal rate should be ~30%; only then can the probabilities be used for planning and risk scoring, not just ranking.")


def plot_confusion(book: FigureBook, m: dict):
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap="Blues")
    ax.grid(False)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black", fontsize=11)
    ax.set_xticks([0, 1], ["No medal", "Medal"])
    ax.set_yticks([0, 1], ["No medal", "Medal"])
    ax.set(xlabel="Predicted", ylabel="Actual", title=f"Confusion matrix (threshold {m['threshold']:.3f})")
    return book.save(fig, "23_confusion_matrix", "Confusion matrix on the test set", "Counts of correct and wrong calls at the chosen decision threshold.",
                     "Shows the two kinds of mistake separately: false alarms (top right) vs missed medallists (bottom left).")


def plot_threshold(book: FigureBook, table: pd.DataFrame, chosen: float):
    t = table.sort_values("threshold")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t["threshold"], t["precision"], label="Precision")
    ax.plot(t["threshold"], t["recall"], label="Recall")
    f1 = 2 * t["precision"] * t["recall"] / (t["precision"] + t["recall"]).replace(0, np.nan)
    ax.plot(t["threshold"], f1, label="F1", color=PALETTE[3])
    ax.axvline(chosen, ls="--", color="grey", label=f"chosen = {chosen:.3f}")
    ax.set(xlabel="Decision threshold", title="Threshold optimisation (validation set)")
    ax.legend(frameon=False)
    return book.save(fig, "24_threshold_optimisation", "Choosing the decision threshold", "Precision, recall and F1 as the cut-off moves; the dashed line is the value picked on validation data.",
                     "A fixed 0.5 cut-off ignores class imbalance and the different costs of false alarms and misses; this makes the trade-off explicit and tunable.")


def plot_importance(book: FigureBook, imp: pd.DataFrame):
    r = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.barh(r["feature"], r["importance_mean"], xerr=r["importance_std"], color=PALETTE[2])
    ax.set_xlabel("Drop in PR-AUC when the feature is shuffled")
    ax.set_title("Permutation importance (validation set)")
    return book.save(fig, "25_permutation_importance", "What drives the predictions", "How much model quality falls when each input is randomly shuffled.",
                     "Confirms the model relies on sensible signals (country/sport history, event size) and flags suspicious dependence on identifiers or leakage-prone fields.")


def plot_slices(book: FigureBook, sl: pd.DataFrame):
    s = sl[sl["n"] >= 300].copy()
    s["label"] = s["slice"] + ": " + s["value"].astype(str)
    s = s.sort_values("pr_auc").tail(30)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(s) + 1)))
    ax.barh(s["label"], s["pr_auc"], color=PALETTE[4])
    ax.set_xlabel("PR-AUC on the test set")
    ax.set_title("Performance by slice")
    return book.save(fig, "26_slice_performance", "Does the model work equally well everywhere?", "Test PR-AUC for subgroups (sex, season, era, sport).",
                     "A good overall score can hide a weak subgroup; reporting slices is the basic fairness and robustness check before deployment.")
