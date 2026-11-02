"""Builds one self-contained Markdown + HTML analysis report from the saved metrics, tables and figures."""
from __future__ import annotations

import base64
import html
from pathlib import Path

import pandas as pd

from .config import Paths
from .io_utils import read_json, write_text


def _tbl(path: Path, n: int = 10, cols: list[str] | None = None, fmt: int = 3) -> str:
    if not path.exists():
        return "_not available_\n"
    df = pd.read_csv(path)
    df = df[cols] if cols else df
    return df.head(n).round(fmt).to_markdown(index=False) + "\n" if _has_tabulate() else "```\n" + df.head(n).round(fmt).to_string(index=False) + "\n```\n"


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def _fig(paths: Paths, name: str, idx: dict) -> str:
    meta = idx.get(name)
    if not meta:
        return ""
    return (f"![{meta['title']}](figures/{meta['file']})\n\n**What it shows:** {meta['what_it_shows']}  \n"
            f"**Why it matters:** {meta['why_it_matters']}\n\n")


def build_report(paths: Paths, cfg: dict) -> Path:
    idx = read_json(paths.figures / "figure_index.json", {})
    qa, fm = read_json(paths.metrics / "data_quality.json", {}), read_json(paths.metrics / "final_model_metrics.json", {})
    ha, pv = read_json(paths.metrics / "host_advantage.json", {}), read_json(paths.metrics / "participation_vs_medals.json", {})
    age, fs = read_json(paths.metrics / "age_across_sports_tests.json", {}), read_json(paths.metrics / "forecast_summary.json", {})
    mon, T = read_json(paths.metrics / "monitoring_summary.json", {}), paths.tables
    L = ["# Olympics Analytics — Analysis Report", ""]

    L += ["## 1. Data quality", ""]
    if qa:
        s = qa["summary"]
        L += [f"Quality score **{s['quality_score']}/100** on {s['rows']:,} rows ({s['critical']} critical, {s['warnings']} warnings).", "",
              _tbl(T / "data_quality_checks.csv", 40, ["id", "severity", "count", "pct_of_rows"]), _fig(paths, "01_missingness", idx)]
    L += ["## 2. Descriptive statistics", "", _tbl(T / "descriptive_numeric.csv", 10, ["variable", "n", "mean", "median", "std", "iqr", "skew"]),
          _fig(paths, "02_participation_growth", idx), _fig(paths, "03_women_share", idx), _fig(paths, "04_country_medal_trends", idx)]
    L += ["## 3. Statistical inference", ""]
    if age:
        L += [f"**Age across sports** — {age.get('interpretation')} Kruskal-Wallis H={age.get('kruskal_H', 0):,.0f}, "
              f"ε²={age.get('epsilon_squared', 0):.3f}; ANOVA η²={age.get('eta_squared', 0):.3f}. {age.get('assumption_note', '')}", "", _fig(paths, "05_age_by_sport", idx)]
    L += ["**Medallists vs the rest** (Welch t, Holm-adjusted; with Cohen's d):", "", _tbl(T / "medalist_vs_rest_tests.csv", 6, ["variable", "mean_diff", "diff_ci_low", "diff_ci_high", "cohens_d", "effect_magnitude"]),
          _fig(paths, "06_height_weight_medal", idx), _fig(paths, "10_chi_square_residuals", idx), _fig(paths, "07_medal_rate_ci", idx)]
    if pv:
        L += [f"**Participation vs medals:** {pv.get('interpretation', '')}", "", _fig(paths, "11_participation_vs_medals", idx)]
    if ha.get("interpretation"):
        L += [f"**Host advantage:** {ha['interpretation']}", "", _fig(paths, "08_host_advantage", idx)]
    L += [_fig(paths, "09_correlation_heatmap", idx)]

    L += ["## 4. Forecasting", ""]
    if fs:
        L += [f"Rolling-origin backtest over editions {fs['backtest_years']}. Best method: **{fs['best_model']}** (MAE {fs['best_mae']:.2f} vs naive {fs['naive_mae']:.2f}). "
              f"90% interval covered {100 * fs['interval_coverage_in_backtest']:.0f}% of backtest outcomes.", "",
              _tbl(T / "forecast_model_comparison.csv", 12, ["model", "mae", "rmse", "smape", "rel_mae_vs_naive", "bias"]),
              _fig(paths, "13_forecast_model_comparison", idx), _fig(paths, "12_forecast_fan", idx)]

    L += ["## 5. Medal prediction model", ""]
    if fm:
        t, b = fm["test_metrics"], fm["baseline_test_metrics"]
        L += [f"Champion **{fm['algorithm']}** ({fm['version']}), calibration `{fm['calibration']}`, threshold {fm['threshold']:.3f} ({fm['threshold_strategy']}).", "",
              "| Metric | Test | No-skill baseline |", "|---|---:|---:|"]
        for k in ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "mcc", "roc_auc", "pr_auc", "log_loss", "brier", "ece"):
            L.append(f"| {k} | {t[k]:.4f} | {b.get(k, float('nan')):.4f} |")
        ci = fm.get("test_confidence_intervals_95", {})
        L += ["", "95% cluster-bootstrap intervals: " + ", ".join(f"{k} {v[0]:.3f}–{v[1]:.3f}" for k, v in ci.items()), ""]
        tb = fm.get("temporal_backtest")
        if tb:
            L += [f"Temporal check (train before {tb['test_editions']}, test on them): PR-AUC {tb['pr_auc']:.3f} vs baseline {tb['baseline_pr_auc']:.3f}.", ""]
        L += ["### Model comparison (validation)", "", _tbl(T / "model_comparison_validation.csv", 25, ["rank", "model", "pr_auc", "roc_auc", "f1_best_threshold", "brier", "ece", "fit_seconds"], 4),
              _fig(paths, "20_model_comparison", idx), _fig(paths, "21_pr_roc_curves", idx), _fig(paths, "22_calibration", idx), _fig(paths, "23_confusion_matrix", idx),
              _fig(paths, "24_threshold_optimisation", idx), _fig(paths, "25_permutation_importance", idx), _fig(paths, "26_slice_performance", idx)]
    L += ["## 6. Monitoring", ""]
    if mon:
        L += [f"Drift: **{mon['drift']['overall']}** (worst feature `{mon['drift']['worst_feature']}`, PSI {mon['drift']['worst_psi']:.3f}). "
              f"Performance on newest editions: **{mon['performance']['status']}**.", "", _tbl(T / "drift_report.csv", 8, ["feature", "psi", "ks_stat", "status"])]
    md = "\n".join(L)
    write_text(paths.root / "reports" / "REPORT.md", md)
    body = _markdown_to_html(md, paths)
    out = paths.root / "reports" / "REPORT.html"
    write_text(out, body)
    return out


def _markdown_to_html(md: str, paths: Paths) -> str:
    try:
        import markdown  # type: ignore
        inner = markdown.markdown(md, extensions=["tables"])
    except ImportError:
        inner = "<pre>" + html.escape(md) + "</pre>"
    # inline images so the HTML is a single shareable file
    import re

    def embed(m):
        f = paths.root / "reports" / m.group(1)
        if f.exists():
            return f'src="data:image/png;base64,{base64.b64encode(f.read_bytes()).decode()}"'
        return m.group(0)
    inner = re.sub(r'src="(figures/[^"]+)"', embed, inner)
    css = "body{font-family:system-ui,sans-serif;max-width:980px;margin:2rem auto;padding:0 1rem;line-height:1.5}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:4px 8px}img{max-width:100%}"
    return f"<!doctype html><meta charset='utf-8'><title>Olympics Analytics Report</title><style>{css}</style>{inner}"
