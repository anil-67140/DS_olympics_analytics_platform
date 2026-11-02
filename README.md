# Olympics Analytics Platform

An end-to-end, production-style data-science project built on 120+ years of Olympic data:

**data quality → statistics → SQL layer → forecasting → 14-model medal-prediction comparison → calibrated, versioned model → FastAPI + Gradio dashboard → monitoring, human feedback and retraining**, with tests, Docker, CI/CD, DVC and MLflow.

Everything runs three ways: in **Google Colab** (outputs saved to Drive), from the **command line**, or in **Docker**.

---

## 1 · Quick start

### Google Colab (recommended for you)
1. Put `olympics-analytics-platform.zip` in `MyDrive/Olympics-Analytics/`.
2. Add `KAGGLE_USERNAME` and `KAGGLE_KEY` to Colab **Secrets** and accept the dataset terms on kaggle.com.
3. Open `notebooks/Olympics_Analytics_Colab.ipynb` in Colab → **Runtime → Run all**.

All outputs (data, tables, figures, metrics, models, logs) land in `MyDrive/Olympics-Analytics/workspace/`. If Colab disconnects, run again: finished stages and finished models are skipped.

### Local
```bash
pip install -r requirements-dev.txt
make test                                   # 76 tests
make demo                                   # entire pipeline on synthetic data, ~30 s, no Kaggle
export KAGGLE_USERNAME=... KAGGLE_KEY=...
make run                                    # real data, resumable
make dashboard                              # http://localhost:7860
make api                                    # http://localhost:8000/docs
```

### Docker
```bash
docker compose up --build                   # api :8000, dashboard :7860, MLflow UI :5000 (mounts ./workspace)
```

---

## 2 · What each requirement maps to

| Requirement | Where |
|---|---|
| Data quality validation (missing, duplicates, inconsistent countries, invalid records) | `olympics/data/validate.py` → `reports/tables/data_quality_report.md`, scored 0-100, with a configurable fail-gate |
| Data documentation | `docs/DATA_DICTIONARY.md` (columns, definitions, sources, limitations) |
| Reproducible pipeline | `olympics/pipeline.py` + `params.yaml` + checkpointing; also `dvc.yaml` |
| Visual storytelling | `analysis/viz.py` — every chart saved with *what it shows* / *why it matters* (`reports/figures/FIGURES.md`) |
| SQL analytical layer | `olympics/sql/` — 8 parameterised queries (country, athlete, event, sport, host, participation) using window functions |
| Interactive dashboard | `apps/dashboard/app.py` (Gradio): KPIs, filters, country comparison, country/sport drill-downs, athlete search, forecast, model report, SQL console |
| Descriptive statistics | mean, median, std, IQR, skew, kurtosis, CV, by group |
| Correlation | Spearman + Pearson (log-log) with Fisher-z CIs; participation vs medals |
| Hypothesis testing | Welch t, Mann-Whitney (rank-biserial), Wilcoxon (host advantage), Holm/BH multiple-test correction |
| Confidence intervals | Wilson (medal rates), t, bootstrap, cluster-bootstrap (model metrics) |
| Chi-square | Sex × Sport and Medal × Season, with bias-corrected Cramér's V and standardised residuals |
| ANOVA / Kruskal-Wallis | Age across sports: classic + Welch ANOVA, Kruskal-Wallis, η²/ε², Levene, Shapiro, pairwise tests |
| Time series | Mann-Kendall + Theil-Sen trend, autocorrelation, ADF; participation and medal trends |
| Predictive analytics / forecasting | `forecasting/medals.py` — 10 methods incl. naïve, EWMA, Theil-Sen, Poisson GLM, ridge, RF, HistGBM, ETS, ARIMA, ensemble; **rolling-origin backtest**, prediction intervals |
| Performance optimisation | vectorised pandas, parquet, cached stages, per-model caches, indexed SQL |
| Automated tests | `tests/` (76) — cleaning, SQL vs pandas, statistics vs known values, leakage, splits, metrics, thresholds, checkpoints, API |
| CI/CD | `.github/workflows/ci.yml` (lint, tests, pipeline smoke run, Docker build + health check), `cd.yml` (image publish on tag), `retrain.yml` (scheduled monitoring/retraining) |
| **ML requirements** | |
| Leakage-safe pipeline | history features from earlier editions only; encoders/imputers inside the Pipeline; unit test that flips one edition's labels |
| Stratified split | stratified **and grouped** by `(Games, NOC)` (`ml/split.py`); optional temporal split; overlap is asserted to be zero |
| Threshold optimisation | F-beta, cost-based, precision@recall, Youden — all reported, chosen on data not used for fitting |
| Precision-recall analysis | PR curves, PR-AUC as the primary metric, no-skill baseline |
| Model & preprocessing versioning | `ml/registry.py` — one object (pipeline + calibrator + threshold), schema hash, library versions, model card, rollback pointer |
| FastAPI inference | `apps/api/main.py` — `/predict`, `/predict/batch`, `/feedback`, `/model`, `/monitoring`, `/health` |
| Docker, CI/CD, DVC, MLflow, Evidently | `Dockerfile`, `docker-compose.yml`, workflows, `dvc.yaml`, `tracking.py` (MLflow + JSONL fallback), `monitoring/drift.py` (PSI/KS built in; Evidently HTML when installed) |
| Explainability | permutation importance (PR-AUC drop) + per-prediction contributions; SHAP when installed |
| Drift detection / model monitoring | PSI, KS, chi-square, Jensen-Shannon, prediction drift, performance degradation vs tolerance |
| Human feedback + retraining | `feedback/store.py` + `ml/retrain.py` — champion/challenger on the same test split |
| Compare ≥ 10 models | 14 always available (+ XGBoost, LightGBM, CatBoost if installed) |
| Full metric set | accuracy, balanced accuracy, precision, recall, specificity, NPV, F1/F2/F0.5, MCC, Cohen's κ, ROC-AUC, PR-AUC, log-loss, Brier (+ skill score), ECE/MCE, lift@10%, confusion matrix, calibration curve, slice metrics |

### Models compared
Baseline (prior) · Logistic regression · Linear SVM (calibrated) · LDA · Gaussian NB · k-NN · Decision tree · Random forest · Extra trees · Bagging · AdaBoost · Gradient boosting · HistGradientBoosting · MLP neural network · XGBoost · LightGBM · CatBoost.

---

## 3 · Choosing the dataset

The default is the **120 years of Olympic history** dataset (the one your original notebook uses): it is what the pipeline was tested against end-to-end. It ends at Rio 2016.

For data beyond 2016 set `data.source: olympedia` in `params.yaml` (or `--set data.source=olympedia`). That adapter is **written but not verified against the real files** (Kaggle was not reachable while building this). First run:
```bash
python -m olympics.cli inspect-source --root workspace --set data.source=olympedia
```
It prints every file and column. If a name differs from what `load_olympedia` expects it raises an error listing what exists; adjust the alias list in `olympics/data/ingest.py`. Everything downstream is source-agnostic.

Already have the CSVs? Skip Kaggle: `--set data.local_dir=/path/to/folder`.

---

## 4 · Configuration (`params.yaml`)

Single source of truth. Override from the CLI: `--set ml.profile=fast --set ml.threshold.strategy=cost --set ml.threshold.cost_fn=5`. Changing a section re-runs only the stages that depend on it.

Common knobs: `ml.profile` (`fast`/`full`), `ml.models` (subset), `ml.split.mode` (`group_stratified`/`temporal`), `ml.threshold.*`, `forecast.holdout_editions`, `data.sample_frac`, `monitoring.psi_*`, `retrain.min_feedback_rows`.

## 5 · Operating the model

```bash
python -m olympics.cli status --root workspace            # which stages are done
python -m olympics.cli run --root workspace --force compare final   # recompute selected stages
python -m olympics.cli retrain --root workspace           # fold in human feedback (champion/challenger)
python -m olympics.cli sql country_trend --root workspace --param country=India --param season=Summer
```
Rollback: write the older version name into `workspace/models/latest.json` (or call `registry.promote`). Then `POST /admin/reload`.

## 6 · Honest limitations

- Predictions describe *historical patterns*; they are not suitable for decisions about individual athletes (see the generated model card).
- Team entries share an outcome, so entries are not independent; splits and bootstrap intervals are grouped to account for that.
- Country identity changes (USSR/Russia, Germany) — see `docs/DATA_DICTIONARY.md`.
- The FastAPI service and Gradio dashboard are covered by an API test and by tested data logic, but the Gradio UI itself needs a visual check on first launch.
