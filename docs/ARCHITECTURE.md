# Architecture

```
Kaggle ──► ingest ──► validate ──► clean ──► medals table ──┬─► SQL layer (SQLite)  ──► dashboard SQL console
 (adapter)  raw        report     rules      (dedup teams)  ├─► statistics + figures ──► REPORT.html
                                                            ├─► forecasting (rolling backtest)
                                                            └─► ML frame (leakage-safe history features)
                                                                  │
            group-stratified split ◄─────────────────────────────┘
                  │
   compare 14+ models (validation, PR-AUC) ─► grouped CV (top-k) ─► calibrate ─► tune threshold
                  │                                                         │
                  └──────────────► test set touched ONCE ◄─────────────────┘
                                         │
                       versioned model (model + calibrator + threshold in one object)
                                         │
              ┌──────────────────────────┼──────────────────────────┐
         FastAPI service           Gradio dashboard        monitoring (PSI/KS/perf, Evidently)
              └──────── predictions + human feedback (JSONL) ◄─────────┘
                                         │
                      retrain: challenger vs champion on the same test split ─► promote / reject
```

## Design decisions

| Concern | Decision | Why |
|---|---|---|
| Resumability | Every stage writes a checkpoint (`checkpoints/<stage>.json`) keyed by a hash of the relevant config; each model's result is cached individually; every file write is atomic (temp file + rename). | A Colab disconnect can lose at most the model that was training. Changing a parameter invalidates exactly the stages that depend on it. |
| Single workspace | All outputs live under one root (`--root`, e.g. a Drive folder). | Download or integrate by copying one folder. |
| Leakage | History features look only at earlier editions; label-dependent encoders and imputers are inside the sklearn `Pipeline`; splits are grouped by `(Games, NOC)`. A unit test flips labels of one edition and asserts earlier features do not move. | Teammates share a medal; a random split would put the same delegation on both sides and inflate scores. |
| Evaluation | PR-AUC is the primary metric; all of accuracy, balanced accuracy, precision, recall, specificity, F1/F2/F0.5, MCC, kappa, ROC-AUC, PR-AUC, log-loss, Brier (+skill), ECE/MCE, lift@10% and the confusion matrix are reported with cluster-bootstrap CIs. | ~85% of entries win nothing, so accuracy alone is meaningless. |
| Calibration | Raw, sigmoid and isotonic are compared on held-out data; the best by log-loss is used. | Probabilities are used for planning, not just ranking. |
| Threshold | Chosen on data not used for fitting (F-beta / cost / precision-at-recall / Youden; all four are reported). | 0.5 ignores class imbalance and unequal error costs. |
| One artifact | `MedalModel` = preprocessing + classifier + calibrator + threshold, with schema hash and library versions in `metadata.json`. | The API cannot apply a different pipeline from the one that was validated; a schema change is refused at load time. |
| Promotion | `latest.json` is a pointer; rollback = point it at an older version. Retraining only promotes if the challenger is within `retrain.min_gain` PR-AUC of the champion on the same test split. | Safe, auditable updates. |
| Forecast validation | Rolling-origin backtest against a naive baseline. | Random CV would leak the future. |
| SQL on Drive | The SQLite file is built on local disk and copied; readers use a local copy. | SQLite journaling is unreliable on network file systems. |

## Module map

| Package | Responsibility |
|---|---|
| `olympics.data` | ingest (Kaggle + adapters), validate, clean, schema, synthetic generator |
| `olympics.sql` | database build + named parameterised queries (`queries/*.sql`) |
| `olympics.analysis` | descriptive stats, inference (CI, tests, effect sizes), trends, figures with captions |
| `olympics.forecasting` | 10 forecasters, rolling backtest, intervals |
| `olympics.ml` | features, splits, model zoo, metrics, calibration, thresholds, training, explainability, registry, retraining |
| `olympics.monitoring` / `olympics.feedback` | drift + performance monitoring; prediction/feedback store |
| `apps/api`, `apps/dashboard` | FastAPI service, Gradio dashboard |
