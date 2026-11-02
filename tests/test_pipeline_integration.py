"""End-to-end tests: one full demo pipeline run (module-scoped fixture), then model/serving/dashboard/API checks."""
import numpy as np
import pandas as pd
import pytest

from olympics.dashboard_logic import DashboardData
from olympics.io_utils import read_json
from olympics.ml.registry import latest_version, list_versions, load_model, promote
from olympics.serving import Predictor


def test_all_artifacts_are_produced(pipeline_ws):
    p = pipeline_ws.paths
    must = [p.tables / n for n in ["data_quality_report.md", "model_comparison_validation.csv", "forecast_next_edition.csv", "slice_metrics.csv",
                                   "threshold_strategies.csv", "calibration_comparison.csv", "permutation_importance.csv", "drift_report.csv",
                                   "cross_validation_top_models.csv", "medalist_vs_rest_tests.csv"]]
    must += [p.metrics / n for n in ["final_model_metrics.json", "monitoring_summary.json", "temporal_backtest.json", "age_across_sports_tests.json"]]
    must += [p.db_file, p.root / "reports" / "REPORT.md", p.root / "reports" / "REPORT.html", p.figures / "FIGURES.md"]
    assert [str(m) for m in must if not m.exists()] == []
    assert len(list(p.figures.glob("*.png"))) >= 18
    assert len(pipeline_ws.ck.status()) >= 9


def test_figures_have_storytelling_captions(pipeline_ws):
    idx = read_json(pipeline_ws.paths.figures / "figure_index.json")
    assert len(idx) >= 18
    for name, meta in idx.items():
        assert len(meta["what_it_shows"]) > 20 and len(meta["why_it_matters"]) > 20, name


def test_model_beats_baseline_and_metrics_are_complete(pipeline_ws):
    m = read_json(pipeline_ws.paths.metrics / "final_model_metrics.json")
    t = m["test_metrics"]
    assert t["pr_auc"] > m["baseline_test_metrics"]["pr_auc"] + 0.05 and t["roc_auc"] > 0.6
    for k in ("accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc", "brier", "ece", "mcc", "log_loss", "balanced_accuracy", "tn", "fp", "fn", "tp"):
        assert k in t
    assert m["temporal_backtest"]["pr_auc"] > m["temporal_backtest"]["baseline_pr_auc"]
    assert set(m["test_confidence_intervals_95"]) >= {"pr_auc", "f1"}


def test_comparison_covers_many_models(pipeline_ws):
    comp = pd.read_csv(pipeline_ws.paths.tables / "model_comparison_validation.csv")
    assert {"logistic_regression", "decision_tree", "hist_gradient_boosting", "baseline_prior"} <= set(comp["model"])
    assert comp["rank"].tolist() == list(range(1, len(comp) + 1)) and comp["pr_auc"].is_monotonic_decreasing


def test_zoo_defines_at_least_ten_candidates():
    from olympics.ml.models import build_models
    names = [s.name for s in build_models(1, "fast") if not s.is_baseline]
    assert len(names) >= 12 and len(set(names)) == len(names)
    assert {"logistic_regression", "random_forest", "extra_trees", "adaboost", "gradient_boosting", "hist_gradient_boosting", "knn", "gaussian_nb", "mlp_neural_net", "lda"} <= set(names)


def test_split_has_no_leakage_across_delegations(pipeline_ws):
    assert all(v == 0 for v in read_json(pipeline_ws.paths.metrics / "split_group_overlap.json").values())


# ---------------------------------------------------------------- versioning
def test_model_versioning_and_rollback(pipeline_ws):
    mdir = pipeline_ws.paths.models
    v = latest_version(mdir)
    vdir = mdir / v
    assert {"model.joblib", "metadata.json", "MODEL_CARD.md", "history_lookup.json"} <= {f.name for f in vdir.iterdir()}
    meta = read_json(vdir / "metadata.json")
    assert meta["schema_hash"] and meta["sklearn_version"] and meta["data_fingerprint"] and meta["threshold"] > 0
    model, _ = load_model(mdir)
    assert 0 < model.threshold < 1
    assert "Known limitations" in (vdir / "MODEL_CARD.md").read_text()
    promote(mdir, v)
    with pytest.raises(FileNotFoundError):
        promote(mdir, "v-does-not-exist")


def test_schema_change_is_rejected(pipeline_ws, monkeypatch):
    import olympics.ml.registry as reg
    monkeypatch.setattr(reg, "ALL_FEATURES", reg.ALL_FEATURES[:-1])
    with pytest.raises(RuntimeError, match="schema"):
        reg.load_model(pipeline_ws.paths.models)


# ---------------------------------------------------------------- serving
def test_predictor_outputs_and_explanations(pipeline_ws):
    P = Predictor(pipeline_ws.paths.models)
    row = pipeline_ws.ml_frame().iloc[0]
    req = {k: (None if row[k] != row[k] else row[k]) for k in ["Age", "Height", "Weight", "Sex", "NOC", "Sport", "Season", "Year"]}
    r = P.predict([req], explain=True)[0]
    assert 0 <= r["probability"] <= 1 and isinstance(r["predicted_medal"], bool) and len(r["explanation"]) == 6
    assert (r["predicted_medal"]) == (r["probability"] >= r["threshold"])
    assert P.predict([req] * 5)[0]["probability"] == pytest.approx(P.predict([req])[0]["probability"])      # deterministic, batch-invariant


def test_predictor_handles_unseen_country_and_missing_values(pipeline_ws):
    P = Predictor(pipeline_ws.paths.models)
    r = P.predict([{"Sex": "F", "NOC": "ZZZ", "Sport": "Curling", "Season": "Winter", "Year": 2030}])[0]
    assert 0 <= r["probability"] <= 1


def test_host_country_gets_higher_probability(pipeline_ws):
    P = Predictor(pipeline_ws.paths.models)
    base = {"Age": 26, "Height": 178, "Weight": 72, "Sex": "M", "NOC": "AUS", "Sport": "Athletics", "Season": "Summer", "noc_event_entrants": 1, "event_entrants": 40}
    assert P.build_features({**base, "Year": 2000})["is_host"] == 1 and P.build_features({**base, "Year": 2004})["is_host"] == 0


def test_retrain_skips_without_enough_feedback(pipeline_ws, cfg):
    from olympics.ml.retrain import retrain_with_feedback
    out = retrain_with_feedback(cfg, pipeline_ws.paths, pipeline_ws.ml_frame(), pipeline_ws.tracker, "fp")
    assert out["status"] == "skipped"


# ---------------------------------------------------------------- dashboard logic
def test_dashboard_logic(pipeline_ws):
    D = DashboardData(pipeline_ws.paths.root)
    k = D.kpis("Summer", D.year_range)
    assert k["Athletes"] > 0 and 0 <= k["Women %"] <= 100
    assert D.kpis("Summer", (D.year_range[1], D.year_range[1]))["Editions"] == 1
    t = D.medal_table("Both", D.year_range, top=5)
    assert t["Rank"].tolist() == [1, 2, 3, 4, 5] and t["Gold"].is_monotonic_decreasing
    top2 = list(t["Country"][:2])
    assert set(D.compare_countries(top2)["Country"]) == set(top2)
    prof = D.country_profile(top2[0])
    assert prof["kpis"]["Medals awarded"] == int(D.medals[D.medals["region"] == top2[0]].shape[0])
    assert len(D.sport_drilldown(D.sports[0])["events"]) > 0
    assert len(D.athlete_career("ath")) > 0 and "hint" in D.athlete_career("a").columns
    assert len(D.sql("SELECT COUNT(*) n FROM medals")) == 1
    with pytest.raises(ValueError):
        D.sql("SELECT 1; DROP TABLE medals")
    assert len(D.figures()) >= 18


# ---------------------------------------------------------------- API
def test_api_end_to_end(pipeline_ws, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    monkeypatch.setenv("OLYMPICS_ROOT", str(pipeline_ws.paths.root))
    from apps.api.main import app
    body = {"Age": 24, "Height": 185, "Weight": 78, "Sex": "M", "NOC": "usa", "Sport": "Swimming", "Season": "Summer", "Year": 2016}
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "ok"
        assert c.get("/model").json()["algorithm"]
        r = c.post("/predict?explain=true", json=body)
        assert r.status_code == 200 and 0 <= r.json()["probability"] <= 1 and len(r.json()["explanation"]) == 6
        assert c.post("/predict", json={**body, "Age": 500}).status_code == 422
        assert c.post("/predict", json={**body, "Sex": "X"}).status_code == 422
        assert len(c.post("/predict/batch", json={"entries": [body, body]}).json()) == 2
        pid = r.json()["prediction_id"]
        assert c.post("/feedback", json={"prediction_id": pid, "actual_medal": True}).json()["labelled"] >= 1
        assert c.post("/feedback", json={"prediction_id": "nope", "actual_medal": True}).status_code == 404
        assert "feedback" in c.get("/monitoring").json() and "nocs" in c.get("/catalog").json()


# ---------------------------------------------------------------- monitoring correctness
def test_monitoring_uses_only_heldout_rows_and_ignores_year(pipeline_ws):
    from olympics.ml.train import load_splits
    m = read_json(pipeline_ws.paths.metrics / "monitoring_summary.json")
    ml, splits = pipeline_ws.ml_frame(), load_splits(pipeline_ws.paths)
    held = np.zeros(len(ml), bool)
    held[np.concatenate([splits["val"], splits["test"]])] = True
    key = ml["Year"].astype(str) + "|" + ml["Season"]
    expected = int((held & key.isin(set(m["current"])).to_numpy()).sum())
    assert m["performance"]["n_current"] == expected                      # not the (training-contaminated) full window
    assert m["performance"]["current"]["pr_auc"] < 0.95                   # a train-contaminated random forest would score ~0.99
    assert "Year" not in set(pd.read_csv(pipeline_ws.paths.tables / "drift_report.csv")["feature"])
