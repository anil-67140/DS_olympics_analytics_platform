import numpy as np
import pandas as pd
import pytest
from sklearn import metrics as skm

from olympics.data.schema import ALL_FEATURES
from olympics.ml.features import add_history_features, build_ml_frame, to_model_frame
from olympics.ml.metrics import calibration_table, cluster_bootstrap_ci, compute_metrics, expected_calibration_error
from olympics.ml.model_wrapper import ProbabilityCalibrator
from olympics.ml.split import group_stratified_split, split_report, temporal_split
from olympics.ml.threshold import optimize_threshold

HIST = ["athlete_prev_games", "athlete_prev_medals", "noc_prev_entries", "noc_prev_medal_rate", "sport_noc_prev_medal_rate"]


# ---------------------------------------------------------------- leakage guarantees
def test_history_features_ignore_current_and_future_labels(cleaned):
    """Flipping every label in year Y must not change history features of any row in years <= Y."""
    df, *_ = cleaned
    year = int(df["Year"].quantile(0.6))
    base = add_history_features(df)
    flipped = df.copy()
    flipped.loc[flipped["Year"] == year, "won_medal"] = 1 - flipped.loc[flipped["Year"] == year, "won_medal"]
    alt = add_history_features(flipped)
    mask = base["Year"] <= year
    pd.testing.assert_frame_equal(base.loc[mask, HIST].reset_index(drop=True), alt.loc[mask, HIST].reset_index(drop=True))
    assert not base.loc[~mask, HIST].reset_index(drop=True).equals(alt.loc[~mask, HIST].reset_index(drop=True))  # later years DO see the change


def test_first_appearance_has_no_history(cleaned):
    df, *_ = cleaned
    f = add_history_features(df)
    first_year = f.groupby("ID")["Year"].transform("min")
    first = f[f["Year"] == first_year]
    assert (first["athlete_prev_games"] == 0).all() and (first["athlete_prev_medals"] == 0).all()
    earliest = f["Year"].min()
    assert (f.loc[f["Year"] == earliest, "noc_prev_entries"] == 0).all()


def test_history_rates_are_valid_probabilities(cleaned):
    df, *_ = cleaned
    f = add_history_features(df)
    for c in ("noc_prev_medal_rate", "sport_noc_prev_medal_rate"):
        assert f[c].between(0, 1).all() and f[c].notna().all()


def test_target_not_in_features():
    assert "won_medal" not in ALL_FEATURES and "Medal" not in ALL_FEATURES
    assert not {"Gold", "Silver", "Bronze"} & set(ALL_FEATURES)


def test_model_frame_has_exact_schema(cleaned):
    df, *_ = cleaned
    X = to_model_frame(build_ml_frame(df).head(20))
    assert list(X.columns) == ALL_FEATURES
    assert to_model_frame(pd.DataFrame([{"Sex": "M"}]))[ALL_FEATURES].shape == (1, len(ALL_FEATURES))   # serving: missing inputs tolerated


# ---------------------------------------------------------------- splits
def test_group_split_has_no_group_overlap_and_is_stratified(cleaned, cfg):
    df, *_ = cleaned
    ml = build_ml_frame(df)
    sp = group_stratified_split(ml, "won_medal", ["Games", "NOC"], 0.7, 0.15, 0.15, seed=1)
    rep = split_report(ml, sp, "won_medal", ["Games", "NOC"])
    assert all(v == 0 for v in rep.attrs["group_overlap"].values())
    allidx = np.concatenate(list(sp.values()))
    assert len(allidx) == len(set(allidx)) == len(ml)                      # partition: nothing lost or duplicated
    rates = rep["positive_rate"]
    assert rates.max() - rates.min() < 0.06
    assert 0.6 < rep.set_index("split").loc["train", "rows"] / len(ml) < 0.8


def test_split_is_reproducible(cleaned):
    ml = build_ml_frame(cleaned[0])
    a = group_stratified_split(ml, "won_medal", ["Games", "NOC"], 0.7, 0.15, 0.15, seed=5)
    b = group_stratified_split(ml, "won_medal", ["Games", "NOC"], 0.7, 0.15, 0.15, seed=5)
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_temporal_split_orders_time(cleaned):
    ml = build_ml_frame(cleaned[0])
    sp = temporal_split(ml)
    assert ml.iloc[sp["train"]]["Year"].max() <= ml.iloc[sp["val"]]["Year"].min() <= ml.iloc[sp["test"]]["Year"].min()


# ---------------------------------------------------------------- metrics
def test_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.15).astype(int)
    p = np.clip(0.15 + 0.5 * (y - 0.15) + rng.normal(0, 0.2, 2000), 0.001, 0.999)
    m = compute_metrics(y, p, 0.3)
    pred = (p >= 0.3).astype(int)
    assert m["roc_auc"] == pytest.approx(skm.roc_auc_score(y, p))
    assert m["pr_auc"] == pytest.approx(skm.average_precision_score(y, p))
    assert m["f1"] == pytest.approx(skm.f1_score(y, pred))
    assert m["precision"] == pytest.approx(skm.precision_score(y, pred)) and m["recall"] == pytest.approx(skm.recall_score(y, pred))
    assert m["mcc"] == pytest.approx(skm.matthews_corrcoef(y, pred))
    assert m["tp"] + m["fp"] + m["tn"] + m["fn"] == len(y)
    assert m["brier"] == pytest.approx(skm.brier_score_loss(y, p))


def test_accuracy_is_misleading_but_pr_auc_is_not():
    y = np.r_[np.zeros(850), np.ones(150)].astype(int)
    m = compute_metrics(y, np.full(1000, 0.15), 0.5)           # "never predicts a medal"
    assert m["accuracy"] == pytest.approx(0.85) and m["recall"] == 0 and m["f1"] == 0
    assert m["pr_auc"] == pytest.approx(0.15) and m["roc_auc"] == pytest.approx(0.5)


def test_constant_predictions_do_not_crash_calibration():
    t = calibration_table(np.array([0, 1, 0, 1]), np.full(4, 0.5))
    assert len(t) == 1
    assert expected_calibration_error(np.array([0, 1, 0, 1]), np.full(4, 0.5))[0] == pytest.approx(0)


def test_ece_detects_miscalibration():
    rng = np.random.default_rng(0)
    p = rng.random(20000)
    y_good = (rng.random(20000) < p).astype(int)
    y_bad = (rng.random(20000) < p**3).astype(int)
    assert expected_calibration_error(y_good, p)[0] < 0.02 < expected_calibration_error(y_bad, p)[0]


def test_calibrators_fix_overconfident_probabilities():
    rng = np.random.default_rng(1)
    true_p = rng.random(30000)
    y = (rng.random(30000) < true_p).astype(int)
    over = np.clip(true_p**0.4, 0.001, 0.999)                  # distorted, overconfident scores
    before = expected_calibration_error(y, over)[0]
    for method in ("sigmoid", "isotonic"):
        cal = ProbabilityCalibrator(method).fit(over[:15000], y[:15000])
        assert expected_calibration_error(y[15000:], cal.transform(over[15000:]))[0] < before
    ident = ProbabilityCalibrator("none").fit(over, y)
    assert np.array_equal(ident.transform(over), over)


def test_cluster_bootstrap_interval_is_sane():
    rng = np.random.default_rng(2)
    groups = np.repeat(np.arange(200), 10)
    y = (rng.random(2000) < 0.2).astype(int)
    p = np.clip(y * 0.3 + rng.random(2000) * 0.6, 0, 1)
    lo, hi = cluster_bootstrap_ci(y, p, groups, "roc_auc", n_boot=100)
    assert lo < compute_metrics(y, p)["roc_auc"] < hi


# ---------------------------------------------------------------- threshold optimisation
@pytest.fixture()
def scores():
    rng = np.random.default_rng(3)
    y = (rng.random(5000) < 0.15).astype(int)
    return y, np.clip(0.12 + 0.35 * y + rng.normal(0, 0.15, 5000), 0.001, 0.999)


def test_f1_threshold_beats_default_half(scores):
    y, p = scores
    best = optimize_threshold(y, p, "fbeta", 1.0)
    assert compute_metrics(y, p, best["threshold"])["f1"] > compute_metrics(y, p, 0.5)["f1"]
    assert best["threshold"] < 0.5


def test_recall_constraint_and_cost_and_beta(scores):
    y, p = scores
    r = optimize_threshold(y, p, "precision_at_recall", target_recall=0.8)
    assert compute_metrics(y, p, r["threshold"])["recall"] >= 0.79
    cheap_fn = optimize_threshold(y, p, "cost", cost_fp=1, cost_fn=1)["threshold"]
    dear_fn = optimize_threshold(y, p, "cost", cost_fp=1, cost_fn=10)["threshold"]
    assert dear_fn < cheap_fn                                   # costlier misses -> lower threshold -> more flags
    assert optimize_threshold(y, p, "fbeta", beta=2)["threshold"] <= optimize_threshold(y, p, "fbeta", beta=0.5)["threshold"]
    assert 0 < optimize_threshold(y, p, "youden")["threshold"] < 1
    with pytest.raises(ValueError):
        optimize_threshold(y, p, "magic")
