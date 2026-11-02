import json

import numpy as np
import pandas as pd
import pytest

from olympics.checkpoint import Checkpointer, cached_json, fingerprint
from olympics.config import Paths, config_fingerprint, load_config
from olympics.feedback.store import FeedbackStore
from olympics.io_utils import frame_path, load_frame, read_json, save_frame, write_json
from olympics.monitoring import drift
from olympics.tracking import Tracker


# ---------------------------------------------------------------- config
def test_config_overrides_and_fingerprint():
    c = load_config(overrides=["ml.profile=fast", "ml.threshold.beta=2", "data.sample_frac=0.5", "ml.models=[a,b]"])
    assert c["ml"]["profile"] == "fast" and c["ml"]["threshold"]["beta"] == 2 and c["data"]["sample_frac"] == 0.5 and c["ml"]["models"] == ["a", "b"]
    assert config_fingerprint(c, ["ml"]) != config_fingerprint(load_config(), ["ml"])
    assert config_fingerprint(c, ["ml"]) == config_fingerprint(json.loads(json.dumps(c)), ["ml"])
    with pytest.raises(ValueError):
        load_config(overrides=["badoverride"])


# ---------------------------------------------------------------- checkpoints / atomic IO
def test_checkpoint_skips_then_reruns_when_inputs_change(tmp_path):
    ck, out, calls = Checkpointer(tmp_path / "ck"), tmp_path / "out.txt", []

    def work():
        calls.append(1)
        out.write_text("x")
        return {"n": 1}
    assert ck.run("s", work, "fp1", [out]) == {"n": 1} and len(calls) == 1
    assert ck.run("s", work, "fp1", [out]) == {"n": 1} and len(calls) == 1          # skipped
    ck.run("s", work, "fp2", [out]); assert len(calls) == 2                          # config changed -> rerun
    out.unlink(); ck.run("s", work, "fp2", [out]); assert len(calls) == 3            # output lost -> rerun
    ck.run("s", work, "fp2", [out], force=True); assert len(calls) == 4
    assert "s" in ck.status()


def test_failed_stage_leaves_no_marker(tmp_path):
    ck = Checkpointer(tmp_path)
    with pytest.raises(RuntimeError):
        ck.run("boom", lambda: (_ for _ in ()).throw(RuntimeError("disconnect")), "f")
    assert "boom" not in ck.status()


def test_item_cache_resumes_after_crash(tmp_path):
    p, calls = tmp_path / "m.json", []
    assert cached_json(p, lambda: calls.append(1) or {"a": 1}) == {"a": 1}
    assert cached_json(p, lambda: calls.append(1) or {"a": 2}) == {"a": 1} and len(calls) == 1


def test_frame_roundtrip_and_atomic_write(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", None]})
    save_frame(df, tmp_path / "t")
    assert frame_path(tmp_path / "t") is not None and load_frame(tmp_path / "t").shape == (2, 2)
    write_json(tmp_path / "j.json", {"v": np.float32(1.5), "n": np.int64(3), "nan": float("nan")})
    assert read_json(tmp_path / "j.json") == {"v": 1.5, "n": 3, "nan": None}
    assert not list(tmp_path.glob("*.tmp"))                                           # no temp-file debris
    (tmp_path / "bad.json").write_text("{not json")
    assert read_json(tmp_path / "bad.json", default="fallback") == "fallback"


def test_paths_are_all_under_root(tmp_path):
    p = Paths.from_env(tmp_path).ensure()
    for attr in ("raw", "interim", "processed", "db", "figures", "tables", "metrics", "models", "mlruns", "checkpoints", "feedback", "logs"):
        assert str(getattr(p, attr)).startswith(str(tmp_path)) and getattr(p, attr).is_dir()


# ---------------------------------------------------------------- tracking
def test_tracker_always_writes_jsonl(tmp_path):
    t = Tracker(tmp_path / "runs")
    t.log_run("r1", {"model": "x"}, {"pr_auc": 0.5, "note": "text"}, tags={"stage": "t"})
    runs = t.runs()
    assert len(runs) == 1 and runs[0]["metrics"] == {"pr_auc": 0.5} and runs[0]["params"]["model"] == "x"


# ---------------------------------------------------------------- feedback loop
def _feat():
    return {"Age": 24, "Sex": "M", "NOC": "USA", "Sport": "Swimming", "Season": "Summer", "Year": 2016}


def test_feedback_store_join_and_summary(tmp_path):
    s = FeedbackStore(tmp_path)
    a = s.log_prediction(_feat(), 0.8, 1, "v1", 0.3)
    b = s.log_prediction(_feat(), 0.1, 0, "v1", 0.3)
    s.log_prediction(_feat(), 0.5, 1, "v1", 0.3)                                      # never reviewed
    s.log_feedback(a, True, "r1")
    s.log_feedback(b, True, "r1")
    s.log_feedback(b, False, "r2", "corrected")                                       # latest feedback wins
    lab = s.labelled().set_index("prediction_id")
    assert len(lab) == 2 and lab.loc[b, "actual"] == 0 and lab.loc[a, "NOC"] == "USA"
    assert s.summary()["predictions"] == 3 and s.summary()["labelled"] == 2
    with pytest.raises(KeyError):
        s.log_feedback("missing", True)


def test_feedback_store_survives_torn_line(tmp_path):
    s = FeedbackStore(tmp_path)
    s.log_prediction(_feat(), 0.8, 1, "v1", 0.3)
    with open(s.pred_path, "a") as fh:
        fh.write('{"prediction_id": "half-writ')                                      # simulated crash mid-write
    assert len(s.predictions()) == 1


# ---------------------------------------------------------------- drift
def test_psi_zero_for_same_distribution_high_for_shift():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 20000), rng.normal(0, 1, 20000)
    assert drift.psi_numeric(a, b) < 0.02
    assert drift.psi_numeric(a, rng.normal(1.5, 1, 20000)) > 0.25
    cat_ref, cat_cur = pd.Series(["a"] * 800 + ["b"] * 200), pd.Series(["a"] * 200 + ["b"] * 800)
    assert drift.psi_categorical(cat_ref, cat_ref) == pytest.approx(0, abs=1e-9) and drift.psi_categorical(cat_ref, cat_cur) > 0.25


def test_detect_drift_flags_shifted_feature(cleaned, cfg):
    from olympics.ml.features import build_ml_frame, to_model_frame
    X = to_model_frame(build_ml_frame(cleaned[0]))
    cur = X.copy()
    cur["Age"] = cur["Age"] + 8
    t = drift.detect_drift(X, cur, cfg).set_index("feature")
    assert t.loc["Age", "status"] == "alert" and t.loc["Sex", "status"] == "ok"
    assert drift.drift_summary(drift.detect_drift(X, cur, cfg))["overall"] == "alert"
    assert drift.drift_summary(drift.detect_drift(X, X, cfg))["overall"] == "ok"


def test_performance_monitor_flags_degradation(cfg):
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.2).astype(int)
    good = np.clip(0.1 + 0.6 * y + rng.normal(0, 0.15, 4000), 0.01, 0.99)
    bad = np.clip(0.3 + 0.05 * y + rng.normal(0, 0.2, 4000), 0.01, 0.99)
    assert drift.performance_monitor(y, good, y, good, 0.4, cfg)["status"] == "healthy"
    assert drift.performance_monitor(y, good, y, bad, 0.4, cfg)["status"] == "degraded"
