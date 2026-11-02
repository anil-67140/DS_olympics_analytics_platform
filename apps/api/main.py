"""FastAPI inference service.

Run:  OLYMPICS_ROOT=/path/to/workspace uvicorn apps.api.main:app --host 0.0.0.0 --port 8000
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from olympics.config import Paths, load_config
from olympics.feedback.store import FeedbackStore
from olympics.io_utils import read_json
from olympics.ml.registry import list_versions
from olympics.serving import Predictor

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    paths = Paths.from_env(os.environ.get("OLYMPICS_ROOT", "workspace")).ensure()
    STATE.update(paths=paths, cfg=load_config(), store=FeedbackStore(paths.feedback), started=time.time(), requests=0)
    try:
        STATE["predictor"] = Predictor(paths.models, os.environ.get("MODEL_VERSION") or None)
    except FileNotFoundError:
        STATE["predictor"] = None  # service starts; /health reports "no model" until the pipeline has run
    yield


app = FastAPI(title="Olympics Medal Prediction API", version="1.0.0", lifespan=lifespan,
              description="Probability that an athlete entry wins a medal, with explanations and a human feedback loop.")


# ------------------------------------------------------------------ schemas
class AthleteEntry(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "Age": 24, "Height": 185, "Weight": 78, "Sex": "M", "NOC": "USA", "Sport": "Swimming", "Season": "Summer", "Year": 2016,
        "event_entrants": 40, "noc_event_entrants": 2, "athlete_prev_games": 2, "athlete_prev_medals": 3}})
    Age: float | None = Field(None, ge=10, le=100)
    Height: float | None = Field(None, ge=120, le=240, description="cm")
    Weight: float | None = Field(None, ge=25, le=250, description="kg")
    Sex: Literal["M", "F"]
    NOC: str = Field(min_length=2, max_length=4, description="3-letter National Olympic Committee code, e.g. USA")
    Sport: str = Field(min_length=2)
    Season: Literal["Summer", "Winter"]
    Year: int = Field(ge=1896, le=2100)
    event_entrants: int | None = Field(None, ge=1, description="Athletes in the event (all countries)")
    noc_event_entrants: int | None = Field(None, ge=1, description="Athletes from this country in the event (team size)")
    athlete_prev_games: int | None = Field(None, ge=0)
    athlete_prev_medals: int | None = Field(None, ge=0)

    def to_request(self) -> dict:
        d = self.model_dump()
        d["NOC"] = d["NOC"].upper()
        return d


class Prediction(BaseModel):
    prediction_id: str
    probability: float
    predicted_medal: bool
    threshold: float
    model_version: str
    explanation: list[dict] | None = None


class BatchRequest(BaseModel):
    entries: list[AthleteEntry] = Field(min_length=1, max_length=500)


class FeedbackIn(BaseModel):
    prediction_id: str
    actual_medal: bool
    reviewer: str = "anonymous"
    note: str = ""


def _predictor() -> Predictor:
    if STATE.get("predictor") is None:
        raise HTTPException(503, "No trained model is available. Run the pipeline first.")
    return STATE["predictor"]


# ------------------------------------------------------------------ endpoints
@app.get("/health")
def health():
    p = STATE.get("predictor")
    return {"status": "ok" if p else "no_model", "model_version": p.version if p else None, "uptime_s": round(time.time() - STATE["started"], 1)}


@app.get("/model")
def model_info():
    p = _predictor()
    m = p.meta
    return {"version": p.version, "algorithm": m.get("algorithm"), "calibration": m.get("calibration"), "threshold": p.model.threshold,
            "test_metrics": {k: m["test_metrics"][k] for k in ("pr_auc", "roc_auc", "f1", "precision", "recall", "brier", "ece")},
            "created_at": m.get("created_at"), "available_versions": list_versions(STATE["paths"].models)}


@app.get("/catalog")
def catalog(season: str | None = Query(None, pattern="^(Summer|Winter)$")):
    p = _predictor()
    return {"nocs": p.nocs, "sports": p.sports(season)}


def _score(entries: list[AthleteEntry], explain: bool) -> list[Prediction]:
    p, store = _predictor(), STATE["store"]
    results = p.predict([e.to_request() for e in entries], explain=explain)
    out = []
    for r in results:
        pid = store.log_prediction(r["features_used"], r["probability"], int(r["predicted_medal"]), r["model_version"], r["threshold"])
        out.append(Prediction(prediction_id=pid, probability=round(r["probability"], 5), predicted_medal=r["predicted_medal"], threshold=r["threshold"],
                              model_version=r["model_version"], explanation=r.get("explanation")))
    STATE["requests"] += len(out)
    return out


@app.post("/predict", response_model=Prediction)
def predict(entry: AthleteEntry, explain: bool = False):
    return _score([entry], explain)[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(req: BatchRequest):
    return _score(req.entries, False)


@app.post("/feedback")
def feedback(fb: FeedbackIn):
    try:
        STATE["store"].log_feedback(fb.prediction_id, fb.actual_medal, fb.reviewer, fb.note)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"status": "recorded", **STATE["store"].summary()}


@app.get("/monitoring")
def monitoring():
    paths = STATE["paths"]
    return {"service": {"requests_served": STATE["requests"]}, "feedback": STATE["store"].summary(),
            "offline_monitoring": read_json(paths.metrics / "monitoring_summary.json", {})}


@app.post("/admin/reload")
def reload_model():
    STATE["predictor"] = Predictor(STATE["paths"].models)
    return {"status": "reloaded", "model_version": STATE["predictor"].version}
