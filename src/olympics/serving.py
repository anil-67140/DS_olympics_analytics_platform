"""Inference core shared by the FastAPI service and the Gradio dashboard (one code path = identical predictions)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data.schema import ALL_FEATURES, HOST_NOC
from .io_utils import read_json
from .ml.explain import baseline_row, local_explanation
from .ml.features import to_model_frame
from .ml.registry import latest_version, load_model


class Predictor:
    def __init__(self, models_dir: Path, version: str | None = None):
        self.models_dir = Path(models_dir)
        self.model, self.meta = load_model(self.models_dir, version)
        self.version = self.meta["version"]
        vdir = self.models_dir / self.version
        self.lookup = read_json(vdir / "history_lookup.json") or read_json(self.models_dir / latest_version(self.models_dir) / "history_lookup.json", {})
        self._baseline = self._make_baseline()

    # ------------------------------------------------------------------ catalogue (for dropdowns / validation)
    @property
    def nocs(self) -> list[str]:
        return sorted({n for s in self.lookup.get("noc", {}).values() for n in s})

    def sports(self, season: str | None = None) -> list[str]:
        seasons = [season] if season else list(self.lookup.get("sport_noc", {}))
        return sorted({sp for s in seasons for d in self.lookup.get("sport_noc", {}).get(s, {}).values() for sp in d})

    # ------------------------------------------------------------------ feature construction
    def build_features(self, req: dict) -> dict:
        r = dict(req)
        season, noc, sport = r["Season"], r["NOC"], r["Sport"]
        prior, g = self.lookup.get("prior", 20.0), self.lookup.get("global_rate", 0.15)
        noc_h = self.lookup.get("noc", {}).get(season, {}).get(noc, {"entries": 0, "medals": 0})
        sp_h = self.lookup.get("sport_noc", {}).get(season, {}).get(noc, {}).get(sport, {"entries": 0, "medals": 0})
        noc_rate = (noc_h["medals"] + prior * g) / (noc_h["entries"] + prior)
        sp_rate = (sp_h["medals"] + prior * noc_rate) / (sp_h["entries"] + prior)
        h, w = r.get("Height"), r.get("Weight")
        r.setdefault("noc_prev_entries", noc_h["entries"])
        r["noc_prev_entries"] = r.get("noc_prev_entries") if r.get("noc_prev_entries") is not None else noc_h["entries"]
        if r.get("noc_prev_medal_rate") is None:
            r["noc_prev_medal_rate"] = noc_rate
        if r.get("sport_noc_prev_medal_rate") is None:
            r["sport_noc_prev_medal_rate"] = sp_rate
        r["BMI"] = (w / (h / 100) ** 2) if h and w else np.nan
        r["is_host"] = int(HOST_NOC.get((int(r["Year"]), season)) == noc) if r.get("is_host") is None else int(r["is_host"])
        r["noc_event_entrants"] = r.get("noc_event_entrants") or 1
        r["event_entrants"] = r.get("event_entrants") or max(r["noc_event_entrants"], 20)
        r["athlete_prev_games"] = r.get("athlete_prev_games") or 0
        r["athlete_prev_medals"] = r.get("athlete_prev_medals") or 0
        return {k: r.get(k) for k in ALL_FEATURES}

    def _frame(self, feats: list[dict]) -> pd.DataFrame:
        return to_model_frame(pd.DataFrame(feats))

    def predict(self, requests: list[dict], explain: bool = False) -> list[dict]:
        feats = [self.build_features(r) for r in requests]
        X = self._frame(feats)
        proba = self.model.predict_proba(X)
        out = []
        for i, (f, p) in enumerate(zip(feats, proba)):
            item = {"probability": float(p), "predicted_medal": bool(p >= self.model.threshold),
                    "threshold": self.model.threshold, "model_version": self.version, "features_used": f}
            if explain:
                ex = local_explanation(self.model, X.iloc[[i]].reset_index(drop=True), self._baseline, top=6)
                item["explanation"] = [{"feature": r.feature, "value": _plain(r.value), "typical_value": _plain(r.baseline_value),
                                        "effect_on_probability": float(r.effect_on_probability)} for r in ex.itertuples()]
            out.append(item)
        return out

    def _make_baseline(self) -> pd.Series:
        base = {"Age": 25.0, "Height": 175.0, "Weight": 70.0, "BMI": 22.9, "Year": float(self.lookup.get("last_year", 2016)),
                "athlete_prev_games": 0.0, "athlete_prev_medals": 0.0, "noc_prev_entries": 500.0, "noc_prev_medal_rate": self.lookup.get("global_rate", 0.15),
                "sport_noc_prev_medal_rate": self.lookup.get("global_rate", 0.15), "event_entrants": 60.0, "noc_event_entrants": 1.0,
                "is_host": 0.0, "Sex": "M", "Season": "Summer", "NOC": "UNKNOWN", "Sport": "UNKNOWN"}
        return pd.Series({k: base[k] for k in ALL_FEATURES})


def _plain(v):
    if isinstance(v, (np.floating, float)):
        return None if v != v else round(float(v), 4)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v
