"""Leakage-safe feature engineering.

Two kinds of leakage are prevented here:

1. **History features use strictly earlier editions only** (athlete's past medals, country and country-sport
   past medal rates). They never look at the row's own Games, so they are knowable before the event.
2. **Anything that learns from the label (target encoding of NOC / Sport, imputation medians, scaling) lives
   inside the sklearn Pipeline** and is therefore fitted on the training split only, never on validation/test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, TargetEncoder

from ..data.schema import (ALL_FEATURES, HIGH_CARD_CATEGORICAL, LOW_CARD_CATEGORICAL, NUMERIC_FEATURES, TARGET)

BASE_RATE = 0.15  # fallback prior before any history exists (overall medal rate is ~14.7%)


def _prior_cumulative(agg: pd.DataFrame, keys: list[str], entries: str, medals: str) -> pd.DataFrame:
    """Cumulative entries/medals over *earlier* years only (one row per key x Year)."""
    agg = agg.sort_values(keys + ["Year"]).copy()
    g = agg.groupby(keys, sort=False)
    agg["prev_entries"] = g[entries].cumsum() - agg[entries]
    agg["prev_medals"] = g[medals].cumsum() - agg[medals]
    return agg


def add_history_features(df: pd.DataFrame, prior: float = 20.0) -> pd.DataFrame:
    out = df.copy()

    # global rate from earlier years only
    yr = out.groupby("Year")["won_medal"].agg(entries="size", medals="sum").sort_index()
    yr["prev_rate"] = ((yr["medals"].cumsum() - yr["medals"]) / (yr["entries"].cumsum() - yr["entries"]).replace(0, np.nan)).fillna(BASE_RATE)
    out["_global_prev_rate"] = out["Year"].map(yr["prev_rate"])

    # athlete history (across both seasons, strictly earlier years)
    a = out.groupby(["ID", "Year"]).agg(a_entries=("won_medal", "size"), a_medals=("won_medal", "sum")).reset_index()
    a = a.sort_values(["ID", "Year"])
    a["athlete_prev_games"] = a.groupby("ID").cumcount()
    a["athlete_prev_medals"] = a.groupby("ID")["a_medals"].cumsum() - a["a_medals"]
    out = out.merge(a[["ID", "Year", "athlete_prev_games", "athlete_prev_medals"]], on=["ID", "Year"], how="left")

    # country history within the same season (Summer and Winter are different competitions)
    n = out.groupby(["NOC", "Season", "Year"]).agg(e=("won_medal", "size"), m=("won_medal", "sum")).reset_index()
    n = _prior_cumulative(n, ["NOC", "Season"], "e", "m")
    n = n.rename(columns={"prev_entries": "noc_prev_entries", "prev_medals": "_noc_prev_medals"})
    out = out.merge(n[["NOC", "Season", "Year", "noc_prev_entries", "_noc_prev_medals"]], on=["NOC", "Season", "Year"], how="left")
    out["noc_prev_medal_rate"] = (out["_noc_prev_medals"] + prior * out["_global_prev_rate"]) / (out["noc_prev_entries"] + prior)

    # country x sport history (hierarchical shrinkage toward the country's own rate)
    s = out.groupby(["NOC", "Sport", "Season", "Year"]).agg(e=("won_medal", "size"), m=("won_medal", "sum")).reset_index()
    s = _prior_cumulative(s, ["NOC", "Sport", "Season"], "e", "m")
    s = s.rename(columns={"prev_entries": "_sp_prev_entries", "prev_medals": "_sp_prev_medals"})
    out = out.merge(s[["NOC", "Sport", "Season", "Year", "_sp_prev_entries", "_sp_prev_medals"]],
                    on=["NOC", "Sport", "Season", "Year"], how="left")
    out["sport_noc_prev_medal_rate"] = ((out["_sp_prev_medals"] + prior * out["noc_prev_medal_rate"])
                                         / (out["_sp_prev_entries"] + prior))
    return out.drop(columns=[c for c in out.columns if c.startswith("_")])


def build_ml_frame(clean: pd.DataFrame, prior: float = 20.0) -> pd.DataFrame:
    df = add_history_features(clean, prior)
    keep = list(dict.fromkeys(["ID", "Games", "Event", "Name", *ALL_FEATURES, TARGET]))
    return df[[c for c in keep if c in df.columns]].reset_index(drop=True)


def build_history_lookup(clean: pd.DataFrame, prior: float = 20.0) -> dict:
    """Snapshot of cumulative history after the last edition; the API uses it to fill history features."""
    base = clean["won_medal"].mean()
    noc = clean.groupby(["Season", "NOC"])["won_medal"].agg(e="size", m="sum")
    sp = clean.groupby(["Season", "NOC", "Sport"])["won_medal"].agg(e="size", m="sum")
    lookup = {"global_rate": float(base), "prior": prior, "noc": {}, "sport_noc": {}, "last_year": int(clean["Year"].max())}
    for (season, n), r in noc.iterrows():
        lookup["noc"].setdefault(season, {})[n] = {"entries": int(r.e), "medals": int(r.m)}
    for (season, n, sport), r in sp.iterrows():
        lookup["sport_noc"].setdefault(season, {}).setdefault(n, {})[sport] = {"entries": int(r.e), "medals": int(r.m)}
    return lookup


# ---------------------------------------------------------------------------------------------- model input
def to_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Select + type the exact columns the pipeline expects (used identically in training and serving)."""
    X = pd.DataFrame(index=df.index)
    for c in NUMERIC_FEATURES:
        X[c] = pd.to_numeric(df[c], errors="coerce").astype("float64") if c in df else np.nan
    for c in LOW_CARD_CATEGORICAL + HIGH_CARD_CATEGORICAL:
        X[c] = df[c].astype(object).where(df[c].notna(), "UNKNOWN") if c in df else "UNKNOWN"
    return X[ALL_FEATURES]


def _to_str(X):
    return pd.DataFrame(X).astype(str).to_numpy(dtype=object)


def make_preprocessor(seed: int = 42, target_encode: bool = True) -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())])
    low = Pipeline([("str", FunctionTransformer(_to_str)), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    if target_encode:
        high = Pipeline([("str", FunctionTransformer(_to_str)),
                         ("te", TargetEncoder(target_type="binary", smooth="auto", cv=5, shuffle=True, random_state=seed))])
    else:
        high = Pipeline([("str", FunctionTransformer(_to_str)), ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False))])
    return ColumnTransformer([("num", numeric, NUMERIC_FEATURES), ("low", low, LOW_CARD_CATEGORICAL),
                              ("high", high, HIGH_CARD_CATEGORICAL)], remainder="drop", verbose_feature_names_out=False)
