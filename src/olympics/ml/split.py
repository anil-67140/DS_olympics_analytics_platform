"""Train / validation / test splitting without leakage.

``group_stratified``: stratified on the medal label AND grouped so that all entries of a delegation at one
Games (default groups = Games x NOC) land in the same split. Without grouping, teammates who share a medal
would sit on both sides of the split and the model could memorise "Country X at Games Y wins" - a classic
source of inflated scores.

``temporal``: train on the oldest editions, validate on the next, test on the most recent (the strictest check).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def _carve(y: np.ndarray, groups: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    k = max(2, int(round(1 / fraction)))
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    rest, held = next(iter(sgkf.split(np.zeros(len(y)), y, groups)))
    return rest, held


def group_stratified_split(df: pd.DataFrame, target: str, group_cols: list[str], train: float, val: float,
                           test: float, seed: int = 42) -> dict[str, np.ndarray]:
    assert abs(train + val + test - 1) < 1e-6, "split fractions must sum to 1"
    groups = df[group_cols].astype(str).agg("|".join, axis=1).to_numpy()
    y = df[target].to_numpy()
    idx = np.arange(len(df))
    rest, test_idx = _carve(y, groups, test, seed)
    val_frac_of_rest = val / (train + val)
    tr_rel, va_rel = _carve(y[rest], groups[rest], val_frac_of_rest, seed + 1)
    return {"train": idx[rest][tr_rel], "val": idx[rest][va_rel], "test": idx[test_idx]}


def temporal_split(df: pd.DataFrame, val_editions: int = 1, test_editions: int = 1) -> dict[str, np.ndarray]:
    editions = (df[["Year", "Season"]].drop_duplicates().sort_values(["Year", "Season"]).reset_index(drop=True))
    cut_test = editions.iloc[-test_editions:]
    cut_val = editions.iloc[-(test_editions + val_editions):-test_editions]
    key = df["Year"].astype(str) + "|" + df["Season"].astype(str)
    mk = lambda e: set((e["Year"].astype(str) + "|" + e["Season"].astype(str)))  # noqa: E731
    test_mask, val_mask = key.isin(mk(cut_test)), key.isin(mk(cut_val))
    idx = np.arange(len(df))
    return {"train": idx[~(test_mask | val_mask).to_numpy()], "val": idx[val_mask.to_numpy()], "test": idx[test_mask.to_numpy()]}


def make_splits(df: pd.DataFrame, cfg: dict) -> dict[str, np.ndarray]:
    s = cfg["ml"]["split"]
    if s["mode"] == "temporal":
        return temporal_split(df)
    return group_stratified_split(df, cfg["ml"]["target"], s["group_cols"], s["train"], s["val"], s["test"], cfg["ml"]["seed"])


def split_report(df: pd.DataFrame, splits: dict[str, np.ndarray], target: str, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    g = df[group_cols].astype(str).agg("|".join, axis=1)
    sets = {k: set(g.iloc[v]) for k, v in splits.items()}
    for name, idx in splits.items():
        rows.append({"split": name, "rows": len(idx), "positive_rate": float(df[target].iloc[idx].mean()),
                     "groups": len(sets[name]), "min_year": int(df["Year"].iloc[idx].min()), "max_year": int(df["Year"].iloc[idx].max())})
    out = pd.DataFrame(rows)
    names = list(splits)
    overlaps = {f"{a}&{b}": len(sets[a] & sets[b]) for i, a in enumerate(names) for b in names[i + 1:]}
    out.attrs["group_overlap"] = overlaps
    return out
