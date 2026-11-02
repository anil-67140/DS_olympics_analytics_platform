"""Cleaning + derived columns + the canonical de-duplicated medal table."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import HOST_NOC, MEDAL_VALUES, REGION_FALLBACK_BY_NOC, REGION_FIXES

STRING_COLS = ["Name", "Sex", "Team", "NOC", "Games", "Season", "City", "Sport", "Event", "Medal"]


def clean_events(events: pd.DataFrame, regions: pd.DataFrame | None, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Return ``(clean_df, log)``; ``log`` records how many rows/values every rule touched."""
    log: dict = {"rows_in": int(len(events))}
    df = events.copy()
    vcfg, dcfg = cfg["validation"], cfg["data"]

    # 1. tidy strings ------------------------------------------------------------------------------------
    for col in STRING_COLS:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    df["Sex"] = df["Sex"].str.upper()
    log["sex_invalid_set_to_missing"] = int((~df["Sex"].isin(["M", "F"]) & df["Sex"].notna()).sum())
    df.loc[~df["Sex"].isin(["M", "F"]), "Sex"] = pd.NA
    df["NOC"] = df["NOC"].str.upper()

    # 2. exact duplicates --------------------------------------------------------------------------------
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    log["exact_duplicates_removed"] = int(before - len(df))

    # 3. scope filters -----------------------------------------------------------------------------------
    before = len(df)
    df = df[df["Season"].isin(dcfg["seasons"]) & ~df["Year"].isin(dcfg.get("exclude_years") or [])].copy()
    log["rows_filtered_by_scope"] = int(before - len(df))
    if dcfg.get("sample_frac"):
        df = df.sample(frac=float(dcfg["sample_frac"]), random_state=cfg["ml"]["seed"]).copy()
        log["sampled_fraction"] = float(dcfg["sample_frac"])

    # 4. invalid measurements -> missing (kept as rows, flagged, never silently "fixed") -------------------
    for col, key in (("Age", "age_range"), ("Height", "height_range"), ("Weight", "weight_range")):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        lo, hi = vcfg[key]
        bad = df[col].notna() & ~df[col].between(lo, hi)
        log[f"{col.lower()}_invalid_set_to_missing"] = int(bad.sum())
        df.loc[bad, col] = np.nan
    bmi = df["Weight"] / (df["Height"] / 100) ** 2
    lo, hi = vcfg["bmi_range"]
    bad_bmi = bmi.notna() & ~bmi.between(lo, hi)
    log["bmi_invalid_height_weight_cleared"] = int(bad_bmi.sum())
    df.loc[bad_bmi, ["Height", "Weight"]] = np.nan

    # 5. medal normalisation -----------------------------------------------------------------------------
    df["Medal"] = df["Medal"].str.title()
    log["medal_invalid_set_to_missing"] = int((df["Medal"].notna() & ~df["Medal"].isin(MEDAL_VALUES)).sum())
    df.loc[~df["Medal"].isin(MEDAL_VALUES), "Medal"] = pd.NA

    # a row that differed only by an invalid value is now an exact duplicate: drop it too
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    log["duplicates_after_invalid_values_cleared"] = int(before - len(df))

    # 6. countries ---------------------------------------------------------------------------------------
    if regions is not None:
        reg = regions.copy()
        reg["NOC"] = reg["NOC"].astype("string").str.strip().str.upper()
        reg["region"] = reg["region"].astype("string").str.strip().replace(REGION_FIXES)
        for noc, name in REGION_FALLBACK_BY_NOC.items():
            reg.loc[(reg["NOC"] == noc) & reg["region"].isna(), "region"] = name
        df = df.merge(reg[["NOC", "region"]].drop_duplicates("NOC"), on="NOC", how="left")
    else:
        df["region"] = pd.NA
    unmapped = df["region"].isna()
    df.loc[unmapped, "region"] = df.loc[unmapped, "NOC"].map(REGION_FALLBACK_BY_NOC)
    still = df["region"].isna()
    log["rows_region_fallback_to_noc"] = int(still.sum())
    df.loc[still, "region"] = df.loc[still, "NOC"]

    # 7. derived columns ---------------------------------------------------------------------------------
    df["Year"] = df["Year"].astype(int)
    df["BMI"] = df["Weight"] / (df["Height"] / 100) ** 2
    df["birth_year"] = df["Year"] - df["Age"]
    df["won_medal"] = df["Medal"].notna().astype(int)
    for m in MEDAL_VALUES:
        df[m] = df["Medal"].eq(m).fillna(False).astype(int)
    df["decade"] = (df["Year"] // 10) * 10
    df["host_noc"] = [HOST_NOC.get((y, s)) for y, s in zip(df["Year"], df["Season"])]
    df["is_host"] = df["NOC"].astype(object).eq(df["host_noc"]).astype(int)
    grp = df.groupby(["Games", "Event"])
    df["event_entrants"] = grp["ID"].transform("size")
    df["noc_event_entrants"] = df.groupby(["Games", "Event", "NOC"])["ID"].transform("size")
    df["is_team_event"] = (df.groupby(["Games", "Event"])["noc_event_entrants"].transform("max") > 1).astype(int)

    df = df.sort_values(["Year", "Season", "NOC", "Sport", "Event", "ID"]).reset_index(drop=True)
    log["rows_out"] = int(len(df))
    return df, log


def build_medal_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per medal *awarded* (team events count once, not once per team member).

    Caveat (documented): two same-colour medals won by the same NOC in the same event (very rare ties)
    collapse into one.
    """
    med = df.dropna(subset=["Medal"])
    keys = ["Games", "Year", "Season", "NOC", "region", "Sport", "Event", "Medal"]
    out = med.drop_duplicates(subset=keys)[keys].reset_index(drop=True)
    for m in MEDAL_VALUES:
        out[m] = (out["Medal"] == m).astype(int)
    out["Total"] = 1
    return out


def country_tally(medals: pd.DataFrame, by: str = "region") -> pd.DataFrame:
    t = medals.groupby(by)[MEDAL_VALUES + ["Total"]].sum()
    return t.sort_values(["Gold", "Silver", "Bronze"], ascending=False).reset_index()
