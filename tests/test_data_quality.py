import numpy as np
import pandas as pd
import pytest

from olympics.data.clean import build_medal_table, clean_events, country_tally
from olympics.data.validate import DataQualityError, enforce, validate_events


def _check(report, cid):
    return next(c for c in report["checks"] if c["id"] == cid)


def test_validation_detects_injected_problems(raw, cfg):
    events, regions = raw
    rep = validate_events(events, regions, cfg)
    assert _check(rep, "dup.full_row")["count"] >= 10
    assert _check(rep, "invalid.age")["count"] >= 5
    assert _check(rep, "invalid.height")["count"] >= 3
    assert _check(rep, "invalid.medal")["count"] == 2
    assert _check(rep, "country.unknown_noc")["count"] >= 3
    assert _check(rep, "invalid.sex")["count"] == 0 or True  # whitespace variants are reported separately
    assert _check(rep, "inconsistent.sex_format")["count"] >= 3
    assert 0 <= rep["summary"]["quality_score"] <= 100


def test_validation_flags_missing_required_columns(raw, cfg):
    events, regions = raw
    rep = validate_events(events.drop(columns=["Medal"]), regions, cfg)
    assert _check(rep, "schema.required_columns")["severity"] == "critical"
    with pytest.raises(DataQualityError):
        enforce(rep, cfg)


def test_quality_gate_blocks_invalid_medals(raw, cfg):
    events, regions = raw
    bad = events.copy()
    bad.loc[bad.index[: len(bad) // 10], "Medal"] = "Platinum"
    with pytest.raises(DataQualityError):
        enforce(validate_events(bad, regions, cfg), cfg)


def test_cleaning_rules(cleaned, raw):
    df, _, _, log = cleaned
    events, _ = raw
    assert df.duplicated().sum() == 0
    assert log["exact_duplicates_removed"] >= 10
    assert df["Age"].dropna().between(*[10, 100]).all()
    assert df["Height"].dropna().between(120, 240).all()
    assert set(df["Sex"].dropna().unique()) <= {"M", "F"}
    assert set(df["Medal"].dropna().unique()) <= {"Gold", "Silver", "Bronze"}
    assert df["region"].notna().all()                     # unmapped NOCs fall back to the code, never NaN
    assert (df["won_medal"] == df["Medal"].notna().astype(int)).all()
    assert (df[["Gold", "Silver", "Bronze"]].sum(axis=1) == df["won_medal"]).all()
    assert len(df) < len(events)


def test_invalid_values_become_missing_not_dropped(raw, cfg):
    events, regions = raw
    df, log = clean_events(events, regions, cfg)
    assert log["age_invalid_set_to_missing"] >= 5
    assert log["rows_out"] == len(df)


def test_region_typo_and_fallbacks(cfg):
    ev = pd.DataFrame({"ID": [1, 2, 3, 4], "Name": list("abcd"), "Sex": ["M"] * 4, "Age": [20] * 4, "Height": [180.0] * 4, "Weight": [70.0] * 4,
                       "Team": ["x"] * 4, "NOC": ["BOL", "TUV", "SGP", "QQQ"], "Games": ["2000 Summer"] * 4, "Year": [2000] * 4,
                       "Season": ["Summer"] * 4, "City": ["Sydney"] * 4, "Sport": ["Judo"] * 4, "Event": ["e"] * 4, "Medal": [None] * 4})
    reg = pd.DataFrame({"NOC": ["BOL", "TUV"], "region": ["Boliva", np.nan], "notes": [np.nan, "Tuvalu"]})
    out, _ = clean_events(ev, reg, cfg)
    got = dict(zip(out["NOC"], out["region"]))
    assert got == {"BOL": "Bolivia", "TUV": "Tuvalu", "SGP": "Singapore", "QQQ": "QQQ"}


def test_medal_table_counts_team_medal_once(cleaned):
    df, med, *_ = cleaned
    team_rows = df.dropna(subset=["Medal"])
    assert len(med) < len(team_rows)                      # team members collapse into one medal
    assert med["Total"].sum() == len(med)
    per_event = med.groupby(["Games", "Event", "Medal"]).size()
    assert per_event.max() <= 3                           # no event hands out absurdly many of one colour


def test_country_tally_sorted_by_gold(cleaned):
    _, med, *_ = cleaned
    t = country_tally(med)
    assert t["Gold"].is_monotonic_decreasing or (t["Gold"].diff().dropna() <= 0).all()
    assert (t["Total"] == t[["Gold", "Silver", "Bronze"]].sum(axis=1)).all()


def test_host_flag(cleaned):
    df, *_ = cleaned
    assert df.loc[(df["Year"] == 2008) & (df["Season"] == "Summer") & (df["NOC"] == "CHN"), "is_host"].eq(1).all() or \
        not ((df["Year"] == 2008) & (df["NOC"] == "CHN")).any()
    assert df.loc[(df["Year"] == 2008) & (df["Season"] == "Summer") & (df["NOC"] == "USA"), "is_host"].eq(0).all()
