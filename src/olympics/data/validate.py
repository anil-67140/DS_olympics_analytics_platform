"""Data-quality validation: missing values, duplicates, inconsistent countries, invalid records.

``validate_events`` never mutates data; it returns a structured report that is written as JSON, Markdown
and CSV, and can optionally fail the pipeline when critical thresholds are exceeded.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..io_utils import save_table, write_json, write_text
from .schema import MEDAL_VALUES, REGION_FIXES, REQUIRED_COLUMNS


class DataQualityError(RuntimeError):
    pass


class _Report:
    def __init__(self, total: int):
        self.total = max(total, 1)
        self.checks: list[dict] = []

    def add(self, cid: str, title: str, severity: str, count: int, examples=None, note: str = "") -> None:
        self.checks.append({
            "id": cid, "title": title, "severity": severity if count else "pass", "count": int(count),
            "pct_of_rows": round(100 * count / self.total, 4),
            "examples": [str(e) for e in (examples or [])][:5], "note": note,
        })


def _examples(series: pd.Series, n: int = 5) -> list:
    return series.dropna().astype(str).unique()[:n].tolist()


def validate_events(df: pd.DataFrame, regions: pd.DataFrame | None, cfg: dict) -> dict:
    vcfg = cfg["validation"]
    rep = _Report(len(df))

    # 1. schema -------------------------------------------------------------------------------------------
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    rep.add("schema.required_columns", "Required columns present", "critical", len(missing_cols),
            missing_cols, "All of: " + ", ".join(REQUIRED_COLUMNS))
    if missing_cols:
        return _finish(rep, df, {}, regions)

    # 2. missing values -----------------------------------------------------------------------------------
    miss = df.isna().sum()
    missingness = {c: {"missing": int(miss[c]), "pct": round(100 * miss[c] / rep.total, 3)} for c in df.columns}
    for col in ("Age", "Height", "Weight"):
        if col in df.columns:
            rep.add(f"missing.{col}", f"Missing {col}", "warning" if miss[col] / rep.total > 0.05 else "info",
                    int(miss[col]), note="Handled by median imputation + missing-indicator in the model pipeline.")
    for col in ("ID", "NOC", "Year", "Sport", "Event"):
        rep.add(f"missing.{col}", f"Missing {col} (key field)", "critical", int(miss[col]))

    # 3. duplicates ---------------------------------------------------------------------------------------
    full_dup = int(df.duplicated().sum())
    rep.add("dup.full_row", "Exact duplicate rows", "warning", full_dup,
            note="Removed during cleaning.")
    dedup = df.drop_duplicates()
    key_dup = int(dedup.duplicated(subset=["ID", "Games", "Event"]).sum())
    rep.add("dup.athlete_event", "Same athlete listed twice for the same Games+Event", "warning", key_dup,
            note="Usually data-entry repeats; kept after full-row dedup so counts stay conservative.")

    # 4. invalid records ----------------------------------------------------------------------------------
    lo, hi = vcfg["age_range"]
    bad_age = df["Age"].notna() & ~df["Age"].between(lo, hi)
    rep.add("invalid.age", f"Age outside [{lo}, {hi}]", "warning", int(bad_age.sum()), _examples(df.loc[bad_age, "Age"]))
    lo, hi = vcfg["height_range"]
    if "Height" in df.columns:
        bad_h = df["Height"].notna() & ~df["Height"].between(lo, hi)
        rep.add("invalid.height", f"Height outside [{lo}, {hi}] cm", "warning", int(bad_h.sum()),
                _examples(df.loc[bad_h, "Height"]))
    lo, hi = vcfg["weight_range"]
    if "Weight" in df.columns:
        bad_w = df["Weight"].notna() & ~df["Weight"].between(lo, hi)
        rep.add("invalid.weight", f"Weight outside [{lo}, {hi}] kg", "warning", int(bad_w.sum()),
                _examples(df.loc[bad_w, "Weight"]))
    if {"Height", "Weight"} <= set(df.columns):
        bmi = df["Weight"] / (df["Height"] / 100) ** 2
        lo, hi = vcfg["bmi_range"]
        rep.add("invalid.bmi", f"Implied BMI outside [{lo}, {hi}]", "warning", int((bmi.notna() & ~bmi.between(lo, hi)).sum()))

    sex_clean = df["Sex"].astype("string").str.strip().str.upper()
    bad_sex = df["Sex"].notna() & ~sex_clean.isin(["M", "F"])
    rep.add("invalid.sex", "Sex not coded as M/F (after trimming)", "warning", int(bad_sex.sum()), _examples(df.loc[bad_sex, "Sex"]))
    raw_sex_variants = df["Sex"].notna() & (df["Sex"].astype("string") != sex_clean) & sex_clean.isin(["M", "F"])
    rep.add("inconsistent.sex_format", "Sex with stray whitespace/lowercase", "info", int(raw_sex_variants.sum()))

    bad_medal = df["Medal"].notna() & ~df["Medal"].astype("string").str.strip().str.title().isin(MEDAL_VALUES)
    rep.add("invalid.medal", "Medal value outside Gold/Silver/Bronze", "critical", int(bad_medal.sum()),
            _examples(df.loc[bad_medal, "Medal"]))
    bad_season = ~df["Season"].isin(["Summer", "Winter"])
    rep.add("invalid.season", "Season not Summer/Winter", "critical", int(bad_season.sum()), _examples(df.loc[bad_season, "Season"]))

    games_year = pd.to_numeric(df["Games"].astype(str).str.extract(r"^(\d{4})", expand=False), errors="coerce")
    games_season = df["Games"].astype(str).str.extract(r"(Summer|Winter)", expand=False)
    mism = (games_year != df["Year"]) | (games_season != df["Season"])
    rep.add("inconsistent.games", "Games label disagrees with Year/Season", "warning", int(mism.sum()), _examples(df.loc[mism, "Games"]))

    # 5. country consistency ------------------------------------------------------------------------------
    if regions is not None:
        unknown_noc = ~df["NOC"].isin(regions["NOC"])
        rep.add("country.unknown_noc", "NOC codes absent from the regions lookup", "warning", int(unknown_noc.sum()),
                _examples(df.loc[unknown_noc, "NOC"]), "Rows keep the NOC code as their country label.")
        typos = [r for r in regions["region"].dropna().unique() if r in REGION_FIXES]
        rep.add("country.region_typos", "Known typos in region names", "info", len(typos),
                [f"{t} -> {REGION_FIXES[t]}" for t in typos])
        noc_per_region = regions.dropna(subset=["region"]).groupby("region")["NOC"].nunique()
        multi = noc_per_region[noc_per_region > 1]
        rep.add("country.historic_aliases", "Regions covered by several NOC codes (historic / merged states)", "info",
                len(multi), [f"{k}: {v} codes" for k, v in multi.head(5).items()],
                "e.g. West/East Germany (FRG, GDR, GER) -> Germany. Region-level tallies merge them; NOC-level keeps them apart.")
        merged = df.merge(regions[["NOC", "region"]], on="NOC", how="left")
        has = merged["region"].notna() & merged["Team"].notna() if "Team" in merged else pd.Series(False, index=merged.index)
        team_base = merged["Team"].astype(str).str.replace(r"-\d+$", "", regex=True).str.lower()
        disagree = has & (team_base != merged["region"].astype(str).str.lower())
        rep.add("country.team_vs_region", "Team name differs from region of its NOC", "info", int(disagree.sum()),
                _examples(merged.loc[disagree, "Team"]), "Expected for mixed teams, boycotts and renamed countries.")
    for col in ("Team", "Sport", "Event"):
        if col in df.columns:
            s = df[col].dropna().astype(str)
            canon = s.str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
            groups = pd.DataFrame({"raw": s, "canon": canon}).drop_duplicates().groupby("canon")["raw"].nunique()
            rep.add(f"inconsistent.{col.lower()}_spelling", f"{col} values that differ only by case/whitespace",
                    "warning", int((groups > 1).sum()))

    # 6. athlete-level consistency ------------------------------------------------------------------------
    sex_per_id = df.assign(_s=sex_clean).dropna(subset=["_s"]).groupby("ID")["_s"].nunique()
    rep.add("athlete.multiple_sex", "Athlete IDs recorded with more than one Sex", "warning", int((sex_per_id > 1).sum()))
    birth = (df["Year"] - df["Age"]).groupby(df["ID"]).agg(lambda s: s.max() - s.min() if s.notna().any() else 0)
    rep.add("athlete.birth_year_spread", f"Athlete IDs with implied birth year spread > {vcfg['max_birth_year_spread']}",
            "warning", int((birth > vcfg["max_birth_year_spread"]).sum()),
            note="Indicates reused IDs or wrong ages (the same athlete cannot be two different ages).")

    # 7. event / temporal sanity --------------------------------------------------------------------------
    per_games = df.groupby("Games")["ID"].nunique()
    small = per_games[per_games < 0.1 * per_games.median()]
    rep.add("temporal.small_games", "Editions with unusually few athletes (<10% of median)", "info", len(small),
            [f"{g}: {n}" for g, n in small.items()])
    med = df.dropna(subset=["Medal"]).copy()
    gold_noc = med[med["Medal"].astype(str).str.title() == "Gold"].groupby(["Games", "Event"])["NOC"].nunique()
    rep.add("medals.multi_gold", "Events with gold medals for >2 different NOCs", "warning", int((gold_noc > 2).sum()),
            [f"{a} / {b}" for a, b in gold_noc[gold_noc > 2].index[:5]], "Possible mis-keyed event names.")

    return _finish(rep, df, missingness, regions)


def _finish(rep: _Report, df: pd.DataFrame, missingness: dict, regions) -> dict:
    sev = pd.Series([c["severity"] for c in rep.checks])
    penalty = 12 * (sev == "critical").sum() + 2 * (sev == "warning").sum()
    return {
        "summary": {
            "rows": int(len(df)), "columns": int(df.shape[1]),
            "critical": int((sev == "critical").sum()), "warnings": int((sev == "warning").sum()),
            "info": int((sev == "info").sum()), "passed": int((sev == "pass").sum()),
            "quality_score": int(max(0, 100 - penalty)),
        },
        "checks": rep.checks,
        "missingness": missingness,
    }


def enforce(report: dict, cfg: dict) -> None:
    """Raise if the data is too broken to proceed (thresholds in params.yaml -> validation.fail_on)."""
    fail = cfg["validation"]["fail_on"]
    by_id = {c["id"]: c for c in report["checks"]}
    problems = []
    if fail.get("missing_required_columns") and by_id.get("schema.required_columns", {}).get("count"):
        problems.append(f"missing required columns: {by_id['schema.required_columns']['examples']}")
    medal = by_id.get("invalid.medal")
    if medal and medal["pct_of_rows"] > fail["invalid_medal_values_pct"]:
        problems.append(f"{medal['pct_of_rows']}% invalid medal values")
    dup = by_id.get("dup.full_row")
    if dup and dup["pct_of_rows"] > fail["duplicate_pct"]:
        problems.append(f"{dup['pct_of_rows']}% duplicate rows")
    if problems:
        raise DataQualityError("Data quality gate failed: " + "; ".join(problems))


def write_quality_report(report: dict, out_dir_tables: Path, out_dir_metrics: Path) -> dict[str, Path]:
    checks = pd.DataFrame(report["checks"])
    paths = {"csv": save_table(checks.drop(columns=["examples"]), Path(out_dir_tables) / "data_quality_checks.csv"),
             "json": Path(out_dir_metrics) / "data_quality.json"}
    write_json(paths["json"], report)
    miss = pd.DataFrame(report["missingness"]).T.reset_index().rename(columns={"index": "column"}) if report["missingness"] else pd.DataFrame()
    if len(miss):
        save_table(miss, Path(out_dir_tables) / "missingness.csv")
    s = report["summary"]
    icon = {"critical": "🔴", "warning": "🟠", "info": "🔵", "pass": "✅"}
    lines = [f"# Data quality report", "",
             f"**Quality score: {s['quality_score']}/100** — {s['rows']:,} rows, {s['columns']} columns — "
             f"{s['critical']} critical, {s['warnings']} warnings, {s['info']} info, {s['passed']} passed", "",
             "| | Check | Count | % rows | Examples | Note |", "|---|---|---:|---:|---|---|"]
    for c in report["checks"]:
        ex = re.sub(r"\|", "/", ", ".join(c["examples"]))
        lines.append(f"| {icon[c['severity']]} | {c['title']} | {c['count']:,} | {c['pct_of_rows']} | {ex} | {c['note']} |")
    paths["md"] = write_text(Path(out_dir_tables) / "data_quality_report.md", "\n".join(lines) + "\n")
    return paths
