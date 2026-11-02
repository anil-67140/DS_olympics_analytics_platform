"""SQL analytical layer (SQLite, standard SQL with window functions).

Tables
------
athlete_events : one row per athlete x event entry (cleaned)
medals         : one row per medal awarded (team events counted once)
regions        : NOC -> country lookup
Named queries live in ``queries/*.sql`` and take bound parameters (no string formatting -> no injection).
"""
from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd

QUERY_DIR = Path(__file__).parent / "queries"

EVENT_COLUMNS = ["ID", "Name", "Sex", "Age", "Height", "Weight", "BMI", "Team", "NOC", "region", "Games", "Year",
                 "Season", "City", "Sport", "Event", "Medal", "won_medal", "Gold", "Silver", "Bronze", "is_host"]

INDEXES = [
    "CREATE INDEX IF NOT EXISTS ix_ae_noc ON athlete_events(NOC, Year)",
    "CREATE INDEX IF NOT EXISTS ix_ae_id ON athlete_events(ID)",
    "CREATE INDEX IF NOT EXISTS ix_ae_sport ON athlete_events(Sport, Season)",
    "CREATE INDEX IF NOT EXISTS ix_ae_games ON athlete_events(Games, Event)",
    "CREATE INDEX IF NOT EXISTS ix_m_region ON medals(region, Year)",
    "CREATE INDEX IF NOT EXISTS ix_m_noc ON medals(NOC, Year)",
]


def build_database(events: pd.DataFrame, medals: pd.DataFrame, regions: pd.DataFrame | None, db_path: Path) -> Path:
    """Build in a *local* temp file, then copy into place. SQLite journaling is unreliable on network
    filesystems such as Google Drive, so the database is never built directly on Drive."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in EVENT_COLUMNS if c in events.columns]
    with tempfile.TemporaryDirectory() as td:
        local = Path(td) / "olympics.sqlite"
        con = sqlite3.connect(local)
        try:
            events[cols].astype(object).where(events[cols].notna(), None).to_sql("athlete_events", con, index=False)
            medals.to_sql("medals", con, index=False)
            if regions is not None:
                regions.astype(object).where(regions.notna(), None).to_sql("regions", con, index=False)
            for stmt in INDEXES:
                con.execute(stmt)
            con.commit()
        finally:
            con.close()
        staged = db_path.with_name(db_path.name + ".tmp")
        shutil.copyfile(local, staged)
        os.replace(staged, db_path)  # atomic swap: readers never see a half-written database
    return db_path


def local_copy(db_path: Path) -> Path:
    """Copy the database to local disk for fast, lock-safe read-only access (e.g. from a Drive folder)."""
    db_path = Path(db_path)
    target = Path(tempfile.gettempdir()) / f"olympics_{db_path.stat().st_mtime_ns}.sqlite"
    if not target.exists():
        shutil.copyfile(db_path, target)
    return target


def list_queries() -> list[str]:
    return sorted(p.stem for p in QUERY_DIR.glob("*.sql"))


def run_query(db_path: Path, name: str, params: dict | None = None) -> pd.DataFrame:
    sql_file = QUERY_DIR / f"{name}.sql"
    if not sql_file.exists():
        raise KeyError(f"Unknown query {name!r}. Available: {list_queries()}")
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        return pd.read_sql_query(sql_file.read_text(encoding="utf-8"), con, params=params or {})


def run_sql(db_path: Path, sql: str, params: dict | None = None) -> pd.DataFrame:
    """Ad-hoc read-only SQL (used by the dashboard's SQL console)."""
    if not sql.lstrip().lower().startswith(("select", "with")):
        raise ValueError("Only SELECT / WITH queries are allowed.")
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        return pd.read_sql_query(sql, con, params=params or {})
