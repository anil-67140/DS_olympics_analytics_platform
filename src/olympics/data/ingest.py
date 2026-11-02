"""Dataset download (Kaggle API / kagglehub) and adapters that map each source onto the canonical schema."""
from __future__ import annotations

import os
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..io_utils import get_logger
from .schema import CANON_COLUMNS
from .synthetic import make_synthetic_events

log = get_logger()


class DatasetNotAvailable(RuntimeError):
    pass


# --------------------------------------------------------------------------- download
def download_kaggle(slug: str, dest: Path, force: bool = False, retries: int = 3) -> Path:
    """Download + unzip a Kaggle dataset into ``dest``. Skips if already downloaded (checkpoint marker)."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / f".downloaded_{slug.replace('/', '__')}"
    if marker.exists() and not force:
        log.info("Dataset %s already downloaded in %s", slug, dest)
        return dest

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            try:
                from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
                api = KaggleApi()
                api.authenticate()
                api.dataset_download_files(slug, path=str(dest), unzip=True, quiet=False)
            except ImportError:
                import kagglehub  # type: ignore
                cache_dir = Path(kagglehub.dataset_download(slug))
                for item in cache_dir.rglob("*"):
                    if item.is_file():
                        target = dest / item.relative_to(cache_dir)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(item, target)
            marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
            return dest
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            log.warning("Kaggle download attempt %d/%d failed: %s", attempt, retries, exc)
            time.sleep(2 * attempt)
    raise DatasetNotAvailable(
        f"Could not download {slug}: {last_err}. In Colab add KAGGLE_USERNAME / KAGGLE_KEY to Secrets "
        "(or upload kaggle.json), accept the dataset's terms on kaggle.com once, then re-run. "
        "Alternatively place the CSV files in the raw data folder and set data.local_dir."
    )


# --------------------------------------------------------------------------- helpers
def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _find_file(root: Path, *candidates: str) -> Path | None:
    wanted = {_norm(c) for c in candidates}
    for path in sorted(Path(root).rglob("*")):
        if path.is_file() and _norm(path.stem) in wanted and path.suffix.lower() in {".csv", ".tsv", ".txt"}:
            return path
    return None


def _pick(df: pd.DataFrame, *aliases: str) -> str | None:
    lookup = {_norm(c): c for c in df.columns}
    for alias in aliases:
        if _norm(alias) in lookup:
            return lookup[_norm(alias)]
    return None


def inspect_source(raw_dir: Path) -> dict:
    """List every tabular file with its columns / row count so the user can confirm the mapping."""
    out = {}
    for path in sorted(Path(raw_dir).rglob("*")):
        if path.is_file() and path.suffix.lower() in {".csv", ".tsv"}:
            sep = "\t" if path.suffix.lower() == ".tsv" else ","
            head = pd.read_csv(path, nrows=5, sep=sep, encoding_errors="replace")
            with open(path, "rb") as fh:
                n_rows = sum(1 for _ in fh) - 1
            out[str(path.relative_to(raw_dir))] = {"rows": n_rows, "columns": list(head.columns)}
    return out


# --------------------------------------------------------------------------- adapters
def load_heesoo(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ev_path = _find_file(raw_dir, "athlete_events")
    reg_path = _find_file(raw_dir, "noc_regions")
    if ev_path is None or reg_path is None:
        raise DatasetNotAvailable(f"athlete_events.csv / noc_regions.csv not found under {raw_dir}")
    events = pd.read_csv(ev_path)
    regions = pd.read_csv(reg_path)  # handles the file's old-Mac line endings
    missing = [c for c in ("ID", "Sex", "NOC", "Year", "Sport", "Event", "Medal") if c not in events.columns]
    if missing:
        raise DatasetNotAvailable(f"athlete_events is missing columns {missing}; found {list(events.columns)}")
    return events[[c for c in CANON_COLUMNS if c in events.columns]], regions


def load_olympedia(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Adapter for the Olympedia-based Kaggle dataset (athlete bio + athlete-event results + games table).

    The mapping is alias-tolerant; if a required column cannot be found a precise error lists what exists,
    so it can be fixed in minutes (see ``python -m olympics.cli inspect-source``).
    """
    res_p = _find_file(raw_dir, "Olympic_Athlete_Event_Results", "athlete_event_results")
    bio_p = _find_file(raw_dir, "Olympic_Athlete_Bio", "athlete_bio")
    games_p = _find_file(raw_dir, "Olympics_Games", "Olympic_Games", "games")
    ctry_p = _find_file(raw_dir, "Olympics_Country", "Olympic_Country", "country")
    if res_p is None or bio_p is None:
        raise DatasetNotAvailable(
            f"Olympedia files not found under {raw_dir}. Found: {list(inspect_source(raw_dir))[:20]}")
    res, bio = pd.read_csv(res_p), pd.read_csv(bio_p)

    def need(df, label, *aliases):
        col = _pick(df, *aliases)
        if col is None:
            raise DatasetNotAvailable(f"{label}: none of {aliases} in columns {list(df.columns)}")
        return col

    r_id, r_noc = need(res, "results", "athlete_id", "athleteid"), need(res, "results", "country_noc", "noc")
    r_sport, r_event = need(res, "results", "sport", "discipline"), need(res, "results", "event")
    r_medal, r_edition = need(res, "results", "medal"), need(res, "results", "edition", "games")
    r_name = _pick(res, "athlete", "name")

    out = pd.DataFrame({
        "ID": res[r_id], "NOC": res[r_noc], "Sport": res[r_sport], "Event": res[r_event],
        "Medal": res[r_medal], "edition": res[r_edition].astype(str),
    })
    if r_name:
        out["Name"] = res[r_name]
    out["Year"] = out["edition"].str.extract(r"(\d{4})", expand=False).astype(float)
    season = np.where(out["edition"].str.contains("Winter", case=False), "Winter", "Summer")
    out["Season"] = season
    out["Games"] = out["Year"].astype("Int64").astype(str) + " " + out["Season"]

    b_id = need(bio, "bio", "athlete_id", "athleteid")
    bio = bio.drop_duplicates(b_id)
    bio_cols = {"Sex": _pick(bio, "sex", "gender"), "born": _pick(bio, "born", "birth_date", "birthdate"),
                "Height": _pick(bio, "height"), "Weight": _pick(bio, "weight"),
                "Team": _pick(bio, "country", "team"), "bio_name": _pick(bio, "name")}
    b = pd.DataFrame({"ID": bio[b_id]})
    for key, col in bio_cols.items():
        if col:
            b[key] = bio[col].values
    out = out.merge(b, on="ID", how="left")
    if "Name" not in out.columns and "bio_name" in out.columns:
        out["Name"] = out["bio_name"]
    if "Sex" in out.columns:
        out["Sex"] = out["Sex"].astype(str).str.strip().str[0].str.upper().where(lambda s: s.isin(["M", "F"]))
    born_year = (out["born"].astype(str).str.extract(r"(\d{4})", expand=False).astype(float)
                 if "born" in out.columns else np.nan)
    out["Age"] = out["Year"] - born_year
    for col in ("Height", "Weight"):
        out[col] = pd.to_numeric(out.get(col), errors="coerce")
    if "Team" not in out.columns:
        out["Team"] = out["NOC"]

    out["City"] = np.nan
    if games_p is not None:
        g = pd.read_csv(games_p)
        g_ed, g_city = _pick(g, "edition", "games"), _pick(g, "city", "host_city")
        if g_ed and g_city:
            city_map = g.drop_duplicates(g_ed).set_index(g_ed)[g_city]
            out["City"] = out["edition"].map(city_map)

    out["Medal"] = out["Medal"].astype("string").str.strip().str.title()
    out.loc[~out["Medal"].isin(["Gold", "Silver", "Bronze"]), "Medal"] = np.nan
    out = out.dropna(subset=["Year"]).copy()
    out["Year"] = out["Year"].astype(int)

    regions = None
    if ctry_p is not None:
        c = pd.read_csv(ctry_p)
        c_noc, c_name = _pick(c, "noc", "country_noc", "id"), _pick(c, "country", "name", "region")
        if c_noc and c_name:
            regions = c[[c_noc, c_name]].rename(columns={c_noc: "NOC", c_name: "region"}).drop_duplicates("NOC")
            regions["notes"] = np.nan
    if regions is None:
        regions = (out.groupby("NOC")["Team"].agg(lambda s: s.mode().iat[0] if len(s.mode()) else s.iat[0])
                   .rename("region").reset_index().assign(notes=np.nan))
    return out[[c for c in CANON_COLUMNS if c in out.columns]], regions


# --------------------------------------------------------------------------- entry point
def load_source(cfg: dict, raw_dir: Path, demo: bool = False, force_download: bool = False
                ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return ``(events, regions, info)`` for the configured source."""
    if demo:
        events, regions = make_synthetic_events()
        return events, regions, {"source": "synthetic-demo", "rows": len(events)}

    dcfg = cfg["data"]
    source = dcfg["source"]
    local = dcfg.get("local_dir") or os.environ.get("OLYMPICS_LOCAL_DATA")
    if local:
        data_dir = Path(local)
    else:
        data_dir = download_kaggle(dcfg["sources"][source]["kaggle_slug"], raw_dir / source, force=force_download)

    loader = {"heesoo37_120y": load_heesoo, "olympedia": load_olympedia}[source]
    events, regions = loader(data_dir)
    info = {"source": source, "rows": int(len(events)), "files": inspect_source(data_dir)}
    log.info("Loaded %s: %d rows, %d columns", source, len(events), events.shape[1])
    return events, regions, info
