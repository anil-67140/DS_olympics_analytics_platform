"""Pure-pandas logic behind the dashboard (KPIs, filters, comparisons, drill-downs). No UI imports -> unit-testable."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from .config import Paths
from .io_utils import load_frame, read_json
from .sql.layer import local_copy, run_query, run_sql


class DashboardData:
    """Loads the processed tables once and serves cached, filterable views."""

    def __init__(self, root: str | Path):
        self.paths = Paths.from_env(root)
        self.events = load_frame(self.paths.processed / "clean_events")
        self.medals = load_frame(self.paths.processed / "medals")
        self.db = local_copy(self.paths.db_file)
        self._cache: dict = {}

    # ------------------------------------------------------------------ filter options
    @property
    def countries(self) -> list[str]:
        return sorted(self.medals["region"].dropna().unique())

    @property
    def sports(self) -> list[str]:
        return sorted(self.events["Sport"].dropna().unique())

    @property
    def year_range(self) -> tuple[int, int]:
        return int(self.events["Year"].min()), int(self.events["Year"].max())

    @property
    def seasons(self) -> list[str]:
        return sorted(self.events["Season"].unique())

    def _slice(self, season, years, sports=None, countries=None):
        lo, hi = years
        ev, md = self.events, self.medals
        ev = ev[ev["Year"].between(lo, hi)]
        md = md[md["Year"].between(lo, hi)]
        if season and season != "Both":
            ev, md = ev[ev["Season"] == season], md[md["Season"] == season]
        if sports:
            ev, md = ev[ev["Sport"].isin(sports)], md[md["Sport"].isin(sports)]
        if countries:
            ev, md = ev[ev["region"].isin(countries)], md[md["region"].isin(countries)]
        return ev, md

    # ------------------------------------------------------------------ KPIs
    def kpis(self, season="Both", years=None, sports=None, countries=None) -> dict:
        ev, md = self._slice(season, years or self.year_range, sports, countries)
        entries = max(len(ev), 1)
        return {"Athletes": int(ev["ID"].nunique()), "Nations": int(ev["NOC"].nunique()), "Editions": int(ev["Games"].nunique()),
                "Sports": int(ev["Sport"].nunique()), "Events": int(ev["Event"].nunique()), "Medals awarded": int(len(md)),
                "Women %": round(100 * ev.drop_duplicates("ID")["Sex"].eq("F").mean(), 1) if len(ev) else 0.0,
                "Medal rate %": round(100 * ev["won_medal"].sum() / entries, 1), "Mean age": round(float(ev["Age"].mean()), 1) if len(ev) else 0.0}

    # ------------------------------------------------------------------ views
    def medal_table(self, season="Both", years=None, sports=None, top=25) -> pd.DataFrame:
        _, md = self._slice(season, years or self.year_range, sports)
        t = md.groupby("region")[["Gold", "Silver", "Bronze", "Total"]].sum()
        t = t.sort_values(["Gold", "Silver", "Bronze"], ascending=False).head(top).reset_index().rename(columns={"region": "Country"})
        t.insert(0, "Rank", range(1, len(t) + 1))
        return t

    def participation(self, season="Both", years=None, sports=None, countries=None) -> pd.DataFrame:
        ev, _ = self._slice(season, years or self.year_range, sports, countries)
        g = ev.groupby(["Year", "Season"]).agg(Athletes=("ID", "nunique"), Nations=("NOC", "nunique"), Events=("Event", "nunique"),
                                              Medal_rate=("won_medal", "mean"), Mean_age=("Age", "mean")).reset_index()
        women = ev.drop_duplicates(["ID", "Year", "Season"]).groupby(["Year", "Season"])["Sex"].apply(lambda s: 100 * (s == "F").mean())
        g["Women_pct"] = g.set_index(["Year", "Season"]).index.map(women)
        return g.round(3)

    def compare_countries(self, countries: list[str], season="Both", years=None, sports=None) -> pd.DataFrame:
        _, md = self._slice(season, years or self.year_range, sports, countries)
        out = md.groupby(["Year", "region"])[["Gold", "Silver", "Bronze", "Total"]].sum().reset_index().rename(columns={"region": "Country"})
        return out

    def country_profile(self, country: str, season="Both", years=None) -> dict:
        ev, md = self._slice(season, years or self.year_range, countries=[country])
        by_sport = md.groupby("Sport")[["Gold", "Silver", "Bronze", "Total"]].sum().sort_values("Total", ascending=False).reset_index()
        return {"kpis": self.kpis(season, years, countries=[country]), "by_sport": by_sport.head(15),
                "trend": md.groupby("Year")["Total"].sum().reset_index(), "top_athletes": self.top_athletes(country=country, season=season, years=years)}

    def top_athletes(self, country=None, sport=None, season="Both", years=None, top=15) -> pd.DataFrame:
        ev, _ = self._slice(season, years or self.year_range, [sport] if sport else None, [country] if country else None)
        g = ev.groupby("ID").agg(Name=("Name", "first"), Country=("region", "first"), Sport=("Sport", "first"), Games=("Games", "nunique"),
                                 Entries=("ID", "size"), Medals=("won_medal", "sum"), Gold=("Gold", "sum"), First=("Year", "min"), Last=("Year", "max"))
        return g.sort_values(["Medals", "Gold", "Entries"], ascending=False).head(top).reset_index(drop=True)

    def athlete_career(self, name_query: str, limit: int = 30) -> pd.DataFrame:
        if not name_query or len(name_query.strip()) < 3:
            return pd.DataFrame({"hint": ["Type at least 3 letters of a name"]})
        m = self.events[self.events["Name"].fillna("").str.contains(name_query.strip(), case=False, regex=False)]
        return (m[["Name", "Sex", "region", "Games", "Sport", "Event", "Age", "Medal"]].sort_values(["Name", "Games"]).head(limit).reset_index(drop=True))

    def sport_drilldown(self, sport: str, season="Both", years=None) -> dict:
        ev, md = self._slice(season, years or self.year_range, [sport])
        age = ev.dropna(subset=["Age"])
        return {"kpis": self.kpis(season, years, [sport]), "top_countries": md.groupby("region")["Total"].sum().sort_values(ascending=False).head(10).reset_index().rename(columns={"region": "Country"}),
                "age_by_sex": age.groupby("Sex")["Age"].describe().round(2).reset_index(), "ages": age[["Age", "Sex", "won_medal"]],
                "events": ev.groupby("Event").agg(Entries=("ID", "size"), Medal_rate=("won_medal", "mean")).sort_values("Entries", ascending=False).head(15).round(3).reset_index()}

    # ------------------------------------------------------------------ artifacts / SQL
    def table(self, name: str) -> pd.DataFrame:
        p = self.paths.tables / name
        return pd.read_csv(p) if p.exists() else pd.DataFrame({"info": [f"{name} not generated yet"]})

    def metrics(self, name: str) -> dict:
        return read_json(self.paths.metrics / name, {}) or {}

    def figures(self) -> list[tuple[str, str]]:
        idx = read_json(self.paths.figures / "figure_index.json", {}) or {}
        return [(str(self.paths.figures / m["file"]), f"{m['title']} — {m['what_it_shows']}") for _, m in sorted(idx.items()) if (self.paths.figures / m["file"]).exists()]

    def sql(self, query: str, limit: int = 500) -> pd.DataFrame:
        q = query.strip().rstrip(";")
        if ";" in q:
            raise ValueError("Only a single statement is allowed.")
        return run_sql(self.db, f"SELECT * FROM ({q}) LIMIT {int(limit)}")
