"""Synthetic athlete-event data with the same schema and the same *kinds* of defects as the real data.

Used by the unit tests, by ``--demo`` runs (no Kaggle needed) and by the CI smoke pipeline.
Team events share one medal across the whole team (like the real data), medal chances depend on
country strength, home advantage and age, so models have a real signal to find.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import HOST_NOC

_NOCS = ["USA", "CHN", "GBR", "GER", "FRA", "ITA", "JPN", "AUS", "CAN", "RUS", "KOR", "BRA", "ESP", "NED",
         "SWE", "NOR", "KEN", "JAM", "IND", "PAK", "EGY", "ARG", "MEX", "POL", "HUN", "CUB", "ETH", "TUR"]
_REGION = {"USA": "USA", "CHN": "China", "GBR": "UK", "GER": "Germany", "FRA": "France", "ITA": "Italy",
           "JPN": "Japan", "AUS": "Australia", "CAN": "Canada", "RUS": "Russia", "KOR": "South Korea",
           "BRA": "Brazil", "ESP": "Spain", "NED": "Netherlands", "SWE": "Sweden", "NOR": "Norway",
           "KEN": "Kenya", "JAM": "Jamaica", "IND": "India", "PAK": "Pakistan", "EGY": "Egypt",
           "ARG": "Argentina", "MEX": "Mexico", "POL": "Poland", "HUN": "Hungary", "CUB": "Cuba",
           "ETH": "Ethiopia", "TUR": "Turkey"}
# sport -> (season, is_team, team_size, ideal_age, height_mean, weight_mean)
_SPORTS = {
    "Athletics": ("Summer", False, 1, 25, 178, 70), "Swimming": ("Summer", False, 1, 22, 182, 75),
    "Gymnastics": ("Summer", False, 1, 20, 160, 55), "Boxing": ("Summer", False, 1, 25, 175, 70),
    "Judo": ("Summer", False, 1, 25, 172, 73), "Weightlifting": ("Summer", False, 1, 26, 170, 85),
    "Basketball": ("Summer", True, 5, 26, 195, 90), "Football": ("Summer", True, 6, 24, 177, 72),
    "Rowing": ("Summer", True, 4, 26, 188, 85), "Fencing": ("Summer", False, 1, 27, 178, 70),
    "Alpine Skiing": ("Winter", False, 1, 25, 175, 72), "Speed Skating": ("Winter", False, 1, 24, 176, 70),
    "Ice Hockey": ("Winter", True, 5, 26, 182, 85), "Curling": ("Winter", True, 4, 32, 175, 80),
}


def make_synthetic_events(n_nocs: int = 20, first_year: int = 1960, last_year: int = 2016,
                          athletes_per_noc: int = 70, seed: int = 7,
                          inject_issues: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(events, regions)`` shaped like athlete_events.csv / noc_regions.csv."""
    rng = np.random.default_rng(seed)
    nocs = _NOCS[:n_nocs]
    strength = dict(zip(nocs, rng.lognormal(0.0, 0.6, len(nocs))))

    pool, next_id = [], 1
    for noc in nocs:
        for _ in range(athletes_per_noc):
            sport = rng.choice(list(_SPORTS))
            _, _, _, _, h_mu, w_mu = _SPORTS[sport]
            sex = rng.choice(["M", "F"])
            adj = 0.93 if sex == "F" else 1.0
            pool.append({"ID": next_id, "Name": f"Athlete {next_id}", "Sex": sex, "NOC": noc, "Sport": sport,
                         "birth_year": int(rng.integers(first_year - 40, last_year - 16)),
                         "Height": float(np.clip(rng.normal(h_mu * adj, 6), 140, 215)),
                         "Weight": float(np.clip(rng.normal(w_mu * adj, 8), 40, 140)),
                         "skill": float(rng.normal(0, 1))})
            next_id += 1
    pool = pd.DataFrame(pool)

    summer_years = list(range(first_year, last_year + 1, 4))
    winter_years = list(range(first_year, min(last_year, 1992) + 1, 4)) + list(range(1994, last_year + 1, 4))
    games = [(y, "Summer") for y in summer_years] + [(y, "Winter") for y in winter_years]

    rows = []
    for year, season in sorted(games):
        host = HOST_NOC.get((year, season))
        active = pool[(year - pool["birth_year"]).between(15, 40)]
        active = active[rng.random(len(active)) < 0.6]
        for sport, (s_season, is_team, size, ideal_age, *_rest) in _SPORTS.items():
            if s_season != season:
                continue
            for sex in ("M", "F"):
                for e in range(2 if is_team else 3):
                    event = f"{sport} {'Men' if sex == 'M' else 'Women'}'s Event {e + 1}"
                    cand = active[(active["Sport"] == sport) & (active["Sex"] == sex)]
                    entries = []  # (noc, athlete_rows, performance)
                    for noc, grp in cand.groupby("NOC"):
                        host_bonus = 0.6 if noc == host else 0.0
                        if is_team:
                            if len(grp) >= size:
                                team = grp.sample(size, random_state=int(rng.integers(1e9)))
                                perf = np.log(strength[noc]) + team["skill"].mean() + host_bonus + rng.normal(0, 0.7)
                                entries.append((noc, team, perf))
                        else:
                            for _, a in grp.head(2).iterrows():
                                age = year - a["birth_year"]
                                perf = (np.log(strength[noc]) + a["skill"] + host_bonus
                                        - 0.04 * (age - ideal_age) ** 2 / 4 + rng.normal(0, 0.8))
                                entries.append((noc, a.to_frame().T, perf))
                    entries.sort(key=lambda t: -t[2])
                    for rank, (noc, ath, _perf) in enumerate(entries):
                        medal = ["Gold", "Silver", "Bronze"][rank] if rank < 3 else None
                        for _, a in ath.iterrows():
                            rows.append({"ID": a["ID"], "Name": a["Name"], "Sex": a["Sex"],
                                         "Age": float(year - a["birth_year"]), "Height": a["Height"],
                                         "Weight": a["Weight"], "Team": _REGION[noc], "NOC": noc,
                                         "Games": f"{year} {season}", "Year": year, "Season": season,
                                         "City": f"City{year}", "Sport": sport, "Event": event, "Medal": medal})
    events = pd.DataFrame(rows)
    # realistic missingness
    for col, rate in (("Height", 0.12), ("Weight", 0.14), ("Age", 0.02)):
        events.loc[rng.random(len(events)) < rate, col] = np.nan
    regions = pd.DataFrame({"NOC": nocs, "region": [_REGION[n] for n in nocs], "notes": [np.nan] * len(nocs)})

    if inject_issues and len(events) > 200:
        dup = events.sample(15, random_state=1)
        events = pd.concat([events, dup], ignore_index=True)           # exact duplicates
        idx = events.sample(6, random_state=2).index
        events.loc[idx, "Age"] = [250, 3, -5, 140, 7, 199]            # impossible ages
        events.loc[events.sample(4, random_state=3).index, "Height"] = 12.0
        events.loc[events.sample(3, random_state=4).index, "Sex"] = "m "   # inconsistent coding
        events.loc[events.sample(3, random_state=5).index, "NOC"] = "ZZZ"  # unknown country code
        events.loc[events.sample(2, random_state=6).index, "Medal"] = "Platinum"  # invalid medal
        events = events.sample(frac=1.0, random_state=9).reset_index(drop=True)
    return events, regions
