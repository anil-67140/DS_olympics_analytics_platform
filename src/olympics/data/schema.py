"""Canonical schema and domain reference data shared by every module."""
from __future__ import annotations

CANON_COLUMNS = ["ID", "Name", "Sex", "Age", "Height", "Weight", "Team", "NOC", "Games", "Year",
                 "Season", "City", "Sport", "Event", "Medal"]
REQUIRED_COLUMNS = ["ID", "Sex", "Age", "NOC", "Games", "Year", "Season", "Sport", "Event", "Medal"]
MEDAL_VALUES = ["Gold", "Silver", "Bronze"]

# Display/typo fixes for the region lookup table (noc_regions.csv ships with "Boliva").
REGION_FIXES = {"Boliva": "Bolivia"}
# NOCs whose `region` is the literal string "NA" in noc_regions.csv (pandas reads it as missing).
REGION_FALLBACK_BY_NOC = {"ROT": "Refugee Olympic Team", "TUV": "Tuvalu", "UNK": "Unknown",
                          "SGP": "Singapore"}  # SGP is absent from noc_regions.csv (349 rows in the 120y data)

# Host NOC by (Year, Season). Used for the home-advantage feature and analyses.
HOST_NOC: dict[tuple[int, str], str] = {
    (1896, "Summer"): "GRE", (1900, "Summer"): "FRA", (1904, "Summer"): "USA", (1908, "Summer"): "GBR",
    (1912, "Summer"): "SWE", (1920, "Summer"): "BEL", (1924, "Summer"): "FRA", (1928, "Summer"): "NED",
    (1932, "Summer"): "USA", (1936, "Summer"): "GER", (1948, "Summer"): "GBR", (1952, "Summer"): "FIN",
    (1956, "Summer"): "AUS", (1960, "Summer"): "ITA", (1964, "Summer"): "JPN", (1968, "Summer"): "MEX",
    (1972, "Summer"): "FRG", (1976, "Summer"): "CAN", (1980, "Summer"): "URS", (1984, "Summer"): "USA",
    (1988, "Summer"): "KOR", (1992, "Summer"): "ESP", (1996, "Summer"): "USA", (2000, "Summer"): "AUS",
    (2004, "Summer"): "GRE", (2008, "Summer"): "CHN", (2012, "Summer"): "GBR", (2016, "Summer"): "BRA",
    (2020, "Summer"): "JPN", (2024, "Summer"): "FRA", (2028, "Summer"): "USA",
    (1924, "Winter"): "FRA", (1928, "Winter"): "SUI", (1932, "Winter"): "USA", (1936, "Winter"): "GER",
    (1948, "Winter"): "SUI", (1952, "Winter"): "NOR", (1956, "Winter"): "ITA", (1960, "Winter"): "USA",
    (1964, "Winter"): "AUT", (1968, "Winter"): "FRA", (1972, "Winter"): "JPN", (1976, "Winter"): "AUT",
    (1980, "Winter"): "USA", (1984, "Winter"): "YUG", (1988, "Winter"): "CAN", (1992, "Winter"): "FRA",
    (1994, "Winter"): "NOR", (1998, "Winter"): "JPN", (2002, "Winter"): "USA", (2006, "Winter"): "ITA",
    (2010, "Winter"): "CAN", (2014, "Winter"): "RUS", (2018, "Winter"): "KOR", (2022, "Winter"): "CHN",
    (2026, "Winter"): "ITA",
}

# Model feature schema (kept here so the API, training and monitoring agree).
NUMERIC_FEATURES = [
    "Age", "Height", "Weight", "BMI", "Year",
    "athlete_prev_games", "athlete_prev_medals",
    "noc_prev_entries", "noc_prev_medal_rate", "sport_noc_prev_medal_rate",
    "event_entrants", "noc_event_entrants", "is_host",
]
LOW_CARD_CATEGORICAL = ["Sex", "Season"]
HIGH_CARD_CATEGORICAL = ["NOC", "Sport"]
ALL_FEATURES = NUMERIC_FEATURES + LOW_CARD_CATEGORICAL + HIGH_CARD_CATEGORICAL
TARGET = "won_medal"
