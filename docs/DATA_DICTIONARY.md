# Data dictionary, sources and limitations

## Sources

| Source | Kaggle slug | Coverage | Notes |
|---|---|---|---|
| **120 years of Olympic history** (default; the dataset used by the original notebook) | `heesoo37/120-years-of-olympic-history-athletes-and-results` | Athens 1896 – Rio 2016, Summer + Winter, 271,116 athlete-event rows | Originally scraped from Sports-Reference.com (2018). Two files: `athlete_events.csv`, `noc_regions.csv`. |
| **Olympedia-based dataset** (optional, newer) | `josephcheng123456/olympic-historical-dataset-from-olympediaorg` | Extends past 2016 (Tokyo 2020, Paris 2024) | Several tables (athlete bio, athlete-event results, games, countries). The adapter `load_olympedia` maps them onto the canonical schema; **run `python -m olympics.cli inspect-source --set data.source=olympedia` first to confirm the column mapping** (see README, "Choosing the dataset"). |

Both are downloaded through the Kaggle API; you must accept each dataset's terms on kaggle.com once.

## Canonical columns (after ingestion)

| Column | Type | Definition |
|---|---|---|
| `ID` | int | Athlete identifier (unique per athlete across Games). |
| `Name` | text | Athlete name. |
| `Sex` | `M`/`F` | Sex as recorded by the source. Whitespace/case variants are normalised; other codes become missing. |
| `Age` | float | Age in years **at the time of the Games**. Missing for ~3.5% of rows; values outside 10–100 are set to missing. |
| `Height` / `Weight` | float | cm / kg. Missing for ~22–23% of rows (mostly pre-1960). Implausible values and implausible BMI are set to missing. |
| `Team` | text | Team name as listed (can be a club, boat or crew name, not just a country). |
| `NOC` | text | National Olympic Committee 3-letter code. Historic and merged bodies have their own codes (e.g. `URS`, `GDR`, `FRG`, `EUN`). |
| `Games` / `Year` / `Season` / `City` | | Edition label, year, `Summer` or `Winter`, host city. |
| `Sport` / `Event` | text | Sport (discipline) and the specific event. |
| `Medal` | `Gold`/`Silver`/`Bronze`/missing | Medal won in this entry. Missing = no medal. |

## Derived columns (created by `olympics.data.clean`)

| Column | Definition |
|---|---|
| `region` | Country name from `noc_regions.csv` (typo `Boliva`→`Bolivia`; blanks such as `ROT`, `TUV`, `UNK`, and `SGP` which is absent from the lookup, are filled; any still-unmapped NOC falls back to its code). |
| `won_medal` | 1 if `Medal` is present else 0 — the prediction target. |
| `Gold`, `Silver`, `Bronze` | 0/1 indicators. |
| `BMI` | `Weight / (Height/100)²`. |
| `birth_year`, `decade` | `Year − Age`; decade of the Games. |
| `host_noc`, `is_host` | Host NOC of that edition and whether the athlete's NOC was the host. |
| `event_entrants` | Athletes entered in that event at that edition (all countries). |
| `noc_event_entrants` | Athletes from the same NOC in that event (≈ team size). |
| `is_team_event` | 1 when some country entered more than one athlete in the event. |

## Model features (leakage-safe)

History features use **strictly earlier editions only** (never the row's own Games): `athlete_prev_games`, `athlete_prev_medals`, `noc_prev_entries`, `noc_prev_medal_rate`, `sport_noc_prev_medal_rate` (shrunk toward the country's rate, then toward the global rate, so tiny samples are not over-trusted). Plus `Age, Height, Weight, BMI, Year, event_entrants, noc_event_entrants, is_host` and categoricals `Sex, Season, NOC, Sport`.

## Medal counting rule

The raw file has **one row per athlete**, so a team medal appears once per team member. A *medal table* counts a medal once per `(Games, NOC, Sport, Event, Medal)`. Rare ties where one NOC wins two medals of the same colour in the same event collapse into one (documented approximation).

## Known limitations

- **Coverage ends in 2016** in the default dataset; forecasts target the *next edition after the last one in the data* (2020 for the default). Switch to the Olympedia source to include later Games.
- **Country identity changes over time**: West/East Germany, USSR/Russia, Yugoslavia, Czechoslovakia. `region` merges some of these (e.g. all German NOCs → "Germany") while `NOC` keeps them apart. Choose deliberately per analysis.
- **Boycotts and wars** (1980, 1984; no Games 1916/1940/1944) distort trends and country comparisons.
- **Event/medal inflation**: the number of events grew from 43 (1896) to 300+, so raw medal totals are not comparable across eras. Prefer medal *share* or *rate*.
- **Missing physique data** is not random (older eras are far less complete), so averages of height/weight are biased toward recent athletes.
- **Art competitions** (1912–1948) and the **1906 Intercalated Games** (excluded by default) are not standard sporting results.
- **Duplicates**: the source contains exact duplicate rows (removed) and some athlete-event repeats (reported).
- The medal-prediction model describes *historical patterns*; it must not be used to make decisions about individual athletes.
