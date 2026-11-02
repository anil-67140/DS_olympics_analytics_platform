import pytest

from olympics.sql.layer import build_database, list_queries, run_query, run_sql


@pytest.fixture(scope="module")
def db(cleaned, tmp_path_factory):
    df, med, regions, _ = cleaned
    return build_database(df, med, regions, tmp_path_factory.mktemp("db") / "o.sqlite")


ALL = {"season": None, "year_min": 1896, "year_max": 2100}


def test_sql_matches_pandas_medal_table(db, cleaned):
    _, med, *_ = cleaned
    sql = run_query(db, "country_medal_summary", ALL).set_index("country")
    pdf = med.groupby("region")[["Gold", "Silver", "Bronze"]].sum()
    for country in pdf.index[:10]:
        assert sql.loc[country, "gold"] == pdf.loc[country, "Gold"]
        assert sql.loc[country, "total"] == pdf.loc[country].sum()


def test_rank_is_dense_in_order(db):
    r = run_query(db, "country_medal_summary", ALL)
    assert r["rank"].iloc[0] == 1
    assert r["gold"].is_monotonic_decreasing


def test_every_named_query_runs(db):
    params = {"country_medal_summary": ALL, "country_trend": {"country": "USA", "season": None}, "medal_rank_by_games": {"season": None, "top_n": 3},
              "athlete_career": {"min_games": 1, "limit": 5}, "event_medals": {"sport": "Athletics", "season": None},
              "participation_trend": {"season": None}, "sport_profile": {"season": None, "min_entries": 1}, "host_effect": {"season": None}}
    assert set(params) == set(list_queries())
    for name, p in params.items():
        assert run_query(db, name, p) is not None


def test_rank_per_games_top_n(db):
    r = run_query(db, "medal_rank_by_games", {"season": "Summer", "top_n": 2})
    assert r["rank_in_games"].min() == 1 and r["rank_in_games"].max() <= 2 + 1


def test_participation_query_counts(db, cleaned):
    df, *_ = cleaned
    p = run_query(db, "participation_trend", {"season": "Summer"})
    g = df[df["Season"] == "Summer"].groupby("Year")["ID"].nunique()
    assert dict(zip(p["Year"], p["athletes"])) == g.to_dict()


def test_sql_is_read_only_and_parameterised(db):
    with pytest.raises(ValueError):
        run_sql(db, "DROP TABLE medals")
    with pytest.raises(Exception):
        run_sql(db, "SELECT 1; DROP TABLE medals")
    # injection via bound parameter is just a literal that matches nothing
    r = run_query(db, "country_trend", {"country": "x' OR '1'='1", "season": None})
    assert len(r) == 0
    assert len(run_sql(db, "SELECT COUNT(*) AS n FROM medals")) == 1


def test_unknown_query_name(db):
    with pytest.raises(KeyError):
        run_query(db, "nope")
