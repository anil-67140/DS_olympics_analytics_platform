-- Medals won by each host nation at its home Games vs its average at all other Games (same season).
-- Params: :season (NULL = both)
WITH per_games AS (
    SELECT m.region AS country, m.Games, m.Year, m.Season, COUNT(*) AS medals
    FROM medals m GROUP BY m.region, m.Games, m.Year, m.Season
), totals AS (
    SELECT Games, COUNT(*) AS all_medals FROM medals GROUP BY Games
), hosts AS (
    SELECT DISTINCT Games, NOC AS host_noc, region AS host_country
    FROM athlete_events WHERE is_host = 1
)
SELECT
    h.host_country AS country, h.Games AS home_games,
    COALESCE(home.medals, 0) AS home_medals,
    ROUND(100.0 * COALESCE(home.medals, 0) / t.all_medals, 2) AS home_share_pct,
    ROUND(AVG(other.medals), 2) AS avg_medals_other_games
FROM hosts h
JOIN totals t ON t.Games = h.Games
LEFT JOIN per_games home  ON home.country = h.host_country AND home.Games = h.Games
LEFT JOIN per_games other ON other.country = h.host_country AND other.Games <> h.Games
                          AND other.Season = home.Season
WHERE (:season IS NULL OR home.Season = :season)
GROUP BY h.host_country, h.Games, home.medals, t.all_medals
ORDER BY home.Year;
