-- Career summary per athlete. Params: :min_games (minimum number of editions), :limit
SELECT
    ID,
    MAX(Name)                         AS name,
    MAX(Sex)                          AS sex,
    GROUP_CONCAT(DISTINCT region)     AS countries,
    GROUP_CONCAT(DISTINCT Sport)      AS sports,
    MIN(Year)                         AS first_year,
    MAX(Year)                         AS last_year,
    MAX(Year) - MIN(Year)             AS career_span_years,
    COUNT(DISTINCT Games)             AS editions,
    COUNT(*)                          AS entries,
    SUM(won_medal)                    AS medals,
    SUM(Gold)                         AS gold,
    ROUND(1.0 * SUM(won_medal) / COUNT(*), 3) AS medal_rate,
    ROUND(AVG(Age), 1)                AS avg_age
FROM athlete_events
GROUP BY ID
HAVING COUNT(DISTINCT Games) >= :min_games
ORDER BY medals DESC, gold DESC, entries DESC
LIMIT :limit;
