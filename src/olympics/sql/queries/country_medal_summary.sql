-- All-time (or filtered) medal table by country. Params: :season (NULL = both), :year_min, :year_max
SELECT
    region                                   AS country,
    SUM(Gold)                                AS gold,
    SUM(Silver)                              AS silver,
    SUM(Bronze)                              AS bronze,
    COUNT(*)                                 AS total,
    COUNT(DISTINCT Games)                    AS editions_with_medals,
    MIN(Year)                                AS first_medal_year,
    MAX(Year)                                AS last_medal_year,
    RANK() OVER (ORDER BY SUM(Gold) DESC, SUM(Silver) DESC, SUM(Bronze) DESC) AS rank
FROM medals
WHERE (:season IS NULL OR Season = :season)
  AND Year BETWEEN :year_min AND :year_max
GROUP BY region
ORDER BY rank, country;
