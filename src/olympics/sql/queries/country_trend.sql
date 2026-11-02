-- Medals per edition for one country, with change vs previous edition and share of all medals awarded.
-- Params: :country, :season (NULL = both)
WITH totals AS (
    SELECT Games, COUNT(*) AS all_medals FROM medals GROUP BY Games
)
SELECT
    m.Games, m.Year, m.Season,
    SUM(m.Gold) AS gold, SUM(m.Silver) AS silver, SUM(m.Bronze) AS bronze, COUNT(*) AS total,
    COUNT(*) - LAG(COUNT(*)) OVER (PARTITION BY m.Season ORDER BY m.Year) AS change_vs_prev_edition,
    ROUND(100.0 * COUNT(*) / t.all_medals, 2) AS share_of_all_medals_pct
FROM medals m
JOIN totals t ON t.Games = m.Games
WHERE m.region = :country AND (:season IS NULL OR m.Season = :season)
GROUP BY m.Games, m.Year, m.Season, t.all_medals
ORDER BY m.Year, m.Season;
