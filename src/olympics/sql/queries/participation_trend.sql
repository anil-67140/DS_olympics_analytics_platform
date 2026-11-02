-- Participation KPIs per edition. Params: :season (NULL = both)
SELECT
    Games, Year, Season,
    COUNT(DISTINCT ID)                                              AS athletes,
    COUNT(DISTINCT NOC)                                             AS nations,
    COUNT(DISTINCT Sport)                                           AS sports,
    COUNT(DISTINCT Event)                                           AS events,
    ROUND(100.0 * COUNT(DISTINCT CASE WHEN Sex = 'F' THEN ID END) / COUNT(DISTINCT ID), 2) AS women_pct,
    ROUND(AVG(Age), 2)                                              AS mean_age,
    COUNT(DISTINCT ID) - LAG(COUNT(DISTINCT ID)) OVER (PARTITION BY Season ORDER BY Year) AS athletes_change
FROM athlete_events
WHERE (:season IS NULL OR Season = :season)
GROUP BY Games, Year, Season
ORDER BY Year, Season;
