-- Physical and performance profile per sport. Params: :season (NULL = both), :min_entries
SELECT
    Sport,
    COUNT(*)                         AS entries,
    COUNT(DISTINCT ID)               AS athletes,
    ROUND(AVG(Age), 2)               AS mean_age,
    ROUND(AVG(Height), 1)            AS mean_height_cm,
    ROUND(AVG(Weight), 1)            AS mean_weight_kg,
    ROUND(100.0 * SUM(CASE WHEN Sex = 'F' THEN 1 ELSE 0 END) / COUNT(*), 1) AS women_pct,
    ROUND(100.0 * SUM(won_medal) / COUNT(*), 2) AS medal_rate_pct
FROM athlete_events
WHERE (:season IS NULL OR Season = :season)
GROUP BY Sport
HAVING COUNT(*) >= :min_entries
ORDER BY entries DESC;
