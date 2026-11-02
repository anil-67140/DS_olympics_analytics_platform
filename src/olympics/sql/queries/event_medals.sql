-- Medallists per event across editions. Params: :sport, :season (NULL = both)
SELECT Games, Year, Event, Medal, region AS country
FROM medals
WHERE Sport = :sport AND (:season IS NULL OR Season = :season)
ORDER BY Year, Event,
         CASE Medal WHEN 'Gold' THEN 1 WHEN 'Silver' THEN 2 ELSE 3 END;
