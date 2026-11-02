-- Country rank within every edition (window function). Params: :season (NULL = both), :top_n
WITH tally AS (
    SELECT Games, Year, Season, region AS country,
           SUM(Gold) AS gold, SUM(Silver) AS silver, SUM(Bronze) AS bronze, COUNT(*) AS total
    FROM medals
    WHERE (:season IS NULL OR Season = :season)
    GROUP BY Games, Year, Season, region
), ranked AS (
    SELECT *, RANK() OVER (PARTITION BY Games ORDER BY gold DESC, silver DESC, bronze DESC) AS rank_in_games
    FROM tally
)
SELECT * FROM ranked WHERE rank_in_games <= :top_n ORDER BY Year, rank_in_games;
