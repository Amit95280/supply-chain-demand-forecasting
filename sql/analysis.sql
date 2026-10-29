
-- SQLite examples for business analysis.
-- Create the database using: python sql/setup_database.py

-- 1. Total sales by store
SELECT store, SUM(sales) AS total_sales
FROM sales
GROUP BY store
ORDER BY total_sales DESC;

-- 2. Top 10 products
SELECT item, SUM(sales) AS total_sales
FROM sales
GROUP BY item
ORDER BY total_sales DESC
LIMIT 10;

-- 3. Monthly sales
SELECT strftime('%Y-%m', date) AS month,
       SUM(sales) AS total_sales
FROM sales
GROUP BY month
ORDER BY month;

-- 4. Average daily sales by store
SELECT store, AVG(sales) AS avg_daily_sales
FROM sales
GROUP BY store
ORDER BY avg_daily_sales DESC;

-- 5. Promotion impact
SELECT promotion,
       AVG(sales) AS avg_sales,
       COUNT(*) AS observations
FROM sales
GROUP BY promotion;

-- 6. Store-item performance using a window function
WITH item_sales AS (
    SELECT store, item, SUM(sales) AS total_sales
    FROM sales
    GROUP BY store, item
)
SELECT *,
       RANK() OVER (PARTITION BY store ORDER BY total_sales DESC) AS item_rank
FROM item_sales;
