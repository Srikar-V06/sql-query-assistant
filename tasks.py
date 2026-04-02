"""
Task definitions for the SQL Query Assistant environment.

Each task defines:
  - A plain-English description the agent sees
  - A ground truth SQL query (executed at reset() to generate the answer key)
  - Metadata for grading (difficulty, max steps, column requirements)

Tasks are ordered easy → medium → hard.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Task:
    task_id: str
    difficulty: str                  # "easy" | "medium" | "hard"
    description: str                 # Shown to the agent in Observation
    ground_truth_sql: str            # Never shown to agent; used to compute answer key
    max_steps: int                   # Episode length limit
    required_columns: list[str]      # Column names the result MUST contain (lowercase)
    order_matters: bool = False      # True if row order is part of correctness (e.g. TOP-N)
    hint: Optional[str] = None       # Optional nudge shown after step 3 (not step 1)
    tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# TASK 1 — Easy
# Straightforward filter + projection. No joins needed.
# ---------------------------------------------------------------------------
TASK_EASY = Task(
    task_id="customer_filter",
    difficulty="easy",
    description="""
Find all customers who signed up on or after 2023-01-01 AND are premium members (is_premium = 1).

Return the following columns (in any order):
  - customer_id
  - name
  - city
  - signup_date

Order results by signup_date ascending, then by customer_id ascending.
""".strip(),
    ground_truth_sql="""
SELECT
    customer_id,
    name,
    city,
    signup_date
FROM customers
WHERE signup_date >= '2023-01-01'
  AND is_premium = 1
ORDER BY signup_date ASC, customer_id ASC
""".strip(),
    max_steps=5,
    required_columns=["customer_id", "name", "city", "signup_date"],
    order_matters=True,
    hint="Filter the customers table using WHERE on signup_date and is_premium.",
    tags=["filter", "sort", "single-table"],
)


# ---------------------------------------------------------------------------
# TASK 2 — Medium
# Requires JOIN across 3 tables + GROUP BY + date range filter + TOP-N.
# ---------------------------------------------------------------------------
TASK_MEDIUM = Task(
    task_id="top_products_revenue",
    difficulty="medium",
    description="""
Find the top 5 products by total revenue generated from DELIVERED orders
placed in Q3 2023 (July 1 2023 – September 30 2023).

Revenue for a line item = quantity × unit_price (from order_items).

Return the following columns:
  - product_id
  - product_name
  - category_name
  - total_revenue  (rounded to 2 decimal places)

Order by total_revenue descending (highest revenue first).
""".strip(),
    ground_truth_sql="""
SELECT
    p.product_id,
    p.name          AS product_name,
    c.name          AS category_name,
    ROUND(SUM(oi.quantity * oi.unit_price), 2) AS total_revenue
FROM order_items oi
JOIN orders     o  ON oi.order_id   = o.order_id
JOIN products   p  ON oi.product_id = p.product_id
JOIN categories c  ON p.category_id = c.category_id
WHERE o.order_date >= '2023-07-01'
  AND o.order_date <= '2023-09-30'
  AND o.status = 'delivered'
GROUP BY p.product_id, p.name, c.name
ORDER BY total_revenue DESC
LIMIT 5
""".strip(),
    max_steps=8,
    required_columns=["product_id", "product_name", "category_name", "total_revenue"],
    order_matters=True,
    hint="Join order_items → orders → products → categories. Filter by order_date range and status = 'delivered'.",
    tags=["join", "group-by", "aggregation", "date-filter", "top-n"],
)


# ---------------------------------------------------------------------------
# TASK 3 — Hard
# Requires CTE or subquery logic to identify "lost" customers:
# ordered in Q1 2023 but placed zero orders after 2023-03-31.
# Also needs a join to get their last order value.
# ---------------------------------------------------------------------------
TASK_HARD = Task(
    task_id="churned_customers",
    difficulty="hard",
    description="""
Identify churned customers: customers who placed at least one DELIVERED order
in Q1 2023 (January 1 2023 – March 31 2023) but have placed NO orders of ANY
status after March 31 2023.

For each churned customer return:
  - customer_id
  - name
  - city
  - total_q1_orders       (count of their delivered orders in Q1 2023)
  - last_order_value      (total revenue of their most recent Q1 2023 delivered order,
                           rounded to 2 decimal places; if tie on date pick highest order_id)

Order by last_order_value descending, then customer_id ascending.
""".strip(),
    ground_truth_sql="""
WITH q1_delivered AS (
    -- customers with at least one delivered order in Q1 2023
    SELECT
        o.customer_id,
        o.order_id,
        o.order_date,
        ROUND(SUM(oi.quantity * oi.unit_price), 2) AS order_value
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    WHERE o.order_date BETWEEN '2023-01-01' AND '2023-03-31'
      AND o.status = 'delivered'
    GROUP BY o.order_id, o.customer_id, o.order_date
),
post_q1_any AS (
    -- customers who placed ANY order after Q1 2023
    SELECT DISTINCT customer_id
    FROM orders
    WHERE order_date > '2023-03-31'
),
churned AS (
    -- customers in q1_delivered but NOT in post_q1_any
    SELECT customer_id
    FROM q1_delivered
    GROUP BY customer_id
    HAVING customer_id NOT IN (SELECT customer_id FROM post_q1_any)
),
q1_stats AS (
    SELECT
        qd.customer_id,
        COUNT(DISTINCT qd.order_id) AS total_q1_orders,
        -- most recent Q1 order value (latest date, then highest order_id on tie)
        qd.order_value AS last_order_value
    FROM q1_delivered qd
    WHERE qd.order_id = (
        SELECT order_id
        FROM q1_delivered qd2
        WHERE qd2.customer_id = qd.customer_id
        ORDER BY qd2.order_date DESC, qd2.order_id DESC
        LIMIT 1
    )
    GROUP BY qd.customer_id, qd.order_value
)
SELECT
    c.customer_id,
    c.name,
    c.city,
    qs.total_q1_orders,
    qs.last_order_value
FROM churned ch
JOIN customers  c  ON ch.customer_id = c.customer_id
JOIN q1_stats   qs ON ch.customer_id = qs.customer_id
ORDER BY qs.last_order_value DESC, c.customer_id ASC
""".strip(),
    max_steps=10,
    required_columns=["customer_id", "name", "city", "total_q1_orders", "last_order_value"],
    order_matters=True,
    hint="Use CTEs: first find Q1 delivered orders, then exclude customers with any order after March 31, then join for stats.",
    tags=["cte", "subquery", "churn", "multi-join", "window-logic"],
)
# ---------------------------------------------------------------------------
# TASK 4 — Medium-Hard
# Requires JOIN across reviews + products + categories + GROUP BY + HAVING.
# Tests aggregation with filtering on minimum review count.
# ---------------------------------------------------------------------------
TASK_MEDIUM_HARD = Task(
    task_id="category_review_summary",
    difficulty="medium-hard",
    description="""
For each product category, calculate review statistics — but only include
categories that have received at least 50 reviews in total.

Return the following columns:
  - category_name
  - total_reviews       (total number of reviews in that category)
  - avg_rating          (average rating across all reviews, rounded to 2 decimal places)
  - pct_five_star       (percentage of reviews that are 5-star, rounded to 1 decimal place)

Order by avg_rating descending, then category_name ascending.
""".strip(),
    ground_truth_sql="""
SELECT
    c.name                                          AS category_name,
    COUNT(r.review_id)                              AS total_reviews,
    ROUND(AVG(r.rating), 2)                         AS avg_rating,
    ROUND(100.0 * SUM(CASE WHEN r.rating = 5 THEN 1 ELSE 0 END) / COUNT(r.review_id), 1) AS pct_five_star
FROM reviews r
JOIN products   p ON r.product_id   = p.product_id
JOIN categories c ON p.category_id  = c.category_id
GROUP BY c.category_id, c.name
HAVING COUNT(r.review_id) >= 50
ORDER BY avg_rating DESC, c.name ASC
""".strip(),
    max_steps=7,
    required_columns=["category_name", "total_reviews", "avg_rating", "pct_five_star"],
    order_matters=True,
    hint="JOIN reviews → products → categories, then GROUP BY category and use HAVING to filter on review count.",
    tags=["join", "group-by", "having", "aggregation", "percentage"],
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_TASKS: dict[str, Task] = {
    TASK_EASY.task_id:   TASK_EASY,
    TASK_MEDIUM.task_id: TASK_MEDIUM,
    TASK_HARD.task_id:   TASK_HARD,
    TASK_MEDIUM_HARD.task_id: TASK_MEDIUM_HARD,
}

TASK_ORDER = [TASK_EASY.task_id, TASK_MEDIUM.task_id, TASK_HARD.task_id,TASK_MEDIUM_HARD.task_id]


def get_task(task_id: str) -> Task:
    if task_id not in ALL_TASKS:
        raise ValueError(f"Unknown task_id '{task_id}'. Valid: {list(ALL_TASKS.keys())}")
    return ALL_TASKS[task_id]