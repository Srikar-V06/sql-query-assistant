---
title: SQL Query Assistant
emoji: 🗄️
colorFrom: blue
colorTo: teal
sdk: docker
pinned: false
tags:
  - openenv
---

# SQL Query Assistant — OpenEnv Environment

An RL environment where AI agents write SQL queries against a seeded
in-memory SQLite e-commerce database to solve progressively harder
analytical tasks.

## Environment Description

The agent interacts with a live SQLite database containing 500 customers,
50 products, 1200 orders, 3000+ order items, and 800 reviews — all
seeded deterministically (seed=42) for full reproducibility.

## Action Space
```json
{
  "query": "A valid SQLite SELECT (or WITH...SELECT) statement"
}
```

Only SELECT statements are permitted. Write operations are safety-blocked.

## Observation Space

| Field | Type | Description |
|---|---|---|
| task_id | string | Current task identifier |
| task_description | string | Plain-English task description |
| schema_info | string | Full CREATE TABLE DDL |
| sample_rows | dict | 3 sample rows per table |
| last_query | string | Previous SQL query submitted |
| last_result | list | Rows returned by last query |
| last_error | string | Error message if query failed |
| steps_taken | int | Steps used so far |
| max_steps | int | Maximum steps allowed |
| done | bool | Whether episode has ended |

## Reward Function

Scores are computed across 4 dimensions:

| Dimension | Weight | Criteria |
|---|---|---|
| syntax | 0.10 | Query runs without error |
| columns | 0.20 | All required columns present |
| partial_rows | 0.30 | ≥50% of ground-truth rows matched |
| exact_match | 0.40 | Result exactly equals ground truth |

Shaping rules applied on top:
- Efficiency bonus: +0.05 × (remaining_steps / max_steps) on correct answer
- Step penalty: −0.02 per incorrect step beyond step 3

## Tasks

### Task 1 — customer_filter (Easy, max_steps=5)
Filter premium customers who signed up on or after 2023-01-01.
Return customer_id, name, city, signup_date ordered by signup_date ASC.

### Task 2 — top_products_revenue (Medium, max_steps=8)
Find top 5 products by revenue from delivered orders in Q3 2023.
Requires JOIN across 4 tables + GROUP BY + date filter.

### Task 3 — churned_customers (Hard, max_steps=10)
Identify customers with delivered orders in Q1 2023 but zero orders after.
Requires CTEs, subqueries, and multi-table joins.

### Task 4 — category_review_summary (Medium-Hard, max_steps=7)
Compute review statistics per category with at least 50 reviews.
Requires JOIN + GROUP BY + HAVING + percentage calculation.

## Setup & Usage

### Run locally
```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Run with Docker
```bash
docker build -t sql-query-assistant .
docker run -p 7860:7860 sql-query-assistant
```

### API Endpoints

- `POST /reset` — Start new episode
- `POST /step` — Submit SQL query
- `GET /state` — Current episode state
- `GET /tasks` — List all tasks
- `GET /grader` — Current grader scores

### Run inference
```bash
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.3-70b-versatile"
export HF_TOKEN="your-groq-api-key"
python inference.py
```

## Baseline Scores

| Task | Score | Correct |
|---|---|---|
| customer_filter | 1.00 | ✅ |
| top_products_revenue | 1.00 | ✅ |
| churned_customers | 0.46 | ❌ |
| category_review_summary | 1.00 | ✅ |

Model: llama-3.3-70b-versatile via Groq API
## Motivation

Training AI agents to write correct SQL is a high-value real-world
problem — data analysts, business intelligence tools, and 
natural language interfaces to databases all depend on it.

This environment provides:
- A reproducible benchmark for text-to-SQL agents
- Dense reward signals that teach agents to improve iteratively
- Progressive difficulty from simple filters to complex churn analysis
- A realistic e-commerce schema agents will encounter in the wild