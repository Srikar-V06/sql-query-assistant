"""
FastAPI server for the SQL Query Assistant OpenEnv environment.

Endpoints
─────────
POST /reset         — Start new episode, returns Observation
POST /step          — Submit SQL Action, returns StepResult
GET  /state         — Current episode state (EpisodeState)
GET  /tasks         — List all tasks + action schema
GET  /grader        — Grader score for current episode
POST /baseline      — Run full baseline (all 4 tasks) with GPT-4o, returns scores

All request/response bodies are JSON. The environment is kept as a single
server-side singleton — this is appropriate for hackathon / eval use.
"""

from __future__ import annotations

import os
import json
import time
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel as PydanticBaseModel, Field

from environment import SQLQueryEnvironment
from models      import Action
from tasks       import ALL_TASKS, TASK_ORDER


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "SQL Query Assistant — OpenEnv",
    description = "An RL environment where agents write SQL queries against a seeded e-commerce database.",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# Single environment instance shared across requests
env = SQLQueryEnvironment()
_episode_started = False   # guard: warn if /step called before /reset


# ---------------------------------------------------------------------------
# Request / Response schemas (FastAPI-native Pydantic, separate from env models)
# ---------------------------------------------------------------------------

class ResetRequest(PydanticBaseModel):
    task_id: Optional[str] = Field(
        default=None,
        description="One of 'customer_filter', 'top_products_revenue', 'churned_customers'. "
                    "Leave null to cycle through tasks in order.",
        examples=["customer_filter", "top_products_revenue", "churned_customers"],
    )

class StepRequest(PydanticBaseModel):
    query: str = Field(
        description="A valid SQLite SELECT statement to run against the database.",
        examples=["SELECT customer_id, name FROM customers WHERE city = 'Mumbai' LIMIT 5"],
    )

class BaselineRequest(PydanticBaseModel):
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key. If omitted, falls back to OPENAI_API_KEY env var.",
    )
    model: str = Field(
        default="gpt-4o",
        description="OpenAI model to use for the baseline run.",
    )
    max_steps_override: Optional[int] = Field(
        default=None,
        description="Override max_steps for each task (useful for quick testing).",
    )


# ---------------------------------------------------------------------------
# Serialisation helper
# (env model objects use a lightweight BaseModel without .dict() / .model_dump)
# ---------------------------------------------------------------------------

def _to_dict(obj) -> Any:
    """Recursively convert env model objects to plain dicts for JSON response."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_dict(i) for i in obj]
    if hasattr(obj, "__dict__"):
        return {k: _to_dict(v) for k, v in obj.__dict__.items()}
    return obj


# ---------------------------------------------------------------------------
# POST /reset
# ---------------------------------------------------------------------------

@app.post("/reset", summary="Start a new episode")
def reset(req: ResetRequest = ResetRequest()) -> dict:
    """
    Resets the environment and starts a new episode.

    Optionally specify a `task_id` to run a specific task. If omitted,
    tasks cycle in order: easy → medium → hard → easy …

    Returns the initial **Observation**.
    """
    global _episode_started
    try:
        obs = env.reset(task_id=req.task_id)
        _episode_started = True
        return {"status": "ok", "observation": _to_dict(obs)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {e}")


# ---------------------------------------------------------------------------
# POST /step
# ---------------------------------------------------------------------------

@app.post("/step", summary="Submit a SQL query action")
def step(req: StepRequest) -> dict:
    """
    Executes the given SQL query against the live database and returns
    a **StepResult** containing the new observation, reward, done flag,
    and an info dict with scoring details.

    Only SELECT statements are permitted. Write operations (INSERT, UPDATE,
    DELETE, DROP, etc.) are safety-blocked and return score=0.
    """
    if not _episode_started:
        raise HTTPException(
            status_code=400,
            detail="Call POST /reset before POST /step.",
        )
    try:
        action = Action(query=req.query)
        result = env.step(action)
        return {
            "status":      "ok",
            "observation": _to_dict(result.observation),
            "reward":      _to_dict(result.reward),
            "done":        result.done,
            "info":        result.info,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {e}")


# ---------------------------------------------------------------------------
# GET /state
# ---------------------------------------------------------------------------

@app.get("/state", summary="Get current episode state")
def state() -> dict:
    """
    Returns the full internal **EpisodeState** — useful for debugging,
    logging, and verifying the agent's progress.
    """
    if not _episode_started:
        raise HTTPException(
            status_code=400,
            detail="No active episode. Call POST /reset first.",
        )
    try:
        s = env.state()
        return {"status": "ok", "state": _to_dict(s)}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# GET /tasks
# ---------------------------------------------------------------------------

@app.get("/tasks", summary="List all available tasks")
def tasks() -> dict:
    """
    Returns metadata for all 4 tasks — descriptions, difficulty,
    max_steps, required columns, and the action schema.
    """
    task_list = []
    for tid in TASK_ORDER:
        t = ALL_TASKS[tid]
        task_list.append({
            "task_id":          t.task_id,
            "difficulty":       t.difficulty,
            "description":      t.description,
            "max_steps":        t.max_steps,
            "required_columns": t.required_columns,
            "order_matters":    t.order_matters,
            "tags":             t.tags,
        })

    return {
        "status": "ok",
        "tasks":  task_list,
        "action_schema": {
            "type": "object",
            "fields": {
                "query": {
                    "type":        "string",
                    "description": "A valid SQLite SELECT (or WITH…SELECT) statement.",
                    "example":     "SELECT * FROM customers LIMIT 5",
                }
            },
        },
    }


# ---------------------------------------------------------------------------
# GET /grader
# ---------------------------------------------------------------------------

@app.get("/grader", summary="Get grader score for the current episode")
def grader() -> dict:
    """
    Returns the last reward signal, breakdown, and best score so far
    for the active episode — handy for evaluation dashboards.
    """
    if not _episode_started:
        raise HTTPException(
            status_code=400,
            detail="No active episode. Call POST /reset first.",
        )
    try:
        s = env.state()
        last_reward = _to_dict(env._last_reward)
        return {
            "status":            "ok",
            "task_id":           s.task_id,
            "steps_taken":       s.steps_taken,
            "best_score_so_far": s.best_score_so_far,
            "cumulative_score":  s.cumulative_score,
            "done":              s.done,
            "last_reward":       last_reward,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# POST /baseline
# ---------------------------------------------------------------------------

@app.post("/baseline", summary="Run the GPT-4o baseline across all 4 tasks")
def baseline(req: BaselineRequest = BaselineRequest()) -> dict:
    """
    Runs a full baseline evaluation using the OpenAI API (GPT-4o by default).

    For each task:
      1. Resets the environment
      2. Feeds the observation to the model with a system prompt
      3. Parses the SQL from the response
      4. Calls step() in a loop until done=True or max_steps
      5. Records the final score

    Returns a score table for all 4 tasks.

    Requires OPENAI_API_KEY as an environment variable or in the request body.
    """
    import re

    api_key = req.openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="No OpenAI API key provided. Pass 'openai_api_key' in the request body "
                   "or set the OPENAI_API_KEY environment variable.",
        )

    try:
        from openai import OpenAI
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="openai package not installed. Add 'openai' to requirements.txt.",
        )

    client = OpenAI(api_key=api_key)

    SYSTEM_PROMPT = """You are an expert SQL analyst working with a SQLite database.
Your task is to write a single correct SQL query that answers the question provided.

Rules:
- Only write SELECT statements (or WITH ... SELECT).
- Do not include any explanation — output ONLY the SQL query.
- Wrap your SQL in ```sql ... ``` code fences.
- Use the schema and sample rows provided to write accurate queries.
- Pay close attention to column names, table relationships, and filter conditions.
"""

    def extract_sql(text: str) -> str:
        """Pull SQL from ```sql ... ``` fences, or return raw text."""
        match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return text.strip()

    def build_user_prompt(obs) -> str:
        return f"""## Task
{obs.task_description}

## Database Schema
```sql
{obs.schema_info}
```

## Sample Data
{json.dumps(obs.sample_rows, indent=2, default=str)}

## Previous Attempt
Query:  {obs.last_query or 'None'}
Result: {json.dumps(obs.last_result, default=str) if obs.last_result else 'None'}
Error:  {obs.last_error or 'None'}

Write the correct SQL query now:"""

    results = []

    for task_id in TASK_ORDER:
        task_meta = ALL_TASKS[task_id]
        max_steps = req.max_steps_override or task_meta.max_steps

        obs    = env.reset(task_id=task_id)
        done   = False
        steps  = 0
        final_score     = 0.0
        final_correct   = False
        last_feedback   = ""
        episode_history = []

        while not done and steps < max_steps:
            steps += 1
            t0 = time.time()

            # ── Call the model ────────────────────────────────────────────
            try:
                response = client.chat.completions.create(
                    model       = req.model,
                    max_tokens  = 800,
                    temperature = 0.0,
                    messages    = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": build_user_prompt(obs)},
                    ],
                )
                raw_text = response.choices[0].message.content or ""
                sql      = extract_sql(raw_text)
            except Exception as e:
                sql = f"-- model error: {e}"

            # ── Step the environment ──────────────────────────────────────
            try:
                step_result = env.step(Action(query=sql))
            except RuntimeError:
                break

            obs          = step_result.observation
            done         = step_result.done
            final_score  = step_result.reward.score
            final_correct= step_result.reward.is_correct
            last_feedback= step_result.reward.feedback

            episode_history.append({
                "step":     steps,
                "query":    sql[:200] + ("…" if len(sql) > 200 else ""),
                "score":    round(final_score, 4),
                "correct":  final_correct,
                "latency_s": round(time.time() - t0, 2),
            })

        results.append({
            "task_id":      task_id,
            "difficulty":   task_meta.difficulty,
            "final_score":  round(final_score, 4),
            "is_correct":   final_correct,
            "steps_used":   steps,
            "max_steps":    max_steps,
            "last_feedback": last_feedback,
            "history":      episode_history,
        })

    # Summary row
    avg_score = round(sum(r["final_score"] for r in results) / len(results), 4)

    return {
        "status":    "ok",
        "model":     req.model,
        "results":   results,
        "summary": {
            "avg_score":     avg_score,
            "tasks_correct": sum(1 for r in results if r["is_correct"]),
            "total_tasks":   len(results),
        },
    }


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/", summary="Health check")
def root() -> dict:
    return {
        "status":      "ok",
        "environment": "sql-query-assistant",
        "version":     "1.0.0",
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/grader", "/baseline"],
    }
# if __name__ == "__main__":
#     import uvicorn
#     # This forces Uvicorn to run directly from this file, 
#     # ensuring all your local imports work perfectly.
#     uvicorn.run(app, host="0.0.0.0", port=7860)