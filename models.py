"""
Typed Pydantic models for the SQL Query Assistant OpenEnv environment.
These define the contract between the environment and the agent.
"""

from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field


class Observation(BaseModel):
    """
    What the agent sees at each step.

    Includes the task description, database schema, the result of the
    last query executed, and episode progress counters.
    """

    task_id: str = Field(description="Unique identifier for the current task")
    task_description: str = Field(description="Plain-English description of what the agent must accomplish")
    schema_info: str = Field(description="CREATE TABLE statements for all tables in the database")
    sample_rows: dict[str, list[dict]] = Field(
        description="3 sample rows per table so the agent understands data format and ranges",
        default_factory=dict,
    )
    last_query: Optional[str] = Field(default=None, description="The SQL query submitted in the previous step")
    last_result: Optional[list[dict]] = Field(
        default=None,
        description="Rows returned by the last query (capped at 50 rows for context length)",
    )
    last_error: Optional[str] = Field(
        default=None, description="SQL error message if the last query failed to execute"
    )
    steps_taken: int = Field(default=0, description="Number of step() calls made so far this episode")
    max_steps: int = Field(default=10, description="Maximum allowed steps before episode terminates")
    done: bool = Field(default=False, description="True if the episode has ended")


class Action(BaseModel):
    """
    What the agent submits each step — a single SQL query string.

    The query is executed against a live SQLite database.
    Only SELECT statements are permitted; write operations are rejected.
    """

    query: str = Field(
        description="A valid SQLite SELECT statement to run against the database",
        examples=[
            "SELECT customer_id, name FROM customers WHERE city = 'Mumbai' LIMIT 10",
            "SELECT p.name, SUM(oi.quantity * oi.unit_price) AS revenue FROM order_items oi JOIN products p ON oi.product_id = p.product_id GROUP BY p.product_id ORDER BY revenue DESC LIMIT 5",
        ],
    )


class Reward(BaseModel):
    """
    Per-step reward signal returned alongside each Observation.

    Designed to give the agent partial credit so it can learn
    from intermediate progress, not just final success/failure.
    """

    score: float = Field(description="Final grader score for this step (0.0 = wrong, 1.0 = perfect)", ge=0.0, le=1.0)
    partial_credit: float = Field(
        description="Breakdown of partial credit earned this step (syntax, columns, rows, correctness)",
        ge=0.0,
        le=1.0,
    )
    is_correct: bool = Field(description="True if the agent's result exactly matches the ground truth")
    feedback: str = Field(description="Human-readable explanation of why this score was awarded")
    breakdown: dict[str, float] = Field(
        description="Score breakdown by dimension: syntax, columns, partial_rows, exact_match",
        default_factory=dict,
    )


class StepResult(BaseModel):
    """
    The full return value of step(action).
    Bundles together everything the agent needs to continue.
    """

    observation: Observation
    reward: Reward
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


class EpisodeState(BaseModel):
    """
    Returned by state() — the full internal state of the environment.
    Useful for debugging, checkpointing, and evaluation.
    """

    task_id: str
    steps_taken: int
    max_steps: int
    done: bool
    cumulative_score: float
    best_score_so_far: float
    ground_truth_row_count: int
    last_query: Optional[str] = None
    last_score: Optional[float] = None