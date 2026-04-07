"""
Graders for the SQL Query Assistant environment.

Each grader compares the agent's query result against the pre-computed
ground truth and returns a Reward object with:
  - score           (0.0 – 1.0)  final weighted score for this step
  - partial_credit  (0.0 – 1.0)  same as score (kept for API symmetry)
  - is_correct      bool          True only on exact match
  - feedback        str           human-readable explanation
  - breakdown       dict          per-dimension scores for transparency

Scoring dimensions (applied to all tasks):
  ┌────────────────┬────────┬────────────────────────────────────────────────┐
  │ Dimension      │ Weight │ When awarded                                   │
  ├────────────────┼────────┼────────────────────────────────────────────────┤
  │ syntax         │  0.10  │ Query ran without error                        │
  │ columns        │  0.20  │ All required columns present in output         │
  │ partial_rows   │  0.30  │ ≥50 % of ground-truth rows present (set match)│
  │ exact_match    │  0.40  │ Result == ground truth (rows + order if needed)│
  └────────────────┴────────┴────────────────────────────────────────────────┘

Total possible = 1.0.
A safety-blocked query (write op) earns 0.0 flat with no partial credit.
"""

from __future__ import annotations

import re
from typing import Optional

from models import Reward


# ---------------------------------------------------------------------------
# Weight constants — change these in one place if you tune the rubric
# ---------------------------------------------------------------------------
W_SYNTAX       = 0.10
W_COLUMNS      = 0.20
W_PARTIAL_ROWS = 0.30
W_EXACT        = 0.40

# Partial-rows threshold: agent must return at least this fraction of GT rows
PARTIAL_ROW_THRESHOLD = 0.50

# For medium task: partial credit on partial_rows if agent gets top-K correct
TOP_K_PARTIAL = 3   # getting top-3 of top-5 still earns partial_rows credit


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_row(row: dict) -> dict:
    """
    Lowercase all keys; convert numeric strings to float for comparison.
    This lets us compare rows regardless of minor formatting differences.
    """
    out = {}
    for k, v in row.items():
        key = k.lower().strip()
        if isinstance(v, str):
            try:
                v = float(v)
            except ValueError:
                v = v.strip().lower()
        elif isinstance(v, float):
            v = round(v, 2)
        out[key] = v
    return out


def _rows_as_tuples(rows: list[dict]) -> list[tuple]:
    """Convert list-of-dicts to list-of-sorted-tuples for set comparison."""
    return [tuple(sorted(_normalise_row(r).items())) for r in rows]


def _required_columns_present(result: list[dict], required: list[str]) -> tuple[bool, list[str]]:
    """Return (all_present, missing_columns)."""
    if not result:
        return False, required
    actual_cols = {k.lower() for k in result[0].keys()}
    missing = [c for c in required if c not in actual_cols]
    return len(missing) == 0, missing


def _uses_cte_or_subquery(query: str) -> bool:
    """Heuristic check: does the SQL use a CTE or subquery? (no AST parser needed)"""
    q = query.upper()
    has_cte      = bool(re.search(r'\bWITH\b', q))
    has_subquery = bool(re.search(r'\(\s*SELECT\b', q))
    return has_cte or has_subquery


def _uses_join(query: str) -> bool:
    """Heuristic check: does the SQL contain a JOIN?"""
    return bool(re.search(r'\bJOIN\b', query.upper()))


def _uses_group_by(query: str) -> bool:
    return bool(re.search(r'\bGROUP\s+BY\b', query.upper()))


def _safe_feedback(parts: list[str]) -> str:
    return " | ".join(parts) if parts else "No issues."


# ---------------------------------------------------------------------------
# Core grading engine (shared across all 3 tasks)
# ---------------------------------------------------------------------------

def _base_grade(
    agent_result:    Optional[list[dict]],
    agent_error:     Optional[str],
    ground_truth:    list[dict],
    required_cols:   list[str],
    order_matters:   bool,
    agent_query:     str,
    style_bonus_fn=None,        # optional callable(query) → (bonus: float, note: str)
) -> Reward:
    """
    Shared grading logic used by all three task graders.

    Returns a fully populated Reward object.
    style_bonus_fn: if provided, can push exact_match score from 0.9 → 1.0.
    """

    breakdown: dict[str, float] = {
        "syntax":       0.0,
        "columns":      0.0,
        "partial_rows": 0.0,
        "exact_match":  0.0,
    }
    feedback_parts: list[str] = []

    # ── 0. Safety block / write operation ───────────────────────────────────
    if agent_error and any(
        kw in agent_error for kw in ("Forbidden operation", "Only SELECT")
    ):
        return Reward(
            score=0.0,
            partial_credit=0.0,
            is_correct=False,
            feedback=f"Safety block: {agent_error}",
            breakdown=breakdown,
        )

    # ── 1. Syntax (query ran without error) ─────────────────────────────────
    if agent_error:
        feedback_parts.append(f"Syntax error: {agent_error}")
        return Reward(
            score=0.0,
            partial_credit=0.0,
            is_correct=False,
            feedback=_safe_feedback(feedback_parts),
            breakdown=breakdown,
        )

    breakdown["syntax"] = W_SYNTAX
    feedback_parts.append("✓ Query executed successfully")

    # ── 2. Required columns present ─────────────────────────────────────────
    all_cols_present, missing = _required_columns_present(agent_result or [], required_cols)

    if all_cols_present:
        breakdown["columns"] = W_COLUMNS
        feedback_parts.append("✓ All required columns present")
    else:
        feedback_parts.append(f"✗ Missing columns: {missing}")

    # ── 3. Partial rows (set-based, ≥50 % of GT rows found) ─────────────────
    gt_tuples    = _rows_as_tuples(ground_truth)
    agent_tuples = _rows_as_tuples(agent_result or [])
    gt_set       = set(gt_tuples)
    agent_set    = set(agent_tuples)

    overlap = len(gt_set & agent_set)
    gt_len  = len(gt_set) or 1          # avoid div-by-zero

    overlap_frac = overlap / gt_len

    if overlap_frac >= PARTIAL_ROW_THRESHOLD:
        breakdown["partial_rows"] = W_PARTIAL_ROWS
        feedback_parts.append(
            f"✓ Partial rows: {overlap}/{len(gt_set)} GT rows matched ({overlap_frac:.0%})"
        )
    else:
        feedback_parts.append(
            f"✗ Partial rows: only {overlap}/{len(gt_set)} GT rows matched ({overlap_frac:.0%})"
        )

    # ── 4. Exact match ───────────────────────────────────────────────────────
    is_correct = False

    if order_matters:
        # Must match row-for-row in correct order
        is_correct = (agent_tuples == gt_tuples)
    else:
        # Order-insensitive set equality
        is_correct = (agent_set == gt_set)

    if is_correct:
        base_exact = W_EXACT
        style_note = ""

        # Optional style bonus (e.g. using CTE on hard task)
        if style_bonus_fn:
            bonus, style_note = style_bonus_fn(agent_query)
            base_exact = min(W_EXACT, base_exact + bonus)

        breakdown["exact_match"] = base_exact
        feedback_parts.append(
            f"✓ Exact match — result is correct!{(' ' + style_note) if style_note else ''}"
        )
    else:
        # Diagnose common mistakes for richer feedback
        if len(agent_result or []) == 0:
            feedback_parts.append("✗ Exact match: query returned no rows")
        elif len(agent_result) != len(ground_truth):
            feedback_parts.append(
                f"✗ Exact match: expected {len(ground_truth)} rows, got {len(agent_result)}"
            )
        elif order_matters and agent_set == gt_set:
            feedback_parts.append("✗ Exact match: rows correct but ORDER BY is wrong")
        else:
            extra   = agent_set - gt_set
            missing_rows = gt_set - agent_set
            feedback_parts.append(
                f"✗ Exact match: {len(missing_rows)} GT rows missing, "
                f"{len(extra)} extra rows returned"
            )

    # ── Final score ──────────────────────────────────────────────────────────
    total = sum(breakdown.values())
    total = round(min(total, 1.0), 4)
    total = max(0.001, min(0.999, total))

    return Reward(
        score=total,
        partial_credit=total,
        is_correct=is_correct,
        feedback=_safe_feedback(feedback_parts),
        breakdown=breakdown,
    )


# ---------------------------------------------------------------------------
# Task-specific graders
# ---------------------------------------------------------------------------

def grade_easy(
    agent_result: Optional[list[dict]],
    agent_error:  Optional[str],
    ground_truth: list[dict],
    agent_query:  str,
) -> Reward:
    """
    TASK 1 — customer_filter (Easy)

    No special style bonuses. Exact ordered match required.
    Required columns: customer_id, name, city, signup_date
    """
    required_cols = ["customer_id", "name", "city", "signup_date"]

    return _base_grade(
        agent_result=agent_result,
        agent_error=agent_error,
        ground_truth=ground_truth,
        required_cols=required_cols,
        order_matters=True,
        agent_query=agent_query,
        style_bonus_fn=None,
    )


def grade_medium(
    agent_result: Optional[list[dict]],
    agent_error:  Optional[str],
    ground_truth: list[dict],
    agent_query:  str,
) -> Reward:
    """
    TASK 2 — top_products_revenue (Medium)

    Extra checks:
      - Partial-row credit: agent gets top-3 of top-5 correct → still earns partial_rows
      - Style check: uses JOIN + GROUP BY (expected technique)

    The TOP-5 order-sensitive requirement means the agent must produce
    rows in exactly the right revenue-descending sequence.
    """
    required_cols = ["product_id", "product_name", "category_name", "total_revenue"]

    # Override partial-row logic: for top-5, getting ≥3 correct rows earns credit
    # We implement this via a patched style_bonus_fn that adjusts feedback only
    # (the main partial_rows logic in _base_grade uses set overlap ≥50 %, which
    # for a 5-row result means ≥3 rows — this already aligns with TOP_K_PARTIAL)

    def medium_style_bonus(query: str) -> tuple[float, str]:
        notes = []
        if _uses_join(query):
            notes.append("uses JOIN ✓")
        else:
            notes.append("no JOIN detected (may be wrong approach)")
        if _uses_group_by(query):
            notes.append("uses GROUP BY ✓")
        else:
            notes.append("no GROUP BY detected")
        # No numeric bonus — style note is informational only for medium
        return 0.0, " | ".join(notes)

    reward = _base_grade(
        agent_result=agent_result,
        agent_error=agent_error,
        ground_truth=ground_truth,
        required_cols=required_cols,
        order_matters=True,
        agent_query=agent_query,
        style_bonus_fn=medium_style_bonus,
    )
    return reward


def grade_hard(
    agent_result: Optional[list[dict]],
    agent_error:  Optional[str],
    ground_truth: list[dict],
    agent_query:  str,
) -> Reward:
    """
    TASK 3 — churned_customers (Hard)

    Style bonus: if the agent uses a CTE or subquery (the expected approach),
    exact_match can reach the full W_EXACT weight (0.40).
    If correct but written as a flat query (unlikely but possible),
    exact_match is capped at W_EXACT − 0.05 = 0.35, giving a max score of 0.95.

    This rewards good SQL craftsmanship without penalising correct flat queries
    too harshly — they still achieve is_correct = True.
    """
    required_cols = ["customer_id", "name", "city", "total_q1_orders", "last_order_value"]

    def hard_style_bonus(query: str) -> tuple[float, str]:
        if _uses_cte_or_subquery(query):
            return 0.0, "uses CTE/subquery ✓ (full marks)"
        else:
            # Correct but no CTE: slight deduction baked into base_exact cap
            return -0.05, "correct but no CTE/subquery (−0.05 style penalty)"

    reward = _base_grade(
        agent_result=agent_result,
        agent_error=agent_error,
        ground_truth=ground_truth,
        required_cols=required_cols,
        order_matters=True,
        agent_query=agent_query,
        style_bonus_fn=hard_style_bonus,
    )
    return reward
def grade_medium_hard(
    agent_result: Optional[list[dict]],
    agent_error:  Optional[str],
    ground_truth: list[dict],
    agent_query:  str,
) -> Reward:
    """
    TASK 4 — category_review_summary (Medium-Hard)

    Requires correct use of HAVING to filter categories by review count.
    Style bonus: uses HAVING clause (expected technique).
    """
    required_cols = ["category_name", "total_reviews", "avg_rating", "pct_five_star"]

    def medium_hard_style_bonus(query: str) -> tuple[float, str]:
        q = query.upper()
        notes = []
        if re.search(r'\bHAVING\b', q):
            notes.append("uses HAVING ✓")
        else:
            notes.append("no HAVING detected — may have filtered incorrectly")
        if _uses_join(query):
            notes.append("uses JOIN ✓")
        if _uses_group_by(query):
            notes.append("uses GROUP BY ✓")
        return 0.0, " | ".join(notes)

    return _base_grade(
        agent_result=agent_result,
        agent_error=agent_error,
        ground_truth=ground_truth,
        required_cols=required_cols,
        order_matters=True,
        agent_query=agent_query,
        style_bonus_fn=medium_hard_style_bonus,
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

GRADER_MAP = {
    "customer_filter":    grade_easy,
    "top_products_revenue": grade_medium,
    "churned_customers":  grade_hard,
    "category_review_summary": grade_medium_hard,
}


def grade(
    task_id:      str,
    agent_result: Optional[list[dict]],
    agent_error:  Optional[str],
    ground_truth: list[dict],
    agent_query:  str,
) -> Reward:
    """
    Main entry point. Dispatches to the correct task grader.

    Args:
        task_id:      One of 'customer_filter', 'top_products_revenue', 'churned_customers'
        agent_result: Rows returned by execute_query() — None if query errored
        agent_error:  Error string from execute_query() — None if query succeeded
        ground_truth: Pre-computed correct answer rows (list of dicts)
        agent_query:  The raw SQL string the agent submitted

    Returns:
        Reward object with score, partial_credit, is_correct, feedback, breakdown
    """
    if task_id not in GRADER_MAP:
        raise ValueError(f"Unknown task_id '{task_id}'. Valid: {list(GRADER_MAP.keys())}")
    return GRADER_MAP[task_id](agent_result, agent_error, ground_truth, agent_query)