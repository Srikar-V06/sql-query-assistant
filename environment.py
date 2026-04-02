"""
Core environment class for the SQL Query Assistant OpenEnv.

Implements the standard OpenEnv interface:
    reset(task_id?)  →  Observation
    step(action)     →  StepResult(observation, reward, done, info)
    state()          →  EpisodeState

Reward shaping
──────────────
The grader produces a base score in [0, 1] each step. On top of that
the environment applies two shaping rules so the agent gets a
meaningful dense signal throughout the episode:

  1. Efficiency bonus  — finishing in fewer steps than max_steps earns
                         a small bonus on the final correct step.
                         bonus = 0.05 × (steps_remaining / max_steps)

  2. Step penalty      — after the 3rd attempt the agent pays a small
                         cost per extra step taken on an incorrect answer.
                         penalty = 0.02 per step beyond step 3 (non-correct)

  3. Best-score carry  — the environment tracks best_score_so_far so
                         the agent can compare progress across steps.

Episode termination
───────────────────
The episode ends when:
  (a) the agent produces a correct answer (is_correct = True), OR
  (b) steps_taken reaches task.max_steps.

The hint from the task definition is injected into the observation's
task_description after step 3 if the agent still hasn't solved it.
"""

from __future__ import annotations

import random
from typing import Optional

from database import build_database, execute_query, get_sample_rows, SCHEMA_INFO, SEED
from tasks    import ALL_TASKS, TASK_ORDER, Task, get_task
from graders  import grade
from models   import Observation, Action, Reward, StepResult, EpisodeState


# ---------------------------------------------------------------------------
# Shaping constants
# ---------------------------------------------------------------------------
HINT_AFTER_STEP    = 3      # inject hint into description from this step onward
PENALTY_AFTER_STEP = 3      # start penalising incorrect answers after this step
PENALTY_PER_STEP   = 0.02   # deducted from shaped score per extra incorrect step
EFFICIENCY_WEIGHT  = 0.05   # max efficiency bonus on a correct final answer


class SQLQueryEnvironment:
    """
    SQL Query Assistant — OpenEnv-compatible RL environment.

    The agent observes a task description + database schema, submits SQL
    queries as actions, and receives shaped reward signals that guide it
    toward producing the exact correct result within a step budget.

    Usage
    ─────
        env = SQLQueryEnvironment()
        obs = env.reset()                        # start episode (random task)
        obs = env.reset(task_id="churned_customers")  # or a specific task

        result = env.step(Action(query="SELECT ..."))
        print(result.reward.score, result.done)

        state = env.state()                      # full internal state dict
    """

    def __init__(self, seed: int = SEED):
        self._db_seed    = seed
        self._conn       = None          # sqlite3 connection (rebuilt on reset)
        self._task: Optional[Task] = None
        self._ground_truth: list[dict]  = []
        self._steps_taken  = 0
        self._done         = False
        self._cumulative_score  = 0.0
        self._best_score        = 0.0
        self._last_query: Optional[str]        = None
        self._last_result: Optional[list[dict]] = None
        self._last_error: Optional[str]         = None
        self._last_reward: Optional[Reward]     = None
        self._task_cycle   = list(TASK_ORDER)   # for round-robin when task_id=None
        self._cycle_index  = 0

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def reset(self, task_id: Optional[str] = None) -> Observation:
        """
        Start a new episode.

        Args:
            task_id: One of 'customer_filter', 'top_products_revenue',
                     'churned_customers'.  If None, cycles through tasks
                     in order (easy → medium → hard → easy …).

        Returns:
            Initial Observation with task description, schema, and sample rows.
        """
        # ── Pick task ────────────────────────────────────────────────────
        if task_id is not None:
            self._task = get_task(task_id)
        else:
            self._task = get_task(self._task_cycle[self._cycle_index % 3])
            self._cycle_index += 1

        # ── Fresh DB ──────────────────────────────────────────────────────
        self._conn = build_database(seed=self._db_seed)

        # ── Compute ground truth ─────────────────────────────────────────
        rows, err = execute_query(self._conn, self._task.ground_truth_sql)
        if err:
            raise RuntimeError(
                f"Ground truth query failed for task '{self._task.task_id}': {err}"
            )
        self._ground_truth = rows or []

        # ── Reset episode state ───────────────────────────────────────────
        self._steps_taken     = 0
        self._done            = False
        self._cumulative_score = 0.0
        self._best_score       = 0.0
        self._last_query       = None
        self._last_result      = None
        self._last_error       = None
        self._last_reward      = None

        return self._build_observation()

    def step(self, action: Action) -> StepResult:
        """
        Execute one agent action (a SQL query) and return the result.

        Args:
            action: Action object with a .query string.

        Returns:
            StepResult with (observation, reward, done, info).

        Raises:
            RuntimeError if called before reset() or after episode is done.
        """
        if self._task is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._steps_taken += 1
        query = action.query.strip()

        # ── Execute query ─────────────────────────────────────────────────
        result, error = execute_query(self._conn, query)
        self._last_query  = query
        self._last_result = result
        self._last_error  = error

        # ── Grade ─────────────────────────────────────────────────────────
        base_reward = grade(
            task_id      = self._task.task_id,
            agent_result = result,
            agent_error  = error,
            ground_truth = self._ground_truth,
            agent_query  = query,
        )

        # ── Shape reward ──────────────────────────────────────────────────
        shaped_reward = self._shape_reward(base_reward)
        self._last_reward = shaped_reward

        # ── Update trackers ───────────────────────────────────────────────
        self._cumulative_score += shaped_reward.score
        if shaped_reward.score > self._best_score:
            self._best_score = shaped_reward.score

        # ── Termination ───────────────────────────────────────────────────
        self._done = (
            base_reward.is_correct
            or self._steps_taken >= self._task.max_steps
        )

        obs = self._build_observation()

        info = {
            "task_id":            self._task.task_id,
            "difficulty":         self._task.difficulty,
            "steps_taken":        self._steps_taken,
            "max_steps":          self._task.max_steps,
            "cumulative_score":   round(self._cumulative_score, 4),
            "best_score_so_far":  round(self._best_score, 4),
            "ground_truth_rows":  len(self._ground_truth),
            "agent_rows_returned": len(result) if result is not None else 0,
            "base_score":         base_reward.score,
            "shaped_score":       shaped_reward.score,
        }

        return StepResult(
            observation=obs,
            reward=shaped_reward,
            done=self._done,
            info=info,
        )

    def state(self) -> EpisodeState:
        """Return the full internal state of the current episode."""
        if self._task is None:
            raise RuntimeError("Call reset() before state().")
        return EpisodeState(
            task_id               = self._task.task_id,
            steps_taken           = self._steps_taken,
            max_steps             = self._task.max_steps,
            done                  = self._done,
            cumulative_score      = round(self._cumulative_score, 4),
            best_score_so_far     = round(self._best_score, 4),
            ground_truth_row_count= len(self._ground_truth),
            last_query            = self._last_query,
            last_score            = self._last_reward.score if self._last_reward else None,
        )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _shape_reward(self, base: Reward) -> Reward:
        """
        Apply efficiency bonus and step penalty on top of the grader score.

        Shaping rules:
          • Step penalty:   for steps > PENALTY_AFTER_STEP on an incorrect answer,
                            deduct PENALTY_PER_STEP to encourage early solutions.
          • Efficiency bonus: if the agent gets it right, reward finishing early.
                            bonus = EFFICIENCY_WEIGHT × (remaining_steps / max_steps)

        The shaped score is clamped to [0.0, 1.0].
        """
        score = base.score
        notes: list[str] = []

        if base.is_correct:
            # Efficiency bonus — more steps remaining = bigger bonus
            steps_remaining = self._task.max_steps - self._steps_taken
            bonus = round(EFFICIENCY_WEIGHT * (steps_remaining / self._task.max_steps), 4)
            score = min(1.0, score + bonus)
            if bonus > 0:
                notes.append(f"(+{bonus:.3f} efficiency bonus: solved in {self._steps_taken} steps)")
        else:
            # Step penalty after threshold
            if self._steps_taken > PENALTY_AFTER_STEP:
                extra_steps = self._steps_taken - PENALTY_AFTER_STEP
                penalty = round(PENALTY_PER_STEP * extra_steps, 4)
                score = max(0.0, score - penalty)
                notes.append(f"(-{penalty:.3f} step penalty: step {self._steps_taken} > {PENALTY_AFTER_STEP})")

        score = round(score, 4)

        shaped_feedback = base.feedback
        if notes:
            shaped_feedback = base.feedback + " || Shaping: " + " ".join(notes)

        return Reward(
            score         = score,
            partial_credit= score,
            is_correct    = base.is_correct,
            feedback      = shaped_feedback,
            breakdown     = base.breakdown,
        )

    def _build_observation(self) -> Observation:
        """Construct an Observation from current episode state."""
        description = self._task.description

        # Inject hint after HINT_AFTER_STEP steps (and only if not yet solved)
        if (
            self._steps_taken >= HINT_AFTER_STEP
            and not self._done
            and self._task.hint
        ):
            description = (
                description
                + f"\n\n💡 Hint (unlocked at step {HINT_AFTER_STEP}): {self._task.hint}"
            )

        return Observation(
            task_id          = self._task.task_id,
            task_description = description,
            schema_info      = SCHEMA_INFO,
            sample_rows      = get_sample_rows(self._conn, n=3),
            last_query       = self._last_query,
            last_result      = self._last_result,
            last_error       = self._last_error,
            steps_taken      = self._steps_taken,
            max_steps        = self._task.max_steps,
            done             = self._done,
        ) 