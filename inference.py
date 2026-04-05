import os
import sys
import json
import time
from openai import OpenAI

# ── Read env vars ─────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "llama-3.3-70b-versatile")
HF_TOKEN     = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("ERROR: HF_TOKEN not set. Please set your Groq API key.", file=sys.stderr)
    sys.exit(1)

# ── OpenAI client pointed at Groq ────────────────────────────────
client = OpenAI(
    api_key  = HF_TOKEN,
    base_url = API_BASE_URL,
)

# ── Import environment ────────────────────────────────────────────
from environment import SQLQueryEnvironment
from models      import Action
from tasks       import TASK_ORDER

SYSTEM_PROMPT = """\
You are an expert SQL analyst working with a SQLite e-commerce database.
Your only job is to write a single correct SQL query that answers the task.

Rules:
- Output ONLY the SQL query — no explanation, no preamble.
- Wrap the query in ```sql ... ``` code fences.
- Use only SELECT statements (or WITH...SELECT CTEs).
- Study the schema and sample data carefully before writing.
- If a previous attempt is shown, analyse what went wrong and fix it.
"""

def extract_sql(text: str) -> str:
    import re
    match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()

def build_prompt(obs) -> str:
    return f"""## Task
{obs.task_description}

## Schema
{obs.schema_info}

## Sample rows
{json.dumps(obs.sample_rows, indent=2, default=str)}

## Previous attempt
Query:  {obs.last_query  or 'None'}
Result: {json.dumps(obs.last_result, default=str) if obs.last_result else 'None'}
Error:  {obs.last_error  or 'None'}

Write the correct SQL query now:"""


def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env=sql-query-assistant model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error) -> None:
    error_val    = error if error else "null"
    done_val     = str(done).lower()
    action_short = action.replace("\n", " ")[:120]
    print(f"[STEP] step={step} action={action_short} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


def run_episode(env, task_id: str):
    obs     = env.reset(task_id=task_id)
    done    = False
    step    = 0
    rewards = []

    log_start(task=task_id, model=MODEL_NAME)

    while not done and step < obs.max_steps:
        step += 1

        try:
            response = client.chat.completions.create(
                model       = MODEL_NAME,
                max_tokens  = 1000,
                temperature = 0.0,
                messages    = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_prompt(obs)},
                ],
            )
            sql = extract_sql(response.choices[0].message.content or "")
        except Exception as e:
            sql = f"-- model error: {e}"

        result  = env.step(Action(query=sql))
        obs     = result.observation
        done    = result.done
        reward  = result.reward.score
        error   = result.observation.last_error

        rewards.append(reward)
        log_step(step=step, action=sql, reward=reward, done=done, error=error)

        time.sleep(1)

    success     = result.reward.is_correct
    score       = sum(rewards) / len(rewards) if rewards else 0.0
    score       = min(max(score, 0.0), 1.0)
    log_end(success=success, steps=step, score=score, rewards=rewards)


def main():
    env = SQLQueryEnvironment()
    for task_id in TASK_ORDER:
        run_episode(env, task_id)

if __name__ == "__main__":
    main()