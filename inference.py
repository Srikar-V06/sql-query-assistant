import os
import sys
import json
import time
from openai import OpenAI

# ── Read env vars (never hardcode these) ─────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL")
MODEL_NAME   = os.environ.get("MODEL_NAME")
HF_TOKEN     = os.environ.get("HF_TOKEN")

if not all([API_BASE_URL, MODEL_NAME, HF_TOKEN]):
    print("ERROR: API_BASE_URL, MODEL_NAME, and HF_TOKEN must all be set.", file=sys.stderr)
    sys.exit(1)

# ── OpenAI client pointed at Groq ────────────────────────────────
client = OpenAI(
    api_key  = HF_TOKEN,
    base_url = API_BASE_URL,
)

# ── Import your environment ───────────────────────────────────────
from environment import SQLQueryEnvironment
from models      import Action
from tasks       import TASK_ORDER, ALL_TASKS

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


def run_episode(env, task_id: str, episode_num: int):
    obs  = env.reset(task_id=task_id)
    done = False
    step = 0

    # ── [START] log ───────────────────────────────────────────────
    print(json.dumps({"log": "START", "task_id": task_id, "episode": episode_num}))
    sys.stdout.flush()

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

        result = env.step(Action(query=sql))
        obs    = result.observation
        done   = result.done

        # ── [STEP] log ────────────────────────────────────────────
        print(json.dumps({
            "log":     "STEP",
            "step":    step,
            "query":   sql[:300],
            "score":   round(result.reward.score, 4),
            "correct": result.reward.is_correct,
        }))
        sys.stdout.flush()

        time.sleep(1)   # avoid Groq rate limits

    # ── [END] log ─────────────────────────────────────────────────
    print(json.dumps({
        "log":         "END",
        "task_id":     task_id,
        "final_score": round(result.reward.score, 4),
        "is_correct":  result.reward.is_correct,
    }))
    sys.stdout.flush()


def main():
    env = SQLQueryEnvironment()
    for i, task_id in enumerate(TASK_ORDER, start=1):
        run_episode(env, task_id, episode_num=i)

if __name__ == "__main__":
    main()