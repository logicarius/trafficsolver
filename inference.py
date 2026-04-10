"""
inference.py — UrbanFlow baseline inference script.

Mandatory stdout format:
  [START] task=<task_name> env=urbanflow model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>

Usage:
  API_BASE_URL=https://api.openai.com/v1 MODEL_NAME=gpt-4o HF_TOKEN=hf_xxx python inference.py
"""

import os
import json
import requests
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")
ENV_URL      = os.getenv("ENV_URL",       "http://localhost:7860")

client = OpenAI(api_key=HF_TOKEN or "sk-dummy", base_url=API_BASE_URL)

TASKS = ["clear_bottleneck", "rebalance_network", "maintain_stability"]


def call_env(endpoint: str, method: str = "GET", data: dict = None) -> dict:
    url = f"{ENV_URL}{endpoint}"
    if method == "POST":
        r = requests.post(url, json=data, timeout=30)
    else:
        r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def get_agent_action(obs: dict, task_id: str) -> dict:
    """Ask the LLM what action to take given the current observation."""
    system_prompt = """You are a traffic engineer AI. You manage urban traffic networks.
You will receive the current network state and must output a JSON action to reduce congestion.

Available action types:
- "add_capacity": increase road capacity. Requires road_id, capacity_increase (1-10)
- "add_road": add a new road. Requires from_node, to_node, new_capacity (1-10)
- "reroute": shift load from most congested road. No extra params needed.
- "adjust_signal": change traffic signal timing. Requires target_node, green_time (10-60)
- "do_nothing": take no action.

IMPORTANT: Respond ONLY with a valid JSON object. No explanation outside JSON.
Example: {"action_type": "add_capacity", "road_id": "r4", "capacity_increase": 3, "reasoning": "r4 is the main bottleneck"}
"""
    user_msg = f"""Task: {task_id}
Current network state:
{json.dumps(obs, indent=2)}

What action do you take? Respond with JSON only."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.2,
            max_tokens=300,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as e:
        # Fallback: reroute action
        return {"action_type": "reroute", "reasoning": f"LLM error fallback: {e}"}


def run_task(task_id: str) -> dict:
    """Run one full episode for a task. Returns final score info."""
    # Reset
    obs = call_env("/reset", "POST", {"task_id": task_id})
    max_steps = obs.get("max_steps", 5)

    print(f"[START] task={task_id} env=urbanflow model={MODEL_NAME}", flush=True)

    rewards = []
    last_error = "null"
    step_num = 0
    done = False
    final_score = 0.0

    for step_num in range(1, max_steps + 1):
        # Get action from LLM
        action = get_agent_action(obs, task_id)
        action_str = action.get("action_type", "do_nothing")

        # Step environment
        try:
            result = call_env("/step", "POST", action)
            reward_val = result["reward"]["total"]
            done = result["done"]
            error = result["info"].get("action_error") or "null"
            obs = result["observation"]
            final_score = reward_val
        except Exception as e:
            reward_val = 0.0
            done = True
            error = str(e)

        rewards.append(reward_val)
        last_error = error

        print(
            f"[STEP] step={step_num} action={action_str} "
            f"reward={reward_val:.2f} done={str(done).lower()} error={error}",
            flush=True,
        )

        if done:
            break

    success = final_score >= 0.5
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={str(success).lower()} steps={step_num} "
        f"score={final_score:.2f} rewards={rewards_str}",
        flush=True,
    )

    return {"task_id": task_id, "score": final_score, "success": success, "steps": step_num}


def main():
    print(f"UrbanFlow Inference | model={MODEL_NAME} | env={ENV_URL}", flush=True)
    print("-" * 60, flush=True)

    results = []
    for task_id in TASKS:
        result = run_task(task_id)
        results.append(result)
        print("-" * 60, flush=True)

    # Summary
    avg_score = sum(r["score"] for r in results) / len(results)
    print(f"\nBaseline Results:", flush=True)
    for r in results:
        status = "✓" if r["success"] else "✗"
        print(f"  {status} {r['task_id']}: score={r['score']:.2f}", flush=True)
    print(f"  Average score: {avg_score:.2f}", flush=True)


if __name__ == "__main__":
    main()
