"""
inference.py — OpenEnv Baseline Inference Script
Adaptive AI Project Manager

Uses the OpenAI client to drive an AI agent through the environment.
Reads sprint state, sends to OpenAI for action decision, executes in env.
Runs all 3 formal tasks: easy, medium, hard.

Required env vars:
  API_BASE_URL   LLM API endpoint  (e.g. https://api.openai.com/v1)
  MODEL_NAME     model identifier  (e.g. gpt-4o-mini)
  HF_TOKEN       API key / HF token

Optional:
  ENV_BASE_URL   environment server (default: http://localhost:7860)

Strict checker log format:
  [START] task=<str> env=<str> model=<str>
  [STEP]  step=<int> action=<str> reward=<float> done=<bool> error=<str|None>
  [END]   success=<bool> steps=<int> score=<float> rewards=<list>
"""

import json
import os
import time
import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "llama-3.3-70b-versatile")
HF_TOKEN     = os.getenv("HF_TOKEN",     os.getenv("OPENAI_API_KEY", ""))
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860").rstrip("/")

BENCHMARK         = "adaptive-ai-project-manager"
MAX_STEPS         = 35
SUCCESS_THRESHOLD = 0.60
SEED              = 42

TASKS = [
    {"task_id": "task_easy",   "scenario": "easy",   "desc": "6 tasks, 3 devs, 20 steps"},
    {"task_id": "task_medium", "scenario": "medium",  "desc": "12 tasks, 4 devs, 30 steps"},
    {"task_id": "task_hard",   "scenario": "hard",    "desc": "18 tasks, 5 devs, 40 steps"},
]

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error) -> None:
    print(f"[STEP] step={step} action={action!r} reward={reward:.4f} done={done} error={error}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    print(f"[END] success={success} steps={steps} score={score:.4f} rewards={rewards}", flush=True)


def env_reset(scenario: str) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/reset",
                      json={"scenario": scenario, "seed": SEED}, timeout=60)
    r.raise_for_status()
    return r.json()

def env_step(session_id: str, action: dict) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/step",
                      json={"session_id": session_id, **action}, timeout=60)
    r.raise_for_status()
    return r.json()

def env_grader(session_id: str) -> dict:
    r = requests.get(f"{ENV_BASE_URL}/grader",
                     params={"session_id": session_id}, timeout=60)
    r.raise_for_status()
    return r.json()


def format_obs(obs: dict) -> str:
    step    = obs.get("step", 0)
    maxs    = obs.get("max_steps", 40)
    metrics = obs.get("metrics", {})
    tasks   = obs.get("tasks", [])
    devs    = obs.get("developers", [])
    events  = obs.get("recent_events", [])

    ready  = [t for t in tasks if t.get("status") == "ready"]
    avdevs = [d for d in devs if d.get("available") and d.get("fatigue", 1) < 0.90]

    lines = [
        f"Step {step}/{maxs} | Done {metrics.get('completed_tasks',0)}/{metrics.get('total_tasks',0)} tasks | "
        f"Delivered {metrics.get('delivered_story_points',0):.1f}/{metrics.get('total_story_points',1):.1f} SP"
    ]
    if events:
        lines.append("EVENTS: " + " | ".join(e.get("description","") for e in events[:3]))
    if ready:
        lines.append("READY TASKS:")
        for t in ready[:6]:
            sk = ",".join(t.get("required_skills",[]))
            tid = t.get("id", "unknown")
            name = t.get("name", "")
            story_points = float(t.get("story_points", 0.0))
            deadline_step = t.get("deadline_step", "-")
            priority = t.get("priority", "MEDIUM")
            lines.append(
                f"  id={tid} {name!r} {story_points:.1f}sp dl={deadline_step} p={priority} skills=[{sk}]"
            )
    if avdevs:
        lines.append("AVAILABLE DEVS:")
        for d in avdevs[:5]:
            sk = ",".join(f"{k}={v:.0%}" for k,v in list(d.get("skills",{}).items())[:3])
            did = d.get("id", "unknown")
            dname = d.get("name", "unknown")
            lines.append(f"  id={did} {dname} fatigue={d.get('fatigue',0):.2f} skills=[{sk}]")
    return "\n".join(lines)


SYSTEM = """You are an AI Agile project manager. Each turn output ONE action as JSON only.

Valid actions:
{"action_type":"assign_task","task_id":"<id>","developer_id":"<id>"}
{"action_type":"rest_developer","developer_id":"<id>"}
{"action_type":"reprioritize","task_id":"<id>","new_priority":"CRITICAL"}
{"action_type":"pair_program","task_id":"<id>","primary_developer_id":"<id>","secondary_developer_id":"<id>"}
{"action_type":"noop"}

Rules: assign ready tasks to matching skill devs first. Rest devs with fatigue>0.75 if idle.
Output ONLY the JSON, no explanation."""

def get_action(client: OpenAI, obs_text: str, history: list) -> dict:
    msgs = [{"role": "system", "content": SYSTEM}]
    for h in history[-3:]:
        msgs.append({"role": "user",      "content": h["obs"]})
        msgs.append({"role": "assistant", "content": json.dumps(h["act"])})
    msgs.append({"role": "user", "content": obs_text})

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME, messages=msgs, max_tokens=150, temperature=0.1)
        text = resp.choices[0].message.content.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"): text = text[4:]
            text = text.strip()
        act = json.loads(text)
        if "action_type" not in act:
            return {"action_type": "noop"}
        return act
    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", flush=True)
        return {"action_type": "noop"}


def run_task(client: OpenAI, task: dict) -> float:
    scenario = task["scenario"]
    task_id  = task["task_id"]

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards, steps_taken, score, success = [], 0, 0.0, False
    session_id, history = None, []

    try:

        rdata      = env_reset(scenario)
        session_id = rdata["data"]["session_id"]
        obs        = rdata["data"]["observation"]
        done       = obs.get("done", False)


        for step_num in range(1, MAX_STEPS + 1):
            if done:
                break

            obs_text   = format_obs(obs)
            action     = get_action(client, obs_text, history)
            action_str = json.dumps(action)
            error      = None
            reward     = 0.0

            try:
                sr     = env_step(session_id, action)
                obs    = sr["data"]["observation"]
                reward = float(obs.get("reward", 0.0))
                done   = obs.get("done", False)
            except Exception as e:
                error = str(e)

            rewards.append(round(reward, 4))
            steps_taken = step_num
            history.append({"obs": obs_text, "act": action})
            log_step(step=step_num, action=action_str, reward=reward, done=done, error=error)

            if done:
                break


        if session_id and done:
            try:
                gr    = env_grader(session_id)
                score = float(gr["data"].get("weighted_total", 0.0))
            except Exception as e:
                print(f"[DEBUG] grader: {e}", flush=True)
                score = min(1.0, max(0.0, sum(rewards) / max(1, MAX_STEPS * 2)))
        else:
            score = min(1.0, max(0.0, sum(rewards) / max(1, MAX_STEPS * 2)))

        score   = round(score, 4)
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] task {task_id} error: {e}", flush=True)
        score, success = 0.0, False

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


def main():
    if not HF_TOKEN:
        print("[ERROR] HF_TOKEN not set. Export HF_TOKEN=<your-api-key>", flush=True)
        raise SystemExit(1)

    print(f"[INFO] API_BASE_URL={API_BASE_URL} MODEL={MODEL_NAME} ENV={ENV_BASE_URL}", flush=True)

    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

    scores = {}
    for task in TASKS:
        print(f"\n{'='*55}\nRunning {task['task_id']}: {task['desc']}\n{'='*55}", flush=True)
        scores[task["task_id"]] = run_task(client, task)
        time.sleep(1)

    avg = sum(scores.values()) / len(scores)
    print(f"\n[FINAL] easy={scores.get('task_easy',0):.4f} "
          f"medium={scores.get('task_medium',0):.4f} "
          f"hard={scores.get('task_hard',0):.4f} "
          f"avg={avg:.4f}", flush=True)

if __name__ == "__main__":
    main()
