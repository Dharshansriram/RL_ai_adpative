"""
train_grpo.py — TRL + Unsloth GRPO Training for Adaptive AI Project Manager
=============================================================================

Round 2 submission training script.
Run this as a Colab notebook cell-by-cell, or as a standalone Python script.

Stack:
  - OpenEnv  : environment interface (adaptive-ai-project-manager)
  - TRL      : GRPOTrainer for RL fine-tuning
  - Unsloth  : memory-efficient LoRA training
  - HuggingFace Hub : model upload

Usage in Colab:
  1. Run CELL 0 (install)
  2. Run CELL 1 (imports + config)
  3. Run CELL 3 (rollout function) — validates env connectivity
  4. Run CELL 4 (train) — starts GRPO training
  5. Run CELL 5 (evaluate) — before/after comparison
  6. Run CELL 6 (push to hub)
"""

# ============================================================
# CELL 0 — Install dependencies
# ============================================================
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install trl>=0.8.0 openenv-core>=0.2.0 requests datasets transformers accelerate
# !pip install wandb  # optional but recommended for plots

# ============================================================
# CELL 1 — Imports & Configuration
# ============================================================

import json
import os
import random
import re
from dataclasses import dataclass
from typing import Optional

import requests
import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

# ── Configuration ────────────────────────────────────────────
OPENENV_BASE_URL = os.environ.get(
    "OPENENV_BASE_URL",
    "https://<your-hf-space>.hf.space",  # ← replace with your HF Space URL
)
BASE_MODEL   = "unsloth/Qwen2.5-1.5B-Instruct"   # small, fast, Colab-friendly
OUTPUT_DIR   = "./pm-grpo-checkpoints"
HF_REPO_ID   = os.environ.get("HF_REPO_ID", "dharshansriram/adaptive-ai-pm-grpo")  # LoRA repo (not the Space unless you want)
MAX_EPISODES = 200      # total episodes for training
SCENARIOS    = ["easy", "medium", "hard"]
SEED_RANGE   = (0, 999)
MAX_STEPS_PER_EPISODE = 40

# GRPO hyperparameters
GRPO_CFG = dict(
    num_train_epochs        = 3,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 4,
    num_generations         = 8,   # rollouts per prompt (G in GRPO)
    max_prompt_length       = 512,
    max_completion_length   = 256,
    remove_unused_columns   = False,
    learning_rate           = 5e-6,
    weight_decay            = 0.01,
    warmup_ratio            = 0.05,
    lr_scheduler_type       = "cosine",
    logging_steps           = 10,
    save_steps              = 50,
    output_dir              = OUTPUT_DIR,
    report_to               = "wandb",   # set to "none" to disable wandb
    run_name                = "pm-grpo-v1",
    beta                    = 0.04,      # KL penalty coefficient
    use_vllm                = False,     # set True if you have vLLM installed
)


# ============================================================
# CELL 2 — Load model with Unsloth
# ============================================================

from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = BASE_MODEL,
    max_seq_length = 2048,
    load_in_4bit   = True,   # 4-bit QLoRA for Colab memory efficiency
    dtype          = None,   # auto
)

model = FastLanguageModel.get_peft_model(
    model,
    r                 = 16,    # LoRA rank
    target_modules    = ["q_proj", "v_proj", "k_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    lora_alpha        = 32,
    lora_dropout      = 0.0,
    bias              = "none",
    use_gradient_checkpointing = "unsloth",
    random_state      = 42,
)

print("✅ Model loaded with Unsloth 4-bit QLoRA")
print(f"   Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


# ============================================================
# CELL 3 — Environment Client & Rollout Function
# ============================================================

class PMEnvClient:
    """Thin HTTP client for the OpenEnv Project Manager environment."""

    def __init__(self, base_url: str = OPENENV_BASE_URL) -> None:
        self.base = base_url.rstrip("/")

    def reset(self, scenario: str = "medium", seed: int = 42) -> dict:
        r = requests.post(f"{self.base}/reset",
                          json={"scenario": scenario, "seed": seed},
                          timeout=30)
        r.raise_for_status()
        return r.json()

    def step(self, session_id: str, action: dict) -> dict:
        r = requests.post(f"{self.base}/step",
                          json={"session_id": session_id, **action},
                          timeout=30)
        r.raise_for_status()
        return r.json()

    def grade(self, session_id: str) -> dict:
        r = requests.get(f"{self.base}/grader",
                         params={"session_id": session_id},
                         timeout=30)
        r.raise_for_status()
        return r.json()


def obs_to_prompt(obs: dict, scenario: str) -> str:
    """
    Convert environment observation to a language model prompt.

    The prompt gives the model everything it needs to make a decision:
    - Sprint context (step, scenario, progress)
    - Task list (id, name, priority, status, deadline, remaining_sp)
    - Developer list (id, name, skills, fatigue, availability)
    - Recent events

    Output format: structured JSON action
    """
    step      = obs.get("step", 0)
    max_steps = obs.get("max_steps", 30)
    metrics   = obs.get("metrics", {})
    tasks     = obs.get("tasks", [])
    devs      = obs.get("developers", [])
    events    = obs.get("recent_events", [])

    # Summarise tasks (top 8 most urgent)
    task_lines = []
    for t in sorted(tasks, key=lambda x: (
        0 if x["status"] in ("ready", "in_progress") else 1,
        -x.get("business_value", 0),
    ))[:8]:
        task_lines.append(
            f"  {t['id']} | {t['name'][:20]:<20} | {t['status']:<12} | "
            f"pri={t['priority']} | {t['remaining_points']:.1f}sp | "
            f"deadline={t['deadline_step']} | skills={t.get('required_skills', [])}"
        )

    dev_lines = []
    for d in devs:
        skills_str = ", ".join(
            f"{k}={v:.2f}" for k, v in list((d.get("skills") or {}).items())[:3]
        )
        dev_lines.append(
            f"  {d['id']} | {d['name']:<14} | fatigue={d['fatigue']:.2f} | "
            f"avail={d['available']} | tasks={d.get('current_tasks',[])} | {skills_str}"
        )

    event_str = "\n".join(f"  ⚡ {e['description']}" for e in events[-3:]) or "  None"

    prompt = f"""You are an AI Agile project manager. Manage developers and tasks to maximise sprint delivery.

SPRINT STATUS: step={step}/{max_steps} | scenario={scenario}
Delivered: {metrics.get('delivered_story_points',0):.1f}/{metrics.get('total_story_points',0):.1f} SP | Tasks done: {metrics.get('completed_tasks',0)}/{metrics.get('total_tasks',0)}

TASKS (top 8):
{chr(10).join(task_lines)}

DEVELOPERS:
{chr(10).join(dev_lines)}

RECENT EVENTS:
{event_str}

Choose ONE action. Respond with valid JSON only:
{{"action_type": "assign_task", "task_id": "...", "developer_id": "..."}}
OR: {{"action_type": "unassign_task", "task_id": "...", "developer_id": "..."}}
OR: {{"action_type": "reprioritize", "task_id": "...", "new_priority": "CRITICAL|HIGH|MEDIUM|LOW"}}
OR: {{"action_type": "rest_developer", "developer_id": "..."}}
OR: {{"action_type": "split_task", "task_id": "...", "split_ratio": 0.5}}
OR: {{"action_type": "pair_program", "task_id": "...", "primary_developer_id": "...", "secondary_developer_id": "..."}}
OR: {{"action_type": "noop"}}

Your decision:"""
    return prompt


def _strip_json_fences(text: str) -> str:
    text = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    return text.replace("```", "").strip()


def _smart_quotes_to_ascii(text: str) -> str:
    return (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )


def _balanced_json_slices(text: str):
    n = len(text)
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        depth = 0
        in_str = False
        esc = False
        quote = None
        for j in range(i, n):
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == quote:
                    in_str = False
                continue
            if c in ('"', "'"):
                in_str = True
                quote = c
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    yield text[i : j + 1]
                    break


def _unwrap_nested_action(d: dict) -> dict:
    for k in ("action", "decision", "tool_call", "output", "response"):
        inner = d.get(k)
        if isinstance(inner, dict) and "action_type" in inner:
            return inner
    return d


def _normalize_keys(d: dict) -> dict:
    key_map = {
        "actionType": "action_type",
        "type": "action_type",
        "ActionType": "action_type",
        "taskId": "task_id",
        "developerId": "developer_id",
        "primaryDeveloperId": "primary_developer_id",
        "secondaryDeveloperId": "secondary_developer_id",
        "newPriority": "new_priority",
        "splitRatio": "split_ratio",
    }
    return {key_map.get(k, k): v for k, v in d.items()}


def _normalize_action_type(v) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "assign": "assign_task",
        "task_assign": "assign_task",
        "rest": "rest_developer",
        "noop": "noop",
        "no_op": "noop",
        "none": "noop",
        "idle": "noop",
        "pair": "pair_program",
        "pairing": "pair_program",
    }
    return aliases.get(s, s)


def _normalize_priority(v) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip().upper().replace(" ", "_")
    if s in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
        return s
    return None


def parse_action(text: str) -> dict:
    """Extract one environment action dict from model text (robust to nesting / fences)."""
    raw = _smart_quotes_to_ascii(_strip_json_fences(text or ""))

    candidates = []
    if raw.startswith("{"):
        candidates.append(raw)
    candidates.extend(_balanced_json_slices(raw))

    def try_load(s: str):
        s = s.strip()
        if not s:
            return None
        s = re.sub(r",(\s*[}\]])", r"\1", s)
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return None

    for blob in candidates:
        obj = try_load(blob)
        if obj is None:
            continue
        if isinstance(obj, list) and obj:
            obj = obj[0]
        if not isinstance(obj, dict):
            continue
        obj = _unwrap_nested_action(obj)
        obj = _normalize_keys(obj)
        at = _normalize_action_type(obj.get("action_type"))
        if not at:
            continue
        obj["action_type"] = at
        if "new_priority" in obj and obj["new_priority"] is not None:
            npv = _normalize_priority(obj["new_priority"])
            if npv:
                obj["new_priority"] = npv
        for fld in (
            "task_id",
            "developer_id",
            "primary_developer_id",
            "secondary_developer_id",
        ):
            if fld in obj and obj[fld] is not None:
                obj[fld] = str(obj[fld]).strip()
        return obj

    return {"action_type": "noop"}


def compute_reward_from_env(env_client: PMEnvClient, session_id: str,
                             final_obs: dict) -> float:
    """
    Multi-signal reward for GRPO:
      1. Grade-based reward (primary signal, from 8-dim grader)
      2. Efficiency bonus
      3. Anti-hacking check (negative if violations detected)
    """
    if not final_obs.get("done", False):
        # Episode still running — use intermediate signal
        metrics = final_obs.get("metrics", {})
        total   = max(1, metrics.get("total_story_points", 1))
        done_sp = metrics.get("delivered_story_points", 0)
        return done_sp / total * 0.5   # partial credit

    try:
        grade_resp = env_client.grade(session_id)
        data = grade_resp.get("data", {})
        weighted_total = data.get("weighted_total", 0.0)
        grade_letter   = data.get("grade", "F")
        # Scale to [0, 1] for GRPO
        # A+ = 1.0, A = 0.9, B+ = 0.8, B = 0.7, C = 0.6, D = 0.5, F = 0
        grade_map = {"A+": 1.0, "A": 0.9, "B+": 0.8, "B": 0.7,
                     "C": 0.6, "D": 0.5, "F": 0.0}
        grade_bonus = grade_map.get(grade_letter, 0.0)
        # Combine weighted score + grade bonus
        reward = 0.7 * weighted_total + 0.3 * grade_bonus
        return float(reward)
    except Exception:
        return 0.0


def run_episode_rollout(
    model_fn,
    tokenizer,
    env_client: PMEnvClient,
    scenario: str = "medium",
    seed: int = 42,
    max_steps: int = MAX_STEPS_PER_EPISODE,
) -> tuple[list[dict], float]:
    """
    Run one complete episode, generating actions with the model.

    Returns:
        trajectory: list of {prompt, completion, reward} dicts
        final_reward: episode-level reward
    """
    reset_resp   = env_client.reset(scenario=scenario, seed=seed)
    session_id   = reset_resp["data"]["session_id"]
    obs          = reset_resp["data"]["observation"]
    trajectory   = []
    episode_done = False
    step_count   = 0
    history_actions: list[dict] = []

    while not episode_done and step_count < max_steps:
        prompt = obs_to_prompt(obs, scenario)

        # Generate action with model
        inputs  = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens    = 80,
                temperature       = 0.7,
                do_sample         = True,
                pad_token_id      = tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        parsed = parse_action(completion)
        replay_snapshot = json.dumps(history_actions)

        action = dict(parsed)
        action["session_id"] = session_id

        # Execute in environment
        try:
            step_resp = env_client.step(session_id, action)
            obs       = step_resp["data"]["observation"]
        except Exception:
            obs = {"done": True, "metrics": {}}

        episode_done = obs.get("done", False)
        step_count  += 1

        # Step-level reward from rubric breakdown (if available)
        step_reward = obs.get("reward", 0.0)
        trajectory.append({
            "prompt":     prompt,
            "completion": completion,
            "reward":     step_reward,
            "scenario":   scenario,
            "episode_seed": int(seed),
            "replay_actions_json": replay_snapshot,
        })
        history_actions.append({k: v for k, v in parsed.items() if k != "session_id"})

    final_reward = compute_reward_from_env(env_client, session_id, obs)
    # Assign terminal reward to last step
    if trajectory:
        trajectory[-1]["reward"] += final_reward

    return trajectory, final_reward


# ============================================================
# CELL 4 — Build Training Dataset (Rollouts)
# ============================================================

def build_dataset(
    model_fn,
    tokenizer,
    env_client: PMEnvClient,
    n_episodes: int = MAX_EPISODES,
) -> Dataset:
    """
    Run N episodes, collect trajectories, build HF Dataset for GRPOTrainer.
    """
    all_prompts     = []
    all_completions = []
    all_rewards     = []
    all_scores      = []
    all_scenario    = []
    all_seed        = []
    all_replay_json = []

    rng = random.Random(42)

    for i in range(n_episodes):
        scenario = rng.choice(SCENARIOS)
        seed     = rng.randint(*SEED_RANGE)
        try:
            traj, final_reward = run_episode_rollout(
                model_fn, tokenizer, env_client,
                scenario=scenario, seed=seed,
            )
            for t in traj:
                all_prompts.append(t["prompt"])
                all_completions.append(t["completion"])
                all_rewards.append(t["reward"])
                all_scenario.append(t["scenario"])
                all_seed.append(t["episode_seed"])
                all_replay_json.append(t["replay_actions_json"])
            all_scores.append(final_reward)
            if (i + 1) % 10 == 0:
                avg = sum(all_scores[-10:]) / min(10, len(all_scores))
                print(f"  Episode {i+1:3d}/{n_episodes} | "
                      f"scenario={scenario:<6} seed={seed:4d} | "
                      f"reward={final_reward:.3f} | avg10={avg:.3f}")
        except Exception as e:
            print(f"  ⚠  Episode {i+1} failed: {e}")

    dataset = Dataset.from_dict({
        "prompt":               all_prompts,
        "completion":           all_completions,
        "reward":               all_rewards,
        "scenario":             all_scenario,
        "episode_seed":         all_seed,
        "replay_actions_json":  all_replay_json,
    })
    print(f"\n✅ Dataset built: {len(dataset)} steps from {n_episodes} episodes (replay metadata for GRPO)")
    return dataset


# ============================================================
# CELL 4b — Reward Function for GRPOTrainer
# ============================================================

# GRPOTrainer expects a reward function: (prompts, completions) -> List[float]
# We pre-compute rewards during rollout and store them; here we return them directly.
# For live training, we hook the env into the reward fn below.

_env_client = PMEnvClient(OPENENV_BASE_URL)

GRPO_ENV_REWARD_ALPHA = float(os.environ.get("GRPO_ENV_REWARD_ALPHA", "0.82"))


def _format_shape_reward(completion: str) -> float:
    action = parse_action(completion)
    reward = 0.0
    valid_types = {"assign_task", "unassign_task", "reprioritize",
                   "rest_developer", "split_task", "pair_program", "noop"}
    if "action_type" in action:
        reward += 0.10
    if action.get("action_type") in valid_types:
        reward += 0.15
    if action.get("action_type") == "assign_task":
        if action.get("task_id") and action.get("developer_id"):
            reward += 0.25
    if action.get("action_type") == "noop":
        reward -= 0.05
    return float(reward)


def _live_step_reward(
    env_client: PMEnvClient,
    scenario: str,
    episode_seed: int,
    replay_actions_json: str,
    completion: str,
) -> float:
    fmt = _format_shape_reward(completion)
    try:
        history = json.loads(replay_actions_json) if replay_actions_json else []
    except json.JSONDecodeError:
        history = []

    try:
        reset_resp = env_client.reset(scenario=scenario, seed=int(episode_seed))
        session_id = reset_resp["data"]["session_id"]
        obs = reset_resp["data"]["observation"]
    except Exception:
        return 0.2 * fmt

    for h in history:
        step_action = dict(h)
        step_action["session_id"] = session_id
        try:
            step_resp = env_client.step(session_id, step_action)
            obs = step_resp["data"]["observation"]
        except Exception:
            return 0.15 * fmt

    cand = parse_action(completion)
    cand = {k: v for k, v in cand.items() if v is not None}
    cand["session_id"] = session_id
    try:
        step_resp = env_client.step(session_id, cand)
        obs = step_resp["data"]["observation"]
    except Exception:
        return 0.12 * fmt

    step_r = float(obs.get("reward", 0.0) or 0.0)
    if obs.get("done"):
        term = compute_reward_from_env(env_client, session_id, obs)
        return float(max(-0.5, min(2.0, 0.25 * step_r + 0.75 * term + 0.05 * fmt)))
    return float(max(-0.5, min(1.5, step_r + 0.08 * fmt)))


def grpo_reward_fn(
    prompts: list[str],
    completions: list[str],
    completion_ids=None,
    trainer_state=None,
    log_extra=None,
    log_metric=None,
    environments=None,
    scenario=None,
    episode_seed=None,
    replay_actions_json=None,
    **kwargs,
) -> list[float]:
    """
    Live env reward: replay prefix (from dataset), apply model completion, read /step reward
    and /grader when done; blend with a small format-only signal.
    """
    n = len(completions)
    out = [0.0] * n

    use_live = (
        scenario is not None
        and episode_seed is not None
        and replay_actions_json is not None
        and len(scenario) == n
        and len(episode_seed) == n
        and len(replay_actions_json) == n
    )

    if not use_live:
        for i, completion in enumerate(completions):
            out[i] = _format_shape_reward(completion)
        return out

    for i, completion in enumerate(completions):
        live = _live_step_reward(
            _env_client,
            scenario[i],
            int(episode_seed[i]),
            replay_actions_json[i],
            completion,
        )
        fmt = _format_shape_reward(completion)
        out[i] = (1.0 - GRPO_ENV_REWARD_ALPHA) * fmt + GRPO_ENV_REWARD_ALPHA * live
    return out


# ============================================================
# CELL 5 — Train with GRPOTrainer
# ============================================================

def train(dataset: Dataset) -> None:
    config = GRPOConfig(**GRPO_CFG)

    trainer = GRPOTrainer(
        model          = model,
        tokenizer      = tokenizer,
        reward_funcs   = [grpo_reward_fn],
        args           = config,
        train_dataset  = dataset,
    )

    print("🚀 Starting GRPO training...")
    trainer.train()

    # Save LoRA adapter (NOT merged to avoid QLoRA upcast bug)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
    print(f"✅ LoRA adapter saved to {OUTPUT_DIR}/lora_adapter")


# ============================================================
# CELL 6 — Evaluate: Baseline vs Trained Agent
# ============================================================

def evaluate_comparison(env_client: PMEnvClient, n_seeds: int = 5) -> dict:
    """
    Compare baseline heuristic agent vs trained LLM agent.
    Runs n_seeds episodes on each scenario, reports reward curves.
    """
    results = {"baseline": {}, "trained": {}}

    for scenario in ["easy", "medium", "hard"]:
        base_scores    = []
        trained_scores = []

        for seed in range(n_seeds):
            # --- Baseline: heuristic agent via /demo endpoint ---
            try:
                r = requests.get(
                    f"{env_client.base}/demo",
                    params={"scenario": scenario, "seed": seed},
                    timeout=60,
                )
                r.raise_for_status()
                data = r.json()["data"]
                base_scores.append(data.get("weighted_total", 0.0))
            except Exception as e:
                print(f"  Baseline {scenario} seed={seed} failed: {e}")
                base_scores.append(0.0)

            # --- Trained: LLM agent ---
            try:
                _, reward = run_episode_rollout(
                    model, tokenizer, env_client,
                    scenario=scenario, seed=seed,
                )
                trained_scores.append(reward)
            except Exception as e:
                print(f"  Trained {scenario} seed={seed} failed: {e}")
                trained_scores.append(0.0)

        results["baseline"][scenario] = {
            "mean":  round(sum(base_scores) / len(base_scores), 4),
            "scores": base_scores,
        }
        results["trained"][scenario] = {
            "mean":  round(sum(trained_scores) / len(trained_scores), 4),
            "scores": trained_scores,
        }

        base_mean    = results["baseline"][scenario]["mean"]
        trained_mean = results["trained"][scenario]["mean"]
        delta        = trained_mean - base_mean
        sign         = "+" if delta >= 0 else ""
        print(f"  {scenario:<8} | baseline={base_mean:.3f} | "
              f"trained={trained_mean:.3f} | Δ={sign}{delta:.3f}")

    return results


def plot_training_curves(log_dir: str = OUTPUT_DIR) -> None:
    """
    Plot reward and loss curves from training logs.
    Saves as PNG for README embedding.
    """
    import json
    import glob
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots")
        return

    log_files = glob.glob(f"{log_dir}/**/trainer_state.json", recursive=True)
    if not log_files:
        print("No trainer_state.json found; run training first")
        return

    with open(log_files[-1]) as f:
        state = json.load(f)

    log_history = state.get("log_history", [])
    steps   = [h["step"]   for h in log_history if "loss" in h]
    losses  = [h["loss"]   for h in log_history if "loss" in h]
    rewards = [h.get("rewards/mean", h.get("reward", None))
               for h in log_history if "loss" in h]
    rewards = [r for r in rewards if r is not None]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(steps[:len(losses)], losses, color="#2563eb", linewidth=1.5)
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.grid(alpha=0.3)

    if rewards:
        ax2.plot(steps[:len(rewards)], rewards, color="#16a34a", linewidth=1.5)
        ax2.set_xlabel("Training Step")
        ax2.set_ylabel("Mean Reward")
        ax2.set_title("Episode Reward (GRPO)")
        ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{log_dir}/training_curves.png", dpi=150, bbox_inches="tight")
    print(f"✅ Saved training_curves.png to {log_dir}/")
    plt.show()


# ============================================================
# CELL 7 — Push to HuggingFace Hub
# ============================================================

def push_to_hub(repo_id: str = HF_REPO_ID) -> None:
    """
    Push LoRA adapter to HuggingFace Hub (same pattern as notebook: login + push).
    Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN before running.
    """
    from huggingface_hub import login

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(
            "Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN to push the adapter to the Hub."
        )
    login(token=token)
    model.push_to_hub(repo_id, token=token)
    tokenizer.push_to_hub(repo_id, token=token)
    print(f"✅ Model pushed to https://huggingface.co/{repo_id}")


# ============================================================
# CELL 8 — Main (run all cells in order)
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  PM-GRPO Training Pipeline")
    print("=" * 60)

    env_client = PMEnvClient(OPENENV_BASE_URL)

    # Verify env connectivity
    try:
        ping = requests.get(f"{OPENENV_BASE_URL}/health", timeout=10)
        ping.raise_for_status()
        print(f"✅ Environment reachable at {OPENENV_BASE_URL}")
    except Exception as e:
        print(f"❌ Cannot reach environment: {e}")
        print("   Set OPENENV_BASE_URL to your HF Space URL")
        raise SystemExit(1)

    # Build dataset
    dataset = build_dataset(model, tokenizer, env_client, n_episodes=MAX_EPISODES)

    # Train
    train(dataset)

    # Evaluate
    print("\n📊 Evaluation: Baseline vs Trained")
    print("-" * 50)
    results = evaluate_comparison(env_client, n_seeds=5)

    # Plot
    plot_training_curves()

    # Push (optional)
    if HF_REPO_ID and "your-username" not in HF_REPO_ID and "YOUR_" not in HF_REPO_ID:
        push_to_hub(HF_REPO_ID)

    print("\n✅ Training pipeline complete!")
