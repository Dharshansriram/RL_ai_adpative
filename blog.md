# Teaching an LLM to Run a Sprint: The Adaptive AI Project Manager

*OpenEnv Hackathon India 2026 · Theme #3.1 — World Modeling / Professional Tasks*

---

## Why this exists

LLMs are everywhere in “engineering lead” work—triage, planning, follow-ups. Most toy environments stop at a single next action. Real product management is **partially observable**, **long-horizon**, and full of **events you did not plan for**: a production bug, a sick teammate, scope creep, an infra outage.

**Adaptive AI Project Manager** is a training environment where the model does **real work** in a live API loop: it reads sprint state, chooses structured actions, and gets **dense, rubric-based feedback** every step, plus a deterministic **episode grader** when the sprint ends. The goal is to move LLM agents toward **coherent state tracking**, **causal task ordering**, and **robust response to disruption**—not one-shot pattern matching.

---

## What you get

### A world model you can call over HTTP (OpenEnv-style)

The simulator exposes a familiar loop:

1. `POST /reset` — new episode (scenario + seed)
2. `POST /step` — one JSON action, advance the clock
3. `GET /state` / `GET /tasks` — read-only context when you need it
4. `GET /grader` — eight weighted dimensions when `done=true`

The agent sees **tasks** (status, story points, deadlines, skills, dependencies), **developers** (fatigue, availability, current load), and **recent events** so the next decision is grounded in the same partially observed sprint your human PM would have.

**Try it live**

- **Space (UI + API):** [huggingface.co/spaces/dharshansriram/openenv-pr-ai](https://huggingface.co/spaces/dharshansriram/openenv-pr-ai)  
- **Base URL for clients / training** (no trailing slash):  
  `https://dharshansriram-openenv-pr-ai.hf.space`  
  Set `OPENENV_BASE_URL` to that value in Colab, Kaggle, or local training so `POST /reset` and `POST /step` hit the running app.

The manifest lives in `openenv.yaml` so the interface stays explicit for tooling and graders. The Space runtime depends on **`openenv-core`** (see `requirements.txt`); training stacks (TRL, Unsloth, etc.) are installed in Colab/Kaggle, not in the slim Space image.

### Seven real actions, not a fake menu

The policy chooses among **assign**, **unassign**, **reprioritize**, **rest**, **split**, **pair program**, and **noop**—each with the fields the backend validates. That keeps the environment **hard to shortcut**: you cannot “solve” it with a single generic string; you have to respect IDs, skills, and timing.

### Stochastic stress

Scenarios range from small sprints to **chaos** settings. Injected events (bugs, sick devs, scope change, urgent work, blocked stacks) push the model off any fixed script. That is intentional: the reward signal should reflect **adaptation**, not memorisation of a static board.

---

## Rewards that teach, not just score

A single number at the end of an episode is too easy to game. This project uses **eight composable rubrics** (see `rubric_rewards.py`): progress, deadlines, skill match, fatigue, dependency order, response to injected work, utilisation, and **anti-hacking** checks (e.g. churn, rest-spam, noop abuse). Each step can expose a **breakdown** in the observation info so you can see *why* a step was credited or penalised.

At episode end, an **8-dimension grader** aggregates delivery, value, timeliness, priority, team health, adaptability, efficiency, and dependencies into a report compatible with the OpenEnv grader story.

> **Note:** Per-step scores can be **negative** when rubrics apply penalties. That is expected; what matters is **trend** and **end-of-episode** quality, not “always positive” numbers.

---

## How we train (GRPO + TRL + Unsloth)

**Group Relative Policy Optimisation (GRPO)** in Hugging Face **TRL** lets the model compare multiple candidate completions for the *same* prompt and shift probability toward what the environment actually rewards. We use **Unsloth** for efficient **4-bit QLoRA** on a small instruct model (e.g. `unsloth/Qwen2.5-1.5B-Instruct`) so training fits typical Colab GPUs.

The training path is not “a CSV of labels.” Rollouts and GRPO rewards can call the **same Space** you deploy: replay the prefix of actions for a state, then score the model’s **next** action against the real `/step` reward (and grader when the episode ends). That keeps the **causal** link between JSON output and sim state—exactly what Theme #3.1 asks for: **real interaction** with a dynamic system.

**Entry points in the repo**

- `PM_GRPO_Training_fixed.ipynb` — **recommended** Colab/Kaggle notebook (install, secrets, config, dataset, GRPO, eval, Hub push)
- `PM_GRPO_Training (3).ipynb` — alternate notebook variant
- `train_grpo.py` — script-shaped pipeline for local or CI-style runs

Set `OPENENV_BASE_URL` to your `*.hf.space` URL before you rely on live rewards; otherwise the reward path may fall back to format-only shaping.

---

## Training evidence on Weights and Biases

This submission includes a **real GRPO training run** logged to **Weights & Biases** so judges can inspect curves interactively (hackathon criterion: *observable* training progress).

| Field | Value |
|--------|--------|
| **W&B entity** | `Dharshansriram-r-ciet` |
| **Project** | `huggingface` |
| **Run name** | `openenv-pm-grpo` |
| **How to open** | In W&B: Workspace → project **huggingface** → run **`openenv-pm-grpo`** → copy the run URL from the browser and paste it into your README checklist (“Training run (W&B)”). |

### What the dashboards show (~500 global steps)

- **`train/loss`** — Training loss stays **small** (on the order of **1e-4**), with normal GRPO noise; at **step 145** the logged loss was about **0.000185**.
- **`train/rewards/grpo_reward_fn/mean`** (and related reward panels) — Mean reward **oscillates** roughly in the **−0.1 … +0.1** band. That is **consistent with a rubric-heavy env** where penalties exist; the important part is stable optimisation, not “always positive reward.” At **step 145**, mean reward was about **−0.043** with reward **std ≈ 0.042**.
- **`train/rewards/grpo_reward_fn/std`** — Higher volatility than the mean; typical for **group-relative** sampling across completions.
- **`train/num_tokens`** — **Linear growth** to on the order of **~1.5M tokens** by the end of the run, i.e. the trainer is doing real work across many tokens, not stalling on step 0.

Together, these panels are the **quantitative** complement to README tables and optional `training_curves.png` exports.

![Summary of W&B panels: loss, GRPO mean reward, token count vs global step](./assets/wandb-openenv-pm-grpo.png)

*Figure: repo-export summary plot aligned with run **`openenv-pm-grpo`** (~500 steps, rubric-style mean reward band). The **same image** is also at the **repository root** as `training_curves.png` so the README can embed it with a single root-level path. For pixel-identical UI to your session, use the interactive W&B link below.*

**Direct link:** open your run in W&B and paste the canonical URL here after publish, e.g.  
`https://wandb.ai/Dharshansriram-r-ciet/huggingface/runs/<RUN_ID>`  
(Replace `<RUN_ID>` with the id from the run’s page.)

---

## What “good” looks like

Hackathon rubrics care about **evidence**: loss and reward curves, and **before/after** comparison against a **baseline** (e.g. heuristic agent or untrained model) on the same seeds and scenarios. The README documents example deltas; this post adds a **logged W&B run** (`openenv-pm-grpo`) as primary curve evidence.

Your proof is: **the Space works**, the **training notebook runs against it**, **W&B shows a long token trajectory + tracked GRPO rewards + loss**, and eval tables show **before/after** behaviour where applicable.

---

## Who this is for

- **Researchers** who want a **PM-shaped** world model with clear APIs and rich rewards  
- **Practitioners** prototyping **LLM + tool** loops for planning and assignments  
- **Judges** who need a **runnable** Space, a **coherent** reward story, and a **reproducible** training path

---

## Links

| Resource | URL |
|----------|-----|
| Hugging Face Space | [huggingface.co/spaces/dharshansriram/openenv-pr-ai](https://huggingface.co/spaces/dharshansriram/openenv-pr-ai) |
| API base | `https://dharshansriram-openenv-pr-ai.hf.space` |
| Weights & Biases (GRPO run) | Open project **`huggingface`** under entity **`Dharshansriram-r-ciet`**, run **`openenv-pm-grpo`**, then paste the run URL here (format: `https://wandb.ai/.../runs/...`) |
| Source code & README | Add your **public GitHub** repo URL when you open-source the submission https://github.com/Dharshansriram/RL_ai_adpative|
| Mini-blog (HF) | After you publish this post on Hugging Face, **replace** this row with that post’s URL (judges expect a link, not only a file in the repo) |

---

## License

MIT — see the repository for full text.

---

*If you post this on Hugging Face, paste the post URL into your project README under “mini-blog / HF writeup” so judges can find it in one click.*
