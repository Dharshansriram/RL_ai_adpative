"""
quickstart.py — Zero-config demo of the Adaptive AI Project Manager.

Run with:
    python quickstart.py

No server, no setup. Shows a complete episode in ~5 seconds:
  • Sprint setup  — tasks, developers, story points, dependencies
  • Live steps    — action taken, decision reason, events fired
  • ASCII timeline — visual replay of the full episode
  • Score report  — 8-dimension deterministic grade
  • Baseline comparison — easy / medium / hard side by side


"""

from environment import ProjectManagerEnv
from demo import PriorityAwareAgent
from baseline_runner import BaselineRunner
from timeline import EpisodeTimeline
from models import TaskPriority, TaskStatus


try:
    import sys
    if sys.platform == "win32":
        raise ImportError
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    RED    = "\033[91m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"
except ImportError:
    GREEN = YELLOW = CYAN = RED = BOLD = RESET = ""

W = 68


def hr(ch="─"):
    print(ch * W)


def header(title: str, ch="═"):
    print(ch * W)
    print(f"  {BOLD}{title}{RESET}")
    print(ch * W)




SCENARIO = "medium"
SEED     = 42

env   = ProjectManagerEnv(scenario=SCENARIO, seed=SEED)
agent = PriorityAwareAgent()
tl    = EpisodeTimeline()
obs   = env.reset()

header("🚀  ADAPTIVE AI PROJECT MANAGER  —  QUICKSTART DEMO")
print(f"  Scenario : {SCENARIO}  |  Seed : {SEED}  |  Steps : {obs.max_steps}")
print(f"  Tasks    : {obs.metrics.total_tasks}  |  Developers : {len(obs.developers)}")
print(f"  Total SP : {obs.metrics.total_story_points:.1f}  "
      f"|  Business Value : {obs.metrics.total_business_value:.1f}")
hr()

print(f"\n  {BOLD}Developers{RESET}")
for d in obs.developers:
    top_skills = sorted(d.skills.items(), key=lambda x: -x[1])[:3]
    skill_str  = "  ".join(f"{k.value}={v:.2f}" for k, v in top_skills)
    print(f"    {CYAN}•{RESET} {d.name:<10}  vel={d.velocity:.1f}  {skill_str}")

print(f"\n  {BOLD}Initial Backlog{RESET}")
for t in sorted(obs.tasks, key=lambda t: -t.priority.value):
    dep_str = f"  deps={len(t.dependencies)}" if t.dependencies else ""
    prio_col = RED if t.priority == TaskPriority.CRITICAL else (
               YELLOW if t.priority == TaskPriority.HIGH else RESET)
    print(f"    {prio_col}•{RESET} {t.name:<28} {t.story_points:.1f}sp  "
          f"deadline=step{t.deadline_step}  {t.priority.name:<8}{dep_str}")

print()
hr()



SHOW_STEPS  = {1, 5, 10, 15, 20, 25, 30}
print(f"\n  {BOLD}Running episode…{RESET}  (key steps shown)\n")

while not obs.done:
    action = agent.act(obs)
    obs    = env.step(action)
    tl.record(obs)

    dr   = obs.info.get("decision_reason", {})
    show = obs.step in SHOW_STEPS or bool(obs.recent_events) or obs.done

    if show:
        sp       = dr.get("sprint_progress", {})
        pct      = sp.get("pct", 0)
        fill     = int(pct / 5)                           # 20-char bar
        bar      = f"{'█' * fill}{'░' * (20 - fill)}"
        sig      = dr.get("reward_signal", "neutral")
        sig_col  = (GREEN  if "positive"   in sig else
                    RED    if "penalty"    in sig else
                    YELLOW if "neutral"    in sig else RESET)
        print(f"  Step {obs.step:>2}/{obs.max_steps}  [{bar}] "
              f"{pct:.0f}%  {sig_col}reward={obs.reward:+.3f}{RESET}")
        print(f"    ▸ {dr.get('action_outcome', '')}")
        for ev in dr.get("events_fired", []):
            print(f"    {YELLOW}🔔 {ev}{RESET}")
        if dr.get("urgent_warning"):
            print(f"    {RED}⚠  {dr['urgent_warning']}{RESET}")



score = env.grade()
print()
hr("═")
print(score.report())



print(tl.to_ascii())



print()
header("📊  BASELINE COMPARISON  —  easy / medium / hard")
runner  = BaselineRunner(seed=42)
result  = runner.run_all()

print(f"\n  Agent: {result.agent}  |  Seed: {result.seed}\n")
dims    = ["delivery","value","timeliness","priority","team_health","adaptability","efficiency","dependency"]
weights = [0.25, 0.20, 0.15, 0.10, 0.10, 0.10, 0.07, 0.03]
sep52   = "─" * 52

print(f"  {'Dimension':<20} {'EASY':>7}  {'MEDIUM':>7}  {'HARD':>7}  w")
print("  " + sep52)
for dim, w in zip(dims, weights):
    vals = [r.dimensions[dim] for r in result.tasks]
    row  = "  ".join(f"{v:>7.3f}" for v in vals)

    best_idx = vals.index(max(vals))
    parts    = [f"{v:>7.3f}" for v in vals]
    parts[best_idx] = f"{GREEN}{parts[best_idx]}{RESET}"
    print(f"  {dim:<20} {'  '.join(parts)}  {w}")
print("  " + sep52)
for label, key in [("TOTAL (weighted)", None), ("Grade", None)]:
    if key is None:
        totals = [f"{r.weighted_total:>7.4f}" for r in result.tasks]
        grades = [f"  [{r.grade:>2}]  " for r in result.tasks]
        print(f"  {'TOTAL':<20} {'  '.join(totals)}")
        print(f"  {'Grade':<20} {'  '.join(grades)}")
        break
print("  " + sep52)

s   = result.summary
print(f"\n  Mean score : {s['mean_score']:.4f}")
print(f"  Range      : {s['min_score']:.4f} – {s['max_score']:.4f}")
print(f"\n  {s['summary']}")

print()
hr("═")
print(f"""
  {BOLD}What this benchmark tests:{RESET}
  The heuristic agent (PriorityAwareAgent) scores D grade on medium.
  An RL agent trained on this environment targets B–A (0.70–0.90+),
  demonstrating real learning value over the heuristic baseline.

  {BOLD}To run the API server:{RESET}
    uvicorn main:app --host 0.0.0.0 --port 7860 --reload
    → http://localhost:7860/docs     (Swagger UI)
    → GET /demo?scenario=medium      (full episode in one call)
    → GET /baseline                  (baseline scores)
    → GET /timeline?session_id=...   (step-by-step replay)

  {BOLD}To run all tests:{RESET}
    python tests.py                  (31 tests)
""")
