"""
baseline_runner.py — Runs the PriorityAwareAgent heuristic baseline
across all three formal OpenEnv tasks and returns structured results.

Tasks:
  easy   → small sprint (6 tasks, 3 devs, 20 steps)
  medium → dependencies + deadlines (12 tasks, 4 devs, 40 steps)
  hard   → chaos + dynamic events (18 tasks, 5 devs, 50 steps)
"""

from __future__ import annotations

from typing import Any

from environment import ProjectManagerEnv
from demo import PriorityAwareAgent
from api_models import BaselineResponse, TaskResultSchema


FORMAL_TASKS = [
    {
        "task_id": "task_easy",
        "scenario": "easy",
        "description": (
            "Small sprint: 6 tasks, 3 developers, 20 steps. "
            "No dependency chains. Baseline for minimal viable delivery."
        ),
    },
    {
        "task_id": "task_medium",
        "scenario": "medium",
        "description": (
            "Medium sprint: 12 tasks, 4 developers, 40 steps. "
            "Dependency chains, varying deadlines, and scope changes."
        ),
    },
    {
        "task_id": "task_hard",
        "scenario": "hard",
        "description": (
            "Hard sprint: 18 tasks, 5 developers, 50 steps. "
            "Dense dependencies, dynamic events (bugs, sick devs, outages), "
            "tight deadlines, and chaos conditions."
        ),
    },
]


_DIMENSION_LABELS = {
    "delivery":     "Story-point delivery rate",
    "value":        "Business-value capture",
    "timeliness":   "On-time delivery rate",
    "priority":     "Priority ordering",
    "team_health":  "Team health (fatigue/churn)",
    "adaptability": "Event-injected task completion",
    "efficiency":   "Dev throughput efficiency",
    "dependency":   "Dependency-order correctness",
}

_IMPROVEMENT_TIPS = {
    "delivery":     "Assign developers earlier; use pair programming on large tasks.",
    "value":        "Prioritise tasks with high business_value × priority; use reprioritize action.",
    "timeliness":   "Watch deadline_step carefully; bump near-deadline tasks to CRITICAL.",
    "priority":     "Never let CRITICAL tasks fail while LOW tasks finish — rank by urgency score.",
    "team_health":  "Rest developers when fatigue ≥ 0.78; avoid unassign churn.",
    "adaptability": "Always immediately assign injected hotfix/urgent tasks upon creation.",
    "efficiency":   "Minimise idle developers; prefer pair programming on CRITICAL tasks.",
    "dependency":   "Check task.dependencies before assigning; complete blockers first.",
}


class BaselineRunner:
    """
    Executes PriorityAwareAgent across all formal tasks.

    All runs are deterministic given the same seed.
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed

    def run_all(self) -> BaselineResponse:
        results: list[TaskResultSchema] = []

        for task_def in FORMAL_TASKS:
            result = self._run_one(task_def)
            results.append(result)

        summary = self._summarise(results)
        return BaselineResponse(
            status="success",
            agent="PriorityAwareAgent",
            seed=self._seed,
            tasks=results,
            summary=summary,
        )



    def _run_one(self, task_def: dict) -> TaskResultSchema:
        env   = ProjectManagerEnv(scenario=task_def["scenario"], seed=self._seed)
        agent = PriorityAwareAgent()
        obs   = env.reset()

        while not obs.done:
            action = agent.act(obs)
            obs    = env.step(action)

        breakdown = env.grade()
        m = obs.metrics

        del_pct = (
            f"{m.delivered_story_points / m.total_story_points * 100:.1f}%"
            if m.total_story_points > 0 else "0.0%"
        )

        dims = {
            "delivery":     round(breakdown.delivery_score,     4),
            "value":        round(breakdown.value_score,        4),
            "timeliness":   round(breakdown.timeliness_score,   4),
            "priority":     round(breakdown.priority_score,     4),
            "team_health":  round(breakdown.team_health_score,  4),
            "adaptability": round(breakdown.adaptability_score, 4),
            "efficiency":   round(breakdown.efficiency_score,   4),
            "dependency":   round(breakdown.dependency_score,   4),
        }


        weakest_dim = min(dims, key=dims.get)
        improvement_hint = (
            f"Weakest: {_DIMENSION_LABELS[weakest_dim]} "
            f"({dims[weakest_dim]:.3f}). "
            f"Tip: {_IMPROVEMENT_TIPS[weakest_dim]}"
        )

        return TaskResultSchema(
            task_id=task_def["task_id"],
            scenario=task_def["scenario"],
            seed=self._seed,
            weighted_total=round(breakdown.weighted_total, 6),
            grade=breakdown.grade,
            dimensions=dims,
            steps_taken=obs.step,
            delivered_pct=del_pct,
            events_handled=m.events_handled,
            improvement_hint=improvement_hint,
        )

    def _summarise(self, results: list[TaskResultSchema]) -> dict[str, Any]:
        score_by_scenario = {r.scenario: r.weighted_total for r in results}
        easy = round(score_by_scenario.get("easy", 0.0), 6)
        medium = round(score_by_scenario.get("medium", 0.0), 6)
        hard = round(score_by_scenario.get("hard", 0.0), 6)
        average = round((easy + medium + hard) / 3.0, 6)

        if easy >= medium and easy >= hard:
            strongest = "easy"
        elif medium >= hard:
            strongest = "medium"
        else:
            strongest = "hard"

        all_scores = [easy, medium, hard]
        return {

            "easy":    easy,
            "medium":  medium,
            "hard":    hard,
            "average": average,

            "mean_score": average,
            "min_score":  round(min(all_scores), 6),
            "max_score":  round(max(all_scores), 6),
            "grades": {r.task_id: r.grade for r in results},
            "scores_by_task": {r.task_id: r.weighted_total for r in results},
            "summary": (
                f"Deterministic baseline run complete. "
                f"Strongest performance on '{strongest}' ({score_by_scenario[strongest]:.4f}). "
                f"Mean across easy/medium/hard: {average:.4f}."
            ),
        }

    @staticmethod
    def _build_comparison_table(results: list[TaskResultSchema]) -> str:
        """
        Render an ASCII table of all 8 dimension scores per task.
        """
        dims = [
            "delivery", "value", "timeliness", "priority",
            "team_health", "adaptability", "efficiency", "dependency",
        ]
        headers = ["Dimension      "] + [r.task_id.replace("task_", "").capitalize() + ("  " if len(r.task_id) < 11 else " ") for r in results]
        sep = "─" * 52

        lines = [
            "",
            "  Baseline Comparison — PriorityAwareAgent (seed=42)",
            "  " + sep,
            "  {:<18} {:>8}  {:>8}  {:>8}".format(
                "Dimension",
                *(r.task_id.replace("task_", "").upper() for r in results),
            ),
            "  " + sep,
        ]

        for dim in dims:
            label = _DIMENSION_LABELS[dim][:17].ljust(17)
            vals  = "  ".join(f"{r.dimensions[dim]:>6.3f}" for r in results)
            lines.append(f"  {label}  {vals}")

        lines += [
            "  " + sep,
            "  {:<18} {:>8}  {:>8}  {:>8}".format(
                "TOTAL (weighted)",
                *(f"{r.weighted_total:.4f}" for r in results),
            ),
            "  {:<18} {:>8}  {:>8}  {:>8}".format(
                "Grade",
                *(f"  [{r.grade}]" for r in results),
            ),
            "  " + sep,
            "",
        ]
        return "\n".join(lines)
