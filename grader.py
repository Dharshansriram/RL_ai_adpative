"""
grader.py — Deterministic episode grader for the Adaptive AI Project Manager.

"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from models import Developer, Observation, SprintMetrics, Task, TaskStatus




@dataclass
class ScoreBreakdown:
    """Per-dimension scores plus the weighted aggregate."""

    delivery_score:     float = 0.0
    value_score:        float = 0.0
    timeliness_score:   float = 0.0
    priority_score:     float = 0.0
    team_health_score:  float = 0.0
    adaptability_score: float = 0.0
    efficiency_score:   float = 0.0
    dependency_score:   float = 0.0
    weighted_total:     float = 0.0
    grade:              str   = "F"
    notes:              list  = field(default_factory=list)


    WEIGHTS: dict = field(default_factory=lambda: {
        "delivery":     0.25,
        "value":        0.20,
        "timeliness":   0.15,
        "priority":     0.10,
        "team_health":  0.10,
        "adaptability": 0.10,
        "efficiency":   0.07,
        "dependency":   0.03,
    })

    def compute_total(self) -> float:
        w = self.WEIGHTS
        self.weighted_total = (
            w["delivery"]     * self.delivery_score
            + w["value"]      * self.value_score
            + w["timeliness"] * self.timeliness_score
            + w["priority"]   * self.priority_score
            + w["team_health"]    * self.team_health_score
            + w["adaptability"]   * self.adaptability_score
            + w["efficiency"]     * self.efficiency_score
            + w["dependency"]     * self.dependency_score
        )
        self.grade = self._letter_grade(self.weighted_total)
        return self.weighted_total

    @staticmethod
    def _letter_grade(s: float) -> str:
        if s >= 0.93: return "A+"
        if s >= 0.85: return "A"
        if s >= 0.77: return "B+"
        if s >= 0.70: return "B"
        if s >= 0.60: return "C"
        if s >= 0.50: return "D"
        return "F"

    def report(self) -> str:
        lines = [
            "=" * 58,
            f"  EPISODE SCORE : {self.weighted_total:.4f}   Grade: [{self.grade}]",
            "=" * 58,
            f"  {'Delivery':<18} {self.delivery_score:.3f}   (w={self.WEIGHTS['delivery']})",
            f"  {'Business Value':<18} {self.value_score:.3f}   (w={self.WEIGHTS['value']})",
            f"  {'Timeliness':<18} {self.timeliness_score:.3f}   (w={self.WEIGHTS['timeliness']})",
            f"  {'Priority Order':<18} {self.priority_score:.3f}   (w={self.WEIGHTS['priority']})",
            f"  {'Team Health':<18} {self.team_health_score:.3f}   (w={self.WEIGHTS['team_health']})",
            f"  {'Adaptability':<18} {self.adaptability_score:.3f}   (w={self.WEIGHTS['adaptability']})",
            f"  {'Efficiency':<18} {self.efficiency_score:.3f}   (w={self.WEIGHTS['efficiency']})",
            f"  {'Dependencies':<18} {self.dependency_score:.3f}   (w={self.WEIGHTS['dependency']})",
            "-" * 58,
        ]
        for note in self.notes:
            lines.append(f"  ⚠  {note}")
        lines.append("=" * 58)
        return "\n".join(lines)




class EpisodeGrader:
    """
    Deterministic episode grader.

    Call grade(final_observation) once the episode has terminated.
    All scorer methods are pure functions with no side effects.
    """

    def __init__(self, dependency_map: dict | None = None) -> None:

        self._deps: dict = dependency_map or {}



    def grade(self, obs: Observation) -> ScoreBreakdown:
        """Compute all dimension scores and return the populated breakdown."""
        bd = ScoreBreakdown()
        tasks   = obs.tasks
        devs    = obs.developers
        metrics = obs.metrics

        bd.delivery_score     = self._delivery(metrics)
        bd.value_score        = self._value(metrics)
        bd.timeliness_score   = self._timeliness(tasks)
        bd.priority_score     = self._priority(tasks)
        bd.team_health_score  = self._team_health(devs, metrics, obs.max_steps)
        bd.adaptability_score = self._adaptability(tasks, metrics)
        bd.efficiency_score   = self._efficiency(devs, metrics, obs.step)
        bd.dependency_score   = self._dependency(tasks)

        self._annotate(bd, metrics, devs, tasks)
        bd.compute_total()
        return bd



    def _delivery(self, m: SprintMetrics) -> float:
        """Fraction of story points delivered (0–1)."""
        if m.total_story_points == 0:
            return 1.0
        return min(1.0, m.delivered_story_points / m.total_story_points)

    def _value(self, m: SprintMetrics) -> float:
        """Fraction of total business value captured."""
        if m.total_business_value == 0:
            return 1.0
        return min(1.0, m.delivered_business_value / m.total_business_value)

    def _timeliness(self, tasks: list) -> float:
        """
        (on_time + 0.4 × late) / gradeable_tasks.
        Failed tasks contribute 0 (no partial credit).
        """
        gradeable = [
            t for t in tasks
            if t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
        ]
        if not gradeable:
            return 1.0

        completed = [t for t in gradeable if t.status == TaskStatus.COMPLETED]
        on_time = sum(
            1 for t in completed
            if t.completed_step is not None and t.completed_step <= t.deadline_step
        )
        late = len(completed) - on_time
        return min(1.0, max(0.0, (on_time + 0.4 * late) / len(gradeable)))

    def _priority(self, tasks: list) -> float:
        """
        Penalise completing lower-priority tasks while higher-priority tasks
        fail. Uses pairwise comparison; violations are amplified ×2.
        """
        completed = sorted(
            [t for t in tasks if t.status == TaskStatus.COMPLETED],
            key=lambda t: t.completed_step or 0,
        )
        failed = [t for t in tasks if t.status == TaskStatus.FAILED]

        if not failed or not completed:
            return 1.0

        violations = 0
        comparisons = 0
        for f_task in failed:
            for c_task in completed:
                if c_task.priority.value < f_task.priority.value:
                    if (c_task.completed_step or 0) > f_task.deadline_step:
                        violations += 1
                comparisons += 1

        if comparisons == 0:
            return 1.0
        return max(0.0, 1.0 - (violations / comparisons) * 2.0)

    def _team_health(self, devs: list, m: SprintMetrics, max_steps: int = 40) -> float:
        """
        Combines end-state average fatigue, overtime frequency, and
        the context-switch (unassign) churn penalty.

        v2.1 fix: overtime_pen was raw_count × 0.005, which always hit the
        0.40 cap in any working sprint (e.g. 4 devs × 30 steps = up to 120
        overtime_steps). This collapsed team_health to 0 regardless of strategy.

        Fix: normalise overtime_steps by (n_devs × max_steps) — the true
        maximum — so the penalty reflects the *fraction* of team-time above
        the 0.80 fatigue threshold, not an unbounded count.
        """
        if not devs:
            return 1.0
        avg_fatigue = sum(d.fatigue for d in devs) / len(devs)


        max_possible  = max(1, len(devs) * max(1, max_steps))
        overtime_frac = min(1.0, m.overtime_steps / max_possible)
        overtime_pen  = overtime_frac * 0.35          # full overtime → −0.35

        unassign_pen  = min(0.20, m.unassign_penalties * 0.02)
        raw = (1.0 - avg_fatigue) - overtime_pen - unassign_pen
        return max(0.0, min(1.0, raw))

    def _adaptability(self, tasks: list, m: SprintMetrics) -> float:
        """Fraction of event-injected tasks that were completed."""
        injected = [t for t in tasks if t.is_injected]
        if not injected:
            return 1.0
        completed_inj = sum(1 for t in injected if t.status == TaskStatus.COMPLETED)
        base = completed_inj / len(injected)
        event_bonus = min(0.20, m.events_handled * 0.04)
        return min(1.0, base + event_bonus)

    def _efficiency(self, devs: list, m: SprintMetrics, total_steps: int) -> float:
        """
        SP delivered / theoretical max SP.
        Uses a negative-exponential transform so moderate utilisation
        still earns a decent score.
        """
        if not devs or total_steps == 0:
            return 1.0
        actual = m.delivered_story_points
        theoretical = sum(d.velocity * total_steps for d in devs)
        if theoretical == 0:
            return 1.0
        utilisation = actual / theoretical

        return min(1.0, max(0.0, 1.0 - math.exp(-3.0 * utilisation)))

    def _dependency(self, tasks: list) -> float:
        """Detects causal-order violations in the completed task set."""
        task_map = {t.id: t for t in tasks}
        violations = 0
        total = 0
        for task in tasks:
            if task.status != TaskStatus.COMPLETED:
                continue
            for dep_id in self._deps.get(task.id, []):
                dep = task_map.get(dep_id)
                if dep is None:
                    continue
                total += 1
                if dep.status != TaskStatus.COMPLETED:
                    violations += 1
                elif (dep.completed_step or 0) > (task.completed_step or 0):
                    violations += 1
        if total == 0:
            return 1.0
        return max(0.0, 1.0 - violations / total)



    def _annotate(
        self, bd: ScoreBreakdown, m: SprintMetrics,
        devs: list, tasks: list,
    ) -> None:
        if m.failed_tasks:
            bd.notes.append(f"{m.failed_tasks} task(s) missed their deadline.")
        if m.unassign_penalties > 3:
            bd.notes.append(
                f"High context-switching churn ({m.unassign_penalties} unassigns)."
            )
        burned = [d for d in devs if d.fatigue > 0.85]
        if burned:
            bd.notes.append(
                "Burnout risk: " + ", ".join(d.name for d in burned)
                + " ended with fatigue > 0.85."
            )
        if bd.delivery_score < 0.5:
            bd.notes.append("Sprint failed: less than 50% of story points delivered.")
        if bd.adaptability_score < 0.4:
            bd.notes.append("Dynamic events were largely unhandled.")
