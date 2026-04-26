"""
rubric_rewards.py — OpenEnv Rubric-Based Multi-Reward System
=============================================================

Round 2 upgrade: replaces monolithic reward with composable rubric checks.

Design principles (from hackathon guide):
  1. Multiple independent reward functions — harder for agent to game any one
  2. Process-aware feedback — intermediate steps are credited, not just outcomes
  3. Anti-reward-hacking checks — detect and penalise shortcuts
  4. Composable: each Rubric is an independent verifier

Rubric dimensions:
  R1  TaskProgressRubric        — per-sp progress toward completion
  R2  DeadlineUrgencyRubric     — urgency-weighted completion bonus
  R3  SkillMatchRubric          — reward appropriate skill matching
  R4  FatigueManagementRubric   — penalise burnout, reward recovery
  R5  DependencyOrderRubric     — reward completing tasks in correct order
  R6  InjectedResponseRubric    — reward fast response to injected events
  R7  CapacityUtilisationRubric — penalise idle developers when work is ready
  R8  AntiHackingRubric         — detect and penalise shortcuts/gaming
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from models import (
    Action, ActionType, Developer,
    Task, TaskPriority, TaskStatus,
)

@dataclass
class RubricResult:
    rubric_name: str
    score: float
    weight: float
    reason: str
    is_violation: bool = False

    @property
    def weighted(self) -> float:
        return self.score * self.weight


class BaseRubric:
    NAME   = "base"
    WEIGHT = 1.0

    def evaluate(
        self,
        prev_tasks: list[Task],
        curr_tasks: list[Task],
        curr_devs: list[Developer],
        action: Action,
        current_step: int,
        max_steps: int,
        **kwargs: Any,
    ) -> RubricResult:
        raise NotImplementedError



class TaskProgressRubric(BaseRubric):
    """Reward actual story-point reduction each step (process signal)."""
    NAME   = "task_progress"
    WEIGHT = 0.25

    PROGRESS_BASE = 0.50

    def evaluate(self, prev_tasks, curr_tasks, curr_devs, action,
                 current_step, max_steps, **_) -> RubricResult:
        prev_map = {t.id: t for t in prev_tasks}
        total_delta = 0.0
        details = []
        for t in curr_tasks:
            prev = prev_map.get(t.id)
            if prev and t.remaining_points < prev.remaining_points:
                delta   = prev.remaining_points - t.remaining_points
                urgency = _urgency_factor(t, current_step, max_steps)
                contrib  = delta * self.PROGRESS_BASE * urgency
                total_delta += contrib
                details.append(f"{t.name}: -{delta:.2f}sp ×{urgency:.1f}urgency")
        reason = "; ".join(details) if details else "No progress this step"
        return RubricResult(self.NAME, total_delta, self.WEIGHT, reason)



class DeadlineUrgencyRubric(BaseRubric):
    """Bonus for completing tasks; amplified when on-time, penalised when failed."""
    NAME   = "deadline_urgency"
    WEIGHT = 0.20

    ONTIME_BONUS  =  1.50
    LATE_BONUS    =  0.40
    FAIL_PENALTY  = -2.00

    def evaluate(self, prev_tasks, curr_tasks, curr_devs, action,
                 current_step, max_steps, **_) -> RubricResult:
        prev_map = {t.id: t for t in prev_tasks}
        total = 0.0
        details = []
        for t in curr_tasks:
            prev = prev_map.get(t.id)
            if not prev:
                continue
            if prev.status != TaskStatus.COMPLETED and t.status == TaskStatus.COMPLETED:
                on_time = t.completed_step is not None and t.completed_step <= t.deadline_step
                bonus   = (self.ONTIME_BONUS if on_time else self.LATE_BONUS) * t.priority.value
                total  += bonus
                details.append(f"{'ON-TIME' if on_time else 'LATE'} {t.name} +{bonus:.2f}")
            elif prev.status != TaskStatus.FAILED and t.status == TaskStatus.FAILED:
                pen = self.FAIL_PENALTY * t.priority.value
                total += pen
                details.append(f"FAILED {t.name} {pen:.2f}")
        reason = "; ".join(details) if details else "No completions/failures this step"
        return RubricResult(self.NAME, total, self.WEIGHT, reason)



class SkillMatchRubric(BaseRubric):
    """Reward assigning developers whose skills match the task well."""
    NAME   = "skill_match"
    WEIGHT = 0.10

    def evaluate(self, prev_tasks, curr_tasks, curr_devs, action,
                 current_step, max_steps, **_) -> RubricResult:
        if action.action_type != ActionType.ASSIGN_TASK or action.assign is None:
            return RubricResult(self.NAME, 0.0, self.WEIGHT, "Not an assign action")

        dev_map  = {d.id: d for d in curr_devs}
        task_map = {t.id: t for t in curr_tasks}
        dev  = dev_map.get(action.assign.developer_id)
        task = task_map.get(action.assign.task_id)

        if dev is None or task is None:
            return RubricResult(self.NAME, 0.0, self.WEIGHT, "Unknown dev/task")

        prof = dev.proficiency_for_task(task)
        # Graduated bonus: great match = +0.3, poor match = -0.1
        if prof >= 0.70:
            score  = 0.30
            reason = f"Excellent skill match: {dev.name}→{task.name} (prof={prof:.2f})"
        elif prof >= 0.40:
            score  = 0.10
            reason = f"Adequate skill match: {dev.name}→{task.name} (prof={prof:.2f})"
        elif prof >= 0.15:
            score  = -0.05
            reason = f"Weak skill match: {dev.name}→{task.name} (prof={prof:.2f})"
        else:
            score  = -0.15
            reason = f"Skill mismatch: {dev.name}→{task.name} (prof={prof:.2f})"

        return RubricResult(self.NAME, score, self.WEIGHT, reason)


# ---------------------------------------------------------------------------
# R4 — Fatigue Management
# ---------------------------------------------------------------------------

class FatigueManagementRubric(BaseRubric):
    """Penalise high fatigue states; reward proactive rest."""
    NAME   = "fatigue_management"
    WEIGHT = 0.10

    BURNOUT_PENALTY  = -0.08
    OVERTIME_PENALTY = -0.04
    REST_REWARD      =  0.15

    def evaluate(self, prev_tasks, curr_tasks, curr_devs, action,
                 current_step, max_steps, **_) -> RubricResult:
        total  = 0.0
        burned = []
        over   = []
        for d in curr_devs:
            if d.fatigue > 0.85:
                total += self.BURNOUT_PENALTY
                burned.append(d.name)
            elif d.fatigue > 0.80:
                total += self.OVERTIME_PENALTY
                over.append(d.name)

        rest_bonus = ""
        if action.action_type == ActionType.REST_DEVELOPER and action.rest:
            # Only reward rest if the developer actually needed it
            dev = next((d for d in curr_devs if d.id == action.rest.developer_id), None)
            if dev and dev.fatigue < 0.70:  # proactive rest, before burnout
                total += self.REST_REWARD
                rest_bonus = f"; proactive rest bonus for {dev.name}"

        parts = []
        if burned: parts.append(f"burnout: {', '.join(burned)}")
        if over:   parts.append(f"overtime: {', '.join(over)}")
        reason = ("; ".join(parts) + rest_bonus) if parts or rest_bonus else "Healthy team"
        return RubricResult(self.NAME, total, self.WEIGHT, reason)


# ---------------------------------------------------------------------------
# R5 — Dependency Order
# ---------------------------------------------------------------------------

class DependencyOrderRubric(BaseRubric):
    """Reward completing tasks in correct dependency order."""
    NAME   = "dependency_order"
    WEIGHT = 0.08

    def evaluate(self, prev_tasks, curr_tasks, curr_devs, action,
                 current_step, max_steps, **_) -> RubricResult:
        prev_map = {t.id: t for t in prev_tasks}
        task_map = {t.id: t for t in curr_tasks}
        violations = 0
        correct = 0

        for t in curr_tasks:
            prev = prev_map.get(t.id)
            if not prev or prev.status == t.status:
                continue
            if t.status == TaskStatus.COMPLETED:
                for dep_id in t.dependencies:
                    dep = task_map.get(dep_id)
                    if dep and dep.status == TaskStatus.COMPLETED:
                        correct += 1
                    else:
                        violations += 1

        score = 0.0
        if correct > 0:
            score += correct * 0.10
        if violations > 0:
            score -= violations * 0.30

        reason = f"{correct} ordered completions, {violations} order violations"
        return RubricResult(self.NAME, score, self.WEIGHT, reason,
                            is_violation=(violations > 0))


# ---------------------------------------------------------------------------
# R6 — Injected Event Response
# ---------------------------------------------------------------------------

class InjectedResponseRubric(BaseRubric):
    """Reward quickly responding to injected tasks (bugs, urgent features)."""
    NAME   = "injected_response"
    WEIGHT = 0.10

    FAST_RESPONSE_BONUS = 0.40  # assigned within 2 steps of injection

    def evaluate(self, prev_tasks, curr_tasks, curr_devs, action,
                 current_step, max_steps, **_) -> RubricResult:
        prev_map = {t.id: t for t in prev_tasks}
        total = 0.0
        details = []
        for t in curr_tasks:
            if not t.is_injected:
                continue
            prev = prev_map.get(t.id)
            # Newly injected task just got assigned
            if prev is None and t.status == TaskStatus.IN_PROGRESS:
                total += self.FAST_RESPONSE_BONUS
                details.append(f"Immediate response to injected {t.name}")
            elif (prev and prev.status == TaskStatus.READY
                  and t.status == TaskStatus.IN_PROGRESS):
                # +bonus if assigned quickly (within 2 steps of creation)
                age = current_step - (t.created_step or current_step)
                if age <= 2:
                    total += self.FAST_RESPONSE_BONUS
                    details.append(f"Fast response ({age}s) to {t.name}")
                elif age <= 5:
                    total += self.FAST_RESPONSE_BONUS * 0.5
                    details.append(f"OK response ({age}s) to {t.name}")
            # Penalty for ignoring a critical injected task
            if (t.is_injected and t.priority == TaskPriority.CRITICAL
                    and t.status == TaskStatus.READY and not t.assigned_to):
                age = current_step - (t.created_step or current_step)
                if age > 3:
                    total -= 0.25
                    details.append(f"Ignoring critical injected {t.name} (age={age})")

        reason = "; ".join(details) if details else "No injected events this step"
        return RubricResult(self.NAME, total, self.WEIGHT, reason)


# ---------------------------------------------------------------------------
# R7 — Capacity Utilisation
# ---------------------------------------------------------------------------

class CapacityUtilisationRubric(BaseRubric):
    """Penalise idle capacity when work is ready."""
    NAME   = "capacity_utilisation"
    WEIGHT = 0.08

    IDLE_WASTE_PENALTY = -0.08

    def evaluate(self, prev_tasks, curr_tasks, curr_devs, action,
                 current_step, max_steps, **_) -> RubricResult:
        ready_tasks = [t for t in curr_tasks
                       if t.status == TaskStatus.READY and not t.assigned_to]
        idle_devs   = [d for d in curr_devs
                       if d.available and not d.current_tasks and d.fatigue < 0.75]

        wasted = min(len(ready_tasks), len(idle_devs))
        score  = wasted * self.IDLE_WASTE_PENALTY

        reason = (f"{wasted} idle dev-task pairs wasted"
                  if wasted > 0 else "Good capacity utilisation")
        return RubricResult(self.NAME, score, self.WEIGHT, reason)


# ---------------------------------------------------------------------------
# R8 — Anti-Hacking (critical)
# ---------------------------------------------------------------------------

class AntiHackingRubric(BaseRubric):
    """
    Detect common reward-hacking patterns:
      - Repeated unassign/reassign cycles (churn gaming)
      - Resting a developer who isn't fatigued (rest-spamming)
      - Splitting tasks repeatedly below minimum size
      - NOOP when critical tasks are ready
    """
    NAME   = "anti_hacking"
    WEIGHT = 0.09

    def __init__(self) -> None:
        self._action_history: list[str] = []
        self._unassign_counts: dict[str, int] = {}
        self._split_counts:    dict[str, int] = {}

    def evaluate(self, prev_tasks, curr_tasks, curr_devs, action,
                 current_step, max_steps, **_) -> RubricResult:
        violations = []
        penalty    = 0.0
        at = action.action_type

        # Track action history
        self._action_history.append(at.value)
        if len(self._action_history) > 10:
            self._action_history.pop(0)

        # 1. Churn detection: unassign same task > 2 times
        if at == ActionType.UNASSIGN_TASK and action.unassign:
            tid = action.unassign.task_id
            self._unassign_counts[tid] = self._unassign_counts.get(tid, 0) + 1
            if self._unassign_counts[tid] > 2:
                penalty -= 0.50
                violations.append(f"Churn abuse: unassigning {tid} x{self._unassign_counts[tid]}")

        # 2. Rest spam: resting a non-fatigued developer
        if at == ActionType.REST_DEVELOPER and action.rest:
            dev = next((d for d in curr_devs if d.id == action.rest.developer_id), None)
            if dev and dev.fatigue < 0.30:
                penalty -= 0.20
                violations.append(f"Rest spam: {dev.name} fatigue={dev.fatigue:.2f}")

        # 3. Split spam: splitting same task repeatedly
        if at == ActionType.SPLIT_TASK and action.split:
            tid = action.split.task_id
            self._split_counts[tid] = self._split_counts.get(tid, 0) + 1
            if self._split_counts[tid] > 1:
                penalty -= 0.30
                violations.append(f"Split spam: {tid} split x{self._split_counts[tid]}")

        # 4. NOOP abuse: 4+ consecutive noops while critical tasks ready
        recent_noops = sum(1 for a in self._action_history[-4:] if a == "noop")
        critical_ready = [t for t in curr_tasks
                          if t.status == TaskStatus.READY
                          and t.priority == TaskPriority.CRITICAL
                          and not t.assigned_to]
        if recent_noops >= 4 and critical_ready:
            penalty -= 0.60
            violations.append(f"NOOP abuse: {recent_noops} consecutive noops, "
                               f"{len(critical_ready)} critical tasks idle")

        # 5. Late-episode waste: unassigning in final 5 steps
        if at == ActionType.UNASSIGN_TASK and (max_steps - current_step) <= 5:
            penalty -= 0.25
            violations.append("Unassign in final 5 steps — destructive")

        reason = ("; ".join(violations)
                  if violations else "No reward-hacking patterns detected")
        is_viol = len(violations) > 0
        return RubricResult(self.NAME, penalty, self.WEIGHT, reason,
                            is_violation=is_viol)


# ---------------------------------------------------------------------------
# Rubric Composer
# ---------------------------------------------------------------------------

@dataclass
class RubricBreakdown:
    results:       list[RubricResult] = field(default_factory=list)
    total_reward:  float = 0.0
    violations:    list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_reward":  round(self.total_reward, 4),
            "violations":    self.violations,
            "rubrics": {
                r.rubric_name: {
                    "score":      round(r.score, 4),
                    "weighted":   round(r.weighted, 4),
                    "weight":     r.weight,
                    "reason":     r.reason,
                    "violation":  r.is_violation,
                }
                for r in self.results
            },
        }


class RubricComposer:
    """
    Composes multiple independent rubric checks into a single reward.

    Usage:
        composer = RubricComposer()
        breakdown = composer.evaluate(prev_tasks, curr_tasks, ...)
        reward = breakdown.total_reward
    """

    def __init__(self) -> None:
        self._rubrics: list[BaseRubric] = [
            TaskProgressRubric(),
            DeadlineUrgencyRubric(),
            SkillMatchRubric(),
            FatigueManagementRubric(),
            DependencyOrderRubric(),
            InjectedResponseRubric(),
            CapacityUtilisationRubric(),
            AntiHackingRubric(),
        ]

    def evaluate(
        self,
        prev_tasks: list,
        curr_tasks: list,
        curr_devs: list,
        action: Action,
        current_step: int,
        max_steps: int,
        **kwargs,
    ) -> RubricBreakdown:
        bd = RubricBreakdown()
        for rubric in self._rubrics:
            result = rubric.evaluate(
                prev_tasks=prev_tasks,
                curr_tasks=curr_tasks,
                curr_devs=curr_devs,
                action=action,
                current_step=current_step,
                max_steps=max_steps,
                **kwargs,
            )
            bd.results.append(result)
            bd.total_reward += result.weighted
            if result.is_violation:
                bd.violations.append(f"[{result.rubric_name}] {result.reason}")

        return bd


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _urgency_factor(task: "Task", step: int, max_steps: int) -> float:
    """Urgency multiplier: 1.0–2.5× as deadline approaches."""
    frac = max(1, task.deadline_step - step) / max(1, max_steps)
    if frac < 0.10: return 2.5
    if frac < 0.25: return 1.8
    if frac < 0.50: return 1.3
    return 1.0
