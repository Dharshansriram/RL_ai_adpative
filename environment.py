"""
environment.py — Adaptive AI Project Manager (OpenEnv).

OpenEnv interface:
  env.reset()       → Observation
  env.step(Action)  → Observation  [includes info.decision_reason every step]
  env.state         → Observation  (read-only snapshot, no side effects)
  env.grade()       → ScoreBreakdown

Internal sub-systems:
  _TaskGraph       — DAG dependency resolution (PENDING→BLOCKED→READY→IN_PROGRESS→DONE)
  _EventEngine     — Stochastic event scheduling + application
  _RewardEngine    — Dense shaped per-step reward (legacy, kept for baseline)
  RubricComposer   — Round 2: 8-rubric composable multi-reward system (anti-hacking)
  Scenario         — Configurable sprint factories (easy/medium/hard/small/large/chaos)

Round 2 changes:
  - RubricComposer replaces monolithic _RewardEngine as default reward
  - rubric_breakdown included in obs.info every step for interpretability
  - anti_hacking rubric detects churn, rest-spam, split-spam, NOOP abuse
  - use_rubrics=True (default) — set False for legacy behaviour
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any

from grader import EpisodeGrader, ScoreBreakdown
from rubric_rewards import RubricComposer, RubricBreakdown
from models import (
    Action, ActionType,
    AssignTaskPayload, UnassignTaskPayload,
    ReprioritizePayload, RestDeveloperPayload,
    SplitTaskPayload, PairProgramPayload,
    Developer, DynamicEvent, EventType,
    Observation, SkillTag,
    SprintMetrics, Task, TaskPriority, TaskStatus,
)




class _TaskGraph:
    """
    Manages the task dependency graph.
    """

    def __init__(self, tasks: list) -> None:
        self._tasks: dict = {t.id: t for t in tasks}

    def register(self, task: Task) -> None:
        """Add a dynamically injected task to the graph."""
        self._tasks[task.id] = task

    def resolve_statuses(self, current_step: int) -> None:
        """Mutate task statuses in-place according to current dependency and deadline state."""
        for task in self._tasks.values():

            if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.IN_PROGRESS):
                continue

            if current_step > task.deadline_step:
                task.status = TaskStatus.FAILED
                continue

            if self._deps_satisfied(task):
                task.status = TaskStatus.READY
            else:
                task.status = TaskStatus.BLOCKED

    def _deps_satisfied(self, task: Task) -> bool:
        for dep_id in task.dependencies:
            dep = self._tasks.get(dep_id)
            if dep is None or dep.status != TaskStatus.COMPLETED:
                return False
        return True

    def dependency_map(self) -> dict:
        return {tid: list(t.dependencies) for tid, t in self._tasks.items()}




class _EventEngine:
    """
    Controls the dynamic event lifecycle:
      1. Pre-schedule guaranteed events at episode start (bug, scope change, sick day)
      2. Probabilistically fire ad-hoc events each step (10% chance)
      3. Apply event effects to tasks/developers
    """

    AD_HOC_EVENT_PROB = 0.10

    def __init__(self, rng: random.Random) -> None:
        self._rng      = rng
        self._pending: list = []
        self._applied: list = []

    def schedule_initial_events(self, tasks: list, developers: list, max_steps: int) -> None:
        """Guarantee at least one bug report, scope change, and sick day per episode."""

        schedule = [
            (EventType.BUG_REPORT,      0.30),
            (EventType.SCOPE_CHANGE,    0.50),
            (EventType.DEVELOPER_SICK,  0.65),
           ]
        for etype, frac in schedule:
            step = int(max_steps * frac) + self._rng.randint(-2, 2)
            step = max(1, min(step, max_steps - 3))
            self._pending.append(DynamicEvent(event_type=etype, trigger_step=step))

    def tick(self, current_step: int, tasks: list, developers: list, max_steps: int) -> list:
        """Fire all due events; possibly inject one ad-hoc event. Returns fired list."""
        fired = []


        for ev in list(self._pending):
            if ev.trigger_step <= current_step and not ev.applied:
                if self._apply(ev, tasks, developers, current_step):
                    ev.applied = True
                    self._applied.append(ev)
                    self._pending.remove(ev)
                    fired.append(ev)


        remaining = max_steps - current_step
        if remaining > 3 and self._rng.random() < self.AD_HOC_EVENT_PROB:
            ev = self._random_event(current_step, developers)
            if ev and self._apply(ev, tasks, developers, current_step):
                ev.applied = True
                self._applied.append(ev)
                fired.append(ev)

        return fired



    def _apply(self, ev: DynamicEvent, tasks: list, devs: list, step: int) -> bool:
        handlers = {
            EventType.BUG_REPORT:           self._bug_report,
            EventType.SCOPE_CHANGE:          self._scope_change,
            EventType.DEVELOPER_SICK:        self._developer_sick,
            EventType.URGENT_FEATURE:        self._urgent_feature,
            EventType.INFRASTRUCTURE_OUTAGE: self._infra_outage,
            EventType.KNOWLEDGE_TRANSFER:    self._knowledge_transfer,
        }
        handler = handlers.get(ev.event_type)
        return handler(ev, tasks, devs, step) if handler else False

    def _bug_report(self, ev, tasks, devs, step) -> bool:
        """Inject a CRITICAL hotfix task with a tight deadline (3–6 steps)."""
        deadline = step + self._rng.randint(3, 6)
        pts = round(self._rng.uniform(1.0, 3.0), 1)
        hotfix = Task(
            name=f"Hotfix-{ev.id}",
            description="Critical production bug; tight SLA.",
            priority=TaskPriority.CRITICAL,
            required_skills=[SkillTag.BACKEND, SkillTag.QA],
            story_points=pts,
            deadline_step=deadline,
            status=TaskStatus.READY,
            business_value=8.0,
            is_injected=True,
            created_step=step,
        )
        tasks.append(hotfix)
        ev.description = f"Bug report → hotfix {hotfix.id} ({pts}sp, deadline step {deadline})"
        return True

    def _scope_change(self, ev, tasks, devs, step) -> bool:
        """Increase story points of an active task (scope creep)."""
        candidates = [
            t for t in tasks
            if t.status in (TaskStatus.READY, TaskStatus.IN_PROGRESS) and not t.is_injected
        ]
        if not candidates:
            return False
        target = self._rng.choice(candidates)
        inc = round(self._rng.uniform(0.5, 2.0), 1)
        target.story_points    += inc
        target.remaining_points += inc
        ev.description = f"Scope change: '{target.name}' +{inc}sp"
        return True

    def _developer_sick(self, ev, tasks, devs, step) -> bool:
        """Remove a developer for 2–4 steps; orphan their assigned tasks."""
        healthy = [d for d in devs if d.available]
        if not healthy:
            return False
        victim   = self._rng.choice(healthy)
        duration = self._rng.randint(2, 4)
        victim.available       = False
        victim.sick_until_step = step + duration
        for tid in list(victim.current_tasks):
            victim.current_tasks.remove(tid)
            task = next((t for t in tasks if t.id == tid), None)
            if task:
                task.assigned_to = [d for d in task.assigned_to if d != victim.id]
                if not task.assigned_to:
                    task.status = TaskStatus.READY
        ev.description = (
            f"{victim.name} sick for {duration} steps "
            f"(returns step {victim.sick_until_step})"
        )
        return True

    def _urgent_feature(self, ev, tasks, devs, step) -> bool:
        """Inject a HIGH-priority feature with a medium deadline (6–10 steps)."""
        deadline = step + self._rng.randint(6, 10)
        pts = round(self._rng.uniform(2.0, 4.5), 1)
        feat = Task(
            name=f"UrgentFeature-{ev.id}",
            description="Stakeholder-requested feature — high visibility.",
            priority=TaskPriority.HIGH,
            required_skills=[SkillTag.FRONTEND, SkillTag.BACKEND],
            story_points=pts,
            deadline_step=deadline,
            status=TaskStatus.READY,
            business_value=7.0,
            is_injected=True,
            created_step=step,
        )
        tasks.append(feat)
        ev.description = f"Urgent feature {feat.id} injected ({pts}sp, deadline step {deadline})"
        return True

    def _infra_outage(self, ev, tasks, devs, step) -> bool:
        """Block DevOps tasks for 2 steps."""
        affected = [
            t for t in tasks
            if SkillTag.DEVOPS in t.required_skills
            and t.status in (TaskStatus.READY, TaskStatus.IN_PROGRESS)
        ]
        ev.payload["blocked_until"]     = step + 2
        ev.payload["affected_task_ids"] = [t.id for t in affected]
        ev.description = f"Infrastructure outage: {len(affected)} devops task(s) blocked 2 steps"
        return True

    def _knowledge_transfer(self, ev, tasks, devs, step) -> bool:
        """Boost a developer's weakest skill temporarily."""
        if not devs:
            return False
        dev = self._rng.choice(devs)
        if not dev.skills:
            return False
        weakest = min(dev.skills, key=lambda s: dev.skills[s])
        boost   = round(self._rng.uniform(0.15, 0.30), 2)
        expires = step + self._rng.randint(3, 7)
        dev.skill_boosts[weakest.value] = boost
        dev.boost_expires_step = expires
        ev.description = (
            f"Knowledge transfer: {dev.name} +{boost} {weakest.value} until step {expires}"
        )
        return True

    def _random_event(self, step: int, devs: list) -> DynamicEvent | None:
        weights = {
            EventType.BUG_REPORT:         3,
            EventType.SCOPE_CHANGE:        2,
            EventType.URGENT_FEATURE:      2,
            EventType.KNOWLEDGE_TRANSFER:  1,
        }
        if any(d.available for d in devs):
            weights[EventType.DEVELOPER_SICK] = 1
        etype = self._rng.choices(list(weights.keys()), weights=list(weights.values()), k=1)[0]
        return DynamicEvent(event_type=etype, trigger_step=step)

    @property
    def pending(self) -> list:
        return [e for e in self._pending if not e.applied]

    @property
    def all_applied(self) -> list:
        return list(self._applied)




class _RewardEngine:
    """
    Dense per-step reward shaping.

    Philosophy:
    - Progress rewards give a signal every step (no sparse reward problem)
    - Urgency multiplier naturally focuses the agent on near-deadline work
    - Completion bonuses reward finishing tasks on time vs. late
    - Penalties discourage burnout, wasted capacity, and context-switching
    - Terminal bonus ties the per-step signal to final sprint outcome
    """


    PROGRESS_BASE      =  0.50
    COMPLETION_ONTIME  =  1.50
    COMPLETION_LATE    =  0.40
    FAILED_PENALTY     = -2.00
    UNASSIGN_PENALTY   = -0.80
    SPLIT_OVERHEAD     = -0.20
    FATIGUE_PENALTY    = -0.05
    OVERTIME_PENALTY   = -0.03
    BLOCKED_PENALTY    = -0.10
    NOOP_PENALTY       = -0.05
    TERMINAL_SCALE     =  5.00

    def step_reward(
        self, prev_tasks: list, curr_tasks: list,
        curr_devs: list, action: Action,
        current_step: int, max_steps: int,
    ) -> float:
        reward   = 0.0
        prev_map = {t.id: t for t in prev_tasks}

        for task in curr_tasks:
            prev = prev_map.get(task.id)


            if prev and task.remaining_points < prev.remaining_points:
                delta   = prev.remaining_points - task.remaining_points
                urgency = self._urgency(task, current_step, max_steps)
                reward += delta * self.PROGRESS_BASE * urgency


            if prev and prev.status != TaskStatus.COMPLETED and task.status == TaskStatus.COMPLETED:
                on_time = (task.completed_step is not None
                           and task.completed_step <= task.deadline_step)
                if on_time:
                    reward += self.COMPLETION_ONTIME * task.priority.value
                else:
                    reward += self.COMPLETION_LATE


            if prev and prev.status != TaskStatus.FAILED and task.status == TaskStatus.FAILED:
                reward += self.FAILED_PENALTY * task.priority.value


            if task.status == TaskStatus.READY and not task.assigned_to:
                reward += self.BLOCKED_PENALTY


        if action.action_type == ActionType.UNASSIGN_TASK:
            reward += self.UNASSIGN_PENALTY
        elif action.action_type == ActionType.SPLIT_TASK:
            reward += self.SPLIT_OVERHEAD
        elif action.action_type == ActionType.NOOP:
            ready = sum(1 for t in curr_tasks if t.status == TaskStatus.READY)
            free  = sum(1 for d in curr_devs
                        if d.available and not d.current_tasks and d.fatigue < 0.75)
            if ready > 0 and free > 0:
                reward += self.NOOP_PENALTY * min(ready, free)


        for dev in curr_devs:
            if dev.fatigue > 0.85:
                reward += self.FATIGUE_PENALTY
            elif dev.fatigue > 0.80:
                reward += self.OVERTIME_PENALTY

        return reward

    def terminal_reward(self, metrics: SprintMetrics, devs: list) -> float:
        """End-of-episode bonus proportional to combined delivery + value capture."""
        if metrics.total_story_points == 0:
            return 0.0
        delivery = metrics.delivered_story_points / metrics.total_story_points
        value = (metrics.delivered_business_value / metrics.total_business_value
                 if metrics.total_business_value > 0 else 0.0)
        return (0.6 * delivery + 0.4 * value) * self.TERMINAL_SCALE

    @staticmethod
    def _urgency(task: Task, step: int, max_steps: int) -> float:
        """Urgency multiplier: 1.0–2.5×, higher as deadline approaches."""
        frac = max(1, task.deadline_step - step) / max(1, max_steps)
        if frac < 0.10: return 2.5
        if frac < 0.25: return 1.8
        if frac < 0.50: return 1.3
        return 1.0




class ProjectManagerEnv:
    """
    OpenEnv-compliant Adaptive AI Project Manager.
    -----
    >>> env = ProjectManagerEnv(scenario="medium", seed=42)
    >>> obs = env.reset()
    >>> while not obs.done:
    ...     action = agent.act(obs)
    ...     obs = env.step(action)
    >>> score = env.grade()
    >>> print(score.report())

    Scenarios
    ---------
    easy   : 6 tasks,  3 devs, 20 steps  — no dependencies
    medium : 12 tasks, 4 devs, 30 steps  — deps + events
    hard   : 18 tasks, 5 devs, 40 steps  — full chaos
    """

    def __init__(
        self,
        scenario: str = "medium",
        seed: int = 42,
        max_steps: int | None = None,
        custom_tasks: list | None = None,
        custom_developers: list | None = None,
        use_rubrics: bool = True,
    ) -> None:
        self._scenario           = scenario
        self._seed               = seed
        self._max_steps_override = max_steps
        self._custom_tasks       = custom_tasks
        self._custom_devs        = custom_developers
        self._use_rubrics        = use_rubrics

        self._tasks:       list = []
        self._devs:        list = []
        self._max_steps:   int  = 0
        self._step:        int  = 0
        self._done:        bool = False
        self._last_reward: float = 0.0
        self._metrics: SprintMetrics = SprintMetrics()
        self._last_rubric_breakdown = None

        self._rng      = random.Random(seed)
        self._events   = _EventEngine(self._rng)
        self._rewards  = _RewardEngine()
        self._rubrics  = RubricComposer()
        self._graph:  _TaskGraph   | None = None
        self._grader: EpisodeGrader | None = None
        self._pair_map: dict = {}


    def seed(self, seed: int = 42) -> int:
        """Set RNG seed for deterministic episode generation."""
        self._seed = seed
        self._rng  = random.Random(seed)
        return seed

    def reset(self) -> Observation:
        """Initialise a fresh episode and return the first observation."""
        random.seed(self._seed)
        self._rng         = random.Random(self._seed)
        self._step        = 0
        self._done        = False
        self._last_reward = 0.0
        self._metrics     = SprintMetrics()
        self._pair_map    = {}

        cfg             = Scenario.get(self._scenario)
        self._max_steps = self._max_steps_override or cfg["max_steps"]

        self._tasks = (
            [copy.deepcopy(t) for t in self._custom_tasks]
            if self._custom_tasks
            else cfg["task_factory"](self._rng, self._max_steps)
        )
        self._devs = (
            [copy.deepcopy(d) for d in self._custom_devs]
            if self._custom_devs
            else cfg["dev_factory"](self._rng)
        )

        self._graph  = _TaskGraph(self._tasks)
        self._events = _EventEngine(self._rng)
        self._events.schedule_initial_events(self._tasks, self._devs, self._max_steps)

        # Seed sprint-level metrics
        self._metrics.total_tasks          = len(self._tasks)
        self._metrics.total_story_points   = sum(t.story_points for t in self._tasks)
        self._metrics.total_business_value = sum(t.business_value for t in self._tasks)

        self._graph.resolve_statuses(self._step)
        self._grader = EpisodeGrader(self._graph.dependency_map())

        return self._observe(reward=0.0, recent=[])

    def step(self, action: Action) -> Observation:
        """
        Advance one time step, returning updated observation with full explainability.

        Returns Observation with:
          obs.reward           — shaped per-step reward float
          obs.done             — True when episode terminates
          obs.info["decision_reason"]  — plain-English explanation of this step
          obs.info["decision_details"] — structured dict for programmatic use
          obs.info["action_valid"]     — whether the submitted action was legal
          obs.info["action_echo"]      — echo of the parsed action
          obs.info["step_summary"]     — one-line step/reward/done summary
        """
        if self._done:
            raise RuntimeError("Episode has terminated. Call reset() first.")

        prev_tasks = copy.deepcopy(self._tasks)


        valid, info = self._apply_action(action)


        self._simulate_work()


        fired = self._events.tick(self._step, self._tasks, self._devs, self._max_steps)
        self._metrics.events_handled += len(fired)


        for t in self._tasks:
            self._graph.register(t)
        self._graph.resolve_statuses(self._step)


        existing_ids = {t.id for t in prev_tasks}
        for t in self._tasks:
            if t.is_injected and t.id not in existing_ids:
                self._metrics.total_tasks          += 1
                self._metrics.total_story_points   += t.story_points
                self._metrics.total_business_value += t.business_value


        self._update_developers()


        if self._use_rubrics:
            rubric_bd = self._rubrics.evaluate(
                prev_tasks=prev_tasks,
                curr_tasks=self._tasks,
                curr_devs=self._devs,
                action=action,
                current_step=self._step,
                max_steps=self._max_steps,
            )
            reward = rubric_bd.total_reward
            self._last_rubric_breakdown = rubric_bd
        else:
            reward = self._rewards.step_reward(
                prev_tasks=prev_tasks, curr_tasks=self._tasks,
                curr_devs=self._devs, action=action,
                current_step=self._step, max_steps=self._max_steps,
            )
            self._last_rubric_breakdown = None


        self._refresh_metrics()


        self._step += 1


        all_terminal = all(
            t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
            for t in self._tasks
        )
        if self._step >= self._max_steps or all_terminal:
            reward    += self._rewards.terminal_reward(self._metrics, self._devs)
            self._done = True

        self._last_reward = reward


        reasoning, decision_details = self._build_reasoning(action, valid, info, fired, reward)

        return self._observe(
            reward=reward,
            recent=fired,
            info={
                "action_valid":    valid,
                "decision_reason": decision_details,
                "reasoning":       reasoning,
                "reward_breakdown": (
                    "Useful progress this step."   if reward > 0.0 else
                    "Penalty: wasted effort or increased risk." if reward < 0.0 else
                    "Neutral step with no major change."
                ),
                "action_echo": {
                    "type":   action.action_type.value,
                    "result": info.get("msg") or info.get("error", ""),
                },
                "step_summary": (
                    f"Step {self._step}/{self._max_steps} | "
                    f"reward={reward:+.3f} | done={self._done}"
                ),
                "rubric_breakdown": (
                    self._last_rubric_breakdown.to_dict()
                    if self._last_rubric_breakdown is not None else {}
                ),
                **info,
            },
        )



    def _build_reasoning(
        self, action: Action, valid: bool,
        info: dict, fired: list, reward: float,
    ) -> tuple:
        """
        Returns (reasoning_str, decision_details_dict).

        reasoning_str      — natural-language paragraph, human-readable
        decision_details   — structured dict for API consumers and visualisers

        Keys in decision_details
        ─────────────────────────
        action_outcome   : str       what the action did / why it was rejected
        events_fired     : list[str] descriptions of events that triggered this step
        reward_signal    : str       band label: strong_positive/positive/neutral/mild_penalty/heavy_penalty
        reward_components: dict      labelled contributing factors to the reward
        sprint_progress  : dict      {done, total, pct, failed}
        urgent_warning   : str|None  set when <= 5 steps remain
        """

        if valid:
            action_outcome = info.get("msg", action.action_type.value)
            outcome_prefix = f"Action succeeded: {action_outcome}."
        else:
            err = info.get("error", "unknown error")
            action_outcome = f"REJECTED — {err}"
            outcome_prefix = f"Action REJECTED ({action.action_type.value}): {err}."


        events_fired = [ev.description for ev in fired]
        event_parts  = [f"Event fired → {d}." for d in events_fired]


        if reward > 1.0:
            reward_signal = "strong_positive"
            reward_hint   = "Strong positive reward: task completed on time or high-value delivery."
        elif reward > 0.0:
            reward_signal = "positive"
            reward_hint   = "Positive reward: forward progress on assigned work."
        elif reward < -1.0:
            reward_signal = "heavy_penalty"
            reward_hint   = "Heavy penalty: task deadline missed or severe team-health degradation."
        elif reward < 0.0:
            reward_signal = "mild_penalty"
            reward_hint   = "Mild penalty: idle capacity wasted, context-switch, or fatigue overage."
        else:
            reward_signal = "neutral"
            reward_hint   = "Neutral step: no significant progress or regression."


        completed_now = sum(1 for t in self._tasks if t.status.value == "completed")
        failed_now    = sum(1 for t in self._tasks if t.status.value == "failed")
        ready_idle    = sum(1 for t in self._tasks if t.status.value == "ready" and not t.assigned_to)
        overtime_devs = sum(1 for d in self._devs if d.fatigue > 0.80)
        reward_components = {
            "total":               round(reward, 4),
            "idle_ready_tasks":    ready_idle,
            "overtime_developers": overtime_devs,
            "action_type":         action.action_type.value,
        }


        total  = len(self._tasks)
        pct    = completed_now / total * 100 if total else 0.0
        sprint_progress = {
            "done":   completed_now,
            "total":  total,
            "pct":    round(pct, 1),
            "failed": failed_now,
        }
        progress_text = f"Sprint progress: {completed_now}/{total} tasks done ({pct:.0f}%)."


        steps_left    = self._max_steps - self._step
        urgent_warning: str | None = None
        urgent_parts:  list[str]   = []
        if steps_left <= 5 and not self._done:
            urgent_warning = f"Only {steps_left} steps remaining — prioritise CRITICAL tasks."
            urgent_parts   = [f"WARNING: {urgent_warning}"]


        reasoning_str = " ".join(
            [outcome_prefix] + event_parts + [reward_hint, progress_text] + urgent_parts
        )

        decision_details = {
            "action_outcome":    action_outcome,
            "events_fired":      events_fired,
            "reward_signal":     reward_signal,
            "reward_components": reward_components,
            "sprint_progress":   sprint_progress,
            "urgent_warning":    urgent_warning,
        }

        return reasoning_str, decision_details



    @property
    def state(self) -> Observation:
        """Read-only snapshot of current state (no side effects)."""
        return self._observe(reward=self._last_reward, recent=[])

    def grade(self) -> ScoreBreakdown:
        """Deterministically grade the completed episode."""
        if not self._done:
            raise RuntimeError("Episode is not done yet. Run until obs.done=True first.")
        return self._grader.grade(self.state)


    def _apply_action(self, action: Action) -> tuple:
        table = {
            ActionType.ASSIGN_TASK:    self._act_assign,
            ActionType.UNASSIGN_TASK:  self._act_unassign,
            ActionType.REPRIORITIZE:   self._act_reprioritize,
            ActionType.REST_DEVELOPER: self._act_rest,
            ActionType.SPLIT_TASK:     self._act_split,
            ActionType.PAIR_PROGRAM:   self._act_pair,
            ActionType.NOOP:           lambda a: (True, {}),
        }
        handler = table.get(action.action_type)
        if handler is None:
            return False, {"error": "Unknown action type"}
        return handler(action)

    def _act_assign(self, action: Action) -> tuple:
        p    = action.assign
        task = self._task(p.task_id)
        dev  = self._dev(p.developer_id)
        if task is None: return False, {"error": f"Task {p.task_id} not found"}
        if dev  is None: return False, {"error": f"Dev {p.developer_id} not found"}
        if not dev.available:
            return False, {"error": f"{dev.name} unavailable"}
        if task.status not in (TaskStatus.READY, TaskStatus.IN_PROGRESS):
            return False, {"error": f"{task.name} not assignable ({task.status})"}
        if p.developer_id in task.assigned_to:
            return False, {"error": f"{dev.name} already on {task.name}"}
        prof = dev.proficiency_for_task(task)
        if prof < 0.15:
            return False, {
                "error": f"{dev.name} lacks required skills for {task.name} (proficiency={prof:.2f}). "
                          f"Required: {[s.value for s in task.required_skills]}"
            }
        task.assigned_to.append(p.developer_id)
        dev.current_tasks.append(task.id)
        task.status = TaskStatus.IN_PROGRESS
        skill_note = " [skill mismatch — low proficiency]" if prof < 0.30 else ""
        return True, {"msg": f"{dev.name} → {task.name}{skill_note}"}

    def _act_unassign(self, action: Action) -> tuple:
        p    = action.unassign
        task = self._task(p.task_id)
        dev  = self._dev(p.developer_id)
        if task is None or dev is None:
            return False, {"error": "Task or dev not found"}
        if p.developer_id not in task.assigned_to:
            return False, {"error": f"{dev.name} not on {task.name}"}
        task.assigned_to.remove(p.developer_id)
        if task.id in dev.current_tasks:
            dev.current_tasks.remove(task.id)
        if not task.assigned_to:
            task.status = TaskStatus.READY
        self._metrics.unassign_penalties += 1
        return True, {"msg": f"Unassigned {dev.name} from {task.name}"}

    def _act_reprioritize(self, action: Action) -> tuple:
        p    = action.reprioritize
        task = self._task(p.task_id)
        if task is None:
            return False, {"error": f"Task {p.task_id} not found"}
        old = task.priority
        task.priority = p.new_priority
        return True, {"msg": f"{task.name}: {old.name}→{p.new_priority.name}"}

    def _act_rest(self, action: Action) -> tuple:
        p   = action.rest
        dev = self._dev(p.developer_id)
        if dev is None:
            return False, {"error": f"Dev {p.developer_id} not found"}
        if dev.current_tasks:
            return False, {"error": f"{dev.name} must be unassigned before resting"}
        recovered   = min(dev.fatigue, 0.40)
        dev.fatigue -= recovered
        return True, {"msg": f"{dev.name} rested; -fatigue {recovered:.2f}"}

    def _act_split(self, action: Action) -> tuple:
        p    = action.split
        task = self._task(p.task_id)
        if task is None:
            return False, {"error": f"Task {p.task_id} not found"}
        if task.status == TaskStatus.IN_PROGRESS:
            return False, {"error": "Cannot split in-progress task"}
        if task.remaining_points < 2.0:
            return False, {"error": "Task too small to split (<2 pts)"}

        r  = max(0.2, min(0.8, p.split_ratio))
        pa = round(task.remaining_points * r, 1)
        pb = round(task.remaining_points * (1.0 - r), 1)

        child_a = Task(
            name=f"{task.name}-A", description=task.description, priority=task.priority,
            required_skills=list(task.required_skills), story_points=pa,
            deadline_step=task.deadline_step, status=TaskStatus.READY,
            dependencies=list(task.dependencies),
            business_value=round(task.business_value * r, 2), created_step=self._step,
        )
        child_b = Task(
            name=f"{task.name}-B", description=task.description, priority=task.priority,
            required_skills=list(task.required_skills), story_points=pb,
            deadline_step=task.deadline_step, status=TaskStatus.BLOCKED,
            dependencies=list(task.dependencies) + [child_a.id],
            business_value=round(task.business_value * (1 - r), 2), created_step=self._step,
        )
        self._tasks.remove(task)
        self._tasks.extend([child_a, child_b])
        self._graph.register(child_a)
        self._graph.register(child_b)
        self._metrics.total_tasks += 1
        return True, {"msg": f"Split {task.name} → {child_a.name}({pa}sp) + {child_b.name}({pb}sp)"}

    def _act_pair(self, action: Action) -> tuple:
        p    = action.pair
        task = self._task(p.task_id)
        prim = self._dev(p.primary_developer_id)
        sec  = self._dev(p.secondary_developer_id)
        if task is None or prim is None or sec is None:
            return False, {"error": "Task or dev not found"}
        if not prim.available or not sec.available:
            return False, {"error": "One or both developers unavailable"}
        if task.status not in (TaskStatus.READY, TaskStatus.IN_PROGRESS):
            return False, {"error": f"{task.name} not assignable ({task.status})"}
        if p.primary_developer_id == p.secondary_developer_id:
            return False, {"error": "Cannot pair a dev with themselves"}
        for dev, did in [(prim, p.primary_developer_id), (sec, p.secondary_developer_id)]:
            if did not in task.assigned_to:
                task.assigned_to.append(did)
                dev.current_tasks.append(task.id)
        task.status = TaskStatus.IN_PROGRESS
        self._pair_map[task.id] = p.secondary_developer_id
        return True, {"msg": f"Pair: {prim.name}+{sec.name} on {task.name}"}



    def _simulate_work(self) -> None:
        """Advance all in-progress tasks by one step of developer effort."""
        dev_map = {d.id: d for d in self._devs}

        for task in self._tasks:
            if task.status != TaskStatus.IN_PROGRESS or task.remaining_points <= 0:
                continue

            total_work = 0.0
            is_pair    = task.id in self._pair_map

            for did in task.assigned_to:
                dev = dev_map.get(did)
                if dev is None or not dev.available:
                    continue
                total_work += dev.work_rate(task)
                dev.fatigue = min(1.0, dev.fatigue + dev.fatigue_increment(is_pair))


            if is_pair and len(task.assigned_to) >= 2:
                total_work *= 1.40

            task.remaining_points = max(0.0, task.remaining_points - total_work)


            if task.remaining_points == 0.0:
                task.status         = TaskStatus.COMPLETED
                task.completed_step = self._step
                for did in task.assigned_to:
                    dev = dev_map.get(did)
                    if dev:
                        dev.tasks_completed        += 1
                        dev.story_points_delivered += task.story_points
                        if task.id in dev.current_tasks:
                            dev.current_tasks.remove(task.id)
                task.assigned_to = []
                self._pair_map.pop(task.id, None)



    def _update_developers(self) -> None:
        """Handle sick recovery, passive fatigue recovery, and skill boost expiry."""
        for dev in self._devs:

            if not dev.available and dev.sick_until_step is not None:
                if self._step >= dev.sick_until_step:
                    dev.available       = True
                    dev.sick_until_step = None


            if dev.available and not dev.current_tasks:
                dev.fatigue = max(0.0, dev.fatigue - 0.05)


            if dev.boost_expires_step is not None and self._step >= dev.boost_expires_step:
                dev.skill_boosts       = {}
                dev.boost_expires_step = None


            if dev.fatigue > 0.80:
                self._metrics.overtime_steps += 1


    def _refresh_metrics(self) -> None:
        completed = [t for t in self._tasks if t.status == TaskStatus.COMPLETED]
        failed    = [t for t in self._tasks if t.status == TaskStatus.FAILED]

        self._metrics.completed_tasks          = len(completed)
        self._metrics.failed_tasks             = len(failed)
        self._metrics.delivered_story_points   = sum(t.story_points for t in completed)
        self._metrics.delivered_business_value = sum(t.business_value for t in completed)
        on_time = sum(
            1 for t in completed
            if t.completed_step is not None and t.completed_step <= t.deadline_step
        )
        self._metrics.on_time_deliveries = on_time
        self._metrics.late_deliveries    = len(completed) - on_time



    def _observe(self, reward: float, recent: list, info: dict | None = None) -> Observation:
        return Observation(
            step           = self._step,
            max_steps      = self._max_steps,
            tasks          = copy.deepcopy(self._tasks),
            developers     = copy.deepcopy(self._devs),
            pending_events = copy.deepcopy(self._events.pending),
            recent_events  = copy.deepcopy(recent),
            metrics        = copy.copy(self._metrics),
            reward         = reward,
            done           = self._done,
            info           = info or {},
        )

    def _task(self, tid: str) -> Task | None:
        return next((t for t in self._tasks if t.id == tid), None)

    def _dev(self, did: str) -> Developer | None:
        return next((d for d in self._devs if d.id == did), None)



class Scenario:
    """
    Registry of pre-built sprint configurations.

    Scenario parameters are carefully balanced so that:
    - A good heuristic agent scores C–B on easy/medium
    - A good RL agent can achieve A–A+ with learned strategies
    - 'hard' is genuinely hard for both, showing the RL improvement gap

    Balance formula:
    - Avg work_rate per dev = velocity(1.1) × proficiency(0.65) ≈ 0.72 sp/step
    - 'easy':   3 devs × 0.72 × 20 steps = 43sp capacity. Tasks = ~22sp → 50% headroom
    - 'medium': 4 devs × 0.72 × 30 steps = 86sp capacity. Tasks = ~50sp → 40% headroom
    - 'hard':   5 devs × 0.72 × 40 steps = 144sp capacity. Tasks = ~80sp → tight with events
    """

    _registry: dict = {}

    @classmethod
    def register(cls, name: str, cfg: dict) -> None:
        cls._registry[name] = cfg

    @classmethod
    def get(cls, name: str) -> dict:
        if name not in cls._registry:
            raise ValueError(
                f"Unknown scenario '{name}'. Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name]


    @staticmethod
    def _devs(rng: random.Random, count: int) -> list:
        """Fixed developer archetypes — deterministic given the same seed."""
        archetypes = [
            {"name": "Dharaneesh",  "skills": {SkillTag.BACKEND: 0.90, SkillTag.DEVOPS: 0.60, SkillTag.QA: 0.50},     "velocity": 1.4},
            {"name": "Ganapathy",    "skills": {SkillTag.FRONTEND: 0.85, SkillTag.MOBILE: 0.70, SkillTag.QA: 0.40},    "velocity": 1.2},
            {"name": "Athish",  "skills": {SkillTag.ML: 0.95, SkillTag.DATA: 0.80, SkillTag.BACKEND: 0.50},        "velocity": 1.1},
            {"name": "karthick",   "skills": {SkillTag.DEVOPS: 0.90, SkillTag.SECURITY: 0.75, SkillTag.BACKEND: 0.60},"velocity": 1.3},
            {"name": "JoesphFabio",    "skills": {SkillTag.QA: 0.90, SkillTag.FRONTEND: 0.60, SkillTag.MOBILE: 0.55},     "velocity": 1.0},
            {"name": "Dharshansriram",  "skills": {SkillTag.BACKEND: 0.70, SkillTag.DATA: 0.65, SkillTag.SECURITY: 0.60},  "velocity": 0.9},
        ]
        selected = rng.sample(archetypes, min(count, len(archetypes)))
        return [
            Developer(
                name=a["name"], skills=a["skills"], velocity=a["velocity"],
                fatigue=round(rng.uniform(0.0, 0.08), 2),
            )
            for a in selected
        ]

    @staticmethod
    def _tasks(rng: random.Random, max_steps: int, n: int, deps: bool) -> list:
        """
        BALANCED task factory — v3.

        Story points: 1.5–5.0sp
        Deadlines:    proportional to task size + generous slack buffer.

        """
        skill_combos = [
            [SkillTag.BACKEND], [SkillTag.FRONTEND],
            [SkillTag.BACKEND, SkillTag.FRONTEND],
            [SkillTag.ML, SkillTag.DATA], [SkillTag.DEVOPS],
            [SkillTag.QA, SkillTag.BACKEND], [SkillTag.SECURITY, SkillTag.DEVOPS],
            [SkillTag.MOBILE, SkillTag.FRONTEND], [SkillTag.DATA], [SkillTag.SECURITY],
        ]
        names = [
            "User Auth", "Payment Gateway", "Dashboard UI", "Data Pipeline",
            "CI/CD Setup", "Rate Limiter", "Mobile Onboarding", "Security Audit",
            "Search Feature", "Notification System", "Analytics Module",
            "Cache Layer", "WebSocket Support", "OAuth Integration",
            "PDF Export", "Batch Processor", "Admin Panel", "Audit Logging",
            "Recommendation Engine", "A/B Testing Framework",
        ]
        rng.shuffle(names)
        result = []
        for i in range(n):
            pts      = round(rng.uniform(1.5, 5.0), 1)
            priority = rng.choices(list(TaskPriority), weights=[10, 40, 35, 15], k=1)[0]

            avg_work_rate    = 0.72
            min_steps_needed = int(pts / avg_work_rate) + 1

            min_slack = max(5, max_steps // 4)
            max_slack = max(min_slack + 2, max_steps // 2)
            slack    = rng.randint(min_slack, max_slack)

            deadline = min(int(max_steps * 0.85), min_steps_needed + slack)
            deadline = max(deadline, min_steps_needed + 3)
            result.append(Task(
                name=names[i % len(names)],
                priority=priority,
                required_skills=list(rng.choice(skill_combos)),
                story_points=pts,
                deadline_step=deadline,
                business_value=round(rng.uniform(3.0, 10.0), 1),
            ))

        if deps and len(result) >= 4:
            chain_len = rng.randint(2, min(3, len(result) // 3))
            chain = rng.sample(result, chain_len)
            chain.sort(key=lambda t: t.deadline_step)
            for i in range(1, len(chain)):
                chain[i].dependencies.append(chain[i - 1].id)
                if chain[i - 1].deadline_step >= chain[i].deadline_step:
                    chain[i].deadline_step = chain[i - 1].deadline_step + 3

        return result



Scenario.register("easy", {
    "max_steps":    20,
    "task_factory": lambda rng, ms: Scenario._tasks(rng, ms, 6,  False),
    "dev_factory":  lambda rng:     Scenario._devs(rng, 3),
})
Scenario.register("medium", {
    "max_steps":    30,
    "task_factory": lambda rng, ms: Scenario._tasks(rng, ms, 12, True),
    "dev_factory":  lambda rng:     Scenario._devs(rng, 4),
})
Scenario.register("hard", {
    "max_steps":    40,
    "task_factory": lambda rng, ms: Scenario._tasks(rng, ms, 18, True),
    "dev_factory":  lambda rng:     Scenario._devs(rng, 5),
})
Scenario.register("small", {
    "max_steps":    20,
    "task_factory": lambda rng, ms: Scenario._tasks(rng, ms, 6,  False),
    "dev_factory":  lambda rng:     Scenario._devs(rng, 3),
})
Scenario.register("large", {
    "max_steps":    50,
    "task_factory": lambda rng, ms: Scenario._tasks(rng, ms, 20, True),
    "dev_factory":  lambda rng:     Scenario._devs(rng, 6),
})
Scenario.register("chaos", {
    "max_steps":    40,
    "task_factory": lambda rng, ms: Scenario._tasks(rng, ms, 18, True),
    "dev_factory":  lambda rng:     Scenario._devs(rng, 5),
})
