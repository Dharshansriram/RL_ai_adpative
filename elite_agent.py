"""
elite_agent.py — EliteProjectAgent
=================================================

ROOT CAUSE ANALYSIS (from step-by-step tracing):

"""

from __future__ import annotations

from environment import ProjectManagerEnv
from models import (
    Action, ActionType,
    AssignTaskPayload, PairProgramPayload,
    ReprioritizePayload, RestDeveloperPayload,
    UnassignTaskPayload,
    Developer, Observation, Task,
    TaskPriority, TaskStatus,
)


class EliteProjectAgent:
    """
    Targets ≥95% reliability (weighted score ≥ 0.70) across all scenario/seed combos.
    """


    MANDATORY_REST_THRESHOLD = 0.70
    FATIGUE_TIER1_CAP        = 0.52
    FATIGUE_TIER2_CAP        = 0.65
    REST_TRIGGER             = 0.48
    PAIR_FATIGUE_CAP         = 0.35


    LARGE_TASK_SP            = 3.0
    URGENCY_HORIZON          = 5

    MIN_PROFICIENCY          = 0.25
    INJECT_PRIORITY_BOOST    = 1.5


    SENIOR_SKILL_THRESHOLD   = 0.70
    MID_SKILL_THRESHOLD      = 0.50

    def __init__(self) -> None:
        self._injected_seen: set[str] = set()
        self._pending_rests: set[str] = set()
        self._rotation_log:  set[str] = set()

    def act(self, obs: Observation) -> Action:
        """Main decision function — called once per environment step."""


        for t in obs.tasks:
            if (t.is_injected
                    and t.id not in self._injected_seen
                    and t.status in (TaskStatus.READY, TaskStatus.IN_PROGRESS)):
                self._injected_seen.add(t.id)


        self._rotation_log = {
            did for did in self._rotation_log
            if any(
                d.id == did and d.fatigue >= self.MANDATORY_REST_THRESHOLD - 0.12
                for d in obs.developers
            )
        }

        for dev in obs.developers:
            if (dev.id in self._pending_rests
                    and dev.available
                    and not dev.current_tasks):
                self._pending_rests.discard(dev.id)
                return Action(
                    action_type=ActionType.REST_DEVELOPER,
                    rest=RestDeveloperPayload(developer_id=dev.id),
                )


        for dev in sorted(obs.developers, key=lambda d: -d.fatigue):
            if (dev.available
                    and dev.fatigue >= self.MANDATORY_REST_THRESHOLD
                    and dev.current_tasks
                    and dev.id not in self._rotation_log):
                task = self._least_urgent_task(dev, obs)
                if task:
                    self._rotation_log.add(dev.id)
                    self._pending_rests.add(dev.id)
                    return Action(
                        action_type=ActionType.UNASSIGN_TASK,
                        unassign=UnassignTaskPayload(
                            task_id=task.id, developer_id=dev.id
                        ),
                    )

        injected_ready = sorted(
            [t for t in obs.tasks
             if t.is_injected
             and t.status == TaskStatus.READY
             and not t.assigned_to],
            key=lambda t: -t.priority.value,
        )
        for task in injected_ready:
            dev = self._best_dev(
                task, obs,
                max_fatigue=self.FATIGUE_TIER2_CAP,
                allow_multitask=True,
            )
            if dev and dev.proficiency_for_task(task) >= self.MIN_PROFICIENCY:
                return Action(
                    action_type=ActionType.ASSIGN_TASK,
                    assign=AssignTaskPayload(task_id=task.id, developer_id=dev.id),
                )

            if dev is None:
                dev = self._best_dev(
                    task, obs,
                    max_fatigue=self.FATIGUE_TIER2_CAP,
                    allow_multitask=True,
                )
            if dev:
                return Action(
                    action_type=ActionType.ASSIGN_TASK,
                    assign=AssignTaskPayload(task_id=task.id, developer_id=dev.id),
                )


        if injected_ready:
            hi_inject = next(
                (t for t in injected_ready if t.priority.value >= TaskPriority.HIGH.value),
                None,
            )
            if hi_inject:
                freed = self._free_dev_from_low_priority(obs, exclude_tasks=injected_ready)
                if freed:
                    dev_id, task_id = freed
                    self._pending_rests.discard(dev_id)
                    return Action(
                        action_type=ActionType.UNASSIGN_TASK,
                        unassign=UnassignTaskPayload(task_id=task_id, developer_id=dev_id),
                    )


        for task in self._rank_tasks(obs):
            allow_mt = (task.priority == TaskPriority.CRITICAL)
            deadline_critical = self._is_deadline_critical(task, obs)

            dev = self._best_dev(
                task, obs,
                max_fatigue=self.FATIGUE_TIER2_CAP,
                allow_multitask=allow_mt or deadline_critical,
            )
            if dev and dev.proficiency_for_task(task) >= self.MIN_PROFICIENCY:
                return Action(
                    action_type=ActionType.ASSIGN_TASK,
                    assign=AssignTaskPayload(task_id=task.id, developer_id=dev.id),
                )


            if deadline_critical:
                fallback_dev = self._best_dev(
                    task, obs,
                    max_fatigue=self.FATIGUE_TIER2_CAP,
                    allow_multitask=True,
                )
                if fallback_dev:
                    return Action(
                        action_type=ActionType.ASSIGN_TASK,
                        assign=AssignTaskPayload(task_id=task.id, developer_id=fallback_dev.id),
                    )


        for task in sorted(obs.tasks, key=lambda t: -t.priority.value):
            if (task.status == TaskStatus.IN_PROGRESS
                    and (task.priority == TaskPriority.CRITICAL
                         or task.story_points >= self.LARGE_TASK_SP)
                    and len(task.assigned_to) == 1):
                sec = self._best_dev(
                    task, obs,
                    exclude=task.assigned_to,
                    max_fatigue=self.PAIR_FATIGUE_CAP,
                    idle_only=True,
                    allow_multitask=False,
                )
                if sec and sec.proficiency_for_task(task) >= self.MIN_PROFICIENCY:
                    return Action(
                        action_type=ActionType.PAIR_PROGRAM,
                        pair=PairProgramPayload(
                            task_id=task.id,
                            primary_developer_id=task.assigned_to[0],
                            secondary_developer_id=sec.id,
                        ),
                    )


        for dev in sorted(obs.developers, key=lambda d: -d.fatigue):
            if (dev.available
                    and dev.fatigue >= self.REST_TRIGGER
                    and not dev.current_tasks):

                emergency = [
                    t for t in obs.ready_tasks()
                    if (t.deadline_step - obs.step) <= 2
                    and dev.proficiency_for_task(t) >= 0.5
                    and not t.assigned_to
                ]
                if not emergency:
                    return Action(
                        action_type=ActionType.REST_DEVELOPER,
                        rest=RestDeveloperPayload(developer_id=dev.id),
                    )


        for task in obs.ready_tasks():
            steps_left = task.deadline_step - obs.step
            if steps_left <= self.URGENCY_HORIZON and task.priority != TaskPriority.CRITICAL:
                return Action(
                    action_type=ActionType.REPRIORITIZE,
                    reprioritize=ReprioritizePayload(
                        task_id=task.id,
                        new_priority=TaskPriority.CRITICAL,
                    ),
                )

        return Action(action_type=ActionType.NOOP)



    def _rank_tasks(self, obs: Observation) -> list[Task]:

        dep_map: dict[str, int] = {}
        for t in obs.tasks:
            for dep_id in t.dependencies:
                dep_map[dep_id] = dep_map.get(dep_id, 0) + 1

        def score(t: Task) -> float:
            steps_left       = max(1, t.deadline_step - obs.step)
            inject_boost     = self.INJECT_PRIORITY_BOOST if t.is_injected else 1.0

            effective_sp     = t.remaining_points if t.remaining_points > 0 else t.story_points
            urgency          = (t.priority.value * t.business_value * inject_boost) / steps_left
            unblock_bonus    = dep_map.get(t.id, 0) * 2.0
            completion_bonus = (1.0 - effective_sp / max(t.story_points, 0.01)) * 3.0
            return urgency + unblock_bonus + completion_bonus

        candidates = [t for t in obs.ready_tasks() if not t.assigned_to]
        return sorted(candidates, key=score, reverse=True)



    def _dev_skill_level(self, dev: Developer) -> float:
        """Return average skill proficiency across a developer's known skills.
        """
        if not dev.skills:
            return 0.0
        return sum(dev.skills.values()) / len(dev.skills)

    def _task_complexity(self, task: Task) -> float:
        """Estimate task complexity for skill-tier matching.

        """
        sp_factor = min(1.0, task.story_points / 8.0)
        prio_factor = (task.priority.value - 1) / 4.0
        return (sp_factor + prio_factor) / 2.0

    def _best_dev(
        self,
        task: Task,
        obs: Observation,
        exclude: list | None = None,
        max_fatigue: float | None = None,
        idle_only: bool = False,
        allow_multitask: bool = False,
    ) -> Developer | None:
        """
        Skill-level-aware developer selection.
        """
        if max_fatigue is None:
            max_fatigue = self.FATIGUE_TIER2_CAP
        excl  = set(exclude or []) | self._pending_rests
        avail = [
            d for d in obs.available_developers()
            if d.id not in excl and d.fatigue < max_fatigue
        ]

        task_complexity = self._task_complexity(task)

        def dev_score(d: Developer) -> float:
            proficiency = d.proficiency_for_task(task)
            dev_level   = self._dev_skill_level(d)

            skill_match = 1.0 - abs(dev_level - task_complexity)
            return (
                proficiency * 2.0
                - 0.35 * (d.fatigue / max(0.01, max_fatigue))
                + skill_match * 0.3
            )

        idle_strict = [
            d for d in avail
            if not d.current_tasks and d.fatigue < self.FATIGUE_TIER1_CAP
        ]
        if idle_strict:
            return max(idle_strict, key=dev_score)

        if idle_only:
            return None

        no_task = [d for d in avail if not d.current_tasks]
        if no_task:
            return max(no_task, key=dev_score)

        if allow_multitask:
            with_task = [d for d in avail if d.current_tasks]
            if with_task:
                return max(with_task, key=dev_score)

        return None



    def _is_deadline_critical(self, task: Task, obs: Observation) -> bool:
        """True when the task will likely fail if not assigned this step.
        """
        steps_left = task.deadline_step - obs.step
        effective_sp = task.remaining_points if task.remaining_points > 0 else task.story_points
        return steps_left <= effective_sp + 1

    def _least_urgent_task(self, dev: Developer, obs: Observation) -> Task | None:
        """Return the lowest-urgency task currently assigned to this dev."""
        tasks = [
            t for t in obs.tasks
            if t.id in dev.current_tasks and t.status == TaskStatus.IN_PROGRESS
        ]
        if not tasks:
            return None
        return min(tasks, key=lambda t: t.priority.value * t.business_value)

    def _free_dev_from_low_priority(
        self,
        obs: Observation,
        exclude_tasks: list[Task],
    ) -> tuple[str, str] | None:
        """
        Find a (dev_id, task_id) where the dev is doing LOW/MEDIUM work
        that can be pre-empted to free capacity for an injected urgent task.
        Picks the freshest available dev (lowest fatigue) first.
        """
        exclude_ids = {t.id for t in exclude_tasks}
        for dev in sorted(obs.developers, key=lambda d: d.fatigue):
            if not dev.available or not dev.current_tasks:
                continue
            for tid in dev.current_tasks:
                task = next((t for t in obs.tasks if t.id == tid), None)
                if (task
                        and task.priority.value <= TaskPriority.MEDIUM.value
                        and task.id not in exclude_ids):
                    return dev.id, task.id
        return None




def run_episode_elite(
    scenario: str = "medium",
    seed: int = 42,
    verbose: bool = True,
) -> float:
    """Run one complete episode with EliteProjectAgent; return weighted score."""
    env   = ProjectManagerEnv(scenario=scenario, seed=seed)
    agent = EliteProjectAgent()
    obs   = env.reset()

    if verbose:
        sep = "=" * 64
        print(f"\n{sep}")
        print(f"  ELITE AGENT v4.1 — {scenario.upper()}  seed={seed}")
        print(
            f"  Steps={obs.max_steps}  Tasks={obs.metrics.total_tasks}"
            f"  Devs={len(obs.developers)}"
        )
        print(
            f"  Total SP: {obs.metrics.total_story_points:.1f}"
            f"  BizValue: {obs.metrics.total_business_value:.1f}"
        )
        print(sep + "\n")

    while not obs.done:
        action = agent.act(obs)
        obs    = env.step(action)
        if verbose and (obs.step % 5 == 0 or obs.done or obs.recent_events):
            print(f"  {obs.summary()}")
            for ev in obs.recent_events:
                print(f"    [EVENT] {ev.description}")

    score = env.grade()
    if verbose:
        print(f"\n{score.report()}")
    return score.weighted_total


def benchmark(seeds: list | None = None, verbose: bool = False) -> dict:
    """
    Run full benchmark across all scenarios and seeds.
    Returns a summary dict with mean, min, max, pass_rate.
    """
    if seeds is None:
        seeds = [1, 7, 13, 42, 99, 123, 200, 500, 999]
    results: dict[str, list[float]] = {"easy": [], "medium": [], "hard": []}
    for scenario in ("easy", "medium", "hard"):
        for seed in seeds:
            s = run_episode_elite(scenario, seed, verbose=False)
            results[scenario].append(s)
            if verbose:
                status = "PASS" if s >= 0.70 else "FAIL"
                print(f"  {scenario:8} seed={seed:4}: {s:.4f}  {status}")
    all_scores = [s for lst in results.values() for s in lst]
    passing    = sum(1 for s in all_scores if s >= 0.70)
    return {
        "mean":         round(sum(all_scores) / len(all_scores), 4),
        "min":          round(min(all_scores), 4),
        "max":          round(max(all_scores), 4),
        "pass_rate":    round(passing / len(all_scores), 4),
        "n_total":      len(all_scores),
        "n_passing":    passing,
        "per_scenario": {
            sc: round(sum(v) / len(v), 4) for sc, v in results.items()
        },
    }


if __name__ == "__main__":
    print("Running benchmark across 27 scenario/seed combinations...\n")
    s = benchmark(verbose=True)
    print(f"\n{'='*60}")
    print(f"  BENCHMARK SUMMARY — EliteProjectAgent v4.1")
    print(f"{'='*60}")
    print(f"  Mean Score : {s['mean']:.4f}")
    print(f"  Min  Score : {s['min']:.4f}")
    print(f"  Max  Score : {s['max']:.4f}")
    print(f"  Pass Rate  : {s['pass_rate']*100:.1f}%  ({s['n_passing']}/{s['n_total']} >= 0.70)")
    print(f"  Easy  avg  : {s['per_scenario']['easy']:.4f}")
    print(f"  Medium avg : {s['per_scenario']['medium']:.4f}")
    print(f"  Hard  avg  : {s['per_scenario']['hard']:.4f}")
    print(f"{'='*60}")
