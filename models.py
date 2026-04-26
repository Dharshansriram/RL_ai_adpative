"""
models.py — Typed data models for the Adaptive AI Project Manager environment.

Zero external dependencies — uses stdlib dataclasses + typing only.
Compatible with Python 3.10+.
"""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any




class SkillTag(str, Enum):
    BACKEND  = "backend"
    FRONTEND = "frontend"
    ML       = "ml"
    DEVOPS   = "devops"
    QA       = "qa"
    DATA     = "data"
    SECURITY = "security"
    MOBILE   = "mobile"


class TaskStatus(str, Enum):
    PENDING     = "pending"
    READY       = "ready"
    IN_PROGRESS = "in_progress"
    BLOCKED     = "blocked"
    COMPLETED   = "completed"
    FAILED      = "failed"


class TaskPriority(int, Enum):
    LOW      = 1
    MEDIUM   = 2
    HIGH     = 3
    CRITICAL = 4


class EventType(str, Enum):
    BUG_REPORT            = "bug_report"
    SCOPE_CHANGE          = "scope_change"
    DEVELOPER_SICK        = "developer_sick"
    URGENT_FEATURE        = "urgent_feature"
    INFRASTRUCTURE_OUTAGE = "infra_outage"
    KNOWLEDGE_TRANSFER    = "knowledge_transfer"


class ActionType(str, Enum):
    ASSIGN_TASK    = "assign_task"
    UNASSIGN_TASK  = "unassign_task"
    REPRIORITIZE   = "reprioritize"
    REST_DEVELOPER = "rest_developer"
    SPLIT_TASK     = "split_task"
    PAIR_PROGRAM   = "pair_program"
    NOOP           = "noop"




@dataclass
class Task:
    """A unit of work in the sprint backlog."""
    name: str
    required_skills: list
    story_points: float
    deadline_step: int
    id: str                      = field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str             = ""
    priority: TaskPriority       = TaskPriority.MEDIUM
    remaining_points: float      = 0.0
    status: TaskStatus           = TaskStatus.PENDING
    assigned_to: list            = field(default_factory=list)
    dependencies: list           = field(default_factory=list)
    completed_step: int | None   = None
    created_step: int            = 0
    business_value: float        = 5.0
    is_injected: bool            = False

    def __post_init__(self) -> None:
        if self.remaining_points == 0.0:
            self.remaining_points = self.story_points

    @property
    def completion_ratio(self) -> float:
        if self.story_points == 0:
            return 1.0
        return 1.0 - (self.remaining_points / self.story_points)

    def urgency_score(self, current_step: int) -> float:
        steps_left = max(1, self.deadline_step - current_step)
        return (self.priority.value * self.business_value) / steps_left


@dataclass
class Developer:
    """A team member with skills, capacity, and fatigue state."""
    name: str
    skills: dict
    velocity: float              = 1.0
    id: str                      = field(default_factory=lambda: str(uuid.uuid4())[:8])
    fatigue: float               = 0.0
    available: bool              = True
    sick_until_step: int | None  = None
    current_tasks: list          = field(default_factory=list)
    tasks_completed: int         = 0
    story_points_delivered: float = 0.0
    skill_boosts: dict           = field(default_factory=dict)
    boost_expires_step: int | None = None

    @property
    def effective_velocity(self) -> float:
        """Velocity degrades linearly with fatigue; 100% fatigue → 30% output."""
        return max(0.1, self.velocity * (1.0 - 0.7 * self.fatigue))

    def proficiency_for_task(self, task: Task) -> float:
        """Average proficiency across required skills. Missing skill → 0.1."""
        if not task.required_skills:
            return 1.0
        total = 0.0
        for skill in task.required_skills:
            base  = self.skills.get(skill, 0.1)
            boost = self.skill_boosts.get(skill.value, 0.0)
            total += min(1.0, base + boost)
        return total / len(task.required_skills)

    def work_rate(self, task: Task) -> float:
        """Story points completed per environment step for this task."""
        return self.effective_velocity * self.proficiency_for_task(task)

    def fatigue_increment(self, pair_programming: bool = False) -> float:
        """Fatigue gained per step of work. Context switching multiplies cost."""
        base = 0.04
        if pair_programming:
            base *= 1.3
        n = len(self.current_tasks)
        if n > 1:
            base *= 1.2 * n
        return min(base, 1.0 - self.fatigue)


@dataclass
class DynamicEvent:
    """A stochastic event that mutates environment state when triggered."""
    event_type: EventType
    trigger_step: int
    id: str                 = field(default_factory=lambda: str(uuid.uuid4())[:8])
    payload: dict           = field(default_factory=dict)
    applied: bool           = False
    description: str        = ""


@dataclass
class SprintMetrics:
    """Accumulated episode-level statistics for reward shaping and grading."""
    total_tasks: int               = 0
    completed_tasks: int           = 0
    failed_tasks: int              = 0
    total_story_points: float      = 0.0
    delivered_story_points: float  = 0.0
    total_business_value: float    = 0.0
    delivered_business_value: float = 0.0
    unassign_penalties: int        = 0
    overtime_steps: int            = 0
    events_handled: int            = 0
    on_time_deliveries: int        = 0
    late_deliveries: int           = 0




@dataclass
class AssignTaskPayload:
    task_id: str
    developer_id: str

@dataclass
class UnassignTaskPayload:
    task_id: str
    developer_id: str

@dataclass
class ReprioritizePayload:
    task_id: str
    new_priority: TaskPriority

@dataclass
class RestDeveloperPayload:
    developer_id: str

@dataclass
class SplitTaskPayload:
    task_id: str
    split_ratio: float = 0.5

@dataclass
class PairProgramPayload:
    task_id: str
    primary_developer_id: str
    secondary_developer_id: str

@dataclass
class Action:
    """
    Structured action consumed by env.step().

    Set action_type and populate the matching payload field; all others stay None.
    """
    action_type: ActionType
    assign:       AssignTaskPayload    | None = None
    unassign:     UnassignTaskPayload  | None = None
    reprioritize: ReprioritizePayload  | None = None
    rest:         RestDeveloperPayload | None = None
    split:        SplitTaskPayload     | None = None
    pair:         PairProgramPayload   | None = None




@dataclass
class Observation:
    """
    Full environment state snapshot returned after reset() and each step().
    Immutable by convention — the env always issues fresh deep copies.
    """
    step: int
    max_steps: int
    tasks: list
    developers: list
    pending_events: list
    recent_events: list
    metrics: SprintMetrics
    reward: float
    done: bool
    info: dict = field(default_factory=dict)

    def ready_tasks(self) -> list:
        return [t for t in self.tasks if t.status == TaskStatus.READY]

    def available_developers(self) -> list:
        return [d for d in self.developers if d.available]

    def summary(self) -> str:
        ready  = len(self.ready_tasks())
        avail  = len(self.available_developers())
        ip     = sum(1 for t in self.tasks if t.status == TaskStatus.IN_PROGRESS)
        done   = sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in self.tasks if t.status == TaskStatus.FAILED)
        return (
            f"Step {self.step:>3}/{self.max_steps} | "
            f"Ready:{ready} InProg:{ip} Done:{done} Failed:{failed} | "
            f"Avail:{avail} devs | "
            f"Reward:{self.reward:+.3f}"
        )
