"""
api_models.py — Request/response schemas for the FastAPI layer.
"""


from dataclasses import dataclass, asdict
from email.policy import default
from typing import Any, Optional

from models import (
    Action, ActionType,
    AssignTaskPayload, UnassignTaskPayload,
    ReprioritizePayload, RestDeveloperPayload,
    SplitTaskPayload, PairProgramPayload,
    Observation, Task, Developer, SprintMetrics,
    TaskStatus, TaskPriority,
)
from grader import ScoreBreakdown

from enum import Enum
from pydantic import BaseModel, Field, ConfigDict
from pydantic import BaseModel, Field, ConfigDict

class ActionTypeEnum(str, Enum):
    assign_task = "assign_task"
    unassign_task = "unassign_task"
    reprioritize = "reprioritize"
    rest_developer = "rest_developer"
    split_task = "split_task"
    pair_program = "pair_program"
    noop = "noop"
    AUTO = "auto"


class ScenarioEnum(str, Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"




try:
    from pydantic import BaseModel, Field as PField
    _PYDANTIC = True
except ImportError:
    _PYDANTIC = False

    class BaseModel:
        pass

    def PField(*a, **kw):
        return None




@dataclass
class ResetRequest:
    scenario: str = ""
    seed: int = 42
    session_id: Optional[str] = None



@dataclass
class ActionRequest:
    session_id: str = ""
    action_type: str = "noop"
    task_id: Optional[str] = None
    developer_id: Optional[str] = None
    primary_developer_id: Optional[str] = None
    secondary_developer_id: Optional[str] = None
    new_priority: Optional[str] = None
    split_ratio: float = 0.5

    def to_action(self):
        atype = self.action_type.value if hasattr(self.action_type, "value") else self.action_type

        if atype == "assign_task":
            if not self.task_id:
                raise ValueError("task_id required")

            return Action(
                action_type=ActionType.ASSIGN_TASK,
                assign=AssignTaskPayload(
                    task_id=self.task_id,
                    developer_id=self.developer_id
                )
            )

        elif atype == "unassign_task":
            return Action(action_type=ActionType.UNASSIGN_TASK)

        elif atype == "noop":
            return Action(action_type=ActionType.NOOP)

        elif atype == "split_task":
            return Action(
                action_type=ActionType.SPLIT_TASK,
                split=SplitTaskPayload(split_ratio=self.split_ratio)
            )

        elif atype == "reprioritize":
            return Action(
                action_type=ActionType.REPRIORITIZE,
                reprioritize=ReprioritizePayload(new_priority=self.new_priority)
            )

        elif atype == "rest_developer":
            return Action(
                action_type=ActionType.REST_DEVELOPER,
                rest=RestDeveloperPayload(developer_id=self.developer_id)
            )

        elif atype == "pair_program":
            return Action(
                action_type=ActionType.PAIR_PROGRAM,
                pair=PairProgramPayload(
                    task_id=self.task_id,
                    primary_developer_id=self.primary_developer_id,
                    secondary_developer_id=self.secondary_developer_id
                )
            )

        raise ValueError(f"Unknown action_type: {atype}")


    def _require(self, *fields: str) -> None:
        for f in fields:
            if getattr(self, f, None) is None:
                raise ValueError(
                    f"Field '{f}' is required for action_type='{self.action_type}'.")




@dataclass
class TaskSchema:
    id: str
    name: str
    description: str
    status: str
    priority: str
    priority_value: int
    required_skills: list
    story_points: float
    remaining_points: float
    completion_ratio: float
    deadline_step: int
    business_value: float
    assigned_to: list
    dependencies: list
    completed_step: Optional[int]
    is_injected: bool
    urgency_score: float

    @classmethod
    def from_task(cls, task: Task, current_step: int) -> "TaskSchema":
        return cls(
            id=task.id,
            name=task.name,
            description=task.description,
            status=task.status.value,
            priority=task.priority.name.lower(),
            priority_value=task.priority.value,
            required_skills=[s.value for s in task.required_skills],
            story_points=round(task.story_points, 2),
            remaining_points=round(task.remaining_points, 2),
            completion_ratio=round(task.completion_ratio, 3),
            deadline_step=task.deadline_step,
            business_value=round(task.business_value, 2),
            assigned_to=list(task.assigned_to),
            dependencies=list(task.dependencies),
            completed_step=task.completed_step,
            is_injected=task.is_injected,
            urgency_score=round(task.urgency_score(current_step), 4),
        )


@dataclass
class DeveloperSchema:
    id: str
    name: str
    velocity: float
    fatigue: float
    fatigue_pct: str
    available: bool
    skills: dict
    current_tasks: list
    tasks_completed: int
    story_points_delivered: float
    effective_velocity: float
    sick_until_step: Optional[int]

    @classmethod
    def from_dev(cls, dev: Developer) -> "DeveloperSchema":
        return cls(
            id=dev.id,
            name=dev.name,
            velocity=round(dev.velocity, 2),
            fatigue=round(dev.fatigue, 3),
            fatigue_pct=f"{dev.fatigue * 100:.1f}%",
            available=dev.available,
            skills={k.value: round(v, 2) for k, v in dev.skills.items()},
            current_tasks=list(dev.current_tasks),
            tasks_completed=dev.tasks_completed,
            story_points_delivered=round(dev.story_points_delivered, 2),
            effective_velocity=round(dev.effective_velocity, 3),
            sick_until_step=dev.sick_until_step,
        )


@dataclass
class MetricsSchema:
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    total_story_points: float
    delivered_story_points: float
    delivery_pct: str
    total_business_value: float
    delivered_business_value: float
    value_pct: str
    on_time_deliveries: int
    late_deliveries: int
    events_handled: int
    unassign_penalties: int
    overtime_steps: int

    @classmethod
    def from_metrics(cls, m: SprintMetrics) -> "MetricsSchema":
        del_pct = (
            f"{m.delivered_story_points / m.total_story_points * 100:.1f}%"
            if m.total_story_points > 0 else "0.0%"
        )
        val_pct = (
            f"{m.delivered_business_value / m.total_business_value * 100:.1f}%"
            if m.total_business_value > 0 else "0.0%"
        )
        return cls(
            total_tasks=m.total_tasks,
            completed_tasks=m.completed_tasks,
            failed_tasks=m.failed_tasks,
            total_story_points=round(m.total_story_points, 2),
            delivered_story_points=round(m.delivered_story_points, 2),
            delivery_pct=del_pct,
            total_business_value=round(m.total_business_value, 2),
            delivered_business_value=round(m.delivered_business_value, 2),
            value_pct=val_pct,
            on_time_deliveries=m.on_time_deliveries,
            late_deliveries=m.late_deliveries,
            events_handled=m.events_handled,
            unassign_penalties=m.unassign_penalties,
            overtime_steps=m.overtime_steps,
        )




@dataclass
class ObservationResponse:
    status: str
    session_id: str
    step: int
    max_steps: int
    done: bool
    reward: float
    tasks: list
    developers: list
    metrics: Any
    recent_events: list
    pending_events: list
    info: dict

    @classmethod
    def from_obs(cls, obs: Observation, session_id: str) -> "ObservationResponse":
        return cls(
            status="success",
            session_id=session_id,
            step=obs.step,
            max_steps=obs.max_steps,
            done=obs.done,
            reward=round(obs.reward, 6),
            tasks=[TaskSchema.from_task(t, obs.step) for t in obs.tasks],
            developers=[DeveloperSchema.from_dev(d) for d in obs.developers],
            metrics=MetricsSchema.from_metrics(obs.metrics),
            recent_events=[
                {"id": e.id, "type": e.event_type.value, "description": e.description}
                for e in obs.recent_events
            ],
            pending_events=[
                {"id": e.id, "type": e.event_type.value, "trigger_step": e.trigger_step}
                for e in obs.pending_events
            ],
            info=obs.info,
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TaskListResponse:
    status: str
    session_id: str
    step: int
    max_steps: int
    total_tasks: int
    ready_count: int
    in_progress_count: int
    blocked_count: int
    completed_count: int
    failed_count: int
    tasks_by_urgency: list
    dependency_graph: dict

    @classmethod
    def from_obs(cls, obs: Observation, session_id: str) -> "TaskListResponse":
        schemas = [TaskSchema.from_task(t, obs.step) for t in obs.tasks]
        sorted_tasks = sorted(schemas, key=lambda t: t.urgency_score, reverse=True)
        counts: dict[str, int] = {}
        for t in obs.tasks:
            counts[t.status.value] = counts.get(t.status.value, 0) + 1
        dep_graph = {t.id: list(t.dependencies) for t in obs.tasks}
        return cls(
            status="success",
            session_id=session_id,
            step=obs.step,
            max_steps=obs.max_steps,
            total_tasks=len(obs.tasks),
            ready_count=counts.get("ready", 0),
            in_progress_count=counts.get("in_progress", 0),
            blocked_count=counts.get("blocked", 0),
            completed_count=counts.get("completed", 0),
            failed_count=counts.get("failed", 0),
            tasks_by_urgency=sorted_tasks,
            dependency_graph=dep_graph,
        )


@dataclass
class GraderResponse:
    status: str
    session_id: str
    weighted_total: float
    grade: str
    dimensions: dict
    weights: dict
    notes: list
    report: str

    @classmethod
    def from_breakdown(cls, bd: ScoreBreakdown, session_id: str) -> "GraderResponse":
        return cls(
            status="success",
            session_id=session_id,
            weighted_total=round(bd.weighted_total, 6),
            grade=bd.grade,
            dimensions={
                "delivery":     round(bd.delivery_score,     4),
                "value":        round(bd.value_score,        4),
                "timeliness":   round(bd.timeliness_score,   4),
                "priority":     round(bd.priority_score,     4),
                "team_health":  round(bd.team_health_score,  4),
                "adaptability": round(bd.adaptability_score, 4),
                "efficiency":   round(bd.efficiency_score,   4),
                "dependency":   round(bd.dependency_score,   4),
            },
            weights=bd.WEIGHTS,
            notes=bd.notes,
            report=bd.report(),
        )


@dataclass
class TaskResultSchema:
    task_id: str
    scenario: str
    seed: int
    weighted_total: float
    grade: str
    dimensions: dict
    steps_taken: int
    delivered_pct: str
    events_handled: int
    improvement_hint: str = ""


@dataclass
class BaselineResponse:
    status: str
    agent: str
    seed: int
    tasks: list
    summary: dict




StateResponse = ObservationResponse




class ResetRequestModel(BaseModel):
    """Request body for POST /reset. All fields optional."""
    scenario:   ScenarioEnum  = Field(default=ScenarioEnum.medium,
                                      description="easy | medium | hard")
    seed:       int           = Field(default=42, ge=0,
                                      description="RNG seed for reproducibility")
    session_id: Optional[str] = Field(default=None,
                                      description="Reuse session or omit to auto-create")

    def to_dc(self) -> "ResetRequest":
        return ResetRequest(
            scenario=self.scenario.value,
            seed=self.seed,
            session_id=self.session_id,
        )


class ActionRequestModel(BaseModel):
    """Request body for POST /step."""
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"session_id": "uuid-here", "action_type": "assign_task",
                 "task_id": "abc12345", "developer_id": "dev67890"},
                {"session_id": "uuid-here", "action_type": "noop"},
                {"session_id": "uuid-here", "action_type": "rest_developer",
                 "developer_id": "dev67890"},
            ]
        }
    )

    session_id:             str            = Field(...,   description="Session UUID from /reset")
    action_type:            ActionTypeEnum = Field(...,   description="Action to execute")
    task_id:                Optional[str]  = Field(None,  description="Task ID")
    developer_id:           Optional[str]  = Field(None,  description="Developer ID")
    primary_developer_id:   Optional[str]  = Field(None,  description="Primary dev (pair_program)")
    secondary_developer_id: Optional[str]  = Field(None,  description="Secondary dev (pair_program)")
    new_priority:           Optional[str]  = Field(None,  description="LOW|MEDIUM|HIGH|CRITICAL")
    split_ratio:            float          = Field(0.5, ge=0.2, le=0.8,
                                                   description="Split fraction for split_task")

    def to_dc(self) -> "ActionRequest":
        return ActionRequest(
            session_id=self.session_id,
            action_type=self.action_type.value,
            task_id=self.task_id,
            developer_id=self.developer_id,
            primary_developer_id=self.primary_developer_id,
            secondary_developer_id=self.secondary_developer_id,
            new_priority=self.new_priority,
            split_ratio=self.split_ratio,
        )

    def to_action(self):
        return self.to_dc().to_action()
