"""
main.py — FastAPI server for the Adaptive AI Project Manager (OpenEnv).

Endpoints
---------
POST /reset     Reset environment, return observation + session_id
POST /step      Execute one action, return obs/reward/done/info
GET  /state     Read-only current state snapshot
GET  /tasks     Urgency-sorted task list with dependency graph
GET  /grader    Deterministic 8-dimension episode grade
GET  /baseline  Heuristic baseline across all 3 formal tasks
GET  /timeline  Full per-step episode timeline
GET  /demo      One-shot full episode (no stepping needed)
GET  /health    Liveness probe



Run with:
    uvicorn main:app --host 0.0.0.0 --port 7860
"""

import uuid
from dataclasses import asdict
from enum import Enum
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from api_models import (
    ActionRequest,
    GraderResponse,
    ObservationResponse,
    ResetRequest,
    ResetRequestModel,
    ActionRequestModel,
    StateResponse,
    TaskListResponse,
)
from environment import ProjectManagerEnv
from session_store import SessionStore
from baseline_runner import BaselineRunner
from timeline import EpisodeTimeline
from demo import PriorityAwareAgent



class ActionTypeEnum(str, Enum):
    assign_task    = "assign_task"
    unassign_task  = "unassign_task"
    reprioritize   = "reprioritize"
    rest_developer = "rest_developer"
    split_task     = "split_task"
    pair_program   = "pair_program"
    noop           = "noop"
    AUTO           = "auto"


class ResetResponse(BaseModel):
    status: str
    data: dict



app = FastAPI(
    title="Adaptive AI Project Manager",
    description=(
        "An OpenEnv-compliant RL environment simulating a realistic Agile sprint. "
        "An AI agent assigns developers to tasks, manages fatigue, responds to "
        "dynamic events, and maximises sprint delivery."
    ),
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

store = SessionStore()


def _without_status(payload: dict) -> dict:
    data = dict(payload)
    data.pop("status", None)
    return data



@app.get("/health")
def health():
    return {"status": "ok"}



@app.post("/openenv/reset")
def reset():
    session_id = str(uuid.uuid4())

    env = ProjectManagerEnv(
        scenario="medium",
        seed=42
    )

    obs = env.reset()

    store.put(session_id, env, EpisodeTimeline())

    return {
        "status": "success",
        "data": {
            "session_id": session_id,
            "observation": _without_status(
                ObservationResponse.from_obs(
                    obs, session_id=session_id
                ).to_dict()
            )
        }
    }



@app.post("/openenv/step")
def step(body: dict = {}):

    sessions = list(store.session_ids())
    if not sessions:
        raise HTTPException(400, "Call /openenv/reset first")

    session_id = sessions[0]
    env = store.get(session_id)

    obs = env.state


    return {
        "status": "success",
        "data": {
            "session_id": session_id,
            "observation": _without_status(
                ObservationResponse.from_obs(
                    obs, session_id=session_id
                ).to_dict()
            ),
            "reward": 0,
            "done": False,
            "info": {}
        }
    }


@app.get("/", response_class=HTMLResponse, tags=["meta"])
def home():
    return """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Adaptive AI Project Manager</title>
<style>
body{margin:0;font-family:-apple-system,sans-serif;background:#f9fafb;color:#111827}
.c{max-width:800px;margin:100px auto;text-align:center}
h1{font-size:32px;font-weight:600}
p{color:#6b7280;margin:10px 0 30px}
.card{background:#fff;padding:36px;border-radius:14px;border:1px solid #e5e7eb}
button{background:#2563eb;color:#fff;border:none;padding:12px 24px;
       border-radius:8px;cursor:pointer;font-size:15px}
button:hover{background:#1d4ed8}
.foot{margin-top:30px;color:#9ca3af;font-size:13px}
</style></head><body>
<div class="c">
  <h1>Adaptive AI Project Manager</h1>
  <p>OpenEnv-compliant Agile Sprint Simulation · RL Environment</p>
  <div class="card">
    <p>AI agent manages sprint tasks, developers, fatigue and dynamic events.</p>
    <button onclick="location.href='/docs'">Open API Docs</button>
  </div>
  <div class="foot">FastAPI · OpenEnv Hackathon · Port 7860</div>
</div></body></html>"""


@app.get("/health", tags=["meta"])
def health() -> dict:
    """Liveness probe."""
    return {"status": "success", "data": {"version": "2.1.0"}}


@app.post("/reset", response_model=ResetResponse, tags=["openenv"])
def reset(body: Optional[ResetRequestModel] = Body(default=None)) -> dict:
    """
    Initialise a new episode.

    Body (all optional):
    - scenario: "easy" | "medium" | "hard"  (default: "medium")
    - seed: int  (default: 42)
    - session_id: str  (omit to auto-generate)

    Returns session_id + initial observation.
    """
    if body is None:
        body = ResetRequestModel()

    dc = body.to_dc()
    session_id = (
        dc.session_id
        if dc.session_id and dc.session_id not in ("string", "", None)
        else str(uuid.uuid4())
    )
    env = ProjectManagerEnv(scenario=dc.scenario, seed=dc.seed)
    obs = env.reset()
    store.put(session_id, env, EpisodeTimeline())
    return {
        "status": "success",
        "data": {
            "session_id": session_id,
            "observation": _without_status(
                ObservationResponse.from_obs(obs, session_id=session_id).to_dict()
            ),
            "hint": {
                "next_endpoint": "POST /step",
                "example_action": {"session_id": session_id, "action_type": "noop"},
            },
        },
    }



@app.post("/step", response_model=dict, tags=["openenv"])
def step(body: ActionRequestModel) -> dict:
    """
    Execute one action and advance the simulation by one step.

    Body fields:
    - session_id        (required) from /reset
    - action_type       (required) assign_task | unassign_task | reprioritize |
                                    rest_developer | split_task | pair_program | noop | auto
    - task_id           for assign/unassign/reprioritize/split/pair
    - developer_id      for assign/unassign/rest
    - new_priority      LOW | MEDIUM | HIGH | CRITICAL  (reprioritize)
    - split_ratio       0.2–0.8  (split_task)
    - primary_developer_id / secondary_developer_id  (pair_program)
    """
    dc = body.to_dc()
    env = store.get(dc.session_id)

    if env is None:
        raise HTTPException(404, "Session not found. Call POST /reset first.")

    if dc.action_type == "auto":
        obs_now = env.state
        task_obj = next((t for t in obs_now.tasks if t.id == dc.task_id), None)
        if task_obj:
            for dev in obs_now.developers:
                if dev.available and not dev.current_tasks and dev.fatigue < 0.85:
                    if all(s in dev.skills for s in task_obj.required_skills):
                        dc.developer_id = dev.id
                        break
            if not dc.developer_id:
                for dev in obs_now.developers:
                    if dev.available and dev.fatigue < 0.90:
                        dc.developer_id = dev.id
                        break
        dc.action_type = "assign_task"

    try:
        action = dc.to_action()
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    timeline = store.get_timeline(dc.session_id)

    try:
        obs = env.step(action)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc

    if timeline is not None:
        timeline.record(
            obs,
            action={
                "type":                   str(action.action_type).split(".")[-1].lower(),
                "task_id":                getattr(action.assign,       "task_id",               None),
                "developer_id":           getattr(action.assign,       "developer_id",          None),
                "primary_developer_id":   getattr(action.pair,         "primary_developer_id",  None),
                "secondary_developer_id": getattr(action.pair,         "secondary_developer_id",None),
                "new_priority":           getattr(action.reprioritize,  "new_priority",          None),
                "split_ratio":            getattr(action.split,         "split_ratio",           None),
            },
        )

    store.put(dc.session_id, env, timeline)
    return {
        "status": "success",
        "data": {
            "session_id": dc.session_id,
            "observation": _without_status(
                ObservationResponse.from_obs(obs, session_id=dc.session_id).to_dict()
            ),
        },
    }



@app.get("/state", response_model=dict, tags=["openenv"])
def state(session_id: str = Query(..., description="Session UUID from /reset")) -> dict:
    """Read-only state snapshot. No side effects."""
    env = store.get(session_id)
    if env is None:
        raise HTTPException(404, "Session not found.")
    obs = env.state
    return {
        "status": "success",
        "data": {
            "session_id": session_id,
            "observation": _without_status(
                StateResponse.from_obs(obs, session_id=session_id).to_dict()
            ),
        },
    }


@app.get("/tasks", response_model=dict, tags=["openenv"])
def tasks(session_id: str = Query(..., description="Session UUID from /reset")) -> dict:
    """Urgency-sorted task list with dependency graph."""
    env = store.get(session_id)
    if env is None:
        raise HTTPException(404, "Session not found.")
    obs = env.state
    return {
        "status": "success",
        "data": {
            **_without_status(asdict(TaskListResponse.from_obs(obs, session_id=session_id))),
        },
    }


@app.get("/grader", response_model=dict, tags=["openenv"])
def grader(session_id: str = Query(..., description="Session UUID from /reset")) -> dict:
    """
    Deterministic 8-dimension grade. Requires done=true.
    Dimensions: delivery, value, timeliness, priority, team_health,
                adaptability, efficiency, dependency.
    """
    env = store.get(session_id)
    if env is None:
        raise HTTPException(404, "Session not found.")
    try:
        breakdown = env.grade()
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc

    return {
        "status": "success",
        "data": {
            **_without_status(
                asdict(GraderResponse.from_breakdown(breakdown, session_id=session_id))
            ),
        },
    }



@app.get("/baseline", response_model=dict, tags=["openenv"])
def baseline(seed: int = Query(42, description="RNG seed")) -> dict:
    """Run PriorityAwareAgent baseline across all 3 tasks. Fully deterministic."""
    runner = BaselineRunner(seed=seed)
    result = runner.run_all()
    return {"status": "success", "data": _without_status(asdict(result))}



@app.get("/timeline", tags=["openenv"])
def timeline(session_id: str = Query(..., description="Session UUID from /reset")) -> dict:
    """Per-step episode timeline. Grows with every /step call."""
    env = store.get(session_id)
    if env is None:
        raise HTTPException(404, "Session not found.")
    tl = store.get_timeline(session_id)
    if tl is None:
        raise HTTPException(404, "No timeline for this session.")
    data = tl.to_json()
    return {
        "status": "success",
        "data": [
            {
                "step":    s.get("step"),
                "action":  (s.get("action") or {}).get("type", "noop"),
                "reward":  s.get("reward"),
                "summary": s.get("state_summary"),
            }
            for s in data.get("steps", [])
        ],
    }



@app.get("/demo", tags=["meta"])
def demo(
    scenario: str = Query("medium", description="easy | medium | hard"),
    seed:     int = Query(42,       description="RNG seed"),
) -> dict:
    """
    One-shot complete episode with PriorityAwareAgent.
    No session needed. Returns grade + timeline.
    Perfect for judges and for inference.py testing.
    """
    if not scenario or scenario.strip() == "":
        scenario = "medium"

    env   = ProjectManagerEnv(scenario=scenario, seed=seed)
    agent = PriorityAwareAgent()
    tl    = EpisodeTimeline()
    obs   = env.reset()

    while not obs.done:
        action = agent.act(obs)
        obs    = env.step(action)
        tl.record(obs)

    bd = env.grade()

    return {
        "status": "success",
        "data": {
            "scenario":       scenario,
            "seed":           seed,
            "steps_taken":    obs.step,
            "grade":          bd.grade,
            "weighted_total": round(bd.weighted_total, 6),
            "dimensions": {
                "delivery":     round(bd.delivery_score,     4),
                "value":        round(bd.value_score,        4),
                "timeliness":   round(bd.timeliness_score,   4),
                "priority":     round(bd.priority_score,     4),
                "team_health":  round(bd.team_health_score,  4),
                "adaptability": round(bd.adaptability_score, 4),
                "efficiency":   round(bd.efficiency_score,   4),
                "dependency":   round(bd.dependency_score,   4),
            },
            "score_report":   bd.report(),
            "timeline":       tl.to_json(),
            "ascii_timeline": tl.to_ascii(),
        },
    }
