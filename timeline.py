"""
timeline.py — Episode timeline recorder for the Adaptive AI Project Manager.
Usage
-----
    timeline = EpisodeTimeline()
    obs = env.reset()
    while not obs.done:
        obs = env.step(agent.act(obs))
        timeline.record(obs)          # <-- one call per step

    print(timeline.to_ascii())        # pretty terminal output
    data = timeline.to_json()         # structured dict for the API
"""

from __future__ import annotations

from dataclasses import dataclass, field




@dataclass
class StepSnapshot:
    """Compact record of the environment state at one step."""
    step: int
    reward: float
    cumulative_reward: float


    ready:       int = 0
    in_progress: int = 0
    blocked:     int = 0
    completed:   int = 0
    failed:      int = 0


    assignments: dict = field(default_factory=dict)


    events: list = field(default_factory=list)

    delivery_pct: float = 0.0


    action_outcome: str  = ""
    reward_signal:  str  = "neutral"
    urgent_warning: str | None = None
    action: dict = field(default_factory=dict)
    state_change: dict = field(default_factory=dict)




class EpisodeTimeline:
    """
    Records and renders the complete trajectory of one episode.
    """

    def __init__(self) -> None:
        self._snapshots: list[StepSnapshot] = []
        self._cumulative: float = 0.0



    def record(self, obs, action: dict | None = None) -> None:
        """Capture a lightweight snapshot from the observation."""
        from models import TaskStatus

        self._cumulative += obs.reward


        counts = {s.value: 0 for s in TaskStatus}
        for t in obs.tasks:
            counts[t.status.value] = counts.get(t.status.value, 0) + 1


        dev_map = {d.id: d.name for d in obs.developers}
        assignments: dict[str, list] = {}
        for t in obs.tasks:
            if t.assigned_to:
                for did in t.assigned_to:
                    name = dev_map.get(did, did)
                    assignments.setdefault(name, []).append(t.name)


        m = obs.metrics
        delivery_pct = (
            m.delivered_story_points / m.total_story_points * 100
            if m.total_story_points > 0 else 0.0
        )


        events = [ev.description for ev in obs.recent_events]


        decision_reason = obs.info.get("decision_reason", "")
        details = obs.info.get("decision_details", {})
        if isinstance(decision_reason, str):
            action_outcome = decision_reason or obs.info.get("step_summary", "")
        else:
            action_outcome = details.get("action_outcome", obs.info.get("step_summary", ""))
        reward_signal  = details.get("reward_signal", "neutral")
        urgent_warning = details.get("urgent_warning")

        prev = self._snapshots[-1] if self._snapshots else None
        current_counts = {
            "ready": counts.get("ready", 0),
            "in_progress": counts.get("in_progress", 0),
            "blocked": counts.get("blocked", 0),
            "completed": counts.get("completed", 0),
            "failed": counts.get("failed", 0),
        }
        if prev is None:

            state_change = {
                "task_counts_delta": {k: current_counts[k] for k in current_counts},
                "delivery_pct_delta": round(delivery_pct, 1),
                "events_count_delta": len(events),
            }
        else:
            prev_counts = {
                "ready": prev.ready,
                "in_progress": prev.in_progress,
                "blocked": prev.blocked,
                "completed": prev.completed,
                "failed": prev.failed,
            }
            state_change = {
                "task_counts_delta": {
                    k: current_counts[k] - prev_counts[k] for k in current_counts
                },
                "delivery_pct_delta": round(delivery_pct - prev.delivery_pct, 1),
                "events_count_delta": len(events) - len(prev.events),
            }

        action_payload = action or obs.info.get("action_echo", {})

        self._snapshots.append(StepSnapshot(
            step               = obs.step,
            reward             = round(obs.reward, 3),
            cumulative_reward  = round(self._cumulative, 3),
            ready              = counts.get("ready", 0),
            in_progress        = counts.get("in_progress", 0),
            blocked            = counts.get("blocked", 0),
            completed          = counts.get("completed", 0),
            failed             = counts.get("failed", 0),
            assignments        = assignments,
            events             = events,
            delivery_pct       = round(delivery_pct, 1),
            action_outcome     = action_outcome,
            reward_signal      = reward_signal,
            urgent_warning     = urgent_warning,
            action             = action_payload,
            state_change       = state_change,
        ))



    def to_json(self) -> dict:
        """Return the full timeline as a serialisable dict."""
        return {
            "episode_steps": len(self._snapshots),
            "total_reward":  round(self._cumulative, 4),
            "final_delivery_pct": (
                self._snapshots[-1].delivery_pct if self._snapshots else 0.0
            ),
            "reward_sparkline": self._sparkline(
                [s.reward for s in self._snapshots]
            ),
            "cumulative_sparkline": self._sparkline(
                [s.cumulative_reward for s in self._snapshots]
            ),
            "steps": [self._snapshot_to_dict(s) for s in self._snapshots],
        }

    def _snapshot_to_dict(self, s: StepSnapshot) -> dict:
        state_summary = (
            f"{s.completed} completed, {s.in_progress} in progress, "
            f"{s.ready} ready, {s.blocked} blocked, {s.failed} failed"
        )
        return {
            "step":               s.step,
            "reward":             s.reward,
            "cumulative_reward":  s.cumulative_reward,
            "reward_signal":      s.reward_signal,
            "task_counts": {
                "ready":        s.ready,
                "in_progress":  s.in_progress,
                "blocked":      s.blocked,
                "completed":    s.completed,
                "failed":       s.failed,
            },
            "assignments":    s.assignments,
            "events":         s.events,
            "delivery_pct":   s.delivery_pct,
            "action":         s.action,
            "state_change":   s.state_change,
            "state_summary":  state_summary,
            "action_outcome": s.action_outcome,
            "urgent_warning": s.urgent_warning,
        }



    def to_ascii(self) -> str:
        """
        Render a compact ASCII timeline suitable for terminal output or README.
        """
        if not self._snapshots:
            return "(empty timeline — record() was never called)"

        lines = [
            "",
            "═" * 72,
            "  EPISODE TIMELINE",
            "═" * 72,
            f"  {'Step':<6} {'Tasks':^18} {'Reward':>8}  {'Cumul':>8}  Notes",
            "─" * 72,
        ]

        for s in self._snapshots:
            task_bar  = self._task_bar(s)
            rew_str   = f"{s.reward:+.3f}"
            cum_str   = f"{s.cumulative_reward:+.3f}"
            event_str = "  🔔 " + ", ".join(s.events) if s.events else ""
            warn_str  = "  ⚠ " + s.urgent_warning if s.urgent_warning else ""
            notes     = event_str + warn_str

            lines.append(
                f"  {s.step:<6} {task_bar:<18} {rew_str:>8}  {cum_str:>8} {notes}"
            )


        last = self._snapshots[-1]
        lines += [
            "─" * 72,
            f"  Total steps: {last.step}  |  "
            f"Cumulative reward: {self._cumulative:+.3f}  |  "
            f"Delivery: {last.delivery_pct:.1f}%",
            f"  Reward sparkline: {self._sparkline([s.reward for s in self._snapshots])}",
            "═" * 72,
            "",
        ]
        return "\n".join(lines)



    @staticmethod
    def _task_bar(s: StepSnapshot) -> str:
        """
        Compact representation of task states:
        """
        symbols = (
            "✓" * s.completed
            + "→" * s.in_progress
            + "○" * s.ready
            + "·" * s.blocked
            + "✗" * s.failed
        )
        return symbols[:18] or "·"

    @staticmethod
    def _sparkline(values: list) -> str:
        """
        Convert a list of floats to a Unicode sparkline string.
        """
        if not values:
            return ""
        bars = "▁▂▃▄▅▆▇█"
        mn, mx = min(values), max(values)
        span = mx - mn if mx != mn else 1.0
        result = []
        for v in values:
            if v < 0:
                result.append("▁")
            else:
                idx = int((v - max(0.0, mn)) / span * (len(bars) - 1))
                result.append(bars[min(idx, len(bars) - 1)])
        return "".join(result)
