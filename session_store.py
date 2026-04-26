"""
session_store.py — In-memory session registry for the API server.

Maps session_id (UUID string) → (ProjectManagerEnv, EpisodeTimeline) pairs.
Thread-safe for single-worker uvicorn; add a Redis backend for multi-worker.
"""

from __future__ import annotations

import threading
from typing import Optional, Tuple

from environment import ProjectManagerEnv
from timeline import EpisodeTimeline


class SessionStore:
    """
    Thread-safe in-memory session registry.
    """

    def __init__(self, max_sessions: int = 512) -> None:

        self._store: dict[str, Tuple[ProjectManagerEnv, EpisodeTimeline]] = {}
        self._lock  = threading.Lock()
        self._max   = max_sessions

    def put(
        self,
        session_id: str,
        env: ProjectManagerEnv,
        timeline: EpisodeTimeline | None = None,
    ) -> None:
        """Store or overwrite a session. Creates a fresh timeline if not supplied."""
        with self._lock:
            if len(self._store) >= self._max and session_id not in self._store:
                # Evict the oldest entry (FIFO approximation)
                oldest = next(iter(self._store))
                del self._store[oldest]
            tl = timeline if timeline is not None else EpisodeTimeline()
            self._store[session_id] = (env, tl)

    def get(self, session_id: str) -> Optional[ProjectManagerEnv]:
        """Retrieve the env for a session, or None if it does not exist."""
        with self._lock:
            pair = self._store.get(session_id)
            return pair[0] if pair else None

    def get_timeline(self, session_id: str) -> Optional[EpisodeTimeline]:
        """Retrieve the timeline for a session, or None if it does not exist."""
        with self._lock:
            pair = self._store.get(session_id)
            return pair[1] if pair else None

    def delete(self, session_id: str) -> bool:
        """Remove a session. Returns True if it existed."""
        with self._lock:
            return self._store.pop(session_id, None) is not None

    def count(self) -> int:
        """Return number of active sessions."""
        with self._lock:
            return len(self._store)

    def session_ids(self) -> list[str]:
        """Return all active session IDs (for admin/debug)."""
        with self._lock:
            return list(self._store.keys())

