import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass, field

from config import Config

SESSIONS_DIR = os.path.join(Config.DATA_DIR, "sessions")


@dataclass
class Session:
    session_id: str
    repo_url: str = ""
    local_path: str = ""
    repo_name: str = ""
    conversation: list[dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    def add_turn(self, role: str, content: str) -> None:
        self.conversation.append({"role": role, "content": content})
        self.last_active = time.time()
        # Keep last 40 turns — KV cache keeps Mistral's attention hot
        if len(self.conversation) > 40:
            self.conversation = self.conversation[-40:]

    def has_repo(self) -> bool:
        return bool(self.local_path)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        return cls(**data)


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        self._load_all()

    def _session_path(self, session_id: str) -> str:
        # Sanitize session_id for filesystem safety
        safe_id = "".join(c for c in session_id if c.isalnum() or c in "-_")
        return os.path.join(SESSIONS_DIR, f"{safe_id}.json")

    def _load_all(self) -> None:
        """Load all persisted sessions from disk on startup."""
        for fname in os.listdir(SESSIONS_DIR):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(SESSIONS_DIR, fname)
            try:
                with open(path) as f:
                    data = json.load(f)
                session = Session.from_dict(data)
                self._sessions[session.session_id] = session
            except Exception:
                pass

    def _save(self, session: Session) -> None:
        """Persist session to disk atomically."""
        path = self._session_path(session.session_id)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(session.to_dict(), f, indent=2)
        os.replace(tmp, path)

    async def get_or_create(self, session_id: str) -> Session:
        async with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = Session(session_id=session_id)
                self._save(self._sessions[session_id])
            return self._sessions[session_id]

    async def get(self, session_id: str) -> Session | None:
        async with self._lock:
            return self._sessions.get(session_id)

    async def add_turn(self, session_id: str, role: str, content: str) -> None:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.add_turn(role, content)
                self._save(session)

    async def set_repo(
        self, session_id: str, url: str, local_path: str, repo_name: str
    ) -> None:
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                session = Session(session_id=session_id)
                self._sessions[session_id] = session
            session.repo_url = url
            session.local_path = local_path
            session.repo_name = repo_name
            session.last_active = time.time()
            self._save(session)

    async def list_sessions(self) -> list[dict]:
        async with self._lock:
            return [
                {
                    "session_id": s.session_id,
                    "repo": s.repo_name or "no repo",
                    "turns": len(s.conversation),
                    "last_active": s.last_active,
                }
                for s in sorted(
                    self._sessions.values(),
                    key=lambda s: s.last_active,
                    reverse=True,
                )
            ]

    async def cleanup_stale(self, max_age_hours: float = 72.0) -> int:
        """Remove sessions inactive for more than max_age_hours. Default 72h."""
        cutoff = time.time() - (max_age_hours * 3600)
        async with self._lock:
            stale = [
                sid for sid, s in self._sessions.items()
                if s.last_active < cutoff
            ]
            for sid in stale:
                del self._sessions[sid]
                path = self._session_path(sid)
                if os.path.exists(path):
                    os.unlink(path)
            return len(stale)


# Global singleton — loaded from disk on import
store = SessionStore()
