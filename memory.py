import asyncio
import json
import os
import time

import httpx

from config import Config

BUBBLE_QUEUE_DIR = os.path.join(Config.DATA_DIR, "bubble_queue")
os.makedirs(BUBBLE_QUEUE_DIR, exist_ok=True)

MMCP_URL = Config.MMCP_URL
OWNER_ACCOUNT_ID = Config.OWNER_ACCOUNT_ID


def _mmcp_headers() -> dict:
    h = {"Content-Type": "application/json"}
    if Config.MMCP_UI_TOKEN:
        h["Authorization"] = f"Bearer {Config.MMCP_UI_TOKEN}"
    return h


async def write_bubble(
    session_id: str,
    content: str,
    source: str,
    context: str,
    repo_name: str,
    tier: str,
) -> None:
    """Write execution bubble to MMCP. Falls back to disk queue if unreachable."""
    payload = {
        "session_id": session_id,
        "event_type": "CODE_CLAW_EXECUTION",
        "segment_type": "EXECUTION",
        "trust": "HIGH",
        "actor_aio": "AIO_CODE_CLAW",
        "payload": {
            "content": content[:2000],
            "source": source,
            "context": context[:500],
            "repo_name": repo_name,
            "tier": tier,
            "owner_account_id": OWNER_ACCOUNT_ID,
        },
    }
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.post(f"{MMCP_URL}/mmcp/bubbles/event", json=payload, headers=_mmcp_headers())
            resp.raise_for_status()
    except Exception:
        # Queue to disk for later flush
        path = os.path.join(BUBBLE_QUEUE_DIR, f"{int(time.time() * 1000)}.json")
        try:
            with open(path, "w") as f:
                json.dump(payload, f)
        except Exception:
            pass


async def query_mag(query: str, top_k: int = 5) -> str:
    """Query MAG for contextual memory. Returns answer_context string or empty."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.post(
                f"{MMCP_URL}/mmcp/mag/query",
                headers=_mmcp_headers(),
                json={
                    "query": query,
                    "top_k": top_k,
                    "actor_aio": "AIO_CODE_CLAW",
                    "envelope_id": "GE_BUILD",
                    "owner_account_id": OWNER_ACCOUNT_ID,
                },
            )
            resp.raise_for_status()
            return resp.json().get("answer_context", "")
    except Exception:
        return ""


async def flush_queue() -> int:
    """Flush all queued bubbles to MMCP. Returns count flushed."""
    flushed = 0
    try:
        filenames = [f for f in os.listdir(BUBBLE_QUEUE_DIR) if f.endswith(".json")]
    except Exception:
        return 0

    for fname in filenames:
        path = os.path.join(BUBBLE_QUEUE_DIR, fname)
        try:
            with open(path) as f:
                payload = json.load(f)
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.post(f"{MMCP_URL}/mmcp/bubbles/event", json=payload, headers=_mmcp_headers())
                resp.raise_for_status()
            os.unlink(path)
            flushed += 1
        except Exception:
            pass
    return flushed


async def promote_session(session_id: str) -> str | None:
    """Promote a session's bubbles to a marble. Returns marble_id or None."""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{MMCP_URL}/mmcp/marbles/promote",
                headers=_mmcp_headers(),
                json={"session_id": session_id},
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("marble_id") or (data.get("marbles") or [{}])[0].get("marble_id")
    except Exception:
        return None
