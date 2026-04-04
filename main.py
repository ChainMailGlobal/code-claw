import asyncio
import json
import subprocess

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from claude_process import claude_manager
from config import Config
from executor import execute
from github_handler import clone, commit
from memory import flush_queue, promote_session, query_mag, write_bubble
from planner import plan_implementation
from router import route_instruction
from session_store import store
from tts_engine import synthesize
from vision import qwen_observe

app = FastAPI(title="code-claw")


@app.on_event("startup")
async def startup():
    asyncio.create_task(flush_queue())


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

async def verify_token(authorization: str = Header(default="")) -> None:
    if not Config.CODE_CLAW_SECRET:
        return
    if authorization != f"Bearer {Config.CODE_CLAW_SECRET}":
        raise HTTPException(status_code=401, detail="Unauthorized")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class VisionRequest(BaseModel):
    image_base64: str

class VisionResponse(BaseModel):
    description: str


class ExecuteRequest(BaseModel):
    session_id: str
    instruction: str
    image_base64: str | None = None  # set when Spectacles button pressed
    active_file: str | None = None   # "I'm in auth.py"

class ExecuteResponse(BaseModel):
    session_id: str
    action: str          # "executed" or "asked"
    text: str
    audio_base64: str


class SwarmTask(BaseModel):
    session_id: str
    instruction: str
    repo_url: str | None = None   # optional — clones fresh if provided
    active_file: str | None = None

class SwarmRequest(BaseModel):
    tasks: list[SwarmTask]
    image_base64: str | None = None  # shared visual context for all tasks

class SwarmTaskResult(BaseModel):
    session_id: str
    action: str
    text: str

class SwarmResponse(BaseModel):
    results: list[SwarmTaskResult]
    total: int
    succeeded: int


class RepoRequest(BaseModel):
    session_id: str
    url: str

class RepoResponse(BaseModel):
    session_id: str
    path: str
    status: str
    repo_name: str


class CommitRequest(BaseModel):
    session_id: str
    summary: str

class CommitResponse(BaseModel):
    session_id: str
    sha: str
    message: str


class SessionsResponse(BaseModel):
    sessions: list[dict]


# ---------------------------------------------------------------------------
# Repo context builder
# ---------------------------------------------------------------------------

def _build_repo_context(local_path: str, active_file: str | None) -> str:
    """Build a brief repo snapshot for Mistral context injection."""
    lines = []

    # Git diff --stat (recent changes)
    try:
        diff = subprocess.run(
            ["git", "-C", local_path, "diff", "--stat", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if diff.stdout.strip():
            lines.append(f"Recent changes:\n{diff.stdout.strip()[:500]}")
    except Exception:
        pass

    # File tree (top level only — keep it short)
    try:
        tree = subprocess.run(
            ["git", "-C", local_path, "ls-files", "--others", "--cached", "--exclude-standard"],
            capture_output=True, text=True, timeout=5
        )
        files = tree.stdout.strip().split("\n")[:30]
        lines.append(f"Files: {', '.join(files)}")
    except Exception:
        pass

    if active_file:
        lines.append(f"User is currently working in: {active_file}")

    return "\n".join(lines) if lines else ""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    sessions = await store.list_sessions()
    return {
        "status": "ok",
        "tts_engine": Config.TTS_ENGINE,
        "active_sessions": len(sessions),
        "sessions": sessions,
        "mistral_url": Config.MISTRAL_URL,
        "qwen_url": Config.QWEN_URL,
    }


@app.get("/sessions", response_model=SessionsResponse)
async def list_sessions(_: None = Depends(verify_token)):
    return SessionsResponse(sessions=await store.list_sessions())


@app.post("/vision", response_model=VisionResponse)
async def vision_endpoint(
    body: VisionRequest, _: None = Depends(verify_token)
):
    description = await qwen_observe(
        instruction="Describe this image.", image_base64=body.image_base64
    )
    return VisionResponse(description=description)


@app.post("/execute", response_model=ExecuteResponse)
async def execute_endpoint(
    body: ExecuteRequest, _: None = Depends(verify_token)
):
    loop = asyncio.get_running_loop()

    # 1. Get or create session
    session = await store.get_or_create(body.session_id)

    # 2. Git pull latest if repo is set
    if session.has_repo():
        try:
            await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    ["git", "-C", session.local_path, "pull", "--ff-only"],
                    capture_output=True, text=True, timeout=30
                )
            )
        except Exception:
            pass

    # 3. Build repo context for Mistral
    repo_context = None
    if session.has_repo():
        repo_context = await loop.run_in_executor(
            None, _build_repo_context, session.local_path, body.active_file
        )

    # 3b. MAG query — inject relevant memory into context
    mag_context = await query_mag(body.instruction)
    if mag_context:
        prefix = f"Memory context (prior work):\n{mag_context}"
        repo_context = f"{prefix}\n\n{repo_context}" if repo_context else prefix

    # 4. Mistral + Qwen listen in parallel
    visual_context = None
    mistral_task = route_instruction(
        body.instruction,
        visual_context=None,
        conversation=session.conversation,
        repo_context=repo_context,
    )
    qwen_task = qwen_observe(body.instruction, body.image_base64)
    routing, qwen_context = await asyncio.gather(mistral_task, qwen_task)

    # 5. Store user turn in conversation (persisted to disk)
    await store.add_turn(body.session_id, "user", body.instruction)

    # 6. Decision loop — ask or execute
    if routing.get("action") == "ask":
        question = routing.get("question", "Can you clarify what you'd like to do?")
        await store.add_turn(body.session_id, "assistant", f"[asked]: {question}")

        audio_b64 = await loop.run_in_executor(None, synthesize, question)
        return ExecuteResponse(
            session_id=body.session_id,
            action="asked",
            text=question,
            audio_base64=audio_b64,
        )

    # 7. Token Factory planning pass — Mistral classifies, TF model writes spec
    mistral_prompt = routing.get("prompt", body.instruction)
    plan = await plan_implementation(
        mistral_prompt,
        session.conversation,
        repo_context,
        qwen_context,
    )

    if plan.get("action") == "ask":
        question = plan["question"]
        await store.add_turn(body.session_id, "assistant", f"[planner-asked]: {question}")
        audio_b64 = await loop.run_in_executor(None, synthesize, question)
        return ExecuteResponse(
            session_id=body.session_id,
            action="asked",
            text=question,
            audio_base64=audio_b64,
        )

    # "direct" uses Mistral prompt as-is; "plan" uses Token Factory spec
    claude_prompt = plan.get("spec", mistral_prompt) if plan.get("action") == "plan" else mistral_prompt

    # 8. Execute Claude Code (blocking → threadpool)
    repo_path = session.local_path if session.has_repo() else Config.WORKSPACE_DIR
    result = await loop.run_in_executor(None, execute, claude_prompt, repo_path)

    # 9. Store assistant turn (persisted to disk)
    await store.add_turn(body.session_id, "assistant", f"[executed]: {result['output'][:500]}")

    # 9b. Write bubble to MMCP memory
    asyncio.create_task(write_bubble(
        session_id=body.session_id,
        content=result["output"],
        source=body.instruction,
        context=repo_context or "",
        repo_name=session.repo_name or "",
        tier=plan.get("tier", "direct"),
    ))

    # 10. Build spoken response
    if result["returncode"] == 0:
        spoken = f"Done. {result['output']}"
    else:
        spoken = f"Error. {result['output']}"

    # 11. TTS
    audio_b64 = await loop.run_in_executor(None, synthesize, spoken)

    return ExecuteResponse(
        session_id=body.session_id,
        action="executed",
        text=spoken,
        audio_base64=audio_b64,
    )


@app.post("/execute/stream")
async def execute_stream_endpoint(
    body: ExecuteRequest, _: None = Depends(verify_token)
):
    """
    Streaming version of /execute.
    Returns Server-Sent Events — each chunk is a JSON line:
      {"type": "chunk", "text": "..."}         — Claude Code output chunk
      {"type": "question", "text": "..."}      — Mistral needs clarification
      {"type": "audio", "audio_base64": "..."}  — final TTS audio
      {"type": "done"}                          — stream complete
    Lens Studio reads these and appends to HUD in real time.
    """
    loop = asyncio.get_running_loop()
    session = await store.get_or_create(body.session_id)

    # Git pull latest
    if session.has_repo():
        try:
            await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    ["git", "-C", session.local_path, "pull", "--ff-only"],
                    capture_output=True, text=True, timeout=30
                )
            )
        except Exception:
            pass

    # Repo context
    repo_context = None
    if session.has_repo():
        repo_context = await loop.run_in_executor(
            None, _build_repo_context, session.local_path, body.active_file
        )

    # MAG query — inject relevant memory
    mag_context = await query_mag(body.instruction)
    if mag_context:
        prefix = f"Memory context (prior work):\n{mag_context}"
        repo_context = f"{prefix}\n\n{repo_context}" if repo_context else prefix

    # Mistral + Qwen in parallel
    routing, qwen_context = await asyncio.gather(
        route_instruction(
            body.instruction,
            visual_context=None,
            conversation=session.conversation,
            repo_context=repo_context,
        ),
        qwen_observe(body.instruction, body.image_base64),
    )

    await store.add_turn(body.session_id, "user", body.instruction)

    async def event_stream():
        # Mistral asks for clarification
        if routing.get("action") == "ask":
            question = routing.get("question", "Can you clarify?")
            await store.add_turn(body.session_id, "assistant", f"[asked]: {question}")
            yield f"data: {json.dumps({'type': 'question', 'text': question})}\n\n"
            audio_b64 = await loop.run_in_executor(None, synthesize, question)
            yield f"data: {json.dumps({'type': 'audio', 'audio_base64': audio_b64})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        # Token Factory planning pass
        mistral_prompt = routing.get("prompt", body.instruction)
        plan = await plan_implementation(
            mistral_prompt,
            session.conversation,
            repo_context,
            qwen_context,
        )

        if plan.get("action") == "ask":
            question = plan["question"]
            await store.add_turn(body.session_id, "assistant", f"[planner-asked]: {question}")
            yield f"data: {json.dumps({'type': 'question', 'text': question})}\n\n"
            audio_b64 = await loop.run_in_executor(None, synthesize, question)
            yield f"data: {json.dumps({'type': 'audio', 'audio_base64': audio_b64})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        claude_prompt = plan.get("spec", mistral_prompt) if plan.get("action") == "plan" else mistral_prompt
        repo_path = session.local_path if session.has_repo() else Config.WORKSPACE_DIR

        # Stream Claude Code output chunk by chunk → HUD
        full_output = []
        async for chunk in claude_manager.execute_streaming(claude_prompt, repo_path):
            full_output.append(chunk)
            yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"

        full_text = "".join(full_output)
        await store.add_turn(body.session_id, "assistant", f"[executed]: {full_text[:500]}")

        # Write bubble to MMCP memory
        asyncio.create_task(write_bubble(
            session_id=body.session_id,
            content=full_text,
            source=body.instruction,
            context=repo_context or "",
            repo_name=session.repo_name or "",
            tier=plan.get("tier", "direct"),
        ))

        # TTS of summary
        spoken = f"Done. {full_text}" if full_text else "Done."
        audio_b64 = await loop.run_in_executor(None, synthesize, spoken)
        yield f"data: {json.dumps({'type': 'audio', 'audio_base64': audio_b64})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )


@app.post("/execute/swarm", response_model=SwarmResponse)
async def execute_swarm_endpoint(
    body: SwarmRequest, _: None = Depends(verify_token)
):
    """
    Parallel swarm execution — up to 100 tasks simultaneously.
    Each task runs the full pipeline: Mistral → planner → Claude Code.
    All tasks run concurrently via asyncio.gather().
    """
    loop = asyncio.get_running_loop()

    async def run_task(task: SwarmTask) -> SwarmTaskResult:
        try:
            session = await store.get_or_create(task.session_id)

            # Git pull if repo set
            if session.has_repo():
                try:
                    await loop.run_in_executor(
                        None,
                        lambda: __import__("subprocess").run(
                            ["git", "-C", session.local_path, "pull", "--ff-only"],
                            capture_output=True, text=True, timeout=30,
                        )
                    )
                except Exception:
                    pass

            repo_context = None
            if session.has_repo():
                repo_context = await loop.run_in_executor(
                    None, _build_repo_context, session.local_path, task.active_file
                )

            # Mistral routing + Qwen vision in parallel
            routing, qwen_context = await asyncio.gather(
                route_instruction(
                    task.instruction,
                    visual_context=None,
                    conversation=session.conversation,
                    repo_context=repo_context,
                ),
                qwen_observe(task.instruction, body.image_base64),
            )

            await store.add_turn(task.session_id, "user", task.instruction)

            if routing.get("action") == "ask":
                question = routing.get("question", "Can you clarify?")
                await store.add_turn(task.session_id, "assistant", f"[asked]: {question}")
                return SwarmTaskResult(session_id=task.session_id, action="asked", text=question)

            mistral_prompt = routing.get("prompt", task.instruction)
            plan = await plan_implementation(
                mistral_prompt, session.conversation, repo_context, qwen_context
            )

            if plan.get("action") == "ask":
                question = plan["question"]
                await store.add_turn(task.session_id, "assistant", f"[planner-asked]: {question}")
                return SwarmTaskResult(session_id=task.session_id, action="asked", text=question)

            claude_prompt = plan.get("spec", mistral_prompt) if plan.get("action") == "plan" else mistral_prompt
            repo_path = session.local_path if session.has_repo() else Config.WORKSPACE_DIR

            result = await loop.run_in_executor(None, execute, claude_prompt, repo_path)
            output = result["output"]
            await store.add_turn(task.session_id, "assistant", f"[executed]: {output[:500]}")

            status = "Done." if result["returncode"] == 0 else f"Error. {output[:200]}"
            return SwarmTaskResult(session_id=task.session_id, action="executed", text=status)

        except Exception as e:
            return SwarmTaskResult(session_id=task.session_id, action="error", text=str(e))

    results = await asyncio.gather(*[run_task(t) for t in body.tasks])
    succeeded = sum(1 for r in results if r.action == "executed")
    return SwarmResponse(results=list(results), total=len(results), succeeded=succeeded)


@app.post("/repo", response_model=RepoResponse)
async def repo_endpoint(
    body: RepoRequest, _: None = Depends(verify_token)
):
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, clone, body.url)
    await store.set_repo(
        body.session_id,
        body.url,
        result["path"],
        result["repo_name"],
    )
    return RepoResponse(
        session_id=body.session_id,
        path=result["path"],
        status=result["status"],
        repo_name=result["repo_name"],
    )


@app.post("/commit", response_model=CommitResponse)
async def commit_endpoint(
    body: CommitRequest, _: None = Depends(verify_token)
):
    loop = asyncio.get_running_loop()
    session = await store.get_or_create(body.session_id)
    if not session.has_repo():
        raise HTTPException(status_code=400, detail="No repo set for this session. Call POST /repo first.")
    result = await loop.run_in_executor(None, commit, body.summary, session.local_path)
    # Promote session bubbles → marble on every commit
    asyncio.create_task(promote_session(body.session_id))
    return CommitResponse(
        session_id=body.session_id,
        sha=result["sha"],
        message=result["message"],
    )
