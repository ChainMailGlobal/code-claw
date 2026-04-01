"""
Token Factory planning stage.

Mistral classifies the task complexity. Based on that:
  - direct    → skip planner, Claude Code gets Mistral's prompt as-is
  - coding    → DeepSeek-V3 writes surgical implementation spec
  - reasoning → Nemotron-253B reasons through architecture/design
  - thinking  → DeepSeek-R1 for deep chain-of-thought problems
  - ask       → surface one technical question to user before proceeding

All planning calls go to Token Factory API (no VRAM cost).
"""
import json

import httpx

from config import Config

_CLASSIFY_SYSTEM = """You are a task classifier for a voice-driven coding assistant.

Given a coding task, classify it into one of four tiers:

- "direct"    : simple, unambiguous, single-file (rename, add print, fix typo, small bug fix)
- "coding"    : multi-file feature, refactor, new module, API integration, anything surgical
- "reasoning" : architecture decisions, system design, choosing patterns, performance analysis
- "thinking"  : complex debugging, tracing subtle bugs across layers, deep reasoning needed

Also decide: does this need a clarifying question before ANY model touches it?
Only ask if acting wrong would cause real irreversible damage (drop table, delete branch, overwrite prod).

Output JSON only:
{"tier": "direct"|"coding"|"reasoning"|"thinking", "ask": true|false, "question": "..." }
question is only required if ask is true."""

_PLAN_SYSTEM = """You are an expert software engineer producing implementation specs.

You receive a coding task, conversation history, and repo context.
Your output will be executed directly by Claude Code CLI with full file access.

Write a precise, surgical implementation spec:
- Name exact files, functions, line numbers where relevant
- List steps in order
- Include edge cases and error handling expectations
- Flag any destructive operations explicitly

Be thorough. Claude Code will execute exactly what you write.
Output JSON only: {"spec": "step by step implementation..."}"""


async def plan_implementation(
    instruction: str,
    conversation: list[dict],
    repo_context: str | None,
    qwen_context: str | None = None,
) -> dict:
    """
    Returns one of:
      {"action": "direct"}                    — pass instruction straight to Claude Code
      {"action": "plan", "spec": "..."}       — enriched spec from Token Factory
      {"action": "ask", "question": "..."}    — needs user input first
    """
    # Step 1: Mistral classifies the task (local, fast, free)
    classification = await _classify(instruction, conversation, repo_context)

    if classification.get("ask"):
        return {"action": "ask", "question": classification.get("question", "Can you clarify?")}

    tier = classification.get("tier", "direct")

    if tier == "direct":
        return {"action": "direct"}

    # Step 2: Token Factory planning pass with appropriate model
    model = {
        "coding": Config.MODEL_CODING,
        "reasoning": Config.MODEL_REASONING,
        "thinking": Config.MODEL_THINKING,
    }.get(tier, Config.MODEL_CODING)

    spec = await _plan(instruction, conversation, repo_context, qwen_context, model)
    return {"action": "plan", "spec": spec, "model": model, "tier": tier}


async def _classify(
    instruction: str,
    conversation: list[dict],
    repo_context: str | None,
) -> dict:
    """Ask local Mistral to classify task tier. Falls back to 'coding' on error."""
    messages = [{"role": "system", "content": _CLASSIFY_SYSTEM}]
    for turn in conversation[-6:]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    user_parts = [f"Task: {instruction}"]
    if repo_context:
        user_parts.append(f"Repo context:\n{repo_context[:800]}")
    user_parts.append("Output JSON only.")
    messages.append({"role": "user", "content": "\n\n".join(user_parts)})

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                Config.mistral_chat_url(),
                json={
                    "model": Config.MISTRAL_MODEL,
                    "messages": messages,
                    "temperature": 0.0,
                    "max_tokens": 128,
                    "response_format": {"type": "json_object"},
                },
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            result = json.loads(content)
            if result.get("tier") in ("direct", "coding", "reasoning", "thinking"):
                return result
            return {"tier": "coding", "ask": False}
    except Exception:
        return {"tier": "coding", "ask": False}


async def _plan(
    instruction: str,
    conversation: list[dict],
    repo_context: str | None,
    qwen_context: str | None,
    model: str,
) -> str:
    """Call Token Factory with chosen model. Returns spec string."""
    messages = [{"role": "system", "content": _PLAN_SYSTEM}]
    for turn in conversation[-10:]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    user_parts = [f"Task: {instruction}"]
    if repo_context:
        user_parts.append(f"Repo context:\n{repo_context}")
    if qwen_context:
        user_parts.append(f"Visual context:\n{qwen_context}")
    user_parts.append("Output JSON only: {\"spec\": \"...\"}")
    messages.append({"role": "user", "content": "\n\n".join(user_parts)})

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                Config.token_factory_chat_url(),
                headers={"Authorization": f"Bearer {Config.TOKEN_FACTORY_API_KEY}"},
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": 0.1,
                    "max_tokens": 2048,
                    "response_format": {"type": "json_object"},
                },
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            result = json.loads(content)
            return result.get("spec", instruction)
    except Exception:
        # Token Factory unavailable — fall back to raw instruction
        return instruction
