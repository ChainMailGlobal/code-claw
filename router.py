import json
import re

import httpx

from config import Config

_SYSTEM_PROMPT = """You are a coding assistant working with a developer via Snap Spectacles voice interface. Your job is to:

1. Understand their voice instruction in the context of the conversation history
2. If the instruction is clear enough to act on — respond with JSON:
   {"action": "execute", "prompt": "...", "reasoning": "..."}
3. If you need clarification before Claude Code can act safely — respond with JSON:
   {"action": "ask", "question": "...", "reasoning": "..."}

Rules:
- prompt must be specific: filenames, function names, scope, intent
- Only ask ONE clarifying question maximum — don't interrogate
- If repo context is provided, use it to make the prompt more precise
- If visual context is provided, incorporate it
- Write prompts as direct instructions Claude Code can act on immediately
- Respond ONLY with valid JSON — no markdown, no explanation"""


def _build_messages(
    instruction: str,
    visual_context: str | None,
    conversation: list[dict],
    repo_context: str | None,
) -> list[dict]:
    system = _SYSTEM_PROMPT
    if repo_context:
        system += f"\n\nRepo context:\n{repo_context}"

    messages = [{"role": "system", "content": system}]

    # Inject conversation history
    messages.extend(conversation)

    # Current turn
    user_content = f"Voice instruction: {instruction}"
    if visual_context:
        user_content += f"\n\nVisual context: {visual_context}"
    messages.append({"role": "user", "content": user_content})

    return messages


def _parse_response(content: str) -> dict:
    content = content.strip()
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # Fallback: extract first JSON object
        match = re.search(r"\{[\s\S]*\}", content)
        if not match:
            raise ValueError(f"No JSON found in Mistral response: {content[:200]}")
        parsed = json.loads(match.group())

    if "prompt" not in parsed:
        raise ValueError(f"Missing 'prompt' key in response: {parsed}")
    return parsed


async def route_instruction(
    instruction: str,
    visual_context: str | None = None,
    conversation: list[dict] | None = None,
    repo_context: str | None = None,
) -> dict:
    """
    Send instruction to Mistral for intent parsing.
    Returns {"action": "execute"|"ask", "prompt"|"question": str, "reasoning": str}.
    """
    messages = _build_messages(
        instruction,
        visual_context,
        conversation or [],
        repo_context,
    )

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            Config.mistral_chat_url(),
            json={
                "model": Config.MISTRAL_MODEL,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 512,
                "response_format": {"type": "json_object"},
            },
        )
        resp.raise_for_status()

    content = resp.json()["choices"][0]["message"]["content"]
    parsed = _parse_response(content)

    # Normalize — if no action key, assume execute
    if "action" not in parsed:
        parsed["action"] = "execute"
    if parsed["action"] == "execute" and "prompt" not in parsed:
        parsed["prompt"] = instruction

    return parsed
