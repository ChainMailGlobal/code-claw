import httpx

from config import Config


def _build_messages(instruction: str, image_base64: str | None) -> list[dict]:
    if image_base64:
        content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
            },
            {
                "type": "text",
                "text": (
                    f"Voice instruction: {instruction}\n\n"
                    "Describe what you see in this image briefly and factually, "
                    "focusing on anything relevant to the instruction. "
                    "Then add any observations about the instruction itself."
                ),
            },
        ]
    else:
        content = (
            f"Voice instruction: {instruction}\n\n"
            "As a coding assistant, briefly share any observations or context "
            "about this instruction that could help execute it better."
        )

    return [{"role": "user", "content": content}]


async def qwen_observe(instruction: str, image_base64: str | None = None) -> str:
    """
    Qwen always receives the instruction text.
    If image_base64 provided, Qwen also sees the visual frame.
    Returns Qwen's observations as a string.
    """
    messages = _build_messages(instruction, image_base64)

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            Config.qwen_chat_url(),
            json={
                "model": Config.QWEN_MODEL,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 512,
            },
        )
        resp.raise_for_status()

    return resp.json()["choices"][0]["message"]["content"].strip()
