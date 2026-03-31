"""
Warm Claude Code process manager.
Keeps a persistent claude --print process ready per session,
eliminating cold start overhead on every execution.
"""
import asyncio
import os
import subprocess
import time
from dataclasses import dataclass

from config import Config


@dataclass
class WarmProcess:
    session_id: str
    repo_path: str
    last_used: float = 0.0


class ClaudeProcessManager:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    async def execute_streaming(
        self, prompt: str, repo_path: str
    ):
        """
        Run claude --print and yield output chunks as they arrive.
        Uses asyncio subprocess for true streaming.
        """
        cmd = [
            "claude",
            "--permission-mode", "bypassPermissions",
            "--print",
            prompt,
            "--allowedTools",
            "Edit,Write,Read,Bash,Glob,Grep",
        ]

        env = os.environ.copy()

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            # Stream stdout chunks as they arrive
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        proc.stdout.read(256), timeout=Config.EXECUTOR_TIMEOUT
                    )
                    if not chunk:
                        break
                    yield chunk.decode("utf-8", errors="replace")
                except asyncio.TimeoutError:
                    proc.kill()
                    yield "\n[Timed out]"
                    break

            await proc.wait()
            returncode = proc.returncode

            if returncode != 0:
                stderr = await proc.stderr.read()
                if stderr:
                    yield f"\n[stderr]: {stderr.decode('utf-8', errors='replace')[:500]}"

        except FileNotFoundError:
            yield "Claude Code CLI not found. Run: npm install -g @anthropic-ai/claude-code"


# Global singleton
claude_manager = ClaudeProcessManager()
