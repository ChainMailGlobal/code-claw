import subprocess

from config import Config


def execute(prompt: str, repo_path: str) -> dict:
    """
    Run Claude Code CLI against repo_path with the given prompt.
    Returns {"output": str, "returncode": int, "cli_used": str}.
    """
    cmd = [
        "claude",
        "--permission-mode", "bypassPermissions",
        "--print",
        prompt,
        "--allowedTools",
        "Edit,Write,Read,Bash,Glob,Grep",
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=Config.EXECUTOR_TIMEOUT,
        )
        output = _sanitize_output(result.stdout, result.stderr)
        return {"output": output, "returncode": result.returncode, "cli_used": "claude"}

    except subprocess.TimeoutExpired:
        return {
            "output": f"Claude Code timed out after {Config.EXECUTOR_TIMEOUT}s.",
            "returncode": -1,
            "cli_used": "claude",
        }
    except FileNotFoundError:
        return {
            "output": "Claude Code CLI not found. Is 'claude' installed on this machine?",
            "returncode": -1,
            "cli_used": "claude",
        }


def _sanitize_output(stdout: str, stderr: str) -> str:
    parts = []
    if stdout.strip():
        parts.append(stdout.strip())
    if stderr.strip():
        parts.append(f"[stderr] {stderr.strip()}")
    combined = "\n".join(parts)
    # Truncate to keep TTS response manageable
    if len(combined) > 8000:
        combined = combined[:8000] + "\n[output truncated]"
    return combined
