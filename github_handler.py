import json
import os
import subprocess
from urllib.parse import urlparse

from config import Config


def get_active() -> dict | None:
    path = Config.active_repo_path()
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def set_active(url: str, local_path: str, repo_name: str) -> None:
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    tmp = Config.active_repo_path() + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"url": url, "local_path": local_path, "repo_name": repo_name}, f)
    os.replace(tmp, Config.active_repo_path())


def _inject_token(url: str) -> str:
    if not Config.GITHUB_TOKEN:
        return url
    parsed = urlparse(url)
    return parsed._replace(
        netloc=f"{Config.GITHUB_TOKEN}@{parsed.netloc}"
    ).geturl()


def clone(url: str) -> dict:
    parsed = urlparse(url)
    repo_name = parsed.path.rstrip("/").split("/")[-1].removesuffix(".git")
    local_path = os.path.join(Config.WORKSPACE_DIR, repo_name)
    os.makedirs(Config.WORKSPACE_DIR, exist_ok=True)

    if os.path.exists(local_path):
        set_active(url, local_path, repo_name)
        return {"path": local_path, "status": "already_exists", "repo_name": repo_name}

    auth_url = _inject_token(url)
    result = subprocess.run(
        ["git", "clone", auth_url, local_path],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git clone failed: {result.stderr}")

    set_active(url, local_path, repo_name)
    return {"path": local_path, "status": "cloned", "repo_name": repo_name}


def commit(message: str, local_path: str) -> dict:
    path = local_path
    # Get remote URL from git config
    result = subprocess.run(
        ["git", "-C", path, "remote", "get-url", "origin"],
        capture_output=True, text=True,
    )
    url = result.stdout.strip()

    # Inject token into remote for push
    auth_url = _inject_token(url)
    subprocess.run(
        ["git", "-C", path, "remote", "set-url", "origin", auth_url],
        check=True, capture_output=True, text=True,
    )

    subprocess.run(
        ["git", "-C", path, "add", "-A"],
        check=True, capture_output=True, text=True,
    )

    result = subprocess.run(
        ["git", "-C", path, "commit", "-m", message],
        capture_output=True, text=True,
    )
    if result.returncode != 0 and "nothing to commit" not in result.stdout:
        raise RuntimeError(f"git commit failed: {result.stderr}")

    push = subprocess.run(
        ["git", "-C", path, "push"],
        capture_output=True, text=True,
    )
    if push.returncode != 0:
        raise RuntimeError(f"git push failed: {push.stderr}")

    sha = _get_head_sha(path)
    return {"sha": sha, "message": message}


def _get_head_sha(path: str) -> str:
    result = subprocess.run(
        ["git", "-C", path, "rev-parse", "HEAD"],
        capture_output=True, text=True,
    )
    return result.stdout.strip()
