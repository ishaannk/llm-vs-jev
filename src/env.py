"""Keys from env only. Never logged, never written to disk."""

from __future__ import annotations

import os
from pathlib import Path

_ALIASES = {
    "TYPESAFE_API_KEY": ("TYPESAFE_API_KEY", "jev_api", "JEV_API"),
    "OPENAI_API_KEY": ("OPENAI_API_KEY", "openai_api_key"),
    "ANTHROPIC_API_KEY": ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY", "claude_api_key"),
    "ANTHROPIC_WORKSPACE_ID": ("ANTHROPIC_WORKSPACE_ID", "anthropic_workspace_id"),
}


def _load_dotenv() -> dict[str, str]:
    values: dict[str, str] = {}
    for root in (Path.cwd(), *Path.cwd().parents):
        path = root / ".env"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values.setdefault(key.strip(), value.strip().strip("\"'"))
        break
    return values


def key(name: str) -> str:
    dotenv = _load_dotenv()
    for alias in _ALIASES[name]:
        value = os.environ.get(alias) or dotenv.get(alias)
        if value:
            return value
    raise RuntimeError(f"{name} not set (env or .env); see .env.example")


def optional(name: str) -> str | None:
    try:
        return key(name)
    except RuntimeError:
        return None
