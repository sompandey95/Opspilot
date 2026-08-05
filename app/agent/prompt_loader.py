"""Versioned prompt loading.

`prompts/current.txt` is a symlink to the active version file
(`v1_system.txt`, `v2_system.txt`, …). The version string is derived from the
symlink target so it can flow into trace rows and eval runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"


@dataclass(frozen=True)
class SystemPrompt:
    text: str
    version: str


def load_current_prompt() -> SystemPrompt:
    current = PROMPTS_DIR / "current.txt"
    target_name = current.resolve().name          # e.g. "v1_system.txt"
    version = target_name.split("_", 1)[0]        # e.g. "v1"
    return SystemPrompt(text=current.read_text(encoding="utf-8").strip(), version=version)
