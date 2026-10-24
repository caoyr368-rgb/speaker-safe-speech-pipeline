"""Fail if common private, heavyweight, generated, or credential files are present."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".pytest_cache", "__pycache__", ".venv", "venv"}
BLOCKED_DIRS = {"data", "datasets", "models", "checkpoints", "outputs", "results", "runs"}
BLOCKED_SUFFIXES = {
    ".wav", ".mp3", ".flac", ".m4a", ".ogg",
    ".pt", ".pth", ".ckpt", ".safetensors", ".onnx",
    ".tar", ".gz", ".zip", ".7z",
    ".pem", ".key",
}
TEXT_SUFFIXES = {".py", ".md", ".json", ".toml", ".yml", ".yaml", ".txt", ""}
PRIVATE_PATTERNS = {
    "Windows user path": re.compile(r"[A-Za-z]:[\\/]Users[\\/]", re.IGNORECASE),
    "Unix home path": re.compile(r"/(?:home|root)/"),
    "email address": re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    "private key": re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY"),
    "access token": re.compile(r"\b(?:ghp_|github_pat_|hf_|sk-)[A-Za-z0-9_-]{12,}\b"),
}


def main() -> int:
    problems: list[str] = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in relative.parts):
            continue
        if path.is_dir():
            if path.name.casefold() in BLOCKED_DIRS:
                problems.append(f"blocked directory: {relative}")
            continue
        if path.suffix.casefold() in BLOCKED_SUFFIXES:
            problems.append(f"blocked file type: {relative}")
        if path.stat().st_size > 1_000_000:
            problems.append(f"file exceeds 1 MB: {relative}")
        if path.suffix.casefold() not in TEXT_SUFFIXES:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            problems.append(f"non-UTF-8 text file: {relative}")
            continue
        for label, pattern in PRIVATE_PATTERNS.items():
            if pattern.search(content):
                problems.append(f"{label} found in: {relative}")

    if problems:
        print("Pre-publish check failed:")
        for problem in sorted(set(problems)):
            print(f"- {problem}")
        return 1
    print("Pre-publish check passed: no blocked artifacts or common private values found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
