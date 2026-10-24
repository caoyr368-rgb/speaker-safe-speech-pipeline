"""Small text helpers for content-assisted wake candidate selection."""

from __future__ import annotations

import re
import unicodedata


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    return "".join(character for character in value if character.isalnum())


def edit_distance(left: str, right: str) -> int:
    if len(left) > len(right):
        left, right = right, left
    previous = list(range(len(left) + 1))
    for row_index, right_character in enumerate(right, start=1):
        current = [row_index]
        for column_index, left_character in enumerate(left, start=1):
            current.append(min(
                current[-1] + 1,
                previous[column_index] + 1,
                previous[column_index - 1] + (left_character != right_character),
            ))
        previous = current
    return previous[-1]


def keyword_similarity(transcript: str, keyword: str) -> float:
    observed = normalize_text(transcript)
    expected = normalize_text(keyword)
    if not expected:
        raise ValueError("wake_text must not be empty")
    if expected in observed:
        return 1.0
    distance = edit_distance(observed, expected)
    return max(0.0, 1.0 - distance / max(len(observed), len(expected), 1))


def infer_language(keyword: str, default: str) -> str:
    return "en" if re.search(r"[A-Za-z]", keyword) else default

