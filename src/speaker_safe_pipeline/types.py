"""Typed interfaces used by the model-agnostic orchestration layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence


class OverlapDetector(Protocol):
    def has_overlap(self, audio: Path) -> bool: ...


class Separator(Protocol):
    def separate(self, audio: Path, output_dir: Path) -> Sequence[Path]: ...


class Enhancer(Protocol):
    def enhance(self, audio: Path, output_path: Path) -> Path: ...


class SpeechRecognizer(Protocol):
    def transcribe(self, audio: Path, language: str) -> str: ...


class SpeakerVerifier(Protocol):
    def score(self, enrollment_audio: Path, test_audio: Path) -> float: ...


@dataclass(frozen=True)
class ModelBundle:
    overlap_detector: OverlapDetector
    separator: Separator
    enhancer: Enhancer
    wake_asr: SpeechRecognizer
    command_asr: SpeechRecognizer
    primary_speaker_verifier: SpeakerVerifier
    secondary_speaker_verifier: SpeakerVerifier


@dataclass(frozen=True)
class PipelineConfig:
    primary_speaker_threshold: float
    rescue_primary_threshold: float
    rescue_secondary_threshold: float
    require_strict_rescue_agreement: bool = True
    require_both_rescue_scores_to_improve: bool = True

    def __post_init__(self) -> None:
        for name in (
            "primary_speaker_threshold",
            "rescue_primary_threshold",
            "rescue_secondary_threshold",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be calibrated to a numeric value")


@dataclass(frozen=True)
class Sample:
    sample_id: str | int
    wake_audio: Path
    wake_text: str
    command_audio: Path
    language: str = "zh"


@dataclass
class PipelineResult:
    sample_id: str | int
    content: str
    accepted: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class AudioCandidate:
    candidate_id: str
    path: Path
    kind: str
    primary_score: float | None = None
    secondary_score: float | None = None
    transcript: str = ""
    keyword_score: float | None = None

