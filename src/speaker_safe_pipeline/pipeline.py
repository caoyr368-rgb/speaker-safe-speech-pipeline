"""Overlap-aware orchestration built from independently supplied open models."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Sequence

from .text import infer_language, keyword_similarity
from .types import AudioCandidate, ModelBundle, PipelineConfig, PipelineResult, Sample


class SpeakerSafePipeline:
    """Compose overlap detection, separation, ASR, and speaker verification.

    The class owns only routing and safety logic. Model loading, preprocessing,
    weight acquisition, and device placement stay inside user-provided adapters.
    """

    def __init__(self, models: ModelBundle, config: PipelineConfig) -> None:
        self.models = models
        self.config = config

    def run(self, sample: Sample) -> PipelineResult:
        self._validate_sample(sample)
        with tempfile.TemporaryDirectory(prefix="speaker_safe_pipeline_") as temp:
            workdir = Path(temp)
            wake_overlap = self.models.overlap_detector.has_overlap(sample.wake_audio)
            wake_candidates = self._build_candidates(
                sample.wake_audio, wake_overlap, workdir / "wake", "wake"
            )
            enrollment = self._select_enrollment(
                wake_candidates, sample.wake_text, sample.language
            )

            command_overlap = self.models.overlap_detector.has_overlap(sample.command_audio)
            command_candidates = self._build_candidates(
                sample.command_audio, command_overlap, workdir / "command", "command"
            )
            for candidate in command_candidates:
                candidate.primary_score = self.models.primary_speaker_verifier.score(
                    enrollment.path, candidate.path
                )

            selected = self._best_primary(command_candidates)
            rescue = {"triggered": False, "accepted": False}
            gate_passed = selected.primary_score >= self.config.primary_speaker_threshold

            if not command_overlap and not gate_passed:
                selected, rescue = self._attempt_nonoverlap_rescue(
                    enrollment.path,
                    sample.command_audio,
                    selected,
                    workdir / "rescue",
                )
                gate_passed = bool(rescue["accepted"])

            diagnostics = {
                "wake_overlap": wake_overlap,
                "command_overlap": command_overlap,
                "enrollment_candidate": enrollment.candidate_id,
                "command_candidate": selected.candidate_id,
                "primary_score": selected.primary_score,
                "rescue": rescue,
            }
            if not gate_passed:
                return PipelineResult(sample.sample_id, "", False, diagnostics)

            content = self.models.command_asr.transcribe(
                selected.path, sample.language
            ).strip()
            return PipelineResult(sample.sample_id, content, bool(content), diagnostics)

    def _build_candidates(
        self,
        source: Path,
        has_overlap: bool,
        output_dir: Path,
        prefix: str,
    ) -> list[AudioCandidate]:
        candidates = [AudioCandidate(f"{prefix}:raw", source, "raw")]
        if not has_overlap:
            return candidates

        output_dir.mkdir(parents=True, exist_ok=True)
        separated = list(self.models.separator.separate(source, output_dir / "separated"))
        for index, separated_path in enumerate(separated):
            enhanced_path = output_dir / "enhanced" / f"source{index}.wav"
            enhanced_path.parent.mkdir(parents=True, exist_ok=True)
            final_path = self.models.enhancer.enhance(separated_path, enhanced_path)
            candidates.append(AudioCandidate(
                f"{prefix}:separated_enhanced:{index}", final_path, "separated_enhanced"
            ))
        return candidates

    def _select_enrollment(
        self,
        candidates: Sequence[AudioCandidate],
        wake_text: str,
        default_language: str,
    ) -> AudioCandidate:
        language = infer_language(wake_text, default_language)
        for candidate in candidates:
            candidate.transcript = self.models.wake_asr.transcribe(candidate.path, language)
            candidate.keyword_score = keyword_similarity(candidate.transcript, wake_text)
        return max(
            candidates,
            key=lambda candidate: (
                candidate.keyword_score,
                candidate.kind == "raw",
            ),
        )

    @staticmethod
    def _best_primary(candidates: Sequence[AudioCandidate]) -> AudioCandidate:
        return max(
            candidates,
            key=lambda candidate: (
                candidate.primary_score,
                candidate.kind == "raw",
            ),
        )

    def _attempt_nonoverlap_rescue(
        self,
        enrollment_audio: Path,
        command_audio: Path,
        raw_candidate: AudioCandidate,
        output_dir: Path,
    ) -> tuple[AudioCandidate, dict]:
        output_dir.mkdir(parents=True, exist_ok=True)
        rescue_paths = list(self.models.separator.separate(command_audio, output_dir))
        if len(rescue_paths) != 2:
            raise ValueError("non-overlap rescue requires exactly two separated streams")

        raw_secondary = self.models.secondary_speaker_verifier.score(
            enrollment_audio, command_audio
        )
        candidates: list[AudioCandidate] = []
        for index, path in enumerate(rescue_paths):
            candidates.append(AudioCandidate(
                candidate_id=f"command:rescue:{index}",
                path=path,
                kind="separated_unenhanced",
                primary_score=self.models.primary_speaker_verifier.score(
                    enrollment_audio, path
                ),
                secondary_score=self.models.secondary_speaker_verifier.score(
                    enrollment_audio, path
                ),
            ))

        primary_winner = max(range(2), key=lambda index: candidates[index].primary_score)
        secondary_winner = max(range(2), key=lambda index: candidates[index].secondary_score)
        strict = (
            primary_winner == secondary_winner
            and candidates[0].primary_score != candidates[1].primary_score
            and candidates[0].secondary_score != candidates[1].secondary_score
        )
        selected = candidates[primary_winner]
        accepted = bool(
            (strict or not self.config.require_strict_rescue_agreement)
            and selected.primary_score >= self.config.rescue_primary_threshold
            and selected.secondary_score >= self.config.rescue_secondary_threshold
            and (
                not self.config.require_both_rescue_scores_to_improve
                or (
                    selected.primary_score > raw_candidate.primary_score
                    and selected.secondary_score > raw_secondary
                )
            )
        )
        diagnostics = {
            "triggered": True,
            "accepted": accepted,
            "primary_winner": primary_winner,
            "secondary_winner": secondary_winner,
            "strict_agreement": strict,
            "raw_primary": raw_candidate.primary_score,
            "raw_secondary": raw_secondary,
            "candidate_primary": [candidate.primary_score for candidate in candidates],
            "candidate_secondary": [candidate.secondary_score for candidate in candidates],
        }
        return (selected if accepted else raw_candidate), diagnostics

    @staticmethod
    def _validate_sample(sample: Sample) -> None:
        if not sample.wake_text.strip():
            raise ValueError("wake_text must not be empty")
        for name in ("wake_audio", "command_audio"):
            path = getattr(sample, name)
            if not path.is_file():
                raise FileNotFoundError(f"{name} not found: {path}")

