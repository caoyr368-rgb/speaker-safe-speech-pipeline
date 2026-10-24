from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from target_speaker_command import ModelBundle, PipelineConfig, Sample, TargetSpeakerCommandPipeline


class OverlapDetector:
    def has_overlap(self, audio: Path) -> bool:
        return "overlap" in audio.stem


class Separator:
    def __init__(self) -> None:
        self.calls: list[Path] = []

    def separate(self, audio: Path, output_dir: Path) -> list[Path]:
        self.calls.append(audio)
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = [output_dir / "source0.wav", output_dir / "source1.wav"]
        for output in outputs:
            output.write_bytes(b"separated")
        return outputs


class Enhancer:
    def __init__(self) -> None:
        self.calls: list[Path] = []

    def enhance(self, audio: Path, output_path: Path) -> Path:
        self.calls.append(audio)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(audio.read_bytes())
        return output_path


class Asr:
    def __init__(self, text: str) -> None:
        self.text = text

    def transcribe(self, audio: Path, language: str) -> str:
        return self.text


class Verifier:
    def __init__(self, raw: float, source0: float, source1: float) -> None:
        self.scores = {"raw": raw, "source0.wav": source0, "source1.wav": source1}

    def score(self, enrollment_audio: Path, test_audio: Path) -> float:
        if test_audio.name in self.scores:
            return self.scores[test_audio.name]
        return self.scores["raw"]


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.wake = self.root / "wake.wav"
        self.command = self.root / "command.wav"
        self.wake.write_bytes(b"wake")
        self.command.write_bytes(b"command")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def make_pipeline(
        self,
        primary: Verifier,
        secondary: Verifier,
        enhancer: Enhancer | None = None,
        separator: Separator | None = None,
    ) -> tuple[TargetSpeakerCommandPipeline, Enhancer, Separator]:
        enhancer = enhancer or Enhancer()
        separator = separator or Separator()
        models = ModelBundle(
            overlap_detector=OverlapDetector(),
            separator=separator,
            enhancer=enhancer,
            wake_asr=Asr("hello device"),
            command_asr=Asr("turn on the light"),
            primary_speaker_verifier=primary,
            secondary_speaker_verifier=secondary,
        )
        return TargetSpeakerCommandPipeline(models, PipelineConfig(
            primary_speaker_threshold=0.50,
            rescue_primary_threshold=0.65,
            rescue_secondary_threshold=0.65,
        )), enhancer, separator

    def sample(self) -> Sample:
        return Sample("sample-1", self.wake, "hello device", self.command, "en")

    def test_high_score_nonoverlap_accepts_raw(self) -> None:
        pipeline, enhancer, separator = self.make_pipeline(
            Verifier(0.80, 0.10, 0.10), Verifier(0.80, 0.10, 0.10)
        )
        result = pipeline.run(self.sample())
        self.assertTrue(result.accepted)
        self.assertEqual(result.content, "turn on the light")
        self.assertFalse(result.diagnostics["rescue"]["triggered"])
        self.assertEqual(enhancer.calls, [])
        self.assertEqual(separator.calls, [])

    def test_low_score_is_rescued_only_with_dual_agreement(self) -> None:
        pipeline, enhancer, separator = self.make_pipeline(
            Verifier(0.30, 0.78, 0.20), Verifier(0.25, 0.74, 0.18)
        )
        result = pipeline.run(self.sample())
        self.assertTrue(result.accepted)
        self.assertTrue(result.diagnostics["rescue"]["accepted"])
        self.assertEqual(result.diagnostics["command_candidate"], "command:rescue:0")
        self.assertEqual(len(separator.calls), 1)
        self.assertEqual(enhancer.calls, [], "non-overlap rescue must not enhance")

    def test_disagreement_rejects_low_score_sample(self) -> None:
        pipeline, _, _ = self.make_pipeline(
            Verifier(0.30, 0.78, 0.20), Verifier(0.25, 0.18, 0.74)
        )
        result = pipeline.run(self.sample())
        self.assertFalse(result.accepted)
        self.assertEqual(result.content, "")
        self.assertFalse(result.diagnostics["rescue"]["accepted"])

    def test_overlap_path_keeps_raw_and_uses_enhanced_candidates(self) -> None:
        self.command = self.root / "command_overlap.wav"
        self.command.write_bytes(b"command")
        pipeline, enhancer, separator = self.make_pipeline(
            Verifier(0.40, 0.85, 0.20), Verifier(0.40, 0.80, 0.20)
        )
        result = pipeline.run(self.sample())
        self.assertTrue(result.accepted)
        self.assertTrue(result.diagnostics["command_overlap"])
        self.assertEqual(result.diagnostics["command_candidate"], "command:separated_enhanced:0")
        self.assertEqual(len(separator.calls), 1)
        self.assertEqual(len(enhancer.calls), 2)


if __name__ == "__main__":
    unittest.main()
