"""Run the complete routing logic without downloading any model or audio."""

from __future__ import annotations

import tempfile
from pathlib import Path

from target_speaker_command import ModelBundle, PipelineConfig, Sample, TargetSpeakerCommandPipeline


class MockOverlapDetector:
    def has_overlap(self, audio: Path) -> bool:
        return "overlap" in audio.stem


class MockSeparator:
    def separate(self, audio: Path, output_dir: Path) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = [output_dir / "source0.wav", output_dir / "source1.wav"]
        for path in paths:
            path.write_bytes(b"mock")
        return paths


class MockEnhancer:
    def enhance(self, audio: Path, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(audio.read_bytes())
        return output_path


class MockWakeAsr:
    def transcribe(self, audio: Path, language: str) -> str:
        return "hello device"


class MockCommandAsr:
    def transcribe(self, audio: Path, language: str) -> str:
        return "turn on the light"


class MockSpeakerVerifier:
    def __init__(self, raw_score: float, first_stream_score: float) -> None:
        self.raw_score = raw_score
        self.first_stream_score = first_stream_score

    def score(self, enrollment_audio: Path, test_audio: Path) -> float:
        if test_audio.name == "source0.wav":
            return self.first_stream_score
        if test_audio.name == "source1.wav":
            return 0.20
        return self.raw_score


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        wake = root / "wake.wav"
        command = root / "command.wav"
        wake.write_bytes(b"mock")
        command.write_bytes(b"mock")

        models = ModelBundle(
            overlap_detector=MockOverlapDetector(),
            separator=MockSeparator(),
            enhancer=MockEnhancer(),
            wake_asr=MockWakeAsr(),
            command_asr=MockCommandAsr(),
            primary_speaker_verifier=MockSpeakerVerifier(0.30, 0.76),
            secondary_speaker_verifier=MockSpeakerVerifier(0.28, 0.72),
        )
        pipeline = TargetSpeakerCommandPipeline(models, PipelineConfig(
            primary_speaker_threshold=0.50,
            rescue_primary_threshold=0.65,
            rescue_secondary_threshold=0.65,
        ))
        result = pipeline.run(Sample(
            sample_id="mock-1",
            wake_audio=wake,
            wake_text="hello device",
            command_audio=command,
            language="en",
        ))
        print({
            "id": result.sample_id,
            "content": result.content,
            "accepted": result.accepted,
            "diagnostics": result.diagnostics,
        })


if __name__ == "__main__":
    main()
