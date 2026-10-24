# Open-Source Speech Model Pipeline for Target-Speaker Command Recognition

[中文说明](README_zh.md)

A model-agnostic reference implementation for composing open speech models into an overlap-aware, speaker-gated command-recognition pipeline.

The project focuses on the orchestration layer: typed model interfaces, candidate routing, conservative rescue rules, tests, and a runnable mock demo. Each model is connected through a small adapter, so implementations can be replaced without rewriting the decision logic.

## Why this pipeline exists

A single speaker-verification score can reject a true target speaker when the recording is noisy, overlapped, or otherwise degraded. Blindly lowering the threshold can recover true samples but also admit impostors. This pipeline instead keeps a conservative primary gate and allows a low-score sample to be rescued only when two independent speaker verifiers agree on the same separated stream and both scores improve.

## Architecture

```text
Wake audio
  -> overlap detection
  -> raw candidate is always retained
  -> if overlapped: two-stream separation -> enhancement
  -> lightweight ASR + soft wake-text matching
  -> select enrollment audio

Command audio
  -> overlap detection
  -> if overlapped:
       raw + (two-stream separation -> enhancement)
       -> primary speaker verifier selects and gates
  -> if not overlapped:
       raw -> primary speaker verifier
       -> high score: accept raw
       -> low score: two-stream separation without enhancement
          -> primary + secondary speaker verifiers
          -> rescue only if both select the same stream,
             both thresholds pass, and both improve over raw
  -> accepted audio -> final ASR -> text
  -> rejected audio -> empty text
```

The raw signal is never overwritten. Enhancement is used for overlap handling, but deliberately not used in the non-overlap rescue path because enhancement can alter speaker cues.

## Suggested open models

| Role | Suggested model/project |
|---|---|
| Overlap detection | [pyannote segmentation 3.0](https://huggingface.co/pyannote/segmentation-3.0) |
| Two-speaker separation | [TIGER](https://github.com/JusperLee/TIGER) |
| Speech enhancement | [MossFormerGAN in ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio) |
| Wake candidate transcription | [SenseVoice](https://github.com/QwenAudio/SenseVoice) |
| Primary speaker verification | [CAM++ in 3D-Speaker](https://github.com/modelscope/3D-Speaker) |
| Secondary rescue verification | [ERes2NetV2 in 3D-Speaker](https://github.com/modelscope/3D-Speaker/tree/main/egs/voxceleb/sv-eres2netv2) |
| Final command ASR | [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) |

These are suggested adapters, not vendored dependencies. Review each project's current code and model-weight license before use. Some model hosts require accepting access conditions.

## Install and test

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python examples/mock_demo.py
python tools/prepublish_check.py
```

The package itself has no runtime dependency beyond Python. Real adapters will add the dependencies required by their upstream projects.

## Use the API

Implement the five small protocols in `target_speaker_command.types`, then compose them:

```python
from pathlib import Path
from target_speaker_command import (
    ModelBundle, PipelineConfig, Sample, TargetSpeakerCommandPipeline,
)

models = ModelBundle(
    overlap_detector=my_overlap_adapter,
    separator=my_separator_adapter,
    enhancer=my_enhancer_adapter,
    wake_asr=my_wake_asr_adapter,
    command_asr=my_command_asr_adapter,
    primary_speaker_verifier=my_primary_sv_adapter,
    secondary_speaker_verifier=my_secondary_sv_adapter,
)

config = PipelineConfig(
    primary_speaker_threshold=PRIMARY_THRESHOLD,
    rescue_primary_threshold=RESCUE_PRIMARY_THRESHOLD,
    rescue_secondary_threshold=RESCUE_SECONDARY_THRESHOLD,
)

pipeline = TargetSpeakerCommandPipeline(models, config)
result = pipeline.run(Sample(
    sample_id="example-1",
    wake_audio=Path("wake.wav"),
    wake_text="your wake phrase",
    command_audio=Path("command.wav"),
    language="en",
))
print(result.content, result.accepted, result.diagnostics)
```

See [`examples/mock_demo.py`](examples/mock_demo.py) for a runnable, weight-free example.

## Threshold calibration

`config.example.json` intentionally leaves thresholds unset. Calibrate them on an independent development set that represents expected microphones, noise, overlap, speakers, and recording quality. Keep the final test set untouched. Report both false rejection and false acceptance; optimizing only one can hide an unsafe trade-off.

The rescue gate should remain conservative:

1. Both speaker verifiers choose the same stream.
2. Both scores exceed separately calibrated rescue thresholds.
3. Both scores improve over the unprocessed command.
4. Ties do not count as agreement.

## License

The orchestration code is MIT licensed. Upstream code and model weights remain under their respective licenses; this repository does not relicense them.
