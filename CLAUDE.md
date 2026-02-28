# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Training
```bash
# Train with curriculum learning (3 phases, run sequentially)
python train.py --config configs/mls_french_no_curriculum.yaml --phase 1
python train.py --config configs/mls_french_no_curriculum.yaml --phase 2 --resume ./results-moonshine-fr-phase1
python train.py --config configs/mls_french_no_curriculum.yaml --phase 3 --resume ./results-moonshine-fr-phase2

# Train without curriculum (all data at once)
python train.py --config configs/mls_french_no_curriculum.yaml --no-curriculum

# Quick validation (100 samples)
python train.py --config configs/mls_french_no_curriculum.yaml --test-mode
```

### Data Preparation
```bash
# Segment long audio using Whisper V3 forced alignment
python scripts/intelligent_segmentation.py \
    --dataset facebook/multilingual_librispeech \
    --language french \
    --output ./data/mls_french_segmented \
    --max-duration 10.0 --min-duration 1.0
```

### Evaluation & Inference
```bash
# Evaluate WER/CER on test set
python scripts/evaluate.py \
    --model results-moonshine-fr/checkpoint-best \
    --dataset facebook/multilingual_librispeech \
    --language french --split test

# Single file inference
python scripts/inference.py --model results-moonshine-fr/checkpoint-best --audio sample.wav

# Live transcription with VAD
python scripts/inference.py --model results-moonshine-fr/checkpoint-best --live

# Export to ONNX and run
python scripts/convert_for_deployment.py --model checkpoint-best --output moonshine-fr-onnx
python scripts/inference.py --model moonshine-fr-onnx/onnx --audio sample.wav --use-manual-onnx
```

### Monitoring
```bash
tensorboard --logdir results-moonshine-fr/runs
```

### Optional: Live transcription dependencies
```bash
pip install -r requirements-live.txt
```

## Architecture

### High-Level Data Flow
```
Dataset (HuggingFace/CSV/local)
  → MoonshineDataLoader (data_loader.py)
  → CurriculumScheduler filter (curriculum.py)
  → DataCollatorMoonshineSeq2SeqWithPadding (train.py)
  → MoonshineSeq2SeqTrainer (train.py)
```

### `moonshine_ft/` Library

- **`data_loader.py` — `MoonshineDataLoader`**: Unified loader supporting Common Voice, LibriSpeech, MLS, CSV, and `save_to_disk()` local datasets. All formats are normalized to `audio` + `sentence` columns at 16kHz. `prepare_dataset()` runs feature extraction + tokenization, adding `input_values`, `labels`, `duration`, and `input_length` columns.

- **`curriculum.py` — `CurriculumScheduler` / `CurriculumPhase`**: Implements the 3-phase curriculum from the Moonshine paper. Each phase filters training data by `[min_duration, max_duration]` and optionally `max_words`. Phase configs include per-phase learning rate, warmup steps, and generation parameters (repetition penalty, num_beams). When curriculum is disabled, a single "full dataset" phase is created using the paper-recommended [4, 30]s range.

- **`utils/metrics.py`**: WER/CER computation with empty-string handling. Uses the `evaluate` library. `compute_detailed_metrics()` returns per-sample statistics.

- **`utils/preprocessing.py`**: Audio normalization to RMS 0.075 (matching Moonshine training), padding (center/start/end), and basic resampling.

### `train.py` Key Classes

- **`DataCollatorMoonshineSeq2SeqWithPadding`**: Pads audio inputs, pads labels with `-100` (ignored in loss), and constructs `decoder_input_ids` as `[BOS] + labels[:-1]`. Moonshine expects `input_values`, not `input_features` (unlike Whisper).

- **`MoonshineSeq2SeqTrainer`**: Overrides `prediction_step()` to call `model.generate(input_values=..., attention_mask=...)` directly, computing `max_new_tokens` dynamically from audio duration (roughly 6 tokens/second). Avoids Whisper-style forced-decoding.

### Config YAML Structure
Key sections: `model` (name, freeze_encoder), `dataset` (type, language/path), `audio` (min/max duration, sampling_rate), `preprocessing` (num_proc), `curriculum` (enabled, phases), `training` (all HuggingFace `Seq2SeqTrainingArguments` fields), `generation` (beam search params).

The `type` field under `dataset` selects the loader: `common_voice`, `librispeech`, `mls`, `csv`, or `local`.

### Important Constraints from Moonshine Paper
- Audio must be 16kHz mono
- Optimal training range: 4–30 seconds per clip
- Keep <0.5% of data under 1 second (very short clips cause repetitions and >100% WER)
- `use_cache: False` required on the model when gradient checkpointing is enabled
- Token IDs: BOS=1, EOS=2, PAD=2 (set explicitly on `model.config`)
