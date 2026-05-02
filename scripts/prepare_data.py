"""
Data preparation script for ASR benchmark.

Downloads Common Voice Russian dataset, filters by duration,
normalizes text, converts audio to 16kHz mono WAV, and
saves a clean manifest (CSV) for benchmarking.

Usage:
    python scripts/prepare_data.py
"""

import csv
import io
import logging
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# --- Logging ---------------------------------------------------
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger("prepare_data")


# --- Text normalization ---------------------------------------
def normalize_text(text: str) -> str:
    """
    Normalize reference text for WER/CER computation:
    - lowercase
    - ё → е  (if config.NORMALIZE_YO_YE)
    - remove punctuation
    - collapse whitespace
    - keep only allowed characters (Russian letters, digits, space)
    """
    text = text.lower().strip()
    text = unicodedata.normalize("NFC", text)
    if config.NORMALIZE_YO_YE:
        text = text.replace("ё", "е")
    # Remove punctuation and special characters
    text = re.sub(r"[^\w\s]", " ", text)
    # Keep only allowed chars
    allowed = set(ch.replace("ё", "е") if config.NORMALIZE_YO_YE else ch
                  for ch in config.ALLOWED_CHARS)
    text = "".join(ch for ch in text if ch in allowed)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_audio_duration(audio_path: Path) -> float:
    """Get audio duration in seconds."""
    try:
        import wave

        with wave.open(str(audio_path), "rb") as wav:
            return wav.getnframes() / wav.getframerate()
    except Exception:
        try:
            import torchaudio

            metadata = torchaudio.info(str(audio_path))
            return metadata.num_frames / metadata.sample_rate
        except Exception:
            return 0.0


def _resample_audio(audio_array: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio without requiring librosa at import time."""
    if orig_sr == target_sr:
        return audio_array.astype(np.float32, copy=False)

    try:
        from scipy.signal import resample_poly
        from math import gcd

        divisor = gcd(orig_sr, target_sr)
        return resample_poly(
            audio_array.astype(np.float32, copy=False),
            target_sr // divisor,
            orig_sr // divisor,
        ).astype(np.float32, copy=False)
    except Exception:
        import torchaudio
        import torch

        tensor = torch.as_tensor(audio_array.astype(np.float32, copy=False))
        resampled = torchaudio.functional.resample(tensor, orig_sr, target_sr)
        return resampled.cpu().numpy().astype(np.float32, copy=False)


def convert_audio(audio_array: np.ndarray, orig_sr: int, output_path: Path) -> float:
    """
    Convert audio to 16kHz mono WAV and save.
    Returns duration in seconds.
    """
    # Ensure mono
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=-1)

    audio_array = _resample_audio(audio_array, orig_sr, config.SAMPLE_RATE)

    # Normalize amplitude
    peak = np.abs(audio_array).max()
    if peak > 0:
        audio_array = audio_array / peak * 0.95

    output_path.parent.mkdir(parents=True, exist_ok=True)
    from scipy.io import wavfile

    pcm16 = np.clip(audio_array, -1.0, 1.0)
    pcm16 = (pcm16 * np.iinfo(np.int16).max).astype(np.int16)
    wavfile.write(str(output_path), config.SAMPLE_RATE, pcm16)

    duration = len(audio_array) / config.SAMPLE_RATE
    return duration


def decode_audio_feature(audio_data: dict) -> tuple[np.ndarray, int]:
    """
    Decode a HuggingFace audio feature without relying on datasets' torchcodec path.

    Newer datasets versions may route Audio(decode=True) through torchcodec, which
    is brittle on Windows without a matching FFmpeg/libtorchcodec setup. This helper
    accepts both decoded rows and Audio(decode=False) rows.
    """
    if "array" in audio_data and "sampling_rate" in audio_data:
        return np.asarray(audio_data["array"], dtype=np.float32), int(audio_data["sampling_rate"])

    import soundfile as sf

    if audio_data.get("bytes"):
        audio_array, sample_rate = sf.read(io.BytesIO(audio_data["bytes"]), dtype="float32")
        return np.asarray(audio_array, dtype=np.float32), int(sample_rate)

    audio_path = audio_data.get("path")
    if audio_path:
        audio_array, sample_rate = sf.read(audio_path, dtype="float32")
        return np.asarray(audio_array, dtype=np.float32), int(sample_rate)

    raise ValueError("Unsupported audio feature: expected decoded array, bytes, or path")


def _load_ds(ds_id, lang, split):
    """Load a dataset split, respecting HF_ENDPOINT for mirrors."""
    import os
    from datasets import load_dataset
    kwargs = {"split": split}
    if lang:
        return load_dataset(ds_id, lang, **kwargs)
    return load_dataset(ds_id, **kwargs)


def prepare_common_voice():
    """
    Download and prepare Common Voice Russian dataset.
    Tries DATASET_NAME first, then each entry in DATASET_FALLBACKS.

    Tip: if HuggingFace is blocked, set HF_ENDPOINT to a mirror:
        set HF_ENDPOINT=https://hf-mirror.com
    Creates test and dev manifests.
    """
    import os
    logger.info("=" * 60)
    logger.info("Preparing Common Voice Russian dataset")
    logger.info("=" * 60)

    hf_endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    if hf_endpoint != "https://huggingface.co":
        logger.info(f"Using HF mirror: {hf_endpoint}")

    from datasets import Audio, concatenate_datasets

    text_field = "sentence"
    source_dataset = config.DATASET_NAME
    ds_test = ds_dev = None

    # --- Primary dataset ---
    logger.info(f"Loading primary dataset: {config.DATASET_NAME}")
    try:
        ds_test = _load_ds(config.DATASET_NAME, config.DATASET_LANG, "test")
        ds_dev  = _load_ds(config.DATASET_NAME, config.DATASET_LANG, "validation")
        logger.info(f"Primary dataset loaded: test={len(ds_test)}, dev={len(ds_dev)}")
    except Exception as e:
        logger.warning(f"Primary dataset failed: {type(e).__name__}: {e}")
        logger.info(
            "TIP: If HuggingFace is unreachable, set HF_ENDPOINT to a mirror:\n"
            "     set HF_ENDPOINT=https://hf-mirror.com"
        )

    # --- Fallback chain ---
    if ds_test is None:
        for ds_id, ds_lang, ds_text_field in config.DATASET_FALLBACKS:
            logger.info(f"Trying fallback: {ds_id}")
            try:
                splits_collected = []
                for split in ["test", "validation", "train"]:
                    try:
                        s = _load_ds(ds_id, ds_lang, split)
                        splits_collected.append(s)
                        logger.info(f"  {ds_id}/{split}: {len(s)} samples")
                        # Stop collecting splits when we have enough
                        total_so_far = sum(len(x) for x in splits_collected)
                        if total_so_far >= (config.MAX_TEST_SAMPLES or 500) * 1.5:
                            break
                    except Exception:
                        pass  # split doesn't exist

                if not splits_collected:
                    raise RuntimeError(f"No splits found in {ds_id}")

                # Combine all collected splits into one pool
                combined = concatenate_datasets(splits_collected)
                combined = combined.shuffle(seed=config.SEED)
                # Split 80/20 into test/dev from the combined pool
                split_idx = max(1, int(len(combined) * 0.8))
                ds_test = combined.select(range(split_idx))
                ds_dev  = combined.select(range(split_idx, len(combined)))

                text_field = ds_text_field
                source_dataset = ds_id
                logger.info(
                    f"Fallback OK [{ds_id}]: "
                    f"combined {len(combined)} samples → "
                    f"test={len(ds_test)}, dev={len(ds_dev)}"
                )
                logger.warning(
                    "DATA QUALITY WARNING: Using fallback public dataset "
                    f"'{ds_id}'. Domain is narrow (podcast speech). "
                    "Results are a SMOKE TEST only, not a production benchmark."
                )
                break
            except Exception as e2:
                logger.warning(f"Fallback {ds_id} failed: {type(e2).__name__}: {e2}")

    if ds_test is None:
        raise RuntimeError(
            "All dataset sources failed.\n"
            "Options:\n"
            "  1. Set HF_ENDPOINT=https://hf-mirror.com and retry\n"
            "  2. Set HF_HUB_OFFLINE=1 if data is already cached\n"
            "  3. Place your own WAV files + manifest CSV in data/processed/"
        )

    logger.info(f"source_dataset={source_dataset}  text_field={text_field}")


    if "audio" in ds_test.column_names:
        ds_test = ds_test.cast_column("audio", Audio(decode=False))
    if "audio" in ds_dev.column_names:
        ds_dev = ds_dev.cast_column("audio", Audio(decode=False))

    # Process splits
    for split_name, ds, max_samples in [
        ("test", ds_test, config.MAX_TEST_SAMPLES),
        ("dev", ds_dev, config.MAX_DEV_SAMPLES),
    ]:
        logger.info(f"\nProcessing {split_name} split...")
        output_dir = config.PROCESSED_DIR / split_name
        output_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = config.PROCESSED_DIR / f"{split_name}_manifest.csv"
        manifest_rows = []

        # Shuffle with seed for reproducibility, then limit
        ds_shuffled = ds.shuffle(seed=config.SEED)
        if max_samples and len(ds_shuffled) > max_samples:
            overselect = min(len(ds_shuffled), max_samples * 3)
            ds_shuffled = ds_shuffled.select(range(overselect))  # over-select, filter later

        processed = 0
        skipped_short = 0
        skipped_long = 0
        skipped_empty = 0

        for idx, sample in enumerate(tqdm(ds_shuffled, desc=f"Processing {split_name}")):
            if max_samples and processed >= max_samples:
                break

            # Get text
            text = sample.get(text_field, "")
            normalized = normalize_text(text)
            if not normalized or len(normalized) < 2:
                skipped_empty += 1
                continue

            # Get audio
            audio_data = sample.get("audio", {})
            if not audio_data:
                continue

            try:
                audio_array, orig_sr = decode_audio_feature(audio_data)
            except Exception as e:
                logger.warning(f"Failed to decode audio sample {idx}: {e}")
                continue

            # Check duration
            duration = len(audio_array) / orig_sr
            if duration < config.MIN_DURATION_SEC:
                skipped_short += 1
                continue
            if duration > config.MAX_DURATION_SEC:
                skipped_long += 1
                continue

            # Save audio
            audio_filename = f"{split_name}_{processed:05d}.wav"
            audio_path = output_dir / audio_filename

            actual_duration = convert_audio(audio_array, orig_sr, audio_path)

            # Get metadata
            speaker_id = sample.get("client_id", "unknown")
            gender = sample.get("gender", "unknown")
            age = sample.get("age", "unknown")

            manifest_rows.append({
                "audio_path": str(audio_path.resolve()),
                "text": normalized,
                "text_original": text,
                "duration_sec": round(actual_duration, 3),
                "speaker_id": speaker_id[:16] if speaker_id else "unknown",
                "gender": gender or "unknown",
                "age": age or "unknown",
                "split": split_name,
                "source_dataset": source_dataset,
            })
            processed += 1

        # Write manifest
        if manifest_rows:
            with open(manifest_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=manifest_rows[0].keys())
                writer.writeheader()
                writer.writerows(manifest_rows)

            total_duration = sum(r["duration_sec"] for r in manifest_rows)
            logger.info(f"\n{split_name} split summary:")
            logger.info(f"  Processed: {processed}")
            logger.info(f"  Skipped (short <{config.MIN_DURATION_SEC}s): {skipped_short}")
            logger.info(f"  Skipped (long >{config.MAX_DURATION_SEC}s): {skipped_long}")
            logger.info(f"  Skipped (empty text): {skipped_empty}")
            logger.info(f"  Total duration: {total_duration/3600:.2f} hours")
            logger.info(f"  Manifest: {manifest_path}")
        else:
            logger.warning(f"No samples processed for {split_name}!")

    logger.info("\n" + "=" * 60)
    logger.info("Data preparation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    np.random.seed(config.SEED)
    prepare_common_voice()
