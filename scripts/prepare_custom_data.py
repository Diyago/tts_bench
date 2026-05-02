"""
Import a custom local audio dataset for ASR benchmarking.

Accepts:
  - A directory of audio files (WAV/MP3/FLAC/OGG/M4A/OPUS)
  - An optional transcript file (CSV/TSV/JSONL) mapping filename → text

Output:
  - Converted 16 kHz mono WAV files in data/processed/custom/
  - data/processed/custom_manifest.csv  (same format as Common Voice manifest)

Usage examples:
  # Audio only (WER cannot be computed, RTF/latency still work)
  python scripts/prepare_custom_data.py --audio-dir /path/to/wavs

  # Audio + CSV transcripts (columns: filename, text  OR  audio_path, text)
  python scripts/prepare_custom_data.py --audio-dir /path/to/wavs --transcripts /path/to/transcripts.csv

  # Audio + JSONL transcripts (each line: {"file": "001.wav", "text": "привет"})
  python scripts/prepare_custom_data.py --audio-dir /path/to/wavs --transcripts /path/to/transcripts.jsonl

  # Audio + TSV (tab-separated, columns: filename\\ttext)
  python scripts/prepare_custom_data.py --audio-dir /path/to/wavs --transcripts /path/to/transcripts.tsv

  # Limit samples
  python scripts/prepare_custom_data.py --audio-dir /path/to/wavs --max-samples 200
"""

import argparse
import csv
import json
import logging
import sys
import unicodedata
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger("prepare_custom")

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus", ".aac", ".wma"}


# --- Text normalization (same as main pipeline) --------------------------

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    text = unicodedata.normalize("NFC", text)
    if config.NORMALIZE_YO_YE:
        text = text.replace("ё", "е")
    text = re.sub(r"[^\w\s]", " ", text)
    allowed = set(
        ch.replace("ё", "е") if config.NORMALIZE_YO_YE else ch
        for ch in config.ALLOWED_CHARS
    )
    text = "".join(ch for ch in text if ch in allowed)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --- Transcript loading --------------------------------------------------

def load_transcripts(transcript_path: Path) -> dict[str, str]:
    """
    Load filename → text mapping from CSV / TSV / JSONL.

    CSV/TSV: expects columns named one of:
        (filename | audio_path | file | path) and (text | transcript | transcription | sentence)
    JSONL: each line is a JSON object with similar keys.

    Returns dict: stem_or_basename → text  (lowercase stem for robust matching)
    """
    ext = transcript_path.suffix.lower()
    mapping: dict[str, str] = {}

    def _find_col(headers: list[str], candidates: list[str]) -> str | None:
        for c in candidates:
            if c in headers:
                return c
        return None

    FILE_COLS = ["filename", "audio_path", "file", "path", "audio_file"]
    TEXT_COLS = ["text", "transcript", "transcription", "sentence", "reference"]

    if ext == ".jsonl" or (ext == ".json" and transcript_path.stat().st_size < 10_000_000):
        with open(transcript_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                fname_key = next((k for k in FILE_COLS if k in obj), None)
                text_key  = next((k for k in TEXT_COLS if k in obj), None)
                if fname_key and text_key:
                    stem = Path(obj[fname_key]).stem.lower()
                    mapping[stem] = obj[text_key]
        logger.info(f"Loaded {len(mapping)} transcripts from JSONL: {transcript_path}")

    else:  # CSV or TSV
        delimiter = "\t" if ext == ".tsv" else ","
        with open(transcript_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            headers = reader.fieldnames or []
            fname_col = _find_col(headers, FILE_COLS)
            text_col  = _find_col(headers, TEXT_COLS)
            if not fname_col or not text_col:
                # Maybe it's just two columns without header
                f.seek(0)
                reader2 = csv.reader(f, delimiter=delimiter)
                first_row = next(reader2, [])
                if len(first_row) >= 2:
                    logger.warning(
                        f"No recognised column names in {transcript_path}. "
                        "Assuming col0=filename, col1=text."
                    )
                    stem = Path(first_row[0]).stem.lower()
                    mapping[stem] = first_row[1]
                    for row in reader2:
                        if len(row) >= 2:
                            mapping[Path(row[0]).stem.lower()] = row[1]
            else:
                for row in reader:
                    stem = Path(row[fname_col]).stem.lower()
                    mapping[stem] = row[text_col]
        logger.info(f"Loaded {len(mapping)} transcripts from {ext.upper()}: {transcript_path}")

    return mapping


# --- Audio conversion ---------------------------------------------------

def convert_audio(audio_path: Path, output_path: Path) -> float:
    """Convert any audio file to 16 kHz mono WAV. Returns duration in seconds."""
    import soundfile as sf
    from scipy.io import wavfile

    try:
        import librosa
        audio, sr = librosa.load(str(audio_path), sr=config.SAMPLE_RATE, mono=True)
    except Exception:
        # Fallback: soundfile + manual resample
        audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != config.SAMPLE_RATE:
            from math import gcd
            from scipy.signal import resample_poly
            d = gcd(sr, config.SAMPLE_RATE)
            audio = resample_poly(audio, config.SAMPLE_RATE // d, sr // d).astype(np.float32)

    # Normalize amplitude
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.95

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pcm16 = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm16 * np.iinfo(np.int16).max).astype(np.int16)
    wavfile.write(str(output_path), config.SAMPLE_RATE, pcm16)
    return len(audio) / config.SAMPLE_RATE


# --- Main ---------------------------------------------------------------

def prepare_custom_data(
    audio_dir: Path,
    transcript_path: Path | None = None,
    max_samples: int | None = None,
    split_name: str = "custom",
):
    """
    Prepare a custom audio dataset for benchmarking.

    Creates:
        data/processed/{split_name}_manifest.csv
        data/processed/{split_name}/*.wav  (converted audio)
    """
    logger.info("=" * 60)
    logger.info(f"Preparing custom dataset from: {audio_dir}")
    logger.info("=" * 60)

    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    # Collect all audio files
    audio_files = sorted([
        p for p in audio_dir.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ])
    if not audio_files:
        raise RuntimeError(
            f"No audio files found in {audio_dir}. "
            f"Supported formats: {SUPPORTED_EXTENSIONS}"
        )
    logger.info(f"Found {len(audio_files)} audio files")

    # Load transcripts (optional)
    transcripts: dict[str, str] = {}
    if transcript_path:
        transcripts = load_transcripts(transcript_path)
        logger.info(
            f"Transcripts loaded: {len(transcripts)} entries. "
            f"Files without transcript will have empty text (latency-only mode)."
        )
    else:
        logger.info(
            "No transcript file provided. "
            "WER/CER will not be computed – only RTF and latency."
        )

    # Limit samples
    if max_samples and len(audio_files) > max_samples:
        logger.info(f"Limiting to {max_samples} samples (--max-samples)")
        audio_files = audio_files[:max_samples]

    output_dir = config.PROCESSED_DIR / split_name
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.PROCESSED_DIR / f"{split_name}_manifest.csv"

    manifest_rows = []
    skipped_short = skipped_long = skipped_error = 0

    for idx, audio_path in enumerate(tqdm(audio_files, desc="Converting audio")):
        stem = audio_path.stem.lower()
        raw_text  = transcripts.get(stem, transcripts.get(audio_path.name.lower(), ""))
        norm_text = normalize_text(raw_text)

        output_filename = f"{split_name}_{idx:05d}.wav"
        output_path = output_dir / output_filename

        try:
            duration = convert_audio(audio_path, output_path)
        except Exception as e:
            logger.warning(f"Failed to convert {audio_path.name}: {e}")
            skipped_error += 1
            continue

        if duration < config.MIN_DURATION_SEC:
            skipped_short += 1
            output_path.unlink(missing_ok=True)
            continue
        if duration > config.MAX_DURATION_SEC:
            skipped_long += 1
            output_path.unlink(missing_ok=True)
            continue

        manifest_rows.append({
            "audio_path":     str(output_path.resolve()),
            "text":           norm_text,
            "text_original":  raw_text,
            "duration_sec":   round(duration, 3),
            "speaker_id":     "unknown",
            "gender":         "unknown",
            "age":            "unknown",
            "split":          split_name,
            "source_dataset": str(audio_dir.resolve()),
        })

    if not manifest_rows:
        raise RuntimeError("No samples survived filtering! Check audio quality and duration settings.")

    # Write manifest
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)

    total_hours = sum(r["duration_sec"] for r in manifest_rows) / 3600
    has_text    = sum(1 for r in manifest_rows if r["text"])

    logger.info("\n" + "=" * 60)
    logger.info("Custom dataset prepared:")
    logger.info(f"  Total samples:        {len(manifest_rows)}")
    logger.info(f"  With transcripts:     {has_text}  (WER computable)")
    logger.info(f"  Without transcripts:  {len(manifest_rows) - has_text}  (latency only)")
    logger.info(f"  Skipped (short <{config.MIN_DURATION_SEC}s):  {skipped_short}")
    logger.info(f"  Skipped (long >{config.MAX_DURATION_SEC}s):   {skipped_long}")
    logger.info(f"  Skipped (error):      {skipped_error}")
    logger.info(f"  Total audio:          {total_hours:.3f} hours")
    logger.info(f"  Manifest:             {manifest_path}")
    logger.info("=" * 60)
    logger.info(f"\nRun benchmark with:")
    logger.info(f"  python scripts/run_benchmark.py --split {split_name} --skip-data-prep --models whisper,gigaam,silero,nemo")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Import a custom audio dataset for ASR benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--audio-dir", "-d", type=Path, required=True,
        help="Directory containing audio files (WAV/MP3/FLAC/OGG/M4A/OPUS)"
    )
    parser.add_argument(
        "--transcripts", "-t", type=Path, default=None,
        help="Optional: CSV/TSV/JSONL file with filename→text transcripts"
    )
    parser.add_argument(
        "--max-samples", "-n", type=int, default=None,
        help="Limit number of audio files to process"
    )
    parser.add_argument(
        "--split-name", type=str, default="custom",
        help="Name for this split (used in manifest filename, default: custom)"
    )
    args = parser.parse_args()

    np.random.seed(config.SEED)
    prepare_custom_data(
        audio_dir=args.audio_dir,
        transcript_path=args.transcripts,
        max_samples=args.max_samples,
        split_name=args.split_name,
    )
