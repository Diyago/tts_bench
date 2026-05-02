"""
Main benchmark runner for ASR models on Russian speech.

Orchestrates the full pipeline:
  1. Auto-download & prepare data
  2. Load each model (LOCAL ONLY)
  3. Run inference on test set
  4. Compute metrics
  5. Save results (CSV + Excel) & generate visualizations

Usage:
    python scripts/run_benchmark.py [--skip-data-prep] [--models whisper,gigaam]

Everything runs from this single entry point.
"""

import argparse
import gc
import logging
import random
import re
import sys
import time
import warnings
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Suppress NeMo/Lhotse spam at module level before any imports
logging.getLogger("lhotse").setLevel(logging.CRITICAL)
logging.getLogger("lhotse.cut").setLevel(logging.CRITICAL)
logging.getLogger("lhotse.dataset").setLevel(logging.CRITICAL)
logging.getLogger("lhotse.dataloader").setLevel(logging.CRITICAL)
logging.getLogger("nemo_logging").setLevel(logging.CRITICAL)
logging.getLogger("nemo").setLevel(logging.CRITICAL)
logging.getLogger("nemo.collections").setLevel(logging.CRITICAL)
logging.getLogger("nemo.collections.asr").setLevel(logging.CRITICAL)
logging.getLogger("nemo.utils").setLevel(logging.CRITICAL)
logging.getLogger("nemo.core").setLevel(logging.CRITICAL)
logging.getLogger("root").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", message=".*If you intend to do training.*")
warnings.filterwarnings("ignore", message=".*If you intend to do validation.*")
warnings.filterwarnings("ignore", message=".*Please call the ModelPT.setup_test_data.*")
warnings.filterwarnings("ignore", message=".*The following configuration keys are ignored.*")
warnings.filterwarnings("ignore", message=".*You are using a non-tarred dataset.*")
warnings.filterwarnings("ignore", message=".*CTC decoding strategy.*")
warnings.filterwarnings("ignore", message=".*Megatron num_microbatches_calculator.*")
warnings.filterwarnings("ignore", message=".*Found existing object.*")
warnings.filterwarnings("ignore", message=".*Re-using file from.*")
warnings.filterwarnings("ignore", message=".*Instantiating model from pre-trained.*")
warnings.filterwarnings("ignore", message=".*Tokenizer.*initialized.*")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from models.inference import (
    ASRModel,
    FasterWhisperModel,
    GigaAMModel,
    VibeVoiceModel,
    Gemma3nModel,
    Phi4Model,
    NeMoConformerModel,
    SileroSTTModel,
    create_models,
)
from scripts.evaluate import (
    compute_metrics,
    compute_per_sample_results,
    normalize_for_eval,
    save_results,
)

import os as _os
import warnings as _warnings
# Reduce noisy progress bars / third-party log spam
_os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
_os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Suppress NeMo's print-based logger BEFORE any nemo import
_os.environ.setdefault("NEMO_LOGGING_LEVEL", "ERROR")  # silences [NeMo W/I ...] prints
_os.environ.setdefault("NVTE_FRAMEWORK", "pytorch")    # suppress transformer engine noise

# --- Logging ---------------------------------------------------
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger("benchmark")

# Cut down on noisy but harmless warnings
_warnings.filterwarnings("ignore", message="The given buffer is not writable*")
_warnings.filterwarnings("ignore", message="`torch_dtype` is deprecated*")
_warnings.filterwarnings("ignore", message=".*is deprecated.*AutoImageProcessor.*")
_warnings.filterwarnings("ignore", message=".*is deprecated.*AutoFeatureExtractor.*")

# Make logging play nicely with tqdm (avoid breaking progress bars)
class _TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            pass

_root = logging.getLogger()
for _h in list(_root.handlers):
    _root.removeHandler(_h)
_tqdm_handler = _TqdmLoggingHandler()
_tqdm_handler.setLevel(logging.getLevelName(config.LOG_LEVEL))
_tqdm_handler.setFormatter(logging.Formatter(config.LOG_FORMAT))
_root.addHandler(_tqdm_handler)
_root.setLevel(logging.getLevelName(config.LOG_LEVEL))

# Silence chatty third-party loggers (per-sample spam)
# Use ERROR level (not WARNING) to suppress NeMo's dataloader/pretokenize warnings
for _noisy in config.QUIET_LOGGERS:
    _lg = logging.getLogger(_noisy)
    _lg.setLevel(logging.ERROR)
    _lg.propagate = False  # prevent bubbling to root handler



# --- Seed everything ------------------------------------------
def set_seed(seed: int = config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --- Run metadata ---------------------------------------------
def collect_run_metadata(args) -> dict:
    """Collect environment info for reproducibility."""
    import platform
    import sys as _sys

    meta = {
        "python_version": _sys.version.split()[0],
        "torch_version": torch.__version__,
        "command": " ".join(_sys.argv),
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        meta["gpu_name"] = torch.cuda.get_device_name(0)
        meta["gpu_total_mb"] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024**2), 0
        )
        try:
            import pynvml
            pynvml.nvmlInit()
            meta["cuda_driver_version"] = pynvml.nvmlSystemGetDriverVersion()
        except Exception:
            meta["cuda_driver_version"] = "unknown"
    try:
        import faster_whisper
        meta["faster_whisper_version"] = faster_whisper.__version__
    except Exception:
        pass
    try:
        import ctranslate2
        meta["ctranslate2_version"] = ctranslate2.__version__
    except Exception:
        pass
    return meta


# --- Load manifest -------------------------------------------
def load_manifest(split: str = "test") -> pd.DataFrame:
    """Load prepared manifest CSV."""
    manifest_path = config.PROCESSED_DIR / f"{split}_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            f"Run: python scripts/prepare_data.py"
        )
    df = pd.read_csv(manifest_path)
    logger.info(f"Loaded {len(df)} samples from {split} manifest")
    logger.info(f"Total audio: {df['duration_sec'].sum()/3600:.2f} hours")
    return df


# --- Audio augmentations --------------------------------------
DEFAULT_AUDIO_AUGMENTATIONS = [
    "clean",
    "noise_20db",
    "noise_10db",
    "speed_0.95",
    "speed_1.05",
    "gain_-6db",
    "lowpass_4000",
]


def parse_audio_augmentations(value: str | None) -> list[str]:
    """Parse CLI augmentation policy into concrete augmentation names."""
    if not value or value.lower() in {"none", "off", "false"}:
        return ["clean"]

    value = value.strip().lower()
    if value in {"default", "robust", "all"}:
        return DEFAULT_AUDIO_AUGMENTATIONS.copy()

    augmentations = [part.strip().lower() for part in value.split(",") if part.strip()]
    return augmentations or ["clean"]


def _read_audio(audio_path: str) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def _resample_to_16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == config.SAMPLE_RATE:
        return audio.astype(np.float32, copy=False)

    from math import gcd
    from scipy.signal import resample_poly

    divisor = gcd(sample_rate, config.SAMPLE_RATE)
    resampled = resample_poly(audio, config.SAMPLE_RATE // divisor, sample_rate // divisor)
    return np.asarray(resampled, dtype=np.float32)


def _parse_number(pattern: str, value: str, default: float) -> float:
    match = re.search(pattern, value)
    return float(match.group(1)) if match else default


def apply_audio_augmentation(audio: np.ndarray, augmentation: str, seed: int) -> np.ndarray:
    """
    Apply a deterministic single-file augmentation.

    Supported names:
      clean, noise_20db, noise_10db, speed_0.95, speed_1.05, gain_-6db,
      lowpass_4000, reverb
    """
    audio = audio.astype(np.float32, copy=True)

    if augmentation == "clean":
        return audio

    if augmentation.startswith("noise"):
        snr_db = _parse_number(r"noise_?(-?\d+(?:\.\d+)?)db", augmentation, 20.0)
        rng = np.random.default_rng(seed)
        signal_rms = float(np.sqrt(np.mean(audio ** 2)))
        if signal_rms == 0.0:
            return audio
        noise = rng.normal(0.0, 1.0, size=audio.shape).astype(np.float32)
        noise_rms = float(np.sqrt(np.mean(noise ** 2)))
        target_noise_rms = signal_rms / (10 ** (snr_db / 20.0))
        return audio + noise * (target_noise_rms / max(noise_rms, 1e-8))

    if augmentation.startswith("speed"):
        factor = _parse_number(r"speed_?(\d+(?:\.\d+)?)", augmentation, 1.0)
        if factor <= 0:
            raise ValueError(f"Invalid speed factor: {augmentation}")
        from scipy.signal import resample

        new_len = max(1, int(round(len(audio) / factor)))
        return resample(audio, new_len).astype(np.float32)

    if augmentation.startswith("gain"):
        gain_db = _parse_number(r"gain_?(-?\d+(?:\.\d+)?)db", augmentation, -6.0)
        return audio * (10 ** (gain_db / 20.0))

    if augmentation.startswith("lowpass"):
        cutoff_hz = _parse_number(r"lowpass_?(\d+(?:\.\d+)?)", augmentation, 4000.0)
        from scipy.signal import butter, sosfilt

        nyquist = config.SAMPLE_RATE / 2
        normalized_cutoff = min(max(cutoff_hz / nyquist, 0.01), 0.99)
        sos = butter(6, normalized_cutoff, btype="lowpass", output="sos")
        return sosfilt(sos, audio).astype(np.float32)

    if augmentation == "reverb":
        tail_len = int(config.SAMPLE_RATE * 0.25)
        decay = np.exp(-np.linspace(0, 4, tail_len)).astype(np.float32)
        impulse = np.zeros(tail_len, dtype=np.float32)
        impulse[0] = 1.0
        impulse[:: max(1, config.SAMPLE_RATE // 80)] += 0.08 * decay[:: max(1, config.SAMPLE_RATE // 80)]
        wet = np.convolve(audio, impulse, mode="full")[: len(audio)]
        return (wet / max(np.max(np.abs(wet)), 1.0)).astype(np.float32)

    raise ValueError(f"Unknown audio augmentation: {augmentation}")


def build_augmented_manifest(
    manifest: pd.DataFrame,
    augmentations: list[str],
    output_dir: Path,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """Expand manifest with clean and augmented audio variants."""
    augmentations = augmentations or ["clean"]
    invalid = [name for name in augmentations if not name]
    if invalid:
        raise ValueError(f"Invalid empty augmentation names: {invalid}")

    if augmentations == ["clean"]:
        clean_manifest = manifest.copy()
        clean_manifest["augmentation"] = "clean"
        clean_manifest["source_audio_path"] = clean_manifest["audio_path"]
        return clean_manifest

    import soundfile as sf

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for row_idx, row in manifest.reset_index(drop=True).iterrows():
        audio_path = str(row["audio_path"])
        audio, sample_rate = _read_audio(audio_path)
        audio = _resample_to_16k(audio, sample_rate)
        source_duration = len(audio) / config.SAMPLE_RATE

        for augmentation in augmentations:
            out_row = row.copy()
            out_row["augmentation"] = augmentation
            out_row["source_audio_path"] = audio_path

            if augmentation == "clean":
                out_row["duration_sec"] = round(source_duration, 3)
                rows.append(out_row)
                continue

            aug_seed = seed + row_idx * 1009 + zlib.crc32(augmentation.encode("utf-8"))
            augmented = apply_audio_augmentation(audio, augmentation, aug_seed)
            augmented = np.clip(augmented, -1.0, 1.0).astype(np.float32)

            output_path = output_dir / augmentation / Path(audio_path).name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_path), augmented, config.SAMPLE_RATE)

            out_row["audio_path"] = str(output_path.resolve())
            out_row["duration_sec"] = round(len(augmented) / config.SAMPLE_RATE, 3)
            rows.append(out_row)

    augmented_manifest = pd.DataFrame(rows)
    logger.info(
        f"Audio augmentations enabled: {augmentations}; "
        f"{len(manifest)} clean samples expanded to {len(augmented_manifest)} rows"
    )
    return augmented_manifest


# --- Warmup ---------------------------------------------------
def warmup_model(model: ASRModel, manifest: pd.DataFrame, n_warmup: int = 3) -> float:
    """Run a few warmup samples to stabilize GPU timings."""
    logger.info(f"Warming up {model.name} with {n_warmup} samples...")
    start = time.perf_counter()
    warmup_df = manifest.head(min(n_warmup, len(manifest)))
    for _, row in warmup_df.iterrows():
        try:
            model.transcribe(row["audio_path"], row["duration_sec"])
        except Exception as e:
            logger.warning(f"Warmup error: {e}")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - start


# --- Run inference --------------------------------------------
def run_inference(
    model: ASRModel,
    manifest: pd.DataFrame,
    warmup: bool = True,
) -> tuple[list[str], list[str], list[float], list[float], list[float], list[str], list[dict], dict]:
    """
    Run inference on all samples in manifest.

    Returns:
        references, hypotheses, durations, inference_times, gpu_mems, audio_paths,
        per-sample metadata, timing metadata
    """
    if not model._is_loaded:
        model.load()

    warmup_wall_sec = warmup_model(model, manifest) if warmup else 0.0

    references = []
    hypotheses = []
    durations = []
    inference_times = []
    gpu_mems = []
    audio_paths = []
    sample_metadata = []

    errors = 0
    total = len(manifest)
    inference_start = time.perf_counter()

    for idx, row in tqdm(
        manifest.iterrows(),
        total=total,
        desc=f"Inference [{model.name}]",
        unit="sample",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    ):
        audio_path = row["audio_path"]
        ref_text = row["text"]
        duration = row["duration_sec"]

        # Check file exists
        if not Path(audio_path).exists():
            logger.warning(f"Audio file not found: {audio_path}")
            errors += 1
            continue

        try:
            result = model.transcribe(audio_path, duration)

            references.append(ref_text)
            hypotheses.append(result.text)
            durations.append(duration)
            inference_times.append(result.inference_time_sec)
            gpu_mems.append(result.gpu_memory_peak_mb)
            audio_paths.append(audio_path)
            sample_metadata.append({
                "augmentation": row.get("augmentation", "clean"),
                "source_audio_path": row.get("source_audio_path", audio_path),
            })

        except Exception as e:
            import traceback as _tb
            logger.error(f"Error on {audio_path}: {e}\n{_tb.format_exc()}")
            errors += 1

            # OOM recovery
            if "out of memory" in str(e).lower():
                logger.warning("OOM detected! Clearing cache...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

    if errors:
        logger.warning(f"Total errors: {errors}/{total}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    timing = {
        "warmup_wall_sec": warmup_wall_sec,
        "inference_wall_sec": time.perf_counter() - inference_start,
    }

    return references, hypotheses, durations, inference_times, gpu_mems, audio_paths, sample_metadata, timing


# --- Visualization --------------------------------------------
def create_visualizations(results_df: pd.DataFrame, output_dir: Path):
    """Generate comparison plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_theme(style="whitegrid", font_scale=1.2)
        fig_dir = output_dir / "figures"
        fig_dir.mkdir(exist_ok=True)

        # --- WER comparison bar chart ---------------------
        fig, ax = plt.subplots(figsize=(12, 6))
        df_sorted = results_df.sort_values("wer")
        colors = sns.color_palette("viridis", len(df_sorted))
        bars = ax.barh(df_sorted["model_name"], df_sorted["wer"], color=colors)
        ax.set_xlabel("WER")
        ax.set_title("Word Error Rate by Model")
        for bar, val in zip(bars, df_sorted["wer"]):
            ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=10)
        plt.tight_layout()
        plt.savefig(fig_dir / "wer_comparison.png", dpi=150)
        plt.close()

        # --- WER vs RTF scatter ---------------------------
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(
            results_df["rtf"],
            results_df["wer"],
            s=results_df["gpu_memory_peak_mb"] / 10,
            c=range(len(results_df)),
            cmap="viridis",
            alpha=0.8,
            edgecolors="black",
            linewidth=0.5,
        )
        for _, row in results_df.iterrows():
            ax.annotate(
                row["model_name"],
                (row["rtf"], row["wer"]),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=8,
            )
        ax.set_xlabel("RTF (Real-Time Factor) - lower is faster")
        ax.set_ylabel("WER - lower is better")
        ax.set_title("WER vs Speed (bubble size = GPU memory)")
        plt.tight_layout()
        plt.savefig(fig_dir / "wer_vs_rtf.png", dpi=150)
        plt.close()

        # --- Error type breakdown -------------------------
        error_cols = ["substitution_rate", "deletion_rate", "insertion_rate"]
        if all(c in results_df.columns for c in error_cols):
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(results_df))
            width = 0.25
            ax.bar(x - width, results_df["substitution_rate"], width, label="Substitutions", color="#e74c3c")
            ax.bar(x, results_df["deletion_rate"], width, label="Deletions", color="#3498db")
            ax.bar(x + width, results_df["insertion_rate"], width, label="Insertions", color="#2ecc71")
            ax.set_xticks(x)
            ax.set_xticklabels(results_df["model_name"], rotation=45, ha="right")
            ax.set_ylabel("Error Rate")
            ax.set_title("Error Type Breakdown")
            ax.legend()
            plt.tight_layout()
            plt.savefig(fig_dir / "error_breakdown.png", dpi=150)
            plt.close()

        # --- WER by duration bucket ----------------------
        bucket_cols = ["wer_short", "wer_medium", "wer_long"]
        if all(c in results_df.columns for c in bucket_cols):
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(results_df))
            width = 0.25
            for i, (col, label, color) in enumerate(zip(
                bucket_cols,
                ["Short (<5s)", "Medium (5-15s)", "Long (>15s)"],
                ["#f39c12", "#e74c3c", "#9b59b6"],
            )):
                vals = pd.to_numeric(results_df[col], errors="coerce").fillna(0)
                ax.bar(x + i * width - width, vals, width, label=label, color=color)
            ax.set_xticks(x)
            ax.set_xticklabels(results_df["model_name"], rotation=45, ha="right")
            ax.set_ylabel("WER")
            ax.set_title("WER by Segment Duration")
            ax.legend()
            plt.tight_layout()
            plt.savefig(fig_dir / "wer_by_duration.png", dpi=150)
            plt.close()

        # --- Latency distribution -------------------------
        fig, ax = plt.subplots(figsize=(10, 6))
        latency_cols = ["avg_latency_sec", "p50_latency_sec", "p95_latency_sec"]
        if all(c in results_df.columns for c in latency_cols):
            x = np.arange(len(results_df))
            width = 0.25
            ax.bar(x - width, results_df["avg_latency_sec"], width, label="Mean", color="#3498db")
            ax.bar(x, results_df["p50_latency_sec"], width, label="P50", color="#2ecc71")
            ax.bar(x + width, results_df["p95_latency_sec"], width, label="P95", color="#e74c3c")
            ax.set_xticks(x)
            ax.set_xticklabels(results_df["model_name"], rotation=45, ha="right")
            ax.set_ylabel("Latency (sec)")
            ax.set_title("Inference Latency Distribution")
            ax.legend()
            plt.tight_layout()
            plt.savefig(fig_dir / "latency_distribution.png", dpi=150)
            plt.close()

        logger.info(f"Visualizations saved to {fig_dir}/")

    except ImportError as e:
        logger.warning(f"Skipping visualizations (missing dependency: {e})")


# --- Main -----------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ASR Benchmark Runner (LOCAL models only)")
    parser.add_argument(
        "--skip-data-prep", action="store_true",
        help="Skip data preparation step"
    )
    parser.add_argument(
        "--refresh-data", action="store_true",
        help="Force re-download and re-prepare dataset even if manifest exists"
    )
    parser.add_argument(
        "--models", type=str, default="whisper,gigaam,nemo",
        help=(
            "Comma-separated model families to benchmark.\n"
            "Available: whisper, phi4, gigaam, silero, nemo, vibevoice, vibevoice_4bit, gemma, gemma4\n"
            "Default: whisper,gigaam,nemo\n"
            "Models that fail to load (missing install) are skipped automatically.\n"
            "Note: vibevoice requires transformers from source (pip install git+https://github.com/huggingface/transformers.git)\n"
            "Note: silero has decoder issues, use with caution\n"
            "Note: phi4 has meta-tensor issues with PyTorch 2.6, may require PyTorch 2.5.1"
        )
    )
    parser.add_argument(
        "--split", type=str, default="test",
        help="Dataset split to evaluate (default: test)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Override max samples to process"
    )
    parser.add_argument(
        "--whisper-models", type=str, default=None,
        help="Comma-separated keys from config.WHISPER_MODELS to run"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory for CSV, Excel, figures, and analysis output"
    )
    parser.add_argument(
        "--audio-augmentations", "--augment-audio",
        nargs="?",
        const="default",
        default="clean",
        help=(
            "Comma-separated audio variants for robustness checks. "
            "Use without a value for default policy: "
            "clean,noise_20db,noise_10db,speed_0.95,speed_1.05,gain_-6db,lowpass_4000"
        ),
    )
    parser.add_argument(
        "--custom-dataset", type=Path, default=None,
        metavar="AUDIO_DIR",
        help=(
            "Path to a directory of your own audio files (WAV/MP3/FLAC/OGG/M4A/OPUS). "
            "Bypasses HuggingFace download entirely. "
            "Automatically sets --split custom and --skip-data-prep."
        )
    )
    parser.add_argument(
        "--custom-transcripts", type=Path, default=None,
        metavar="TRANSCRIPTS_FILE",
        help=(
            "Optional: CSV/TSV/JSONL file with transcripts for --custom-dataset. "
            "Columns: filename (or audio_path/file) + text (or transcript/sentence)."
        )
    )
    args = parser.parse_args()

    set_seed()
    benchmark_wall_start = time.perf_counter()

    logger.info("=" * 60)
    logger.info("ASR BENCHMARK - Russian Speech Recognition (LOCAL)")
    logger.info("=" * 60)

    # --- GPU info -----------------------------------------
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        logger.warning("No GPU detected! Running on CPU (will be very slow)")

    # --- Step 1: Prepare data --------------------------------
    # Path A: user's own audio dir  (--custom-dataset)
    # Path B: HuggingFace download  (default)
    if args.custom_dataset:
        from scripts.prepare_custom_data import prepare_custom_data
        split_name = "custom"
        args.split = split_name
        manifest_path = config.PROCESSED_DIR / f"{split_name}_manifest.csv"
        if not manifest_path.exists() or args.refresh_data:
            if args.refresh_data and manifest_path.exists():
                manifest_path.unlink()
            prepare_custom_data(
                audio_dir=args.custom_dataset,
                transcript_path=args.custom_transcripts,
                max_samples=args.max_samples,
                split_name=split_name,
            )
        else:
            logger.info(f"Custom dataset manifest already exists: {manifest_path}")
            logger.info("Use --refresh-data to re-import.")

    elif not args.skip_data_prep:
        manifest_path = config.PROCESSED_DIR / f"{args.split}_manifest.csv"
        need_prep = not manifest_path.exists() or args.refresh_data
        if args.refresh_data and manifest_path.exists():
            logger.info("--refresh-data: removing old manifest and re-preparing...")
            manifest_path.unlink()
        if need_prep:
            logger.info("Data not found - downloading and preparing automatically...")
            if args.max_samples and args.split == "test":
                config.MAX_TEST_SAMPLES = min(config.MAX_TEST_SAMPLES or args.max_samples, args.max_samples)
            from scripts.prepare_data import prepare_common_voice
            prepare_common_voice()
        else:
            logger.info(f"Data already prepared: {manifest_path}")

    else:
        logger.info("Skipping data preparation (--skip-data-prep)")

    # --- Step 2: Load manifest ----------------------------
    manifest = load_manifest(args.split)

    if args.max_samples and len(manifest) > args.max_samples:
        manifest = manifest.head(args.max_samples)
        logger.info(f"Limited to {args.max_samples} samples")

    augmentations = parse_audio_augmentations(args.audio_augmentations)
    output_root = args.output_dir or config.RESULTS_DIR
    manifest = build_augmented_manifest(
        manifest,
        augmentations,
        output_root / "augmented_audio",
        seed=config.SEED,
    )
    logger.info(f"Rows to benchmark after augmentation expansion: {len(manifest)}")

    # Count clean samples for small-dataset warning
    if "augmentation" in manifest.columns:
        n_clean = int((manifest["augmentation"] == "clean").sum())
    else:
        n_clean = len(manifest)

    # --- Step 3: Setup models (LOCAL only) ----------------
    model_families = set(args.models.lower().split(","))
    logger.info(f"Model families to benchmark: {model_families}")

    whisper_configs = None
    if args.whisper_models:
        requested = [name.strip() for name in args.whisper_models.split(",") if name.strip()]
        unknown = [name for name in requested if name not in config.WHISPER_MODELS]
        if unknown:
            raise ValueError(
                f"Unknown Whisper model keys: {unknown}. "
                f"Available: {list(config.WHISPER_MODELS)}"
            )
        whisper_configs = {name: config.WHISPER_MODELS[name] for name in requested}
        logger.info(f"Selected Whisper configs: {requested}")

    models = create_models(
        include_whisper="whisper" in model_families,
        include_gigaam="gigaam" in model_families,
        include_vibevoice="vibevoice" in model_families,
        include_vibevoice_4bit="vibevoice_4bit" in model_families,
        include_gemma="gemma" in model_families,
        include_phi4="phi4" in model_families,
        include_nemo="nemo" in model_families,
        include_silero="silero" in model_families,
        whisper_configs=whisper_configs,
    )

    if not models:
        logger.error("No models configured! Check --models argument.")
        return

    logger.info(f"Models to benchmark: {[m.name for m in models]}")

    # Warn when dataset is very small (smoke-test territory)
    n_clean = len(manifest[manifest.get('augmentation', 'clean') == 'clean']) \
        if 'augmentation' in manifest.columns else len(manifest)
    if n_clean < 100:
        logger.warning(
            f"SMALL DATASET WARNING: only {n_clean} clean samples. "
            "Results are a smoke/sanity check, NOT a production benchmark. "
            "Confidence intervals will be very wide."
        )

    run_meta = collect_run_metadata(args)

    # --- Step 4: Run benchmark ----------------------------
    all_metrics = []
    all_per_sample = []

    for i, model in enumerate(models):
        logger.info(f"\n{'='*60}")
        logger.info(f"Model {i+1}/{len(models)}: {model.name}")
        logger.info(f"{'='*60}")

        try:
            model_wall_start = time.perf_counter()

            # Load model
            load_start = time.perf_counter()
            model.load()
            model_load_wall_sec = time.perf_counter() - load_start

            # Run inference
            refs, hyps, durs, times, gpus, paths, sample_metadata, timing = run_inference(
                model, manifest, warmup=True
            )

            if not refs:
                logger.warning(f"No successful transcriptions for {model.name}")
                continue

            # Compute metrics
            metrics = compute_metrics(
                refs, hyps, durs, times, gpus, model.name
            )
            metrics.update({
                "model_load_wall_sec": round(model_load_wall_sec, 2),
                "warmup_wall_sec": round(timing["warmup_wall_sec"], 2),
                "inference_wall_sec": round(timing["inference_wall_sec"], 2),
                "model_benchmark_wall_sec": round(time.perf_counter() - model_wall_start, 2),
                "audio_augmentations": ",".join(augmentations),
                "augmentation_variants": len(augmentations),
                # run metadata for reproducibility
                "run_python": run_meta.get("python_version", ""),
                "run_torch": run_meta.get("torch_version", ""),
                "run_gpu": run_meta.get("gpu_name", "cpu"),
                "run_cuda_driver": run_meta.get("cuda_driver_version", ""),
                "run_faster_whisper": run_meta.get("faster_whisper_version", ""),
                "run_command": run_meta.get("command", ""),
                # dataset quality label
                "dataset_size_warning": "smoke_test" if n_clean < 100 else "ok",
            })
            all_metrics.append(metrics)

            # Per-sample results (with full text details)
            per_sample_df = compute_per_sample_results(
                refs, hyps, paths, durs, times, gpus, model.name,
                sample_metadata=sample_metadata,
            )
            all_per_sample.append(per_sample_df)

            # Log summary
            logger.info(f"\n--- {model.name} Summary ---")
            logger.info(f"  WER:  {metrics.get('wer', 'N/A')}")
            logger.info(f"  CER:  {metrics.get('cer', 'N/A')}")
            logger.info(f"  RTF:  {metrics.get('rtf', 'N/A')}")
            logger.info(f"  Avg Latency: {metrics.get('avg_latency_sec', 'N/A')} sec")
            logger.info(f"  GPU Peak:    {metrics.get('gpu_memory_peak_mb', 'N/A')} MB")
            logger.info(f"  Wall Time:   {metrics.get('model_benchmark_wall_sec', 'N/A')} sec")

        except Exception as e:
            logger.error(f"Failed to benchmark {model.name}: {e}")
            if str(config.LOG_LEVEL).upper() == "DEBUG":
                import traceback
                traceback.print_exc()

        finally:
            # Unload model to free VRAM
            try:
                model.unload()
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Brief pause between models
        time.sleep(2)

    # --- Step 5: Save results (CSV + Excel) ---------------
    if all_metrics:
        benchmark_total_wall_sec = time.perf_counter() - benchmark_wall_start
        results_dir = save_results(all_metrics, all_per_sample, output_dir=args.output_dir)

        # Generate visualizations
        results_df = pd.DataFrame(all_metrics)
        create_visualizations(results_df, results_dir)

        # --- Brief analysis ------------------------------
        print("\n" + "=" * 60)
        print("ANALYSIS")
        print("=" * 60)

        best_wer = min(all_metrics, key=lambda x: x.get("wer", 999))
        best_speed = min(all_metrics, key=lambda x: x.get("rtf", 999))

        print(f"\nBest quality (lowest WER): {best_wer['model_name']}")
        print(f"   WER = {best_wer['wer']:.4f}, CER = {best_wer['cer']:.4f}")

        print(f"\nFastest (lowest RTF): {best_speed['model_name']}")
        print(f"   RTF = {best_speed['rtf']:.4f}")

        print(f"\nTotal benchmark wall time: {benchmark_total_wall_sec:.2f} sec")
        print(f"Audio augmentations: {', '.join(augmentations)}")

        if best_wer["model_name"] != best_speed["model_name"]:
            print(f"\nTrade-off: {best_wer['model_name']} is best quality, "
                  f"{best_speed['model_name']} is fastest")
        else:
            print(f"\n{best_wer['model_name']} wins both quality AND speed!")

        # Save brief analysis
        analysis_path = results_dir / "analysis.txt"
        with open(analysis_path, "w", encoding="utf-8") as f:
            f.write("ASR Benchmark Analysis\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Best quality: {best_wer['model_name']} (WER={best_wer['wer']:.4f})\n")
            f.write(f"Fastest: {best_speed['model_name']} (RTF={best_speed['rtf']:.4f})\n\n")
            f.write(f"Total benchmark wall time: {benchmark_total_wall_sec:.2f} sec\n")
            f.write(f"Audio augmentations: {', '.join(augmentations)}\n\n")
            f.write("Full results:\n")
            for m in sorted(all_metrics, key=lambda x: x.get("wer", 999)):
                f.write(f"  {m['model_name']:40s} WER={m['wer']:.4f}  "
                        f"CER={m['cer']:.4f}  RTF={m['rtf']:.4f}  "
                        f"WALL={m.get('model_benchmark_wall_sec', 0):.2f}s  "
                        f"GPU={m['gpu_memory_peak_mb']:.0f}MB\n")

        logger.info(f"\nAll results saved to: {results_dir}/")
        logger.info(f"Excel report: {results_dir / 'benchmark_report.xlsx'}")

    else:
        logger.error("No results to save! All models failed.")

    logger.info("\n" + "=" * 60)
    logger.info("Benchmark complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
