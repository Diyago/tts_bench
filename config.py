"""
Central configuration for ASR benchmark.
All paths, model settings, and hyperparameters in one place.
"""

import torch
from pathlib import Path

# --- Random seed -----------------------------------------------
SEED = 42

# --- Paths -----------------------------------------------------
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT_DIR / "results"
MODELS_DIR = ROOT_DIR / "models"

for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, RESULTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- Device auto-detection ------------------------------------
# All model configs read _DEVICE so CPU fallback works everywhere.
# Override via env-var: ASR_DEVICE=cpu python scripts/run_benchmark.py
import os as _os
_DEVICE = _os.environ.get("ASR_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# --- Audio settings -------------------------------------------
SAMPLE_RATE = 16_000
MIN_DURATION_SEC = 1.0   # lowered from 3.0 – short commands matter
MAX_DURATION_SEC = 30.0
AUDIO_FORMAT = "wav"

# --- Dataset settings -----------------------------------------
# Primary: Mozilla Common Voice Russian (validated subset)
DATASET_NAME = "mozilla-foundation/common_voice_17_0"
DATASET_LANG = "ru"

# Fallback chain tried in order when the primary dataset fails.
# Each entry: (dataset_id, lang_or_None, text_field)
DATASET_FALLBACKS = [
    ("mozilla-foundation/common_voice_13_0", "ru", "sentence"),
    ("bond005/podlodka_speech", None, "transcription"),
]

MAX_TEST_SAMPLES = 1000
MAX_DEV_SAMPLES = 100

# --- Model configurations -------------------------------------
def _compute_type(fp16: str = "float16", int8_fp16: str = "int8_float16") -> str:
    """Pick the right quantisation depending on whether CUDA is available."""
    if _DEVICE == "cuda":
        return fp16
    return "int8"  # CPU-safe

WHISPER_MODELS = {
    # Best quality: large-v3 fp16, beam=5
    "whisper-large-v3": {
        "model_size": "large-v3",
        "compute_type": _compute_type("float16"),
        "device": _DEVICE,
        "beam_sizes": [5],
        "vad_filter": True,
        "language": "ru",
    },
    # Best speed/quality trade-off: medium int8, beam=5
    "whisper-medium-int8": {
        "model_size": "medium",
        "compute_type": _compute_type("int8_float16"),
        "device": _DEVICE,
        "beam_sizes": [5],
        "vad_filter": True,
        "language": "ru",
    },
    # Fast: small int8, beam=1 — competitive with gigaam_ctc speed
    "whisper-small-int8": {
        "model_size": "small",
        "compute_type": _compute_type("int8_float16"),
        "device": _DEVICE,
        "beam_sizes": [1],
        "vad_filter": True,
        "language": "ru",
    },
}

GIGAAM_CONFIG   = {"model_versions": ["ctc", "rnnt"], "device": _DEVICE}
VIBEVOICE_CONFIG = {"model_path": "microsoft/VibeVoice-ASR-HF", "device": _DEVICE, "use_4bit": True}
GEMMA_CONFIG    = {"model_id": "google/gemma-3n-E4B-it",  "device": _DEVICE, "use_4bit": False}
PHI4_CONFIG     = {
    "model_id": "microsoft/Phi-4-multimodal-instruct",
    "device": _DEVICE,
    # Keep VRAM under ~6 GB: prefer 4-bit quantization when available.
    "use_4bit": True,
}
QWEN_CONFIG     = {
    # Qwen2-Audio is the ASR model. Qwen2.5-Omni is speech-to-speech (not ASR).
    "model_ids": [
        "Qwen/Qwen2-Audio-7B",
    ],
    "device": _DEVICE,
    "use_4bit": True,  # 4-bit required: 7B model needs ~14GB in fp16, RTX 4070 Ti has 12GB
}
NEMO_CONFIG     = {"model_names": ["stt_ru_conformer_ctc_large"], "device": _DEVICE}
SILERO_CONFIG   = {"language": "ru", "device": _DEVICE}

# --- Benchmark settings ---------------------------------------
BATCH_SIZE = 16
GPU_MEMORY_LIMIT_GB = 11.0

# --- Text normalization ---------------------------------------
# Characters to keep after normalization (Russian + English + digits + space)
ALLOWED_CHARS = set(
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789 "
)

# Ё→Е normalisation: ASR often outputs «е» where reference has «ё».
# When True, both reference and hypothesis get ё→е before WER computation.
NORMALIZE_YO_YE = False

# --- Logging ---------------------------------------------------
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"

# Third-party loggers that spam per-sample messages — set to WARNING
QUIET_LOGGERS = [
    "faster_whisper",       # "Processing audio..." / "VAD filter removed..."
    "speechbrain",          # SpeechBrain progress bars
    "transformers",         # model loading verbosity
    "huggingface_hub",      # download progress
    "nemo_logging",         # NeMo internal logger
    "nemo",                 # NeMo top-level
    "nemo.collections",     # NeMo collections
    "nemo.collections.asr", # NeMo ASR module
    "nemo.utils",           # NeMo utils
    "nemo.core",            # NeMo core
    "lhotse",               # Lhotse dataloader spam
    "lhotse.cut",           # Lhotse cut spam
    "lhotse.dataset",       # Lhotse dataset spam
    "nv_one_logger",        # NVIDIA telemetry spam
    "nv_one_logger.api.config",
    "nv_one_logger.exporter.export_config_manager",
    "nv_one_logger.training_telemetry.api.training_telemetry_provider",
    "root",                 # NeMo dataloader "Initializing Lhotse CutSet..." messages
]
