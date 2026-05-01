"""
Unified inference interface for ASR models (LOCAL ONLY).

Provides a common `ASRModel` base class and implementations for:
- faster-whisper (Whisper Large-v3, Medium with various compute types)
- GigaAM-v3 (Sber, salute-developers: SOTA Russian ASR)
- VibeVoice-ASR (Microsoft, 7B multilingual ASR model)
- Gemma 3n E4B (Google, multimodal with native audio input)
- Gemma 4 (Google, next-gen multimodal with native audio)
- NeMo Conformer (NVIDIA, Conformer-CTC/Transducer ASR)
- Silero STT (lightweight Russian speech recognition)

Models NOT included (with reasons):
- KAME (Sakana AI): Speech-to-Speech model, not ASR; does not produce
  text transcripts. Based on Moshi S2S architecture.

Each model implements:
    transcribe(audio_path: str) -> str
    transcribe_batch(audio_paths: list[str]) -> list[str]
"""

import abc
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger("models.inference")


def _nvml_used_mb() -> float:
    """Return current GPU memory used (MB) via pynvml/nvidia-ml-py.

    Works for *any* CUDA backend (PyTorch, CTranslate2, TensorRT, …).
    Returns 0.0 when CUDA is absent or pynvml is not installed.
    """
    try:
        import pynvml  # pip install nvidia-ml-py
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / (1024 ** 2)
    except Exception:
        # pynvml not available – fall through to torch allocator
        return 0.0


@dataclass
class InferenceResult:
    """Result from a single inference call."""
    text: str
    audio_path: str
    duration_sec: float
    inference_time_sec: float
    gpu_memory_peak_mb: float = 0.0
    model_name: str = ""
    extra: dict = field(default_factory=dict)


class ASRModel(abc.ABC):
    """Abstract base class for ASR models."""

    def __init__(self, name: str, device: str = "cuda"):
        self.name = name
        self.device = device
        self._is_loaded = False

    @abc.abstractmethod
    def load(self):
        """Load the model into memory."""
        pass

    @abc.abstractmethod
    def _transcribe_single(self, audio_path: str) -> str:
        """Transcribe a single audio file. Returns raw text."""
        pass

    def transcribe(self, audio_path: str, duration_sec: float = 0.0) -> InferenceResult:
        """Transcribe with timing and memory tracking."""
        if not self._is_loaded:
            self.load()

        # Reset GPU memory tracking (torch allocator – works for PyTorch models)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        gpu_mem_before = _nvml_used_mb()

        start = time.perf_counter()
        text = self._transcribe_single(audio_path)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        # Prefer NVML (works for CTranslate2/faster-whisper too),
        # fall back to torch allocator for pure-PyTorch models.
        gpu_mem_nvml = _nvml_used_mb() - gpu_mem_before
        gpu_mem_torch = 0.0
        if torch.cuda.is_available():
            gpu_mem_torch = torch.cuda.max_memory_allocated() / (1024 ** 2)
        gpu_mem = max(gpu_mem_nvml, gpu_mem_torch)

        return InferenceResult(
            text=text,
            audio_path=audio_path,
            duration_sec=duration_sec,
            inference_time_sec=elapsed,
            gpu_memory_peak_mb=gpu_mem,
            model_name=self.name,
        )

    def transcribe_batch(
        self, audio_paths: list[str], durations: list[float]
    ) -> list[InferenceResult]:
        """Default batch = sequential single calls. Override for true batching."""
        results = []
        for path, dur in zip(audio_paths, durations):
            results.append(self.transcribe(path, dur))
        return results

    def unload(self):
        """Unload model from memory."""
        self._is_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r})"


# ===============================================================
# Faster-Whisper (CTranslate2)
# ===============================================================

class FasterWhisperModel(ASRModel):
    """Whisper model via faster-whisper (CTranslate2)."""

    def __init__(
        self,
        name: str,
        model_size: str = "large-v3",
        compute_type: str = "float16",
        device: str = "cuda",
        beam_size: int = 5,
        vad_filter: bool = True,
        language: str = "ru",
    ):
        super().__init__(name, device)
        self.model_size = model_size
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.language = language
        self._model = None

    def load(self, shared_model=None):
        """Load the WhisperModel, or reuse a pre-loaded shared instance."""
        from faster_whisper import WhisperModel

        if shared_model is not None:
            # Reuse a WhisperModel already loaded for a different beam size
            self._model = shared_model
            self._is_loaded = True
            logger.info(
                f"Model {self.name}: reusing shared faster-whisper "
                f"{self.model_size} ({self.compute_type})"
            )
            return

        logger.info(
            f"Loading faster-whisper {self.model_size} "
            f"(compute={self.compute_type}, beam={self.beam_size})"
        )
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        self._is_loaded = True
        logger.info(f"Model {self.name} loaded successfully")

    def _transcribe_single(self, audio_path: str) -> str:
        segments, info = self._model.transcribe(
            audio_path,
            beam_size=self.beam_size,
            language=self.language,
            vad_filter=self.vad_filter,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=200,
            ),
            without_timestamps=True,
        )
        text = " ".join(seg.text.strip() for seg in segments)
        return text

    def unload(self):
        del self._model
        self._model = None
        super().unload()


# ===============================================================
# GigaAM-v3 (Sber / salute-developers)
# ===============================================================

class GigaAMModel(ASRModel):
    """
    GigaAM-v3: Sber's SOTA Russian ASR model.

    Install:
        git clone https://github.com/salute-developers/GigaAM.git
        cd GigaAM && pip install -e .[torch]

    Models available:
        v3_ctc      - CTC decoder (no punctuation)
        v3_rnnt     - RNN-T decoder (no punctuation)
        v3_e2e_ctc  - End-to-end with punctuation & normalization
        v3_e2e_rnnt - End-to-end with punctuation & normalization

    For benchmark we use v3_ctc (raw text, fair WER comparison)
    and v3_e2e_rnnt (end-to-end, shows full model capability).
    """

    def __init__(
        self,
        name: str = "gigaam_v3",
        model_version: str = "v3_ctc",
        device: str = "cuda",
    ):
        super().__init__(name, device)
        self.model_version = model_version
        self._model = None

    def load(self):
        try:
            import gigaam

            logger.info(f"Loading GigaAM model: {self.model_version}")
            self._model = gigaam.load_model(self.model_version)
            self._model.eval()

            if self.device == "cuda" and torch.cuda.is_available():
                self._model = self._model.cuda()

            self._is_loaded = True
            logger.info(f"GigaAM {self.model_version} loaded successfully")

        except ImportError:
            logger.error(
                "gigaam not installed. Install with:\n"
                "  git clone https://github.com/salute-developers/GigaAM.git\n"
                "  cd GigaAM && pip install -e .[torch]"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load GigaAM: {e}")
            raise

    def _transcribe_single(self, audio_path: str) -> str:
        result = self._model.transcribe(audio_path)
        # transcribe() returns either a string or a TranscriptionResult
        if hasattr(result, 'text'):
            return result.text
        return str(result)

    def unload(self):
        del self._model
        self._model = None
        super().unload()


# ===============================================================
# VibeVoice-ASR (Microsoft)
# ===============================================================

class VibeVoiceModel(ASRModel):
    """
    VibeVoice-ASR: Microsoft's 7B multilingual ASR model.
    Supports 50+ languages including Russian.
    Handles up to 60-minute audio in a single pass.

    Install:
        git clone https://github.com/microsoft/VibeVoice.git
        cd VibeVoice && pip install -e .

    WARNING: 7B model requires ~14GB VRAM in FP16.
    On RTX 4070 Ti (12GB), use int4/int8 quantization or skip.

    Note: The model produces structured output with speaker/timestamp info.
    We extract only the text content for WER evaluation.
    """

    def __init__(
        self,
        name: str = "vibevoice_asr",
        model_path: str = "microsoft/VibeVoice-ASR",
        device: str = "cuda",
        use_4bit: bool = True,  # quantize to fit 12GB VRAM
    ):
        super().__init__(name, device)
        self.model_path = model_path
        self.use_4bit = use_4bit
        self._model = None
        self._processor = None

    def load(self):
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
            import re as _re

            logger.info(f"Loading VibeVoice-ASR: {self.model_path}")

            load_kwargs = {
                "trust_remote_code": True,
                "device_map": "auto",
            }

            if self.use_4bit:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                logger.info("Using 4-bit quantization to fit 12GB VRAM")
            else:
                load_kwargs["torch_dtype"] = torch.float16

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path, **load_kwargs
            )
            self._processor = AutoProcessor.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            self._model.eval()
            self._is_loaded = True
            logger.info("VibeVoice-ASR loaded successfully")

        except ImportError as e:
            logger.error(
                f"VibeVoice dependencies missing: {e}\n"
                "Install with:\n"
                "  git clone https://github.com/microsoft/VibeVoice.git\n"
                "  cd VibeVoice && pip install -e ."
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load VibeVoice: {e}")
            raise

    def _transcribe_single(self, audio_path: str) -> str:
        import re
        import torchaudio

        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)

        # Prepare input
        inputs = self._processor(
            audio=waveform.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        # Generate
        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )

        # Decode
        raw_text = self._processor.decode(outputs[0], skip_special_tokens=True)

        # VibeVoice produces structured output with speaker/timestamp tags.
        # Extract just the spoken content.
        # Pattern: [Speaker X] [00:00.000 --> 00:05.000] Text content
        text_parts = []
        for line in raw_text.strip().split("\n"):
            # Remove speaker and timestamp tags
            cleaned = re.sub(r"\[.*?\]", "", line).strip()
            if cleaned:
                text_parts.append(cleaned)

        return " ".join(text_parts) if text_parts else raw_text

    def unload(self):
        del self._model
        del self._processor
        self._model = None
        self._processor = None
        super().unload()


# ===============================================================
# Gemma 3n E4B (Google: multimodal with audio)
# ===============================================================

class Gemma3nModel(ASRModel):
    """
    Gemma 3n E4B: Google's lightweight multimodal model with native audio.

    Supports text, image, video, AND audio input.
    Audio is encoded at 6.25 tokens/second from a single channel.
    Effective parameters: 4B (actual 8B with selective activation).
    Fits ~4-5 GB VRAM in bfloat16.

    Requires: transformers >= 4.53.0
    Model: google/gemma-3n-E4B-it (or E2B for smaller variant)

    Note: This is a general-purpose multimodal model, not a dedicated ASR.
    We prompt it to transcribe speech, which is useful to compare with specialized models.
    """

    def __init__(
        self,
        name: str = "gemma_3n_e4b",
        model_id: str = "google/gemma-3n-E4B-it",
        device: str = "cuda",
    ):
        super().__init__(name, device)
        self.model_id = model_id
        self._model = None
        self._processor = None

    def load(self):
        try:
            from transformers import AutoProcessor, Gemma3nForConditionalGeneration

            logger.info(f"Loading Gemma 3n: {self.model_id}")

            self._model = Gemma3nForConditionalGeneration.from_pretrained(
                self.model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            ).eval()

            self._processor = AutoProcessor.from_pretrained(self.model_id)

            self._is_loaded = True
            logger.info(f"Gemma 3n loaded successfully")

        except ImportError as e:
            logger.error(
                f"Gemma 3n dependencies missing: {e}\n"
                "Install with: pip install -U 'transformers>=4.53.0'"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load Gemma 3n: {e}")
            raise

    def _transcribe_single(self, audio_path: str) -> str:
        import librosa

        # Load audio as float32 numpy array at 16kHz
        audio_data, sr = librosa.load(audio_path, sr=16000, mono=True)

        # Build chat message with audio input
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_data},
                    {
                        "type": "text",
                        "text": (
                            "Transcribe the speech in this audio exactly as spoken. "
                            "Output only the transcription text, nothing else. "
                            "The language is Russian."
                        ),
                    },
                ],
            }
        ]

        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self._model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

        output_tokens = generation[0][input_len:]
        text = self._processor.decode(output_tokens, skip_special_tokens=True)
        return text.strip()

    def unload(self):
        del self._model
        del self._processor
        self._model = None
        self._processor = None
        super().unload()


# ===============================================================
# Gemma 4 (Google: next-gen multimodal with audio)
# ===============================================================

class Gemma4Model(ASRModel):
    """
    Gemma 4: Google's next-generation multimodal model with native audio.

    Successor to Gemma 3n, with improved audio understanding capabilities.
    Supports text, image, video, and audio input natively.
    Available in multiple sizes; we use the instruction-tuned variant.

    Requires: transformers >= 4.55.0
    Model: google/gemma-4-12b-it (or smaller variants)

    Note: Like Gemma 3n, this is a general-purpose multimodal model.
    We prompt it for transcription, which is useful for quality comparison.
    VRAM: ~8-10 GB with 4-bit quantization (12b variant).
    """

    def __init__(
        self,
        name: str = "gemma_4",
        model_id: str = "google/gemma-4-12b-it",
        device: str = "cuda",
        use_4bit: bool = True,
    ):
        super().__init__(name, device)
        self.model_id = model_id
        self.use_4bit = use_4bit
        self._model = None
        self._processor = None

    def load(self):
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM

            logger.info(f"Loading Gemma 4: {self.model_id}")

            load_kwargs = {
                "device_map": "auto",
                "trust_remote_code": True,
            }

            if self.use_4bit:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                logger.info("Using 4-bit quantization for Gemma 4")
            else:
                load_kwargs["torch_dtype"] = torch.bfloat16

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id, **load_kwargs
            ).eval()

            self._processor = AutoProcessor.from_pretrained(
                self.model_id, trust_remote_code=True
            )

            self._is_loaded = True
            logger.info(f"Gemma 4 loaded successfully")

        except ImportError as e:
            logger.error(
                f"Gemma 4 dependencies missing: {e}\n"
                "Install with: pip install -U 'transformers>=4.55.0' bitsandbytes"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load Gemma 4: {e}")
            raise

    def _transcribe_single(self, audio_path: str) -> str:
        import librosa

        # Load audio as float32 numpy array at 16kHz
        audio_data, sr = librosa.load(audio_path, sr=16000, mono=True)

        # Build chat message with audio input
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_data},
                    {
                        "type": "text",
                        "text": (
                            "Transcribe the speech in this audio exactly as spoken. "
                            "Output only the transcription text, nothing else. "
                            "The language is Russian."
                        ),
                    },
                ],
            }
        ]

        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self._model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

        output_tokens = generation[0][input_len:]
        text = self._processor.decode(output_tokens, skip_special_tokens=True)
        return text.strip()

    def unload(self):
        del self._model
        del self._processor
        self._model = None
        self._processor = None
        super().unload()


# ===============================================================
# NeMo Conformer (NVIDIA)
# ===============================================================

class NeMoConformerModel(ASRModel):
    """
    NVIDIA NeMo Conformer: Conformer-CTC/Transducer ASR model.

    NeMo provides pre-trained Conformer models for many languages,
    including Russian. Conformer combines convolutions with
    self-attention for strong ASR performance.

    Install:
        pip install nemo_toolkit[asr]

    Models available (Russian-capable):
        stt_ru_conformer_ctc_large        - Conformer-CTC Large (Russian)
        stt_ru_conformer_transducer_large - Conformer-Transducer Large (Russian)

    VRAM: ~2-4 GB depending on model size.
    """

    def __init__(
        self,
        name: str = "nemo_conformer",
        model_name: str = "stt_ru_conformer_ctc_large",
        device: str = "cuda",
    ):
        super().__init__(name, device)
        self.model_name = model_name
        self._model = None

    def load(self):
        try:
            import nemo.collections.asr as nemo_asr

            logger.info(f"Loading NeMo Conformer: {self.model_name}")

            # NeMo models are loaded from pre-trained checkpoints
            self._model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=self.model_name
            )

            if self.device == "cuda" and torch.cuda.is_available():
                self._model = self._model.cuda()

            self._model.eval()
            self._is_loaded = True
            logger.info(f"NeMo Conformer {self.model_name} loaded successfully")

        except ImportError:
            logger.error(
                "nemo_toolkit not installed. Install with:\n"
                "  pip install nemo_toolkit[asr]"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load NeMo Conformer: {e}")
            raise

    def _transcribe_single(self, audio_path: str) -> str:
        # NeMo's transcribe() accepts a list of paths and returns a list of strings
        results = self._model.transcribe([audio_path])
        # Handle both old-style (list of str) and new-style (Hypothesis objects)
        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, 'text'):
                return result.text
            return str(result)
        return ""

    def unload(self):
        del self._model
        self._model = None
        super().unload()


# ===============================================================
# Silero STT (Silero Team)
# ===============================================================

class SileroSTTModel(ASRModel):
    """
    Silero STT: Lightweight, fast Russian speech recognition.

    Pre-trained models from Silero team, optimized for Russian.
    Loaded via torch.hub. Very small footprint (~100 MB),
    runs fast even on CPU. Good baseline for Russian ASR.

    No extra install needed beyond torch + torchaudio.
    Models are downloaded automatically via torch.hub.

    VRAM: ~0.5-1 GB (very lightweight, works on CPU too).
    """

    def __init__(
        self,
        name: str = "silero_stt",
        language: str = "ru",
        device: str = "cuda",
    ):
        super().__init__(name, device)
        self.language = language
        self._model = None
        self._decoder = None
        self._read_batch = None
        self._split_into_batches = None

    def load(self):
        try:
            logger.info(f"Loading Silero STT (language={self.language})")

            model, decoder, read_batch, split_into_batches, _, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model='silero_stt',
                language=self.language,
                device=torch.device(self.device if torch.cuda.is_available() else 'cpu'),
            )

            self._model = model
            self._decoder = decoder
            self._read_batch = read_batch
            self._split_into_batches = split_into_batches
            self._is_loaded = True
            logger.info("Silero STT loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Silero STT: {e}")
            raise

    def _transcribe_single(self, audio_path: str) -> str:
        import torchaudio

        # Load and resample audio to 16kHz
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)

        # Ensure mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        device = torch.device(self.device if torch.cuda.is_available() else 'cpu')
        waveform = waveform.to(device)

        # Transcribe
        with torch.inference_mode():
            output = self._model(waveform)

        # Decode
        if self._decoder is not None:
            text = self._decoder(output[0])
        else:
            # Fallback: greedy decode
            text = ""
            if hasattr(output, 'argmax'):
                indices = output.argmax(dim=-1)
                text = str(indices)

        return text if isinstance(text, str) else str(text)

    def unload(self):
        del self._model
        del self._decoder
        del self._read_batch
        del self._split_into_batches
        self._model = None
        self._decoder = None
        self._read_batch = None
        self._split_into_batches = None
        super().unload()


# ===============================================================
# Factory: create all configured models
# ===============================================================

def create_models(
    include_whisper: bool = True,
    include_gigaam: bool = True,
    include_vibevoice: bool = False,
    include_gemma: bool = False,
    include_gemma4: bool = False,
    include_nemo: bool = False,
    include_silero: bool = False,
    whisper_configs: Optional[dict] = None,
) -> list[ASRModel]:
    """
    Create all configured LOCAL models.

    Args:
        include_whisper: Include faster-whisper models
        include_gigaam: Include GigaAM-v3 models
        include_vibevoice: Include VibeVoice-ASR (needs ~14GB VRAM, use 4bit)
        include_gemma: Include Gemma 3n E4B (multimodal with audio)
        include_gemma4: Include Gemma 4 (next-gen multimodal with audio)
        include_nemo: Include NeMo Conformer (NVIDIA)
        include_silero: Include Silero STT (lightweight Russian)
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import config as cfg

    models = []

    # --- Whisper (faster-whisper) -------------------------
    # Group by (model_size, compute_type, device) so that configs that differ
    # only in beam_size share a single WhisperModel instance.  The first
    # FasterWhisperModel in the group loads the weights; subsequent ones call
    # load(shared_model=<existing instance>) for free.
    if include_whisper:
        configs = whisper_configs or cfg.WHISPER_MODELS
        # key → list of FasterWhisperModel objects (sorted by beam)
        weight_groups: dict = {}
        for name, conf in configs.items():
            key = (conf["model_size"], conf["compute_type"], conf["device"])
            for beam in sorted(conf.get("beam_sizes", [5])):
                model_name = f"{name}_beam{beam}"
                m = FasterWhisperModel(
                    name=model_name,
                    model_size=conf["model_size"],
                    compute_type=conf["compute_type"],
                    device=conf["device"],
                    beam_size=beam,
                    vad_filter=conf.get("vad_filter", True),
                    language=conf.get("language", "ru"),
                )
                weight_groups.setdefault(key, []).append(m)

        for key, group in weight_groups.items():
            primary = group[0]
            # Secondary beam-size variants share primary's _model handle.
            # We monkey-patch load() so they call primary.load() if needed.
            for secondary in group[1:]:
                _primary_ref = primary  # closure capture
                def _shared_load(self, shared_model=None, _p=_primary_ref):
                    if not _p._is_loaded:
                        _p.load()
                    self._model = _p._model
                    self._is_loaded = True
                    logger.info(
                        f"{self.name}: sharing weights with {_p.name}"
                    )
                import types
                secondary.load = types.MethodType(_shared_load, secondary)
            models.extend(group)

    # --- GigaAM-v3 ---------------------------------------
    if include_gigaam:
        # v3_ctc: raw text output, best for fair WER comparison
        models.append(
            GigaAMModel(
                name="gigaam_v3_ctc",
                model_version="v3_ctc",
                device=cfg.GIGAAM_CONFIG["device"],
            )
        )
        # v3_e2e_rnnt: end-to-end with punctuation (to show full capability)
        models.append(
            GigaAMModel(
                name="gigaam_v3_e2e_rnnt",
                model_version="v3_e2e_rnnt",
                device=cfg.GIGAAM_CONFIG["device"],
            )
        )

    # --- VibeVoice-ASR ------------------------------------
    if include_vibevoice:
        models.append(
            VibeVoiceModel(
                name="vibevoice_asr",
                model_path="microsoft/VibeVoice-ASR",
                device="cuda",
                use_4bit=True,  # 4-bit quant to fit 12GB VRAM
            )
        )

    # --- Gemma 3n E4B -------------------------------------
    if include_gemma:
        models.append(
            Gemma3nModel(
                name="gemma_3n_e4b",
                model_id=cfg.GEMMA_CONFIG["model_id"],
                device=cfg.GEMMA_CONFIG["device"],
            )
        )

    # --- Gemma 4 ------------------------------------------
    if include_gemma4:
        models.append(
            Gemma4Model(
                name="gemma_4",
                model_id=cfg.GEMMA4_CONFIG["model_id"],
                device=cfg.GEMMA4_CONFIG["device"],
                use_4bit=cfg.GEMMA4_CONFIG.get("use_4bit", True),
            )
        )

    # --- NeMo Conformer -----------------------------------
    if include_nemo:
        for model_name in cfg.NEMO_CONFIG["model_names"]:
            safe_name = model_name.replace("stt_ru_", "").replace("_", "-")
            models.append(
                NeMoConformerModel(
                    name=f"nemo_{safe_name}",
                    model_name=model_name,
                    device=cfg.NEMO_CONFIG["device"],
                )
            )

    # --- Silero STT ---------------------------------------
    if include_silero:
        models.append(
            SileroSTTModel(
                name="silero_stt",
                language=cfg.SILERO_CONFIG["language"],
                device=cfg.SILERO_CONFIG["device"],
            )
        )

    return models
