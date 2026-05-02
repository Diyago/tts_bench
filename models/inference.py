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

    def _record_load_memory(self):
        """Call this AFTER the model weights are loaded to record GPU footprint.
        Uses NVML so it works for both PyTorch and CTranslate2 backends.
        """
        mem_nvml = _nvml_used_mb()
        mem_torch = 0.0
        if torch.cuda.is_available():
            mem_torch = torch.cuda.memory_allocated() / (1024 ** 2)
        # Use whichever backend shows more (one of them will be 0 for non-torch models)
        self._loaded_gpu_mem_mb = max(mem_nvml, mem_torch)

    @abc.abstractmethod
    def _transcribe_single(self, audio_path: str) -> str:
        """Transcribe a single audio file. Returns raw text."""
        pass

    def transcribe(self, audio_path: str, duration_sec: float = 0.0) -> InferenceResult:
        """Transcribe with timing. GPU memory is measured at load time."""
        if not self._is_loaded:
            self.load()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        text = self._transcribe_single(audio_path)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        # Use the footprint recorded at load time (accurate for CTranslate2).
        gpu_mem = getattr(self, "_loaded_gpu_mem_mb", 0.0)

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
        """Load the WhisperModel, or reuse a pre-loaded shared instance.

        Loading strategy:
          1. Try local HF cache (local_files_only=True) — no network required.
          2. On cache miss, attempt download from HuggingFace.
          3. On any network/SSL error, raise with a clear actionable message.
        """
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

        common_kwargs = dict(device=self.device, compute_type=self.compute_type)

        # --- 1. Try local cache first (works offline / behind firewall) ------
        try:
            logger.info(f"  Trying local HF cache for {self.model_size}...")
            self._model = WhisperModel(
                self.model_size,
                local_files_only=True,
                **common_kwargs,
            )
            self._is_loaded = True
            self._record_load_memory()
            logger.info(f"  Loaded {self.name} from local cache (GPU: {self._loaded_gpu_mem_mb:.0f} MB).")
            return
        except Exception as cache_err:
            logger.info(
                f"  Not in local cache ({type(cache_err).__name__}), "
                "will attempt download..."
            )

        # --- 2. Download from HuggingFace ------------------------------------
        try:
            self._model = WhisperModel(self.model_size, **common_kwargs)
            self._is_loaded = True
            self._record_load_memory()
            logger.info(f"Model {self.name} downloaded and loaded (GPU: {self._loaded_gpu_mem_mb:.0f} MB).")
        except Exception as dl_err:
            err_str = str(dl_err)
            if any(kw in err_str.lower() for kw in ("ssl", "timeout", "connection", "proxy")):
                raise RuntimeError(
                    f"Network/SSL error while downloading faster-whisper '{self.model_size}'.\n"
                    "Possible fixes:\n"
                    "  1. Set HF_HUB_OFFLINE=1 and ensure the model is already cached:\n"
                    "       set HF_HUB_OFFLINE=1 (Windows) or export HF_HUB_OFFLINE=1\n"
                    "  2. Download the model manually:\n"
                    f"       huggingface-cli download Systran/faster-whisper-{self.model_size}\n"
                    "  3. Use a VPN or configure HF_ENDPOINT for a mirror.\n"
                    f"  Original error: {dl_err}"
                ) from dl_err
            raise

    # Class-level flag so the numpy/onnxruntime warning prints only once
    _vad_broken_warned: bool = False

    def _transcribe_single(self, audio_path: str) -> str:
        common_kwargs = dict(
            beam_size=self.beam_size,
            language=self.language,
            without_timestamps=True,
        )
        vad_kwargs = dict(
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=200,
            ),
        )

        if self.vad_filter:
            try:
                segments, _ = self._model.transcribe(
                    audio_path, **common_kwargs, **vad_kwargs
                )
                return " ".join(seg.text.strip() for seg in segments)
            except Exception as e:
                err = str(e)
                # onnxruntime compiled against NumPy 1.x breaks on NumPy 2.x
                if "onnxruntime" in err or "_ARRAY_API" in err or "VAD" in err:
                    if not FasterWhisperModel._vad_broken_warned:
                        logger.warning(
                            "VAD filter failed (onnxruntime/NumPy 2.x incompatibility). "
                            "Disabling VAD for ALL remaining samples.\n"
                            "Permanent fix: pip install \"numpy<2\" "
                            "OR pip install --upgrade onnxruntime"
                        )
                        FasterWhisperModel._vad_broken_warned = True
                    # Permanently disable VAD on this instance to prevent
                    # re-triggering the broken onnxruntime import
                    self.vad_filter = False
                    # Fall through to no-VAD transcription below
                else:
                    raise

        # No-VAD path (either vad_filter=False or VAD broke above)
        segments, _ = self._model.transcribe(audio_path, **common_kwargs, vad_filter=False)
        return " ".join(seg.text.strip() for seg in segments)


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

            # PyTorch 2.6 switched torch.load default to weights_only=True.
            # Some GigaAM checkpoints include OmegaConf objects, which require
            # allowlisting under the safe unpickler. We do a *scoped* allowlist
            # only around the GigaAM load call (no global torch.load monkeypatch).
            try:
                from torch.serialization import safe_globals as _safe_globals
                from typing import Any as _Any
                from omegaconf import DictConfig as _DictConfig, ListConfig as _ListConfig
                from omegaconf.base import ContainerMetadata as _ContainerMetadata

                with _safe_globals([_Any, _DictConfig, _ListConfig, _ContainerMetadata]):
                    self._model = gigaam.load_model(self.model_version)
                logger.info("  GigaAM: loaded with torch.serialization.safe_globals allowlist")
            except Exception as _safe_err:
                # Older torch may not have safe_globals, or omega conf may be absent.
                # Best-effort fallback: extend allowlist globally for this process.
                try:
                    import torch.serialization as _ts
                    from typing import Any as _Any
                    from omegaconf import DictConfig as _DictConfig, ListConfig as _ListConfig
                    from omegaconf.base import ContainerMetadata as _ContainerMetadata
                    _ts.add_safe_globals([_Any, _DictConfig, _ListConfig, _ContainerMetadata])
                    self._model = gigaam.load_model(self.model_version)
                    logger.info("  GigaAM: loaded after add_safe_globals allowlist")
                except Exception:
                    # Last-resort compatibility fallback (unsafe): force weights_only=False
                    # only for the duration of gigaam.load_model(). This can execute
                    # arbitrary code embedded in the checkpoint — do this only if you
                    # trust the checkpoint source.
                    import torch as _torch
                    _orig_load = _torch.load

                    def _patched_load(*a, **kw):
                        kw.setdefault("weights_only", False)
                        return _orig_load(*a, **kw)

                    logger.warning(
                        "  GigaAM: safe loading failed; retrying with torch.load(weights_only=False). "
                        "Only safe if you trust the checkpoint."
                    )
                    _torch.load = _patched_load
                    try:
                        self._model = gigaam.load_model(self.model_version)
                        logger.info("  GigaAM: loaded with weights_only=False fallback")
                    finally:
                        _torch.load = _orig_load

            self._model.eval()

            if self.device == "cuda" and torch.cuda.is_available():
                self._model = self._model.cuda()

            self._is_loaded = True
            self._record_load_memory()
            logger.info(f"GigaAM {self.model_version} loaded successfully (GPU: {self._loaded_gpu_mem_mb:.0f} MB)")

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
        try:
            result = self._model.transcribe(audio_path)
        except Exception as e:
            msg = str(e)
            if "Too long wav file" in msg:
                # GigaAM requires a special API for long audio. In some installs,
                # transcribe_longform depends on pyannote.audio (not always present).
                if hasattr(self._model, "transcribe_longform"):
                    try:
                        result = self._model.transcribe_longform(audio_path)
                    except Exception as lf_err:
                        # Fallback: chunk audio and run regular transcribe().
                        # This avoids extra diarization deps.
                        if "pyannote.audio" not in str(lf_err):
                            raise
                        result = self._transcribe_chunked(audio_path)
                else:
                    result = self._transcribe_chunked(audio_path)
            else:
                raise

        # transcribe() returns either a string or a TranscriptionResult
        if hasattr(result, "text"):
            return result.text
        return str(result)

    def _transcribe_chunked(self, audio_path: str, chunk_sec: float = 15.0, overlap_sec: float = 0.5) -> str:
        """Chunk long audio into smaller windows and transcribe sequentially."""
        import tempfile
        from pathlib import Path
        import numpy as _np
        import soundfile as _sf

        audio, sr = _sf.read(audio_path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = _np.asarray(audio, dtype=_np.float32)

        chunk_len = max(1, int(chunk_sec * sr))
        overlap = max(0, int(overlap_sec * sr))
        step = max(1, chunk_len - overlap)

        texts: list[str] = []
        with tempfile.TemporaryDirectory(prefix="gigaam_chunks_") as td:
            td_path = Path(td)
            idx = 0
            for start in range(0, len(audio), step):
                end = min(len(audio), start + chunk_len)
                if end - start < int(0.25 * sr):
                    break
                chunk = audio[start:end]
                chunk_path = td_path / f"chunk_{idx:04d}.wav"
                _sf.write(str(chunk_path), chunk, sr)
                r = self._model.transcribe(str(chunk_path))
                t = r.text if hasattr(r, "text") else str(r)
                t = (t or "").strip()
                if t:
                    texts.append(t)
                idx += 1

        return " ".join(texts).strip()

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
        model_path: str = "microsoft/VibeVoice-ASR-HF",
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
            from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor
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

            self._processor = AutoProcessor.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            
            try:
                self._model = AutoModel.from_pretrained(
                    self.model_path, **load_kwargs
                )
            except Exception as e:
                logger.warning(f"AutoModel failed ({e}), trying AutoModelForCausalLM...")
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, **load_kwargs
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
        # Prepare inputs
        inputs = self._processor.apply_transcription_request(audio=audio_path)
        inputs = {k: v.to(self._model.device, self._model.dtype) if isinstance(v, torch.Tensor) and torch.is_floating_point(v) else v.to(self._model.device) for k, v in inputs.items()}

        # Generate
        with torch.inference_mode():
            output_ids = self._model.generate(**inputs)

        # Extract only the generated transcription tokens
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]

        # Decode to get structured output
        transcription = self._processor.decode(generated_ids, return_format="parsed")[0]
        
        # Extract just the spoken content (What)
        text_parts = []
        if isinstance(transcription, list):
            for segment in transcription:
                if isinstance(segment, dict) and "What" in segment:
                    text_parts.append(segment["What"])
                elif isinstance(segment, str):
                    text_parts.append(segment)
        elif isinstance(transcription, str):
            import re
            for line in transcription.strip().split("\n"):
                cleaned = re.sub(r"\[.*?\]", "", line).strip()
                if cleaned:
                    text_parts.append(cleaned)
                    
        return " ".join(text_parts).strip()

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
# Phi-4 Multimodal (Microsoft)
# ===============================================================

class Phi4Model(ASRModel):
    """Phi-4-multimodal-instruct (Microsoft) for ASR.

    - 5.6B parameters, multimodal (text + vision + audio)
    - Uses AutoModelForCausalLM with trust_remote_code=True
    - Audio loaded via soundfile as (array, samplerate) tuple
    - Audio languages: EN, ZH, DE, FR, IT, JA, ES, PT
    - Russian is NOT officially supported for audio but may work

    Requires:
        pip install transformers torch soundfile accelerate

    VRAM: ~4-5 GB in bfloat16 on GPU.
    """

    _USER = "<|user|>"
    _ASST = "<|assistant|>"
    _END  = "<|end|>"
    _ASR_PROMPT = "Transcribe the audio clip into text."

    def __init__(
        self,
        name: str = "phi4_multimodal",
        model_id: str = "microsoft/Phi-4-multimodal-instruct",
        device: str = "cuda",
        use_4bit: bool = True,
    ):
        super().__init__(name, device)
        self.model_id = model_id
        self.use_4bit = use_4bit
        self._model = None
        self._processor = None
        self._generation_config = None

    def load(self):
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

            logger.info(f"Loading Phi-4 Multimodal: {self.model_id}")

            self._processor = AutoProcessor.from_pretrained(
                self.model_id, trust_remote_code=True
            )

            # IMPORTANT: Phi4MM remote code calls Tensor.item() during __init__.
            # If transformers enables "meta" initialization (via low_cpu_mem_usage /
            # accelerate dispatch), it will crash. We therefore default to
            # low_cpu_mem_usage=False unless we're using a GPU device_map.
            load_kwargs = dict(
                trust_remote_code=True,
                low_cpu_mem_usage=False,
                device_map=None,
                _fast_init=False,  # avoid meta/empty-weight fast init paths
            )
            if torch.cuda.is_available():
                # Keep VRAM under ~6GB: require 4-bit quantization for GPU.
                if self.use_4bit:
                    try:
                        from transformers import BitsAndBytesConfig

                        load_kwargs["device_map"] = "cuda"
                        load_kwargs["low_cpu_mem_usage"] = True
                        load_kwargs["quantization_config"] = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.bfloat16,
                        )
                        # torch_dtype is ignored with some quant configs; keep it explicit.
                        load_kwargs["torch_dtype"] = torch.bfloat16
                        logger.info("Phi-4: using 4-bit quantization (bitsandbytes) to cap VRAM")
                    except Exception as q_err:
                        # bitsandbytes is often unavailable on Windows. To honor the
                        # <=6GB VRAM constraint, fall back to CPU instead of GPU bf16.
                        load_kwargs["torch_dtype"] = torch.float32
                        load_kwargs["_attn_implementation"] = "eager"
                        logger.warning(
                            f"Phi-4: 4-bit quantization unavailable ({type(q_err).__name__}). "
                            "Falling back to CPU to keep VRAM <= 6GB."
                        )
                else:
                    load_kwargs["device_map"] = "cuda"
                    load_kwargs["low_cpu_mem_usage"] = True
                    load_kwargs["torch_dtype"] = "auto"
                try:
                    import flash_attn  # noqa: F401
                    load_kwargs["_attn_implementation"] = "flash_attention_2"
                except ImportError:
                    load_kwargs["_attn_implementation"] = "eager"
            else:
                load_kwargs["torch_dtype"] = torch.float32
                load_kwargs["_attn_implementation"] = "eager"

            # Force default device away from "meta" during init.
            # Some transformers/accelerate paths can leave a global meta-device
            # context active; Phi4MM remote code calls Tensor.item() in __init__
            # which crashes on meta tensors.
            # PyTorch 2.6+ requires explicit handling to avoid meta tensors.
            _get_default_device = getattr(torch, "get_default_device", None)
            _set_default_device = getattr(torch, "set_default_device", None)
            _orig_default = _get_default_device() if callable(_get_default_device) else None
            if callable(_set_default_device):
                _set_default_device("cpu")
            
            # Disable meta device context entirely for PyTorch 2.6+
            _has_set_default_dtype = hasattr(torch, "set_default_dtype")
            _orig_dtype = None
            if _has_set_default_dtype:
                try:
                    _orig_dtype = torch.get_default_dtype()
                    torch.set_default_dtype(torch.float32)
                except Exception:
                    _has_set_default_dtype = False
                    
            # Safe monkey-patch for Tensor.item() to prevent meta tensor crash in Phi4 remote code
            _orig_item = torch.Tensor.item
            def _safe_item(self):
                if getattr(self, "is_meta", False):
                    return 0
                return _orig_item(self)
            
            torch.Tensor.item = _safe_item
            
            try:
                # Load without device_map first to avoid meta tensors
                # Then manually move to target device
                load_kwargs_no_map = {k: v for k, v in load_kwargs.items() if k != "device_map"}
                
                # If using 4bit, we must use device_map="auto" and accept accelerate's meta tensors
                # But our safe_item monkey-patch protects against the init crash
                if self.use_4bit:
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self.model_id, **load_kwargs
                    ).eval()
                else:
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self.model_id, **load_kwargs_no_map
                    ).eval()
                    # Move to target device manually
                    if torch.cuda.is_available() and self.device == "cuda":
                        self._model = self._model.cuda()
            finally:
                torch.Tensor.item = _orig_item  # remove monkey-patch
                if callable(_set_default_device) and _orig_default is not None:
                    try:
                        _set_default_device(str(_orig_default))
                    except Exception:
                        pass
                if _has_set_default_dtype and _orig_dtype is not None:
                    try:
                        torch.set_default_dtype(_orig_dtype)
                    except Exception:
                        pass

            # If we ended up on CPU, ensure weights are actually materialized there.
            # (When device_map isn't provided, HF loads to CPU by default.)
            if not torch.cuda.is_available() or (load_kwargs.get("device_map") != "cuda" and not self.use_4bit):
                self._model = self._model.to("cpu")

            self._generation_config = GenerationConfig.from_pretrained(self.model_id)

            self._is_loaded = True
            self._record_load_memory()
            logger.info(
                f"Phi-4 Multimodal loaded (GPU: {self._loaded_gpu_mem_mb:.0f} MB)"
            )

        except ImportError as e:
            logger.error(
                f"Phi-4 dependencies missing: {e}\n"
                "Install with:\n"
                "  pip install transformers torch soundfile accelerate"
            )
            raise
        except RuntimeError as e:
            # Known failure mode on some Windows + PyTorch 2.6 setups with Phi4MM remote code.
            if "meta tensors" in str(e):
                raise RuntimeError(
                    "Phi-4 Multimodal failed to initialize due to a meta-tensor init path "
                    "(Torch 2.6 + remote code incompatibility).\n"
                    "Pragmatic fixes:\n"
                    "  - Prefer another multimodal model (e.g. gemma-3n) OR\n"
                    "  - Downgrade PyTorch to 2.5.1 for this model.\n"
                    f"Original error: {e}"
                ) from e
            raise
        except Exception as e:
            logger.error(f"Failed to load Phi-4: {e}")
            raise

    def _transcribe_single(self, audio_path: str) -> str:
        import soundfile as sf

        audio, samplerate = sf.read(audio_path)

        prompt = (
            f"{self._USER}<|audio_1|>{self._ASR_PROMPT}"
            f"{self._END}{self._ASST}"
        )

        inputs = self._processor(
            text=prompt,
            audios=[(audio, samplerate)],
            return_tensors="pt",
        ).to(self._model.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generate_ids = self._model.generate(
                **inputs,
                max_new_tokens=256,
                generation_config=self._generation_config,
            )

        output_tokens = generate_ids[0][input_len:]
        text = self._processor.decode(
            output_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return text.strip()

    def unload(self):
        del self._model
        del self._processor
        self._model = None
        self._processor = None
        self._generation_config = None
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

    @staticmethod
    def _suppress_nemo_spam():
        """Silence NeMo's per-sample dataloader / pretokenize / train-config spam."""
        import logging as _log
        import warnings as _warn
        
        # Set Lhotse dataloader logger to CRITICAL to suppress all dataloader warnings
        for _name in (
            "nemo_logging",
            "nemo.collections.asr",
            "nemo.utils",
            "nemo.core",
            "lhotse",
            "lhotse.cut",
            "lhotse.dataset",
            "lhotse.dataloader",
            "root",
        ):
            _lg = _log.getLogger(_name)
            _lg.setLevel(_log.CRITICAL)  # CRITICAL is higher than ERROR
            _lg.propagate = False
            # Disable all handlers for this logger
            _lg.handlers = []
        
        # Filter warnings
        _warn.filterwarnings("ignore", message=".*If you intend to do training.*")
        _warn.filterwarnings("ignore", message=".*If you intend to do validation.*")
        _warn.filterwarnings("ignore", message=".*Please call the ModelPT.setup_test_data.*")
        _warn.filterwarnings("ignore", message=".*The following configuration keys are ignored.*")
        _warn.filterwarnings("ignore", message=".*You are using a non-tarred dataset.*")
        _warn.filterwarnings("ignore", message=".*CTC decoding strategy.*")
        _warn.filterwarnings("ignore", message=".*Megatron num_microbatches_calculator.*")
        _warn.filterwarnings("ignore", message=".*Found existing object.*")
        _warn.filterwarnings("ignore", message=".*Re-using file from.*")
        _warn.filterwarnings("ignore", message=".*Instantiating model from pre-trained.*")
        _warn.filterwarnings("ignore", message=".*Tokenizer.*initialized.*")

    def load(self):
        try:
            import nemo.collections.asr as nemo_asr

            self._suppress_nemo_spam()
            logger.info(f"Loading NeMo Conformer: {self.model_name}")

            # NeMo prints [NeMo I/W ...] directly to stdout/stderr during load.
            # Redirect both to suppress the spam.
            import io as _io
            import sys as _sys
            _old_stdout = _sys.stdout
            _old_stderr = _sys.stderr
            _sys.stdout = _io.StringIO()
            _sys.stderr = _io.StringIO()
            try:
                self._model = nemo_asr.models.ASRModel.from_pretrained(
                    model_name=self.model_name
                )
                if self.device == "cuda" and torch.cuda.is_available():
                    self._model = self._model.cuda()
                self._model.eval()
            finally:
                _sys.stdout = _old_stdout
                _sys.stderr = _old_stderr
            self._is_loaded = True
            self._record_load_memory()
            logger.info(f"NeMo Conformer {self.model_name} loaded successfully (GPU: {self._loaded_gpu_mem_mb:.0f} MB)")

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
        self._suppress_nemo_spam()
        # Use greedy_batch for speed (NeMo warns about 'greedy' being slower)
        try:
            self._model.change_decoding_strategy("greedy_batch")
        except Exception:
            pass
        # NeMo's internal logger prints [NeMo W/I ...] directly to stdout/stderr,
        # bypassing Python logging. Redirect both during transcribe().
        import io as _io
        import sys as _sys
        _old_stdout = _sys.stdout
        _old_stderr = _sys.stderr
        _sys.stdout = _io.StringIO()
        _sys.stderr = _io.StringIO()
        try:
            results = self._model.transcribe([audio_path])
        finally:
            _sys.stdout = _old_stdout
            _sys.stderr = _old_stderr
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
        import shutil
        # Silero hubconf expects a small allowlist of language codes ("ru", "en", ...).
        # Normalize user/config input like "RU", "ru-RU", "ru_RU" → "ru".
        lang = (self.language or "ru").strip().lower().replace("_", "-")
        if "-" in lang:
            lang = lang.split("-", 1)[0]
        self.language = lang or "ru"

        logger.info(f"Loading Silero STT (language={self.language})")
        device = torch.device(self.device if torch.cuda.is_available() else "cpu")

        # The master branch cache has a stale hubconf.py that tries to import
        # silero_denoise from an older v0.4.1 cache → ImportError.
        # Clear both bad caches before loading.
        hub_dir = torch.hub.get_dir()
        for bad_cache in [
            "snakers4_silero-models_master",
            "snakers4_silero-models_v0.3.0",
            "snakers4_silero-models_v0.4.1",
        ]:
            bad_path = Path(hub_dir) / bad_cache
            if bad_path.exists():
                shutil.rmtree(bad_path, ignore_errors=True)
                logger.info(f"  Cleared stale Silero cache: {bad_path.name}")

        # Load pinned v0.4.1 which has stable 6-return-value API for Russian
        last_err = None
        for repo in ["snakers4/silero-models:v0.4.1"]:
            try:
                logger.info(f"  Trying Silero repo: {repo}")
                try:
                    # Preferred path: pass language explicitly.
                    result = torch.hub.load(
                        repo_or_dir=repo,
                        model="silero_stt",
                        language=self.language,
                        device=device,
                        trust_repo=True,
                    )
                except AssertionError as _lang_err:
                    # Some hubconf revisions have different language allowlists.
                    # Retry without language param and let silero pick its default.
                    logger.info(
                        f"  {repo}: language={self.language!r} rejected; retrying without language"
                    )
                    result = torch.hub.load(
                        repo_or_dir=repo,
                        model="silero_stt",
                        device=device,
                        trust_repo=True,
                    )
                # v0.4.1 returns (model, decoder, read_batch, split_into_batches, _, _)
                if isinstance(result, (list, tuple)) and len(result) >= 4:
                    model, decoder, read_batch, split_into_batches = result[:4]
                    self._read_batch = read_batch
                    self._split_into_batches = split_into_batches
                elif isinstance(result, (list, tuple)) and len(result) == 3:
                    model, decoder, utils = result
                    self._read_batch = getattr(utils, "read_batch", None) or utils[0]
                    self._split_into_batches = getattr(utils, "split_into_batches", None) or utils[1]
                else:
                    raise RuntimeError(f"Unexpected Silero return signature: {type(result)}, len={len(result)}")

                self._model = model
                self._decoder = decoder
                self._is_loaded = True
                self._record_load_memory()
                logger.info(
                    f"Silero STT loaded from {repo} "
                    f"(GPU: {self._loaded_gpu_mem_mb:.0f} MB)"
                )
                return  # success
            except Exception as e:
                # If language is invalid (AssertionError), force "ru" and retry once.
                if isinstance(e, AssertionError) and self.language != "ru":
                    logger.info(
                        f"  {repo} failed: invalid language={self.language!r}; retrying with 'ru'"
                    )
                    self.language = "ru"
                    try:
                        result = torch.hub.load(
                            repo_or_dir=repo,
                            model="silero_stt",
                            language=self.language,
                            device=device,
                            trust_repo=True,
                        )
                        if isinstance(result, (list, tuple)) and len(result) >= 4:
                            model, decoder, read_batch, split_into_batches = result[:4]
                            self._read_batch = read_batch
                            self._split_into_batches = split_into_batches
                        elif isinstance(result, (list, tuple)) and len(result) == 3:
                            model, decoder, utils = result
                            self._read_batch = getattr(utils, "read_batch", None) or utils[0]
                            self._split_into_batches = getattr(utils, "split_into_batches", None) or utils[1]
                        else:
                            raise RuntimeError(
                                f"Unexpected Silero return signature: {type(result)}, len={len(result)}"
                            )
                        self._model = model
                        self._decoder = decoder
                        self._is_loaded = True
                        self._record_load_memory()
                        logger.info(
                            f"Silero STT loaded from {repo} "
                            f"(GPU: {self._loaded_gpu_mem_mb:.0f} MB)"
                        )
                        return
                    except Exception as e2:
                        logger.info(f"  {repo} retry failed: {type(e2).__name__}: {e2}")
                        last_err = e2
                        continue

                logger.info(f"  {repo} failed: {type(e).__name__}: {e}")
                last_err = e

        logger.error(
            f"All Silero repo versions failed. Last error: {last_err}\n"
            "Tip: manually clear cache:\n"
            f"  Remove-Item -Recurse -Force '{hub_dir}\\snakers4_silero-models_v0.4.1'"
        )
        raise RuntimeError(f"Failed to load Silero STT: {last_err}") from last_err


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

        # Transcribe using Silero's model directly
        with torch.inference_mode():
            # Get model output (logits/probs)
            output = self._model(waveform)
            
            # Simple greedy decode
            if hasattr(output, 'argmax'):
                indices = output.argmax(dim=-1).squeeze()
                # Silero uses a specific alphabet mapping
                # For now, just return the indices as a string to see what we get
                text = str(indices.tolist())
            else:
                text = str(output)

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
# ===============================================================
# Qwen Omni/Audio (Alibaba)
# ===============================================================

class QwenModel(ASRModel):
    """
    Qwen Audio / Omni series.
    Supports Qwen2-Audio, Qwen2.5-Omni, Qwen3-Omni.
    """

    def __init__(
        self,
        name: str = "qwen_audio",
        model_id: str = "Qwen/Qwen2.5-Omni-3B",
        device: str = "cuda",
        use_4bit: bool = False,
    ):
        super().__init__(name, device)
        self.model_id = model_id
        self.use_4bit = use_4bit
        self._model = None
        self._processor = None

    def load(self):
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
            
            # For Qwen2.5-Omni, HF requires specific native classes in recent transformers
            ModelClass = AutoModelForCausalLM
            ProcessorClass = AutoProcessor
            
            if "Omni" in self.model_id:
                try:
                    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
                    ModelClass = Qwen2_5OmniForConditionalGeneration
                    ProcessorClass = Qwen2_5OmniProcessor
                except ImportError:
                    # Fallback to try AutoModel
                    try:
                        from transformers import AutoModel
                        ModelClass = AutoModel
                    except ImportError:
                        pass

            logger.info(f"Loading Qwen Audio/Omni model: {self.model_id} via {ModelClass.__name__}")

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
            else:
                load_kwargs["torch_dtype"] = torch.float16

            self._processor = ProcessorClass.from_pretrained(
                self.model_id, trust_remote_code=True
            )
            
            try:
                self._model = ModelClass.from_pretrained(
                    self.model_id, **load_kwargs
                )
            except Exception as e:
                logger.warning(f"{ModelClass.__name__} failed ({e}), trying AutoModelForCausalLM...")
                from transformers import AutoModelForCausalLM
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_id, **load_kwargs
                )
                
            self._model.eval()
            self._is_loaded = True
            logger.info(f"{self.model_id} loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Qwen model: {e}")
            raise

    def _transcribe_single(self, audio_path: str) -> str:
        import librosa

        sr = getattr(self._processor.feature_extractor, "sampling_rate", 16000)
        audio_data, _ = librosa.load(audio_path, sr=sr)

        conversation = [
            {"role": "user", "content": [
                {"type": "audio", "audio_url": "audio.wav"},
                {"type": "text", "text": "Transcribe the audio exactly as spoken in Russian. Output only the transcription."}
            ]}
        ]

        text = self._processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

        inputs = self._processor(text=text, audios=[audio_data], return_tensors="pt", padding=True)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

        with torch.inference_mode():
            generate_ids = self._model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )
        
        generate_ids = generate_ids[:, inputs["input_ids"].size(1):]
        response = self._processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return response.strip()

    def unload(self):
        del self._model
        del self._processor
        self._model = None
        self._processor = None
        super().unload()



def create_models(
    include_whisper: bool = True,
    include_gigaam: bool = True,
    include_vibevoice: bool = False,
    include_vibevoice_4bit: bool = False,
    include_gemma: bool = False,
    include_phi4: bool = False,
    include_nemo: bool = False,
    include_silero: bool = False,
    include_qwen: bool = False,
    whisper_configs: Optional[dict] = None,
) -> list[ASRModel]:
    """
    Create all configured LOCAL models.

    Args:
        include_whisper: Include faster-whisper models
        include_gigaam: Include GigaAM-v3 models
        include_vibevoice: Include VibeVoice-ASR (needs ~14GB VRAM)
        include_vibevoice_4bit: Include VibeVoice-ASR with 4-bit quantization (~6GB VRAM)
        include_gemma: Include Gemma 3n E4B (multimodal with audio)
        include_phi4: Include Phi-4 Multimodal (Microsoft, 5.6B)
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

    # --- GigaAM ---------------------------------------
    if include_gigaam:
        for ver in cfg.GIGAAM_CONFIG["model_versions"]:
            models.append(
                GigaAMModel(
                    name=f"gigaam_{ver}",
                    model_version=ver,
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
                use_4bit=False,  # Full precision (needs ~14GB VRAM)
            )
        )
    if include_vibevoice_4bit:
        models.append(
            VibeVoiceModel(
                name="vibevoice_asr_4bit",
                model_path="microsoft/VibeVoice-ASR",
                device="cuda",
                use_4bit=True,  # 4-bit quant to fit ~6GB VRAM
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

    # --- Phi-4 Multimodal ---------------------------------
    if include_phi4:
        models.append(
            Phi4Model(
                name="phi4_multimodal",
                model_id=cfg.PHI4_CONFIG["model_id"],
                device=cfg.PHI4_CONFIG["device"],
                use_4bit=cfg.PHI4_CONFIG.get("use_4bit", True),
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

    # --- Qwen Omni/Audio ----------------------------------
    if include_qwen:
        model_ids = cfg.QWEN_CONFIG.get("model_ids", [])
        if not model_ids and "model_id" in cfg.QWEN_CONFIG:
            model_ids = [cfg.QWEN_CONFIG["model_id"]]
            
        for m_id in model_ids:
            if not m_id: continue
            safe_name = m_id.split("/")[-1].lower().replace("-", "_").replace(".", "_")
            models.append(
                QwenModel(
                    name=f"qwen_{safe_name}",
                    model_id=m_id,
                    device=cfg.QWEN_CONFIG.get("device", "cuda"),
                    use_4bit=cfg.QWEN_CONFIG.get("use_4bit", False),
                )
            )

    return models
