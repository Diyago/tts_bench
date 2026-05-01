"""
Test suite for the ASR benchmark pipeline.

The tests use synthetic audio and mock models, so they do not download
datasets, load real ASR models, or require a GPU.
"""

import csv
import os
import sys
import wave
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def create_test_wav(path: str, duration_sec: float = 5.0, sr: int = 16000) -> str:
    """Create a synthetic mono WAV file with quiet white noise."""
    n_samples = int(duration_sec * sr)
    np.random.seed(config.SEED)
    audio = (np.random.randn(n_samples) * 0.1 * 32767).astype(np.int16)

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio.tobytes())
    return path


def create_test_manifest(output_dir: str, n_samples: int = 10, split: str = "test") -> Path:
    """Create a synthetic manifest with real Russian text."""
    manifest_rows = []
    audio_dir = Path(output_dir) / split

    for i in range(n_samples):
        duration = float(np.random.uniform(3.0, 15.0))
        audio_path = str(audio_dir / f"{split}_{i:05d}.wav")
        create_test_wav(audio_path, duration_sec=duration)

        manifest_rows.append({
            "audio_path": audio_path,
            "text": f"тестовый текст номер {i} с ai словом",
            "text_original": f"Тестовый текст номер {i} с AI словом!",
            "duration_sec": round(duration, 3),
            "speaker_id": f"spk_{i % 3}",
            "gender": ["male", "female", "other"][i % 3],
            "age": "twenties",
            "split": split,
        })

    manifest_path = Path(output_dir) / f"{split}_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)

    return manifest_path


class TestTextNormalization:
    def test_basic_lowercase(self):
        from scripts.evaluate import normalize_for_eval

        assert normalize_for_eval("Привет Мир") == "привет мир"

    def test_remove_punctuation(self):
        from scripts.evaluate import normalize_for_eval

        result = normalize_for_eval("Привет, мир! Как дела?")
        assert result == "привет мир как дела"

    def test_keep_english_letters(self):
        from scripts.evaluate import normalize_for_eval

        result = normalize_for_eval("Используем AI для распознавания")
        assert result == "используем ai для распознавания"

    def test_keep_digits(self):
        from scripts.evaluate import normalize_for_eval

        assert normalize_for_eval("В 2024 году") == "в 2024 году"

    def test_collapse_whitespace(self):
        from scripts.evaluate import normalize_for_eval

        assert normalize_for_eval("привет     мир") == "привет мир"

    def test_empty_string(self):
        from scripts.evaluate import normalize_for_eval

        assert normalize_for_eval("") == ""
        assert normalize_for_eval("   ") == ""

    def test_unicode_normalization_preserves_yo(self):
        from scripts.evaluate import normalize_for_eval

        result = normalize_for_eval("Ёлка и ёж")
        assert result == "ёлка и ёж"

    def test_mixed_ru_en(self):
        from scripts.evaluate import normalize_for_eval

        result = normalize_for_eval("Модель whisper large v3")
        assert result == "модель whisper large v3"


class TestDurationBuckets:
    def test_short(self):
        from scripts.evaluate import get_duration_bucket

        assert get_duration_bucket(3.0) == "short"
        assert get_duration_bucket(4.9) == "short"

    def test_medium(self):
        from scripts.evaluate import get_duration_bucket

        assert get_duration_bucket(5.0) == "medium"
        assert get_duration_bucket(14.9) == "medium"

    def test_long(self):
        from scripts.evaluate import get_duration_bucket

        assert get_duration_bucket(15.0) == "long"
        assert get_duration_bucket(30.0) == "long"


class TestMetrics:
    def test_perfect_match(self):
        from scripts.evaluate import compute_metrics

        metrics = compute_metrics(
            ["привет как дела"],
            ["привет как дела"],
            [5.0],
            [1.0],
            [500.0],
            "test",
        )
        assert metrics["wer"] == 0.0
        assert metrics["cer"] == 0.0

    def test_complete_mismatch(self):
        from scripts.evaluate import compute_metrics

        metrics = compute_metrics(["привет"], ["здравствуйте"], [5.0], [1.0], [500.0], "test")
        assert metrics["wer"] > 0

    def test_rtf_calculation(self):
        from scripts.evaluate import compute_metrics

        metrics = compute_metrics(["привет"], ["привет"], [2.0], [1.0], [500.0], "test")
        assert abs(metrics["rtf"] - 0.5) < 0.01

    def test_error_breakdown(self):
        from scripts.evaluate import compute_metrics

        refs = ["один два три четыре"]
        hyps = ["один три четыре пять"]
        metrics = compute_metrics(refs, hyps, [5.0], [1.0], [500.0], "test")
        assert metrics["substitutions"] >= 0
        assert metrics["deletions"] >= 0
        assert metrics["insertions"] >= 0

    def test_multiple_samples(self):
        from scripts.evaluate import compute_metrics

        refs = ["привет", "как дела", "добрый день"]
        hyps = ["привет", "как дела", "добрый день"]
        metrics = compute_metrics(refs, hyps, [3.0, 4.0, 5.0], [0.5, 0.6, 0.7], [500, 600, 700], "test")
        assert metrics["num_samples"] == 3
        assert metrics["wer"] == 0.0

    def test_empty_references_filtered(self):
        from scripts.evaluate import compute_metrics

        metrics = compute_metrics(["привет", "", "мир"], ["привет", "что-то", "мир"], [3, 3, 3], [1, 1, 1], [500, 500, 500], "test")
        assert metrics["num_samples"] == 2

    def test_wer_by_bucket(self):
        from scripts.evaluate import compute_metrics

        refs = ["короткий текст", "средний текст для теста"]
        hyps = ["короткий текст", "средний текст для теста"]
        metrics = compute_metrics(refs, hyps, [4.0, 10.0], [0.5, 1.0], [500, 500], "test")
        assert metrics["wer_short"] == 0.0
        assert metrics["wer_medium"] == 0.0


class TestPerSampleResults:
    def test_per_sample_columns(self):
        from scripts.evaluate import compute_per_sample_results

        df = compute_per_sample_results(
            references=["привет мир"],
            hypotheses=["привет мир"],
            audio_paths=["/tmp/test.wav"],
            durations=[5.0],
            inference_times=[1.0],
            gpu_memory_peaks=[500.0],
            model_name="test",
        )
        for col in ["model_name", "reference_original", "hypothesis_raw", "is_correct", "wer", "cer"]:
            assert col in df.columns

    def test_is_correct_flag(self):
        from scripts.evaluate import compute_per_sample_results

        df = compute_per_sample_results(
            references=["привет", "мир"],
            hypotheses=["привет", "мар"],
            audio_paths=["/tmp/a.wav", "/tmp/b.wav"],
            durations=[3.0, 3.0],
            inference_times=[0.5, 0.5],
            gpu_memory_peaks=[500.0, 500.0],
            model_name="test",
        )
        assert df.iloc[0]["is_correct"] == True
        assert df.iloc[1]["is_correct"] == False

    def test_keeps_original_text(self):
        from scripts.evaluate import compute_per_sample_results

        df = compute_per_sample_results(
            references=["Привет, AI!"],
            hypotheses=["привет ai"],
            audio_paths=["/tmp/test.wav"],
            durations=[5.0],
            inference_times=[1.0],
            gpu_memory_peaks=[500.0],
            model_name="test",
        )
        assert df.iloc[0]["reference_original"] == "Привет, AI!"
        assert df.iloc[0]["hypothesis_raw"] == "привет ai"

    def test_keeps_augmentation_metadata(self):
        from scripts.evaluate import compute_per_sample_results

        df = compute_per_sample_results(
            references=["тест"],
            hypotheses=["тест"],
            audio_paths=["/tmp/aug/noise.wav"],
            durations=[3.0],
            inference_times=[0.5],
            gpu_memory_peaks=[0.0],
            model_name="test",
            sample_metadata=[{
                "augmentation": "noise_20db",
                "source_audio_path": "/tmp/source.wav",
            }],
        )

        assert df.iloc[0]["augmentation"] == "noise_20db"
        assert df.iloc[0]["source_audio_filename"] == "source.wav"


class TestManifest:
    def test_create_and_load_manifest(self, tmp_path):
        manifest_path = create_test_manifest(str(tmp_path), n_samples=5)
        df = pd.read_csv(manifest_path)

        assert len(df) == 5
        assert {"audio_path", "text", "duration_sec"}.issubset(df.columns)
        for path in df["audio_path"]:
            assert Path(path).exists()

    def test_manifest_text_has_english(self, tmp_path):
        manifest_path = create_test_manifest(str(tmp_path), n_samples=3)
        df = pd.read_csv(manifest_path)
        assert any("ai" in str(t) for t in df["text"])


class TestAudioFiles:
    def test_wav_creation(self, tmp_path):
        wav_path = str(tmp_path / "test.wav")
        create_test_wav(wav_path, duration_sec=3.0, sr=16000)

        with wave.open(wav_path, "r") as wf:
            assert wf.getnchannels() == 1
            assert wf.getframerate() == 16000
            assert abs(wf.getnframes() / wf.getframerate() - 3.0) < 0.1

    @pytest.mark.parametrize("duration", [3.0, 5.5, 10.0, 20.0])
    def test_wav_different_durations(self, tmp_path, duration):
        path = str(tmp_path / f"test_{duration}.wav")
        create_test_wav(path, duration_sec=duration)
        with wave.open(path, "r") as wf:
            assert abs(wf.getnframes() / wf.getframerate() - duration) < 0.1


class TestAudioAugmentations:
    def test_parse_default_audio_augmentations(self):
        from scripts.run_benchmark import parse_audio_augmentations

        augmentations = parse_audio_augmentations("default")
        assert "clean" in augmentations
        assert "noise_20db" in augmentations
        assert "speed_1.05" in augmentations

    def test_parse_audio_augmentations_off(self):
        from scripts.run_benchmark import parse_audio_augmentations

        assert parse_audio_augmentations("none") == ["clean"]
        assert parse_audio_augmentations("") == ["clean"]

    def test_build_augmented_manifest(self, tmp_path):
        from scripts.run_benchmark import build_augmented_manifest

        audio_path = tmp_path / "sample.wav"
        create_test_wav(str(audio_path), duration_sec=2.0)
        manifest = pd.DataFrame([{
            "audio_path": str(audio_path),
            "text": "тест",
            "duration_sec": 2.0,
            "split": "test",
        }])

        augmented = build_augmented_manifest(
            manifest,
            ["clean", "noise_20db", "speed_1.05"],
            tmp_path / "augmented",
        )

        assert len(augmented) == 3
        assert set(augmented["augmentation"]) == {"clean", "noise_20db", "speed_1.05"}
        assert "source_audio_path" in augmented.columns
        for path in augmented["audio_path"]:
            assert Path(path).exists()

        speed_row = augmented[augmented["augmentation"] == "speed_1.05"].iloc[0]
        assert speed_row["duration_sec"] < 2.0


class TestExcelExport:
    def test_excel_report_creation(self, tmp_path):
        from scripts.evaluate import save_results

        metrics = [{
            "model_name": "test_model",
            "wer": 0.15,
            "cer": 0.08,
            "rtf": 0.3,
            "avg_latency_sec": 0.5,
            "p50_latency_sec": 0.4,
            "p95_latency_sec": 0.8,
            "gpu_memory_peak_mb": 500.0,
            "gpu_memory_avg_mb": 450.0,
            "substitution_rate": 0.05,
            "deletion_rate": 0.03,
            "insertion_rate": 0.02,
            "total_audio_hours": 0.5,
            "total_inference_sec": 180.0,
            "num_samples": 100,
            "total_words_ref": 500,
            "substitutions": 25,
            "deletions": 15,
            "insertions": 10,
            "hits": 450,
        }]

        per_sample_df = pd.DataFrame([{
            "model_name": "test_model",
            "audio_path": "/test/audio.wav",
            "audio_filename": "audio.wav",
            "duration_sec": 5.0,
            "duration_bucket": "medium",
            "reference_original": "Привет, мир!",
            "reference_normalized": "привет мир",
            "hypothesis_raw": "Привет мир",
            "hypothesis_normalized": "привет мир",
            "is_correct": True,
            "wer": 0.0,
            "cer": 0.0,
            "rtf": 0.3,
            "inference_time_sec": 1.5,
            "gpu_memory_peak_mb": 500.0,
        }])

        save_results(metrics, [per_sample_df], output_dir=tmp_path)

        assert (tmp_path / "results.csv").exists()
        assert (tmp_path / "per_sample_results.csv").exists()
        excel_path = tmp_path / "benchmark_report.xlsx"
        assert excel_path.exists()
        assert {"Summary", "Details"}.issubset(set(pd.ExcelFile(excel_path).sheet_names))


class TestConfig:
    def test_allowed_chars_include_english(self):
        for ch in "abcdefghijklmnopqrstuvwxyz":
            assert ch in config.ALLOWED_CHARS

    def test_allowed_chars_include_russian(self):
        for ch in "абвгдеёжзийклмнопрстуфхцчшщъыьэюя":
            assert ch in config.ALLOWED_CHARS

    def test_allowed_chars_include_digits(self):
        for ch in "0123456789":
            assert ch in config.ALLOWED_CHARS

    def test_audio_settings(self):
        assert config.SAMPLE_RATE == 16000
        assert config.MIN_DURATION_SEC > 0
        assert config.MAX_DURATION_SEC > config.MIN_DURATION_SEC

    def test_model_configs_exist(self):
        assert "v3_ctc" in config.GIGAAM_CONFIG["model_versions"]
        assert config.VIBEVOICE_CONFIG["model_path"]
        assert config.GEMMA4_CONFIG["model_id"]
        assert len(config.NEMO_CONFIG["model_names"]) > 0
        assert config.SILERO_CONFIG["language"] == "ru"


class TestModelFactory:
    def test_create_whisper_models(self):
        from models.inference import create_models

        models = create_models(include_whisper=True, include_gigaam=False)
        assert len(models) > 0
        assert all("whisper" in m.name.lower() for m in models)

    def test_create_no_models(self):
        from models.inference import create_models

        models = create_models(include_whisper=False, include_gigaam=False)
        assert len(models) == 0

    def test_model_names_unique(self):
        from models.inference import create_models

        models = create_models(include_whisper=True, include_gigaam=False)
        names = [m.name for m in models]
        assert len(names) == len(set(names))

    def test_model_classes_exist(self):
        from models.inference import (
            ASRModel,
            FasterWhisperModel,
            Gemma3nModel,
            Gemma4Model,
            GigaAMModel,
            InferenceResult,
            NeMoConformerModel,
            SileroSTTModel,
            VibeVoiceModel,
        )

        model_instances = [
            FasterWhisperModel(name="test_whisper", device="cpu"),
            GigaAMModel(name="test_gigaam", model_version="v3_ctc", device="cpu"),
            VibeVoiceModel(name="test_vibe", device="cpu", use_4bit=True),
            Gemma3nModel(name="test_gemma", model_id="google/gemma-3n-E4B-it", device="cpu"),
            Gemma4Model(name="test_gemma4", model_id="google/gemma-4-12b-it", device="cpu"),
            NeMoConformerModel(name="test_nemo", model_name="stt_ru_conformer_ctc_large", device="cpu"),
            SileroSTTModel(name="test_silero", language="ru", device="cpu"),
        ]

        assert all(isinstance(model, ASRModel) for model in model_instances)
        assert all(not model._is_loaded for model in model_instances)
        assert InferenceResult(text="привет", audio_path="/test.wav", duration_sec=3.0, inference_time_sec=0.5).text == "привет"


class TestDataPreparation:
    def test_normalize_text_from_prepare(self):
        from scripts.prepare_data import normalize_text

        result = normalize_text("Привет, мир! AI помощник.")
        assert result == "привет мир ai помощник"

    def test_normalize_preserves_yo(self):
        from scripts.prepare_data import normalize_text

        result = normalize_text("Ёлка стоит в углу")
        assert "ё" in result

    def test_convert_audio(self, tmp_path):
        from scripts.prepare_data import convert_audio

        audio = np.random.randn(44100 * 3).astype(np.float32)
        output_path = tmp_path / "converted.wav"
        duration = convert_audio(audio, 44100, output_path)

        assert output_path.exists()
        assert abs(duration - 3.0) < 0.5
        with wave.open(str(output_path), "r") as wf:
            assert wf.getframerate() == 16000
            assert wf.getnchannels() == 1

    def test_decode_audio_feature_from_bytes(self, tmp_path):
        from scripts.prepare_data import decode_audio_feature

        wav_path = tmp_path / "feature.wav"
        create_test_wav(str(wav_path), duration_sec=1.0)

        audio_array, sample_rate = decode_audio_feature({
            "bytes": wav_path.read_bytes(),
            "path": "feature.wav",
        })

        assert sample_rate == 16000
        assert audio_array.dtype == np.float32
        assert audio_array.shape[0] == 16000


class TestFullPipeline:
    def test_pipeline_with_mock_model(self, tmp_path):
        from models.inference import ASRModel
        from scripts.evaluate import compute_metrics, compute_per_sample_results, save_results

        manifest_path = create_test_manifest(str(tmp_path), n_samples=5)
        manifest = pd.read_csv(manifest_path)

        class MockModel(ASRModel):
            def load(self):
                self._is_loaded = True

            def _transcribe_single(self, audio_path: str) -> str:
                return "тестовый текст номер 0 с ai словом"

        model = MockModel(name="mock_model", device="cpu")
        model.load()

        refs, hyps, durs, times, gpus, paths = [], [], [], [], [], []
        for _, row in manifest.iterrows():
            result = model.transcribe(row["audio_path"], row["duration_sec"])
            refs.append(row["text"])
            hyps.append(result.text)
            durs.append(row["duration_sec"])
            times.append(result.inference_time_sec)
            gpus.append(result.gpu_memory_peak_mb)
            paths.append(row["audio_path"])

        metrics = compute_metrics(refs, hyps, durs, times, gpus, "mock_model")
        assert metrics["num_samples"] == 5
        assert {"wer", "cer", "rtf"}.issubset(metrics)

        df_per = compute_per_sample_results(refs, hyps, paths, durs, times, gpus, "mock_model")
        assert len(df_per) == 5
        assert {"is_correct", "reference_original", "hypothesis_raw"}.issubset(df_per.columns)

        out_dir = tmp_path / "results"
        save_results([metrics], [df_per], output_dir=out_dir)
        assert (out_dir / "results.csv").exists()
        assert (out_dir / "per_sample_results.csv").exists()
        assert (out_dir / "benchmark_report.xlsx").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
