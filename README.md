# ASR Benchmark - Russian Speech Recognition

Локальный бенчмарк для сравнения моделей распознавания русской речи.
Пайплайн работает с локальными моделями, без внешних ASR API.

## Результаты

### Сравнение моделей (Common Voice RU, ~65 сэмплов)

| Модель | WER | CER | RTF | Latency (p50) |
|--------|-----|-----|-----|---------------|
| whisper-large-v3 (beam=5) | **7.5%** | **4.3%** | 0.054 | 0.94s |
| gigaam_rnnt | 8.5% | 5.5% | 0.018 | 0.33s |
| whisper-medium-int8 (beam=5) | 8.3% | 3.9% | 0.045 | 0.78s |
| gigaam_ctc | 8.6% | 5.6% | **0.005** | **0.08s** |
| vibevoice_asr_4bit | 11.1% | 5.9% | 0.269 | 4.52s |
| whisper-small-int8 (beam=1) | 13.2% | 7.0% | 0.054 | 0.94s |
| nemo_conformer-ctc-large | 14.4% | 7.1% | 0.005 | 0.07s |
| gemma_3n_e4b | 14.9% | 6.8% | 1.310 | 22.39s |
| qwen_qwen2_audio_7b | 60.3% | 52.0% | 1.100 | 21.11s |

### LLM-постобработка (gemma3-chat через Ollama)

| Модель | WER | WER LLM | Δ | CER | CER LLM |
|--------|-----|---------|---|-----|---------|
| nemo_conformer-ctc-large | 14.4% | **12.7%** | **-11.9%** | 7.1% | 7.3% |
| gigaam_rnnt | 8.5% | **8.0%** | **-6.7%** | 5.5% | **4.9%** |
| whisper-small-int8_beam1 | 13.2% | **12.3%** | **-6.6%** | 7.0% | **6.7%** |
| gigaam_ctc | 8.6% | 8.5% | -1.6% | 5.6% | **5.2%** |
| whisper-large-v3_beam5 | 7.5% | 7.5% | 0.0% | 4.3% | **4.1%** |
| whisper-medium-int8_beam5 | 8.3% | 8.3% | +0.4% | 3.9% | 4.2% |
| vibevoice_asr_4bit | 11.1% | 11.3% | +1.6% | 5.9% | 6.5% |
| gemma_3n_e4b | 14.9% | 15.1% | +1.2% | 6.8% | 7.2% |
| qwen_qwen2_audio_7b | 60.3% | 63.2% | +4.7% | 52.0% | 54.5% |

**Выводы:**
- LLM-постобработка эффективна для моделей с умеренным WER (8-15%): nemo -11.9%, gigaam_rnnt -6.7%, whisper-small -6.6%.
- Для уже качественных моделей (whisper-large 7.5%) эффект нулевой.
- Для слабых моделей (qwen 60%) LLM деградирует результат.
- CER улучшается стабильнее WER — LLM лучше исправляет отдельные символы, чем целые слова.

## Модели

| Модель | Бэкенд | VRAM |
|--------|--------|------|
| whisper-large-v3 | faster-whisper / CTranslate2, FP16 или INT8 | ~4-6 GB |
| whisper-medium | faster-whisper / CTranslate2, FP16 или INT8 | ~2-3 GB |
| GigaAM-v3 CTC | salute-developers/GigaAM `v3_ctc` | ~4 GB |
| GigaAM-v3 E2E | salute-developers/GigaAM `v3_e2e_rnnt` | ~4 GB |
| VibeVoice-ASR | microsoft/VibeVoice-ASR, 7B, 4-bit | ~8-14 GB |
| Gemma 3n E4B | google/gemma-3n-E4B-it, multimodal audio | ~4-5 GB |
| NeMo Conformer | NVIDIA `stt_ru_conformer_ctc_large` | ~2-4 GB |
| Silero STT | snakers4/silero-models через `torch.hub` | ~0.5-1 GB |
| Qwen Omni / Audio | Qwen/Qwen2.5-Omni (3B/7B), multimodal audio | ~4-6 GB |

## Запуск

```bash
pip install -r requirements.txt
python scripts/run_benchmark.py --models whisper,gigaam,nemo,silero,qwen --max-samples 50
```

## Метрики

- **WER**: word error rate, основная метрика качества.
- **CER**: character error rate.
- **RTF**: real-time factor, `inference_time / audio_duration`.
- **Latency**: средняя, p50 и p95 задержка на сегмент.
- **GPU Memory Peak**: пиковое потребление VRAM.

## LLM-постобработка

```bash
ollama pull gemma3-chat:latest
python scripts/llm_postprocess.py
```

Читает `per_sample_results.csv`, исправляет транскрипции через LLM, обновляет `benchmark_report.xlsx` колонками `wer_llm`/`cer_llm`.

## Нормализация текста

Reference и hypothesis приводятся к общему виду: lowercase, Unicode NFC, удаление пунктуации, сохранение русских/английских букв и цифр. Английские буквы сохранены намеренно — в русских датасетах встречаются `ai`, `ok`, `whisper`, названия продуктов.
