# ASR Benchmark - Russian Speech Recognition

Локальный бенчмарк для сравнения моделей распознавания русской речи.
Пайплайн работает с локальными моделями, без внешних ASR API.

## Модели

| Модель | Бэкенд | Оценка VRAM | Статус |
|--------|--------|-------------|--------|
| whisper-large-v3 | faster-whisper / CTranslate2, FP16 или INT8 | ~4-6 GB | основной кандидат |
| whisper-medium | faster-whisper / CTranslate2, FP16 или INT8 | ~2-3 GB | быстрый кандидат |
| GigaAM-v3 CTC | salute-developers/GigaAM `v3_ctc` | ~4 GB | опционально |
| GigaAM-v3 E2E | salute-developers/GigaAM `v3_e2e_rnnt` | ~4 GB | опционально |
| VibeVoice-ASR | microsoft/VibeVoice-ASR, 7B, 4-bit | ~8-14 GB | опционально |
| Gemma 3n E4B | google/gemma-3n-E4B-it, multimodal audio | ~4-5 GB | опционально |
| Gemma 4 | google/gemma-4-12b-it, multimodal audio | ~8-10 GB | опционально |
| NeMo Conformer | NVIDIA `stt_ru_conformer_ctc_large` | ~2-4 GB | опционально |
| Silero STT | snakers4/silero-models через `torch.hub` | ~0.5-1 GB | опционально |
| Qwen Omni / Audio | Qwen/Qwen2.5-Omni (3B/7B), multimodal audio | ~4-6 GB | опционально |

KAME (Sakana AI) не включен в таблицу сравнения: это speech-to-speech модель на базе Moshi, она не возвращает текстовую транскрипцию и поэтому не подходит для WER/CER.

## Метрики

- **WER**: word error rate, основная метрика качества.
- **CER**: character error rate.
- **RTF**: real-time factor, `inference_time / audio_duration`.
- **Latency**: средняя, p50 и p95 задержка на сегмент.
- **GPU Memory Peak**: пиковое потребление VRAM.
- **Error breakdown**: substitution / deletion / insertion.
- **WER by duration**: качество на коротких, средних и длинных фрагментах.

## Быстрый старт

```bash
make setup      # установить зависимости
make test       # запустить тесты без GPU и реальных моделей
make data       # скачать и подготовить датасет
make smoke      # прогон на 10 сэмплах
make quick      # прогон на 50 сэмплах
make quick-aug  # быстрый robustness-прогон с аугментациями
make bench      # полный Whisper-бенчмарк
make report     # показать последние результаты
make clean      # удалить подготовленные данные и результаты
make all        # setup > test > data > bench
```

То же самое без `make`:

```bash
pip install -r requirements.txt
python -m pytest tests/ -v --tb=short
python scripts/prepare_data.py
python scripts/run_benchmark.py --models whisper
```

## Запуск моделей

```bash
# Только Whisper
python scripts/run_benchmark.py --models whisper

# Только один Whisper-конфиг из config.WHISPER_MODELS
python scripts/run_benchmark.py --models whisper --whisper-models whisper-medium-int8

# Whisper + GigaAM-v3
python scripts/run_benchmark.py --models whisper,gigaam

# Whisper + VibeVoice
python scripts/run_benchmark.py --models whisper,vibevoice

# Whisper + Gemma 4
python scripts/run_benchmark.py --models whisper,gemma4

# Whisper + NeMo Conformer
python scripts/run_benchmark.py --models whisper,nemo

# Whisper + Silero STT
python scripts/run_benchmark.py --models whisper,silero

# Whisper + Qwen Omni
python scripts/run_benchmark.py --models whisper,qwen

# Все подключенные семейства моделей
python scripts/run_benchmark.py --models whisper,gigaam,vibevoice,phi4,gemma,gemma4,nemo,silero,qwen

# Ограничить число сэмплов
python scripts/run_benchmark.py --models whisper --max-samples 50

# Сохранить результаты в отдельную папку
python scripts/run_benchmark.py --models whisper --max-samples 50 --output-dir results/quick

# Robustness-прогон: clean + шум + speed/gain/lowpass варианты
python scripts/run_benchmark.py --models whisper --max-samples 50 --augment-audio --output-dir results/quick_aug

# Явно задать список аудио-аугментаций
python scripts/run_benchmark.py --models whisper --max-samples 50 --augment-audio clean,noise_20db,speed_0.95,speed_1.05
```

## Установка опциональных моделей

### GigaAM-v3

```bash
git clone https://github.com/salute-developers/GigaAM.git
cd GigaAM
pip install -e .[torch]
```

### VibeVoice-ASR

```bash
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice
pip install -e .
```

### NeMo Conformer

```bash
pip install nemo_toolkit[asr]
```

### Gemma 4

```bash
pip install -U "transformers>=4.55.0" bitsandbytes
```

Silero STT загружается через `torch.hub` и не требует отдельного пакета, кроме `torch` и `torchaudio`.

## Тесты

Тесты не скачивают датасеты, не грузят реальные ASR-модели и не требуют GPU:

```bash
python -m pytest tests/ -v --tb=short
```

Проверяется нормализация текста, метрики WER/CER/RTF, подготовка WAV, Excel-экспорт, фабрика моделей и полный пайплайн с mock-моделью.

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

После-обработка транскрипций через LLM для исправления ошибок распознавания:

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
- Для уже качественных моделей (whisper-large 7.5%) эффект нулевой — LLM нечего исправлять.
- Для слабых моделей (qwen 60%) LLM деградирует результат — слишком много ошибок, LLM добавляет шум.
- CER улучшается стабильнее WER — LLM лучше исправляет отдельные символы, чем целые слова.
- Словарная постобработка (транслитерация англицизмов) может дать дополнительный прирост.

### Файлы результатов

| Файл | Формат | Назначение |
|------|--------|------------|
| `results/results.csv` | CSV | агрегированные метрики по моделям |
| `results/per_sample_results.csv` | CSV | подробные результаты по каждому аудиофайлу |
| `results/llm_postprocess_results.csv` | CSV | per-sample результаты LLM-постобработки |
| `results/benchmark_report.xlsx` | Excel | отчет с несколькими листами |
| `results/analysis.txt` | TXT | краткий вывод по лучшей модели и скорости |
| `results/figures/*.png` | PNG | графики качества, скорости и ошибок |
| `results/augmented_audio/` | WAV | сгенерированные аудио-варианты для robustness-прогонов |

Excel-отчет содержит:

1. **Summary**: WER, CER, RTF, latency, GPU, error rates + wer_llm/cer_llm.
2. **Details**: reference, hypothesis, augmentation, WER/CER/RTF по каждому сэмплу.
3. **Errors (worst)**: 100 худших распознаваний.
4. **Correct**: примеры полностью совпавших транскрипций.
5. **Per-model Stats**: accuracy, mean/median WER/CER/RTF + wer_llm/cer_llm.
6. **LLM Post-process**: сводка до/после LLM-постобработки по моделям.
7. **bench results**: расширенная таблица — все поля + hypothesis_fixed, wer_llm, cer_llm.

В `results.csv` дополнительно сохраняется wall-clock время:

- `model_load_wall_sec`: сколько заняла загрузка модели.
- `warmup_wall_sec`: сколько занял warmup.
- `inference_wall_sec`: сколько занял inference-loop по всем сэмплам.
- `model_benchmark_wall_sec`: общий wall time по модели.
- `audio_augmentations`: список аудио-вариантов, использованных в прогоне.

## LLM-постобработка

После прогона бенчмарка можно исправить ошибки распознавания через LLM (требует запущенный Ollama):

```bash
# Установить Ollama и модель
ollama pull gemma3-chat:latest

# Постобработка всех моделей
python scripts/llm_postprocess.py
```

Скрипт читает `per_sample_results.csv`, отправляет каждый hypothesis в LLM, пересчитывает WER/CER с тем же нормализатором что и основной бенчмарк, и обновляет `benchmark_report.xlsx` колонками `wer_llm`/`cer_llm`.

## Аугментации

Флаг `--augment-audio` без значения включает набор:

```text
clean, noise_20db, noise_10db, speed_0.95, speed_1.05, gain_-6db, lowpass_4000
```

Это не заменяет нормальный clean-бенчмарк. Лучше сравнивать два прогона: обычный `clean` и отдельный robustness-прогон с аугментациями. Так видно, насколько модель держит шум, небольшое изменение скорости, громкости и полосы частот.

## Структура проекта

```text
tts/
+-- Makefile
+-- config.py
+-- requirements.txt
+-- README.md
+-- run_pipeline_check.py
+-- data/
|   +-- raw/
|   L-- processed/
+-- models/
|   +-- __init__.py
|   L-- inference.py
+-- scripts/
|   +-- __init__.py
|   +-- prepare_data.py
|   +-- run_benchmark.py
|   +-- evaluate.py
|   +-- llm_postprocess.py
|   L-- semantic_eval.py
+-- tests/
|   +-- __init__.py
|   L-- test_pipeline.py
L-- results/
```

## Нормализация текста

Для честного сравнения reference и hypothesis приводятся к общему виду:

- lowercase;
- Unicode NFC;
- удаление пунктуации;
- сохранение русских букв, английских букв, цифр и пробелов;
- схлопывание повторных пробелов.

Английские буквы сохранены намеренно: в русских датасетах встречаются `ai`, `ok`, `whisper`, названия продуктов и другие латинские вставки.

## Рекомендуемое окружение

- GPU: NVIDIA RTX 4070 Ti или аналогичный ускоритель с 12 GB VRAM.
- RAM: 32 GB.
- Disk: около 10 GB под датасеты и модели, плюс место под результаты.
- Python: 3.10+.
