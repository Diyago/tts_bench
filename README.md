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

| Файл | Формат | Назначение |
|------|--------|------------|
| `results/results.csv` | CSV | агрегированные метрики по моделям |
| `results/per_sample_results.csv` | CSV | подробные результаты по каждому аудиофайлу |
| `results/benchmark_report.xlsx` | Excel | отчет с несколькими листами |
| `results/analysis.txt` | TXT | краткий вывод по лучшей модели и скорости |
| `results/figures/*.png` | PNG | графики качества, скорости и ошибок |
| `results/augmented_audio/` | WAV | сгенерированные аудио-варианты для robustness-прогонов |

Excel-отчет содержит:

1. **Summary**: WER, CER, RTF, latency, GPU и error rates.
2. **Details**: reference, hypothesis, augmentation, WER/CER/RTF по каждому сэмплу.
3. **Errors (worst)**: 100 худших распознаваний.
4. **Correct**: примеры полностью совпавших транскрипций.
5. **Per-model Stats**: accuracy, mean/median WER/CER/RTF.

В `results.csv` дополнительно сохраняется wall-clock время:

- `model_load_wall_sec`: сколько заняла загрузка модели.
- `warmup_wall_sec`: сколько занял warmup.
- `inference_wall_sec`: сколько занял inference-loop по всем сэмплам.
- `model_benchmark_wall_sec`: общий wall time по модели.
- `audio_augmentations`: список аудио-вариантов, использованных в прогоне.

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
|   L-- evaluate.py
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
