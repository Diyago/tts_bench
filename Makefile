# ------------------------------------------------------------
# ASR Benchmark - Makefile (Karpathy style)
# ------------------------------------------------------------
# Usage:
#   make setup      - install dependencies
#   make test       - run tests (verify pipeline, no GPU needed)
#   make data       - download & prepare dataset
#   make bench      - run full benchmark
#   make quick      - quick run on 50 samples
#   make clean      - remove generated data & results
#   make all        - setup > test > data > bench
# ------------------------------------------------------------

.PHONY: all setup test data bench bench-all bench-nemo bench-silero bench-full quick quick-aug report clean help

# Default
all: setup test data bench

# --- Setup -------------------------------------------------
setup:
	pip install -r requirements.txt

# --- Tests -------------------------------------------------
# Run the full test suite: verifies every pipeline stage
# without downloading data or loading real models.
test:
	python -m pytest tests/ -v --tb=short

# --- Data --------------------------------------------------
# Download Common Voice RU from HuggingFace & prepare it.
# Requires: huggingface-cli login (for Common Voice license)
data:
	python scripts/prepare_data.py

# --- Benchmark ---------------------------------------------
# Run full benchmark with Whisper models.
# Data is auto-downloaded if not present.
bench:
	python scripts/run_benchmark.py --models whisper --skip-data-prep

# Full benchmark including GigaAM (requires nemo_toolkit)
bench-all:
	python scripts/run_benchmark.py --models whisper,gigaam --skip-data-prep

# Benchmark with NeMo Conformer
bench-nemo:
	python scripts/run_benchmark.py --models whisper,nemo --skip-data-prep

# Benchmark with Silero STT
bench-silero:
	python scripts/run_benchmark.py --models whisper,silero --skip-data-prep

# Full benchmark with ALL models
bench-full:
	python scripts/run_benchmark.py --models whisper,gigaam,vibevoice,gemma,gemma4,nemo,silero --skip-data-prep

# --- Quick run ---------------------------------------------
# Fast test on 50 samples, good for sanity checking.
quick:
	python scripts/run_benchmark.py --models whisper --max-samples 50

# Robustness run on augmented audio variants.
quick-aug:
	python scripts/run_benchmark.py --models whisper --whisper-models whisper-medium-int8 --max-samples 50 --augment-audio --output-dir results/quick_aug

# Even quicker: 10 samples, just to verify everything works.
smoke:
	python scripts/run_benchmark.py --models whisper --max-samples 10

# --- Report ------------------------------------------------
# Just print existing results
report:
	@echo "=== Results ==="
	@python -c "import pandas as pd; df=pd.read_csv('results/results.csv'); print(df.to_string())"
	@echo ""
	@echo "=== Analysis ==="
	@type results\analysis.txt 2>nul || echo "No analysis yet. Run: make bench"

# --- Clean -------------------------------------------------
clean:
	@if exist data\processed rd /s /q data\processed
	@if exist results rd /s /q results
	@echo Cleaned data/processed/ and results/

clean-all: clean
	@if exist data rd /s /q data
	@echo Cleaned all data/

# --- Help --------------------------------------------------
help:
	@echo.
	@echo ASR Benchmark - Available targets:
	@echo.
	@echo   make setup      Install Python dependencies
	@echo   make test       Run test suite (no GPU needed)
	@echo   make data       Download ^& prepare Common Voice RU
	@echo   make bench      Run full Whisper benchmark
	@echo   make bench-all  Run with Whisper + GigaAM
	@echo   make bench-nemo Run with Whisper + NeMo Conformer
	@echo   make bench-silero  Run with Whisper + Silero STT
	@echo   make bench-full Run ALL models
	@echo   make quick      Quick benchmark (50 samples)
	@echo   make quick-aug  Quick robustness run with audio augmentations
	@echo   make smoke      Smoke test (10 samples)
	@echo   make report     Print existing results
	@echo   make clean      Remove processed data ^& results
	@echo   make clean-all  Remove ALL data
	@echo   make all        Full pipeline: setup > test > data > bench
	@echo.
