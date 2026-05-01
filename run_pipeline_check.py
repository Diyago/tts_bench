"""
Pipeline check script: runs tests + smoke + quick benchmark.
Run: python run_pipeline_check.py
"""
import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent
os.chdir(ROOT)
PYTHON = sys.executable

SEPARATOR = "=" * 70


def run(label: str, cmd: list[str]) -> bool:
    print(f"\n{SEPARATOR}")
    print(f"? {label}")
    print(f"  CMD: {' '.join(cmd)}")
    print(SEPARATOR)

    result = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=False,   # print to stdout in real-time
        text=True,
    )

    print(f"\n{'PASSED' if result.returncode == 0 else 'FAILED'} - exit code {result.returncode}")
    return result.returncode == 0


def main():
    print(f"\n{'#'*70}")
    print("# ASR BENCHMARK - PIPELINE CHECK")
    print(f"# Working dir: {ROOT}")
    print(f"# Python: {PYTHON}")
    print(f"{'#'*70}\n")

    # -- STAGE 1: Unit tests -----------------------------------------------
    ok = run(
        "STAGE 1 - Unit tests (no GPU, no data)",
        [PYTHON, "-m", "pytest", "tests/", "-v", "--tb=short"],
    )
    if not ok:
        print("\n[ABORT] Tests failed - fix tests before running pipeline.")
        sys.exit(1)

    # -- STAGE 2: Smoke test (10 samples, Whisper) ----------------------
    ok = run(
        "STAGE 2 - Smoke test (10 samples, Whisper only)",
        [PYTHON, "scripts/run_benchmark.py", "--models", "whisper", "--max-samples", "10"],
    )
    if not ok:
        print("\n[ABORT] Smoke test failed - check model loading and data.")
        sys.exit(2)

    # -- STAGE 3: Quick run (50 samples, Whisper) ----------------------
    ok = run(
        "STAGE 3 - Quick benchmark (50 samples, Whisper)",
        [PYTHON, "scripts/run_benchmark.py", "--models", "whisper", "--max-samples", "50"],
    )
    if not ok:
        print("\n[WARN] Quick benchmark failed. Check logs above.")
        sys.exit(3)

    print(f"\n{'#'*70}")
    print("# ALL STAGES PASSED")
    print(f"# Results in: {ROOT / 'results'}/")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
