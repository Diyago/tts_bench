# Bootstrap script – installs extra runtime dependencies not in requirements.txt
# Usage: python bootstrap.py

import subprocess, sys

print("=== Installing nvidia-ml-py (NVML GPU memory tracking) ===")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "nvidia-ml-py>=12.535.77"],
    check=True,
)
print("\nDone. You can now run the benchmark:")
print("  python scripts/run_benchmark.py --max-samples 50 --models whisper")
