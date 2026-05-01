# Bootstrap script – run once to initialise git and install nvidia-ml-py
# Usage: python bootstrap.py

import subprocess, sys, os

root = os.path.dirname(__file__)

# 1. Initialise git repo
print("=== Initialising git repo ===")
subprocess.run(["git", "init"], cwd=root, check=True)
subprocess.run(["git", "add", "-A"], cwd=root, check=True)
subprocess.run(
    ["git", "commit", "-m", "Initial benchmark codebase"],
    cwd=root, check=True,
)
print("Git repo initialised and first commit made.\n")

# 2. Install nvidia-ml-py if not already present
print("=== Installing nvidia-ml-py (NVML GPU memory tracking) ===")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "nvidia-ml-py>=12.535.77"],
    check=True,
)
print("\nDone. You can now run the benchmark:")
print("  python scripts/run_benchmark.py --max-samples 50 --models whisper")
