#!/usr/bin/env python3
import os
import subprocess
import time
from datetime import datetime

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["DISPLAY"] = ":0"

# === CONFIGURATION ===
COLMAP_EXE = "colmap"                     # or full path if needed
MAGICK_EXE = "magick"                     # for resize (optional)
COLMAP_SCRIPT = "convert.py"     # 현재 디렉터리에 있는 converter 파일
DATASET_ROOT = "/workspace/masked_datasets/DTU_chaewon"   # DTU dataset 루트 경로
USE_GPU = False
DO_RESIZE = True
LOG_DIR = os.path.join(DATASET_ROOT, "colmap_logs")

# ======================

SCAN_LIST = os.listdir(DATASET_ROOT)

os.makedirs(LOG_DIR, exist_ok=True)


def run_cmd(cmd, log_file):
    """명령어 실행 + 실시간 로그 저장"""
    with open(log_file, "a") as f:
        f.write(f"\n[COMMAND] {cmd}\n")
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in process.stdout:
            decoded = line.decode("utf-8", errors="ignore")
            print(decoded, end="")
            f.write(decoded)
        process.wait()
        return process.returncode


def run_colmap_for_scan(scan_name):
    scan_path = os.path.join(DATASET_ROOT, scan_name)
    if not os.path.exists(scan_path):
        print(f"[SKIP] {scan_path} does not exist.")
        return

    log_file = os.path.join(LOG_DIR, f"{scan_name}_{datetime.now().strftime('%m%d_%H%M')}.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    print(f"\n=== Running COLMAP for {scan_name} ===")
    start = time.time()

    cmd = (
        f"python {COLMAP_SCRIPT} "
        f"-s \"{scan_path}\" "
        f"--colmap_executable \"{COLMAP_EXE}\" "
        f"--magick_executable \"{MAGICK_EXE}\" "
    )
    if not USE_GPU:
        cmd += "--no_gpu "
    if DO_RESIZE:
        cmd += "--resize "

    code = run_cmd(cmd, log_file)

    elapsed = time.time() - start
    if code == 0:
        print(f"[SUCCESS] {scan_name} finished in {elapsed:.1f}s")
    else:
        print(f"[ERROR] {scan_name} failed (exit {code}) – see {log_file}")


def main():
    print("=== Automatic COLMAP Runner ===")
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Target scans: {SCAN_LIST}\n")

    for scan in SCAN_LIST:
        run_colmap_for_scan(scan)

    print("\nAll scans processed.")


if __name__ == "__main__":
    main()
