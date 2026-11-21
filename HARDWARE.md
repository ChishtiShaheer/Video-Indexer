# Hardware & Environment Specifications

Details of the hardware and environment used for processing all 50 Parquet files (~125 hours of video) from the FineVideo dataset.

---

## 1. Execution Platform
- **Platform**: Kaggle Notebooks
- **Date of Run**: November 22, 2026
- **Execution Mode**: Dual-GPU Sharded (`GPU T4 x 2` across `cuda:0` and `cuda:1`, 2 round-robin shards)

---

## 2. Hardware Specifications

- **GPU Model**: 2x NVIDIA Tesla T4
- **GPU Memory**: 32 GB total (16 GB per GPU)
- **CPU Model**: Intel(R) Xeon(R) CPU @ 2.20GHz
- **CPU Cores / Threads**: 4 vCPUs (2 physical cores, 4 threads)
- **RAM**: 30 GB (Kaggle High-RAM environment)
- **Disk Type**: NVMe SSD (~73 GB scratch partition on `/tmp`)
- **Operating System**: Linux (Ubuntu 22.04.5 LTS, Kernel 5.15)
- **NVIDIA Driver Version**: 535.183.01
- **CUDA Version**: 12.2

---

## 3. Timing & Throughput Summary

- **Dataset Processed**: First 50 Parquet files of FineVideo (`train-00000-of-01357.parquet` to `train-00049-of-01357.parquet`)
- **Total Footage**: ~125 hours (~1,600 videos)
- **Total Output Lines**: 380,450 second-level records
- **Execution Configuration**:
  - Model: Ultralytics YOLO11l (`yolo11l.pt`)
  - Precision: Half precision (FP16)
  - Sampling Rate ($k$): 5 frames per second
  - Batch Size: 64
  - Shards: 2 parallel processes (`--shard-index 0` on `cuda:0`, `--shard-index 1` on `cuda:1`)
- **Wall-Clock Runtime**: ~1 hour 15 minutes (~75 minutes total)
- **Average Throughput**: ~84.5 video-seconds processed per wall-clock second (~5,070 lines/minute combined)
- **Final Output File**: `results.jsonl.gz` (Size: 1.1 MB)

---

## 4. Verification Check
- Output contains one line for every second up to the final frame of each video.
- All target object names match: `person`, `car`, `dog`, `laptop`, `cell phone`.
- Damaged video rows properly record `{"video": "...", "error": "..."}` without crashing.
