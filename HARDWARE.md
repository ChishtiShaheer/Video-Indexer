# Hardware & Environment Specifications

Record details of the environment where you ran the full 50 files.

---

## 1. Execution Platform
- **Platform**: Kaggle Notebooks
- **Date & Time of Run**: [Fill: e.g., 2026-11-21 21:00 PKT]
- **Execution Mode**: [Single GPU / Dual GPU Sharded (T4 x 2) / TensorRT Engine]

---

## 2. Hardware Specifications

- **GPU Model**: NVIDIA Tesla T4 (or 2x Tesla T4 / Tesla P100)
- **GPU Memory**: 16 GB (or 2x 16 GB)
- **CPU Model**: Intel(R) Xeon(R) CPU @ 2.20GHz
- **CPU Cores / Threads**: 4 vCPUs (2 physical cores, 4 threads)
- **RAM**: 30 GB (Kaggle High-RAM session)
- **Disk Type**: NVMe SSD (73 GB scratch space in `/kaggle/working`)
- **Operating System**: Linux (Ubuntu 20.04.6 LTS / Ubuntu 22.04 LTS, Kernel 5.15)
- **NVIDIA Driver Version**: 535.183.01
- **CUDA Version**: 12.2

---

## 3. Timing & Throughput Summary

- **Dataset Processed**: First 50 Parquet files of FineVideo (`train-00000-of-01357.parquet` to `train-00049-of-01357.parquet`)
- **Total Footage**: ~125 hours (~1,600 videos)
- **Execution Configuration**:
  - Model: `yolo11l.pt` (or `yolo11l.engine`)
  - Sampling Rate ($k$): 5 frames per second
  - Batch Size: 64
  - Shards: 2 (if dual T4) or 1 (single GPU)
- **Start Time**: [Fill: YYYY-MM-DD HH:MM:SS]
- **End Time**: [Fill: YYYY-MM-DD HH:MM:SS]
- **Total Wall-Clock Runtime**: [Fill: Record exact measured time from Kaggle run]
- **Average Throughput**: [Fill: e.g., ~25–35 video-seconds per real second]
- **Final Output File**: `results.jsonl.gz` (Size: [Fill: e.g., ~8.5 MB])

---

## 4. Empirical Trade-off & Baseline Validation (Sample on 20 Videos)

Measure and document actual numbers on a 20-video baseline slice:
- **Baseline (Full Frame, all 30 fps)**: Time: [Fill]s | Objects F1: 1.00 | People Exact: 100%
- **Sampled ($k=4$)**: Time: [Fill]s | Objects F1: [Fill] | People Exact: [Fill]% | Speedup: [Fill]x
- **Sampled ($k=5$)**: Time: [Fill]s | Objects F1: [Fill] | People Exact: [Fill]% | Speedup: [Fill]x
- **Sampled ($k=7$)**: Time: [Fill]s | Objects F1: [Fill] | People Exact: [Fill]% | Speedup: [Fill]x

---

## 5. Verification Check
- Output contains one line for every second up to the final frame of each video.
- All target object names match: `person`, `car`, `dog`, `laptop`, `cell phone`.
- Damaged video rows properly record `{"video": "...", "error": "..."}` without crashing.
