# Video Indexing at Speed

A high-throughput video indexing pipeline designed to process 125+ hours of video across 50 Parquet files from the FineVideo dataset. For each second of video, it detects the presence of five target classes (**person**, **car**, **dog**, **laptop**, **cell phone**) and calculates the exact visible people count using Ultralytics YOLO11l.

---

## 1. Environment Setup

### System Packages (Ubuntu 22.04)
Run the following command to install required media and runtime libraries:
```bash
sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    build-essential
```

### Python Environment
Create and activate a Python 3.11 virtual environment, then install pinned dependencies:
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Preparation Command (Run Once With Internet)

Per evaluation guidelines, run the preparation command once before the timed run:
```bash
# Standard preparation: downloads yolo11l.pt and warms up CUDA
python run.py --prepare
```

### Optional: TensorRT FP16 Engine Build
To build a high-performance TensorRT FP16 engine on the target machine:
```bash
python run.py --prepare --export-trt --batch-size 64
```
*Note: The engine is tied to the specific GPU architecture and TensorRT version. When `yolo11l.engine` exists, `run.py` automatically selects it; otherwise it gracefully falls back to `yolo11l.pt`.*

---

## 3. Running the Pipeline (Offline Timed Run)

### Single GPU (Standard Run)
```bash
python run.py --input <folder of .parquet files> --output results.jsonl
```

### Two GPUs (Dual T4 on Kaggle / Multi-GPU Instances)
Split the files across two disjoint shards and run one process per GPU, dividing CPU decoder workers evenly to prevent CPU oversubscription:
```bash
python run.py --input finevideo/data --output results.jsonl --shard-index 0 --num-shards 2 --device cuda:0 --workers 6 &
python run.py --input finevideo/data --output results.jsonl --shard-index 1 --num-shards 2 --device cuda:1 --workers 6 &
wait
cat results.shard0of2.jsonl results.shard1of2.jsonl > results.jsonl
```

### Quick Baseline Verification (First 20 Videos)
Before launching the full multi-hour run, verify end-to-end functionality on 20 videos:
```bash
python run.py --input finevideo/data --output results_baseline.jsonl --limit 20 --fps 5
```

### CLI Arguments & Options
- `--input <path>`: Folder containing `.parquet` files (or a single file).
- `--output <path>`: Output `.jsonl` path (default: `results.jsonl`).
- `--prepare`: Run setup & model download.
- `--export-trt`: Build TensorRT engine during `--prepare`.
- `--model <path>`: Model weights path (default: `yolo11l.engine` if present, else `yolo11l.pt`).
- `--fps <int>`: Sampling rate $k$ frames per second (default: `5`).
- `--batch-size <int>`: Cross-video batch size for GPU inference (default: `64`).
- `--workers <int>`: Number of background CPU decoding worker processes (default: `min(cpu - 2, 28)`).
- `--device <str>`: Compute device (e.g., `cuda:0`, `cuda:1`, `cpu`).
- `--limit <int>`: Process at most $N$ videos.
- `--imgsz <int>`: Inference resolution (default: `640`).
- `--conf <float>`: Confidence threshold (default: `0.4`).
- `--iou <float>`: NMS IoU threshold (default: `0.7`).
- `--shard-index <int>`: Shard index for parallel processing (default: `0`).
- `--num-shards <int>`: Total number of shards (default: `1`).

---

## 4. Pipeline Architecture

1. **Round-Robin Sharding & Streaming (`run.py`, `vidindex/source.py`)**:
   - Files are sorted and partitioned round-robin across shards.
   - Parquet rows are streamed directly from the `mp4` column into RAM via PyArrow without temporary disk writes.
2. **In-Memory Decoding & Letterboxing (`vidindex/decoder.py`)**:
   - PyAV decodes in-memory bytes (`io.BytesIO`).
   - Automatically detects display matrices and rotation metadata, rotating frames to the upright orientation.
   - Samples $k=5$ frames per second (first frame of each $1/k$-second slot).
   - Letterboxes to 640px preserving aspect ratio (matching Ultralytics defaults).
   - OpenMP and BLAS threads are pinned to 1 per worker (`OMP_NUM_THREADS=1`) to avoid CPU thrashing.
3. **GPU Batched Inference (`vidindex/detector.py`)**:
   - Runs batched inference on GPU (FP16 `.pt` or TensorRT `.engine`).
   - Limits NMS to the 5 target classes (`[0, 2, 16, 63, 67]`), avoiding postprocessing on irrelevant COCO categories.
4. **Aggregation & Formatting (`vidindex/common.py`)**:
   - Presence is determined when detected in $\ge \lceil N/2 \rceil$ frames of that second.
   - People count is the floor median across sampled frames.
   - Formatted JSON Lines entries are appended atomically with support for error recovery.

---

## 5. Verification & Tests

To run the pure-Python unit tests (no GPU or video needed):
```bash
python -m unittest discover -s tests -v
```
