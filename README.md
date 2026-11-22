# Video Indexing at Speed

A high-throughput video indexing pipeline designed to process 125+ hours of video across 50 Parquet files from the FineVideo dataset. For each second of video, it detects the presence of five target classes (**person**, **car**, **dog**, **laptop**, **cell phone**) and calculates the exact visible people count using Ultralytics YOLO11l.

---

## 1. Quick Start / Evaluator Instructions

This project conforms to the evaluation specifications in Section 6 and 7 of the Candidate Brief.

### Step 1: System Packages (Ubuntu 22.04)
Run before installing Python requirements to provide media decoding libraries and OpenGL runtime dependencies:
```bash
sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0
```

### Step 2: Python Environment & Dependencies
In a clean Python virtual environment (e.g. Python 3.10 / 3.11):
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on CUDA/PyTorch**: `requirements.txt` points to the official PyTorch CUDA 12.1 wheel index, which natively supports NVIDIA RTX 3xxx/4xxx/A100/H100 GPUs with CUDA 12.

### Step 3: Preparation Command (Run Once With Internet)
Per Section 6 of the brief, run the preparation command once before the timed run:
```bash
python run.py --prepare
```
This command:
1. Verifies all required dependencies and GPU acceleration (`torch.cuda.is_available()`).
2. Downloads official Ultralytics YOLO11l weights (`yolo11l.pt`, ~50 MB).
3. Executes a warm-up inference pass to eliminate first-batch CUDA compilation latency.

### Step 4: Timed Offline Run
Run `run.py` on the test video folder without internet access:
```bash
python run.py --input <folder of .parquet files> --output results.jsonl
```

---

## 2. CLI Reference & Options

```bash
python run.py [OPTIONS]
```

| Argument | Default | Description |
|---|---|---|
| `--input <path>` | *required* | Path to folder of `.parquet` files (or a single `.parquet` file). |
| `--output <path>` | `results.jsonl` | Output JSON Lines file path. |
| `--prepare` | `False` | Run setup, verify dependencies, download weights, and warm up model. |
| `--export-trt` | `False` | With `--prepare`: also build an FP16 TensorRT engine (`yolo11l.engine`). |
| `--model <path>` | auto | Model weights. Automatically uses `yolo11l.engine` if present, else `yolo11l.pt`. |
| `--fps <int>` | `5` | Sampling rate $k$ frames per 1-second window. |
| `--batch-size <int>` | `64` | Cross-video batch size for GPU inference. |
| `--workers <int>` | auto | Number of background CPU decoding worker processes. Use `0` for direct serial mode. |
| `--device <str>` | `cuda:0` | Compute device (`cuda:0`, `cuda:1`, or `cpu`). |
| `--limit <int>` | none | Process at most $N$ videos (useful for quick verification). |
| `--shard-index <int>` | `0` | Shard index for parallel processing across multiple GPUs. |
| `--num-shards <int>` | `1` | Total number of shards. |

---

## 3. High-Throughput Multi-GPU Run

To run on multiple GPUs (e.g. Dual T4 or multi-GPU workstations), shard the Parquet files across devices:

```bash
# Launch Shard 0 on GPU 0
python run.py --input finevideo/data --output results.jsonl \
  --shard-index 0 --num-shards 2 --device cuda:0 --workers 0 &

# Launch Shard 1 on GPU 1
python run.py --input finevideo/data --output results.jsonl \
  --shard-index 1 --num-shards 2 --device cuda:1 --workers 0 &

wait

# Merge disjoint shard outputs into a single file and compress
cat results.shard0of2.jsonl results.shard1of2.jsonl > results.jsonl
gzip -k -9 results.jsonl
```

---

## 4. Verification & Unit Tests

Run the pure-Python test suite (no GPU or video download required):
```bash
python -m unittest discover -s tests -v
```

Unit tests cover:
- Median calculation rules (floor rounding on even counts).
- Detection presence criteria ($\ge \lceil k/2 \rceil$ frames).
- Display matrix rotation parsing ($0^\circ, 90^\circ, 180^\circ, 270^\circ$).
- Letterbox aspect-ratio preservation and padding geometry.
- Graceful handling of corrupted/damaged video rows.

---

## 5. Output Format Specifications

Outputs strictly adhere to Section 4 of the Candidate Brief:
- **Valid video-second**:
  ```json
  {"video": "train-00003-of-01357.parquet:4", "second": 12, "objects": ["person", "laptop"], "people": 2}
  ```
- **Empty second**:
  ```json
  {"video": "train-00003-of-01357.parquet:4", "second": 13, "objects": [], "people": 0}
  ```
- **Corrupted/Damaged video**:
  ```json
  {"video": "train-00003-of-01357.parquet:5", "error": "could not decode video"}
  ```

---

## 6. Pipeline Architecture

1. **Streaming & Sharding (`run.py`, `vidindex/source.py`)**:
   - Parquet files are sorted and deterministically partitioned round-robin.
   - MP4 video streams are read row-by-row directly from the `mp4` byte column via PyArrow into memory without writing temporary files to disk.
2. **In-Memory Decoding & Frame Sampling (`vidindex/decoder.py`)**:
   - PyAV decodes video bytes in-memory (`io.BytesIO`).
   - Rotation metadata and display matrices are automatically inspected and corrected.
   - Samples $k=5$ frames per second (selecting the first frame of each $1/k$-second bucket).
   - Letterboxes frames to 640px while strictly preserving the original aspect ratio.
3. **Batched FP16 GPU Inference (`vidindex/detector.py`)**:
   - Half-precision (FP16) batched inference using YOLO11l.
   - Restricts NMS output to the 5 target COCO classes (`[0, 2, 16, 63, 67]`) to minimize post-processing overhead.
   - Auto-resolves precision arguments across Ultralytics runtime versions without warnings or deprecation errors.
4. **Aggregation & Error Recovery (`vidindex/common.py`)**:
   - Target objects are flagged present if detected in $\ge \lceil k/2 \rceil$ frames of that second.
   - Visible people count is calculated as the floor median of counts across sampled frames.
   - Partial progress is preserved on decode errors and cleanly finalized.
