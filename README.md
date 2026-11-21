# Video Indexing at Speed

A high-throughput video indexing pipeline designed to process 125+ hours of video across 50 Parquet files from the FineVideo dataset. For each second of video, it detects the presence of five target classes (**person**, **car**, **dog**, **laptop**, **cell phone**) and calculates the exact visible people count using Ultralytics YOLO11l.

---

## 1. How to Run

### Option A — Kaggle (Recommended, fastest)

> **This is how the submission was produced.** Use the included `image-indexer.ipynb` notebook.

1. Go to [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook** → **File → Import Notebook** → upload `image-indexer.ipynb`.
2. **Settings → Accelerator**: Set to **GPU T4 x 2** (or P100).
3. **Settings → Internet**: Toggle **ON**.
4. Add the [data-video](https://www.kaggle.com/datasets/shaheerahmedchishti/data-video) dataset as input.
5. Run all cells top-to-bottom. Cell 2 installs all Python dependencies automatically.

---

### Option B — Local Machine (Linux / WSL2 with NVIDIA GPU)

#### Step 1 — Clone the repo
```bash
git clone https://github.com/ChishtiShaheer/Video-Indexer.git
cd Video-Indexer
```

#### Step 2 — System libraries
```bash
sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 build-essential
```

#### Step 3 — Python environment
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

#### Step 4 — Install PyTorch (GPU)
```bash
# CUDA 12.1 (RTX 3xxx / 4xxx / A100):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU-only (no GPU):
pip install torch torchvision
```

#### Step 5 — Install remaining dependencies
```bash
pip install -r requirements.txt
```

#### Step 6 — Download the FineVideo dataset (50 Parquet files, ~56 GB)
```bash
pip install huggingface_hub
huggingface-cli login          # paste your HF read token
mkdir -p finevideo/data
huggingface-cli download HuggingFaceFV/finevideo \
  --repo-type dataset \
  --include "data/train-000[0-4]*.parquet" \
  --local-dir finevideo
```

> Accept dataset terms first at [huggingface.co/datasets/HuggingFaceFV/finevideo](https://huggingface.co/datasets/HuggingFaceFV/finevideo)

#### Step 7 — Prepare (download weights, warm up)
```bash
python run.py --prepare
```

#### Step 8 — Run the pipeline
```bash
# Single GPU
python run.py --input finevideo/data --output results.jsonl --fps 5 --batch-size 64

# Two GPUs (faster)
python run.py --input finevideo/data --output results.jsonl --shard-index 0 --num-shards 2 --device cuda:0 --workers 0 &
python run.py --input finevideo/data --output results.jsonl --shard-index 1 --num-shards 2 --device cuda:1 --workers 0 &
wait
cat results.shard0of2.jsonl results.shard1of2.jsonl > results.jsonl
gzip -k -9 results.jsonl
```

---

## 2. Running the Pipeline

> See **Option B Step 8** above for the full local run commands.

### Single GPU
```bash
python run.py --input <folder of .parquet files> --output results.jsonl --fps 5 --batch-size 64
```

### Two GPUs
```bash
python run.py --input finevideo/data --output results.jsonl --shard-index 0 --num-shards 2 --device cuda:0 --workers 0 &
python run.py --input finevideo/data --output results.jsonl --shard-index 1 --num-shards 2 --device cuda:1 --workers 0 &
wait
cat results.shard0of2.jsonl results.shard1of2.jsonl > results.jsonl
```

### Quick Baseline Verification (First 20 Videos)
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
- `--workers <int>`: Number of background CPU decoding worker processes (default: `min(cpu - 2, 8)`).
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
