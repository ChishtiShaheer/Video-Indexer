# Video Indexing at Speed

A high-throughput video indexing pipeline that detects five object classes — **person**, **car**, **dog**, **laptop**, **cell phone** — for every second of video across 50 FineVideo Parquet files (~125 hours), using Ultralytics YOLO11l.

---

## Evaluator Quick-Start (4 Steps)

### Step 1 — System packages (run before pip install)

```bash
sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0
```

> `ffmpeg` is required by PyAV for video decoding. `libgl1` / `libglib2.0-0` are required by OpenCV.

### Step 2 — `pip install -r requirements.txt`

`requirements.txt` includes a PyTorch CUDA 12.1 index URL so GPU-enabled torch installs automatically — **no separate torch install needed**:

```bash
pip install -r requirements.txt
```

> This installs: `torch` (CUDA 12.1), `torchvision`, `ultralytics`, `av`, `opencv-python-headless`, `pyarrow`, `huggingface_hub`, `numpy`, `tqdm`.  
> CUDA 12.1 wheels are forward-compatible with CUDA 12.2 / 12.3 / 12.4 and work on RTX 3xxx / 4xxx / A100 / H100.

### Step 3 — Preparation command (run once, internet required)

```bash
python run.py --prepare
```

This downloads `yolo11l.pt` (~50 MB) from Ultralytics CDN and runs a warm-up inference to verify GPU and all dependencies are working. Output looks like:

```
=== Preparation Phase ===
1. Checking Python dependencies...
   [OK] Torch: 2.x.x (CUDA available: True)
   [OK] GPU 0: NVIDIA GeForce RTX ...
   [OK] Ultralytics: available
   ...
2. Downloading / verifying official YOLO11l weights (yolo11l.pt)...
   [OK] YOLO11l loaded and warm-up inference passed.
```

### Step 4 — Run on test videos (no internet, timed)

```bash
python run.py --input <folder-of-parquet-files> --output results.jsonl
```

**Output format** — one JSON line per video-second:
```json
{"video": "train-00000-of-01357.parquet:5", "second": 0, "objects": ["person"], "people": 1}
```

---

## CLI Reference

| Argument | Default | Description |
|---|---|---|
| `--input <path>` | required | Folder of `.parquet` files (or a single file) |
| `--output <path>` | `results.jsonl` | Output `.jsonl` path |
| `--prepare` | — | Download weights and verify setup |
| `--fps <int>` | `5` | Frames sampled per second |
| `--batch-size <int>` | `64` | GPU inference batch size |
| `--workers <int>` | auto | CPU decode worker processes |
| `--device <str>` | `cuda:0` | e.g. `cuda:0`, `cuda:1`, `cpu` |
| `--limit <int>` | — | Process at most N parquet files |
| `--shard-index <int>` | `0` | For multi-GPU sharding |
| `--num-shards <int>` | `1` | Total shards |
| `--export-trt` | — | Build FP16 TensorRT engine during `--prepare` |

---

## Multi-GPU Run (Two GPUs)

```bash
python run.py --input finevideo/data --output results.jsonl \
    --shard-index 0 --num-shards 2 --device cuda:0 --workers 0 &

python run.py --input finevideo/data --output results.jsonl \
    --shard-index 1 --num-shards 2 --device cuda:1 --workers 0 &

wait
cat results.shard0of2.jsonl results.shard1of2.jsonl > results.jsonl
gzip -k -9 results.jsonl
```

---

## Unit Tests (no GPU needed)

```bash
python -m unittest discover -s tests -v
# Ran 17 tests in 0.003s — OK
```

---

## Kaggle Notebook

`image-indexer.ipynb` is the exact notebook used to produce `results.jsonl.gz` on Kaggle Dual T4 GPUs (~75 minutes for 125 hours of video). To re-run on Kaggle:

1. Import `image-indexer.ipynb` into a new Kaggle notebook.
2. Set Accelerator → **GPU T4 x 2**, Internet → **ON**.
3. Attach the [data-video](https://www.kaggle.com/datasets/shaheerahmedchishti/data-video) dataset as input.
4. Run all cells top-to-bottom.

---

## Pipeline Architecture

1. **Sharding & Streaming** (`run.py`, `vidindex/source.py`): Parquet files sorted and partitioned round-robin. MP4 bytes streamed directly from the `mp4` column via PyArrow — no disk writes.
2. **In-Memory Decoding** (`vidindex/decoder.py`): PyAV decodes `io.BytesIO` in-memory. Samples k=5 frames/second. Letterboxes to 640 px preserving aspect ratio. Handles rotation metadata automatically.
3. **Batched GPU Inference** (`vidindex/detector.py`): FP16 YOLO11l inference. NMS restricted to 5 target COCO class IDs `[0, 2, 16, 63, 67]`.
4. **Aggregation** (`vidindex/common.py`): Object present if detected in ≥ ⌈k/2⌉ frames of that second. People count = floor(median across sampled frames).
