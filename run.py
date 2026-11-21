#!/usr/bin/env python3
"""Main entry point for the video indexing take-home task.

Usage:
    Preparation (internet enabled, run once):
        python run.py --prepare
        python run.py --prepare --export-trt --batch-size 64      # also builds FP16 TensorRT engine

    Timed run (offline):
        python run.py --input <folder of .parquet files> --output results.jsonl

    Two-GPU split (one process per GPU, then concatenate the outputs):
        python run.py --input data --output results.jsonl --shard-index 0 --num-shards 2 --device cuda:0 &
        python run.py --input data --output results.jsonl --shard-index 1 --num-shards 2 --device cuda:1 &
        wait; cat results.shard0of2.jsonl results.shard1of2.jsonl > results.jsonl
"""
import os

# Each decoder worker should use one thread; otherwise OpenMP/BLAS oversubscribe
# the CPU and slow decoding down. Must be set before numpy/torch/cv2 are imported.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import atexit
import inspect
import shutil
import sys
import tempfile
from pathlib import Path

# COCO class ids: person, car, dog, laptop, cell phone
TARGET_CLASSES = [0, 2, 16, 63, 67]
DEFAULT_IMGSZ = 640
DEFAULT_CONF = 0.4
DEFAULT_IOU = 0.7
PT_WEIGHTS = "yolo11l.pt"
ENGINE_WEIGHTS = "yolo11l.engine"


# --------------------------------------------------------------------------- #
# Preparation
# --------------------------------------------------------------------------- #
def prepare(batch_size, imgsz, export_trt):
    """Download weights, verify dependencies, optionally build a TensorRT engine."""
    print("=== Preparation Phase ===")
    print("1. Checking Python dependencies...")
    try:
        import av
        import cv2
        import numpy as np
        import pyarrow
        import torch
        from ultralytics import YOLO
        print(f"   [OK] Torch: {torch.__version__} (CUDA available: {torch.cuda.is_available()})")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"   [OK] GPU {i}: {torch.cuda.get_device_name(i)}")
        print("   [OK] Ultralytics: available")
        print(f"   [OK] PyAV: {av.__version__}")
        print(f"   [OK] OpenCV: {cv2.__version__}")
        print(f"   [OK] PyArrow: {pyarrow.__version__}")
    except ImportError as e:
        print(f"   [FAIL] Missing dependency: {e}")
        sys.exit(1)

    print(f"2. Downloading / verifying official YOLO11l weights ({PT_WEIGHTS})...")
    try:
        model = YOLO(PT_WEIGHTS)
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        model.predict(dummy, imgsz=imgsz, classes=TARGET_CLASSES, verbose=False)
        print("   [OK] YOLO11l loaded and warm-up inference passed.")
    except Exception as e:
        print(f"   [FAIL] Could not load model: {e}")
        sys.exit(1)

    if export_trt:
        print(f"3. Exporting FP16 TensorRT engine (max batch {batch_size}, imgsz {imgsz})...")
        print("   NOTE: the engine is tied to this GPU model and TensorRT version.")
        print("   Build it on the same machine/GPU type that runs the timed job.")
        if not torch.cuda.is_available():
            print("   [FAIL] CUDA is required for TensorRT export.")
            sys.exit(1)
        try:
            engine_path = model.export(
                format="engine",
                half=True,
                imgsz=imgsz,
                batch=batch_size,
                dynamic=True,   # allow partial / smaller batches at the tail of a run
                workspace=8,    # GiB for the TensorRT builder
                device=0,
            )
            trt_model = YOLO(engine_path)
            batch = [np.zeros((imgsz, imgsz, 3), dtype=np.uint8)] * min(batch_size, 8)
            trt_model.predict(batch, imgsz=imgsz, classes=TARGET_CLASSES, half=True, verbose=False)
            print(f"   [OK] Engine built and verified: {engine_path}")
        except Exception as e:
            print(f"   [WARN] TensorRT export failed ({e}). The pipeline will fall back to {PT_WEIGHTS}.")
    print("=== Preparation Complete: Ready for timed offline execution ===")


# --------------------------------------------------------------------------- #
# Sharding
# --------------------------------------------------------------------------- #
def discover_parquet(input_path):
    p = Path(input_path)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(p.rglob("*.parquet"))
    return []


def stage_shard(files, shard_index, num_shards):
    """Symlink this shard's files into a temp dir so the pipeline needs no changes.

    Round-robin over the sorted file list: deterministic, disjoint, and the shards
    stay roughly balanced in size.
    """
    mine = files[shard_index::num_shards]
    tmp = Path(tempfile.mkdtemp(prefix=f"vidindex_shard{shard_index}of{num_shards}_"))
    atexit.register(shutil.rmtree, tmp, ignore_errors=True)
    for f in mine:
        (tmp / f.name).symlink_to(f.resolve())
    return str(tmp), len(mine)


def shard_output_path(output, shard_index, num_shards):
    """results.jsonl -> results.shard0of2.jsonl so parallel shards never share a file."""
    if num_shards == 1:
        return output
    p = Path(output)
    return str(p.with_name(f"{p.stem}.shard{shard_index}of{num_shards}{p.suffix}"))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def build_parser():
    parser = argparse.ArgumentParser(
        description="Find five object classes per second across video parquet files."
    )
    parser.add_argument("--input", type=str,
                        help="Path to folder of .parquet files (or single .parquet file).")
    parser.add_argument("--output", type=str, default="results.jsonl",
                        help="Output .jsonl file path (default: results.jsonl).")
    parser.add_argument("--prepare", action="store_true",
                        help="Run preparation steps (download weights, warm up, verify setup).")
    parser.add_argument("--export-trt", action="store_true",
                        help="With --prepare: also export an FP16 TensorRT engine (yolo11l.engine).")
    parser.add_argument("--model", type=str, default=None,
                        help=f"Model weights. Default: {ENGINE_WEIGHTS} if present, else {PT_WEIGHTS}.")
    parser.add_argument("--fps", type=int, default=5,
                        help="Frames sampled per 1-second window (k, default: 5).")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Cross-video batch size for GPU inference (default: 64).")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of CPU decoder workers (default: auto).")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run inference on ('cuda', 'cuda:1', 'cpu').")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of videos to process (optional).")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ,
                        help=f"Inference image size (default: {DEFAULT_IMGSZ}).")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF,
                        help=f"Confidence threshold (default: {DEFAULT_CONF}).")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU,
                        help=f"NMS IoU threshold (default: {DEFAULT_IOU}).")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="Which shard of the parquet files to process (default: 0).")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of shards (default: 1 = no sharding).")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.prepare:
        prepare(args.batch_size, args.imgsz, args.export_trt)
        return

    if not args.input:
        parser.print_help()
        sys.exit(1)

    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        parser.error("--shard-index must satisfy 0 <= shard-index < num-shards")
    if args.fps < 1 or args.batch_size < 1:
        parser.error("--fps and --batch-size must be >= 1")

    # Resolve weights: prefer a prebuilt TensorRT engine when available.
    model_weights = args.model
    if model_weights is None:
        model_weights = ENGINE_WEIGHTS if Path(ENGINE_WEIGHTS).exists() else PT_WEIGHTS
    print(f"[run] model={model_weights} k={args.fps} batch={args.batch_size} "
          f"imgsz={args.imgsz} conf={args.conf} iou={args.iou}")

    # Decoder workers: leave a couple of threads for the main process / GPU feeder.
    if args.workers is None:
        cpu_cnt = os.cpu_count() or 4
        workers = max(1, min(cpu_cnt - 2, 8))
    else:
        workers = args.workers

    # Sharding
    input_path = args.input
    if args.num_shards > 1:
        files = discover_parquet(args.input)
        if not files:
            print(f"[run] No .parquet files found under {args.input}", file=sys.stderr)
            sys.exit(1)
        input_path, n_mine = stage_shard(files, args.shard_index, args.num_shards)
        print(f"[run] shard {args.shard_index + 1}/{args.num_shards}: "
              f"{n_mine} of {len(files)} parquet files")
    output_path = shard_output_path(args.output, args.shard_index, args.num_shards)
    print(f"[run] output -> {output_path}  (workers={workers})")

    from vidindex.pipeline import run_pipeline

    kwargs = dict(
        input_path=input_path,
        output_path=output_path,
        model_weights=model_weights,
        k=args.fps,
        batch_size=args.batch_size,
        num_workers=workers,
        device=args.device,
        limit=args.limit,
        # Optional knobs: forwarded only if vidindex.pipeline.run_pipeline accepts them.
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        classes=TARGET_CLASSES,
    )
    params = inspect.signature(run_pipeline).parameters
    if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        optional = ("imgsz", "conf", "iou", "classes")
        dropped = [k for k in optional if k not in params]
        for k in dropped:
            kwargs.pop(k)
        if dropped:
            print(f"[run] WARNING: run_pipeline() does not accept {dropped}; "
                  "those values are hard-coded inside vidindex/. Check they match "
                  f"imgsz={DEFAULT_IMGSZ}, conf={DEFAULT_CONF}, iou={DEFAULT_IOU}, "
                  f"classes={TARGET_CLASSES}.", file=sys.stderr)

    run_pipeline(**kwargs)


if __name__ == "__main__":
    main()