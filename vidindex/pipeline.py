"""High-throughput pipeline coordinating video reading, decoding, inference, and formatting."""
import collections
import multiprocessing as mp
import os
import queue
import sys
import time

from .common import (CONF_THRESHOLD, IMGSZ, IOU_THRESHOLD, TARGET_CLASSES,
                     finalize_video, short_error)
from .decoder import process_video_frames
from .detector import Detector
from .source import find_parquet_files, iter_videos


def _decode_worker(task_queue, result_queue, k, chunk_size, imgsz):
    """Worker process: reads mp4 bytes from task_queue, decodes, letterboxes, puts chunks in result_queue."""
    while True:
        try:
            item = task_queue.get()
        except (EOFError, KeyboardInterrupt):
            break
        if item is None:
            break
        vid, data, err = item
        if err is not None:
            result_queue.put((vid, None, [], True, None, err))
            continue
        try:
            for chunk in process_video_frames(vid, data, k=k, chunk_size=chunk_size, imgsz=imgsz):
                result_queue.put(chunk)
        except Exception as e:
            result_queue.put((vid, None, [], True, None, short_error(e)))


def load_completed_videos(output_path):
    """Scan an existing output file and return a set of video IDs that have already finished."""
    completed = set()
    if not os.path.exists(output_path):
        return completed
    try:
        import json
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "video" in obj:
                        completed.add(obj["video"])
                except Exception:
                    pass
    except Exception:
        pass
    return completed


def run_pipeline(
    input_path,
    output_path,
    model_weights="yolo11l.pt",
    k=5,
    batch_size=64,
    num_workers=4,
    device=None,
    limit=None,
    imgsz=IMGSZ,
    conf=CONF_THRESHOLD,
    iou=IOU_THRESHOLD,
    classes=None,
    **kwargs,
):
    """Run the complete end-to-end video indexing pipeline."""
    parquet_files = find_parquet_files(input_path)
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in {input_path}")

    print(f"[Pipeline] Found {len(parquet_files)} parquet file(s).")
    completed_videos = load_completed_videos(output_path)
    if completed_videos:
        print(f"[Pipeline] Resuming: {len(completed_videos)} videos already recorded in {output_path}.")

    print(
        f"[Pipeline] Initializing YOLO11l detector (weights={model_weights}, "
        f"device={device or 'auto'}, imgsz={imgsz}, conf={conf}, iou={iou})..."
    )
    detector = Detector(
        weights=model_weights,
        device=device,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        classes=classes,
    )

    # Open output file in append mode for resume support
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_file = open(output_path, "a", encoding="utf-8")

    video_state = {}
    total_videos_processed = len(completed_videos)
    start_time = time.time()

    if num_workers > 0:
        ctx = mp.get_context("spawn")
        task_queue = ctx.Queue(maxsize=num_workers * 2)
        result_queue = ctx.Queue(maxsize=num_workers * 4)

        workers = []
        for _ in range(num_workers):
            p = ctx.Process(
                target=_decode_worker,
                args=(task_queue, result_queue, k, batch_size, imgsz),
                daemon=True,
            )
            p.start()
            workers.append(p)

        import threading

        def feeder():
            for vid, data, err in iter_videos(parquet_files, skip=completed_videos, limit=limit):
                task_queue.put((vid, data, err))
            for _ in range(num_workers):
                task_queue.put(None)

        feed_thread = threading.Thread(target=feeder, daemon=True)
        feed_thread.start()

        active_workers = num_workers
        while active_workers > 0:
            try:
                item = result_queue.get(timeout=300)
            except queue.Empty:
                if not any(w.is_alive() for w in workers):
                    break
                continue

            vid, batch, seconds_meta, is_last, max_sec, err = item
            if vid not in video_state:
                video_state[vid] = {"secs": collections.defaultdict(list), "max_sec": -1}

            if max_sec is not None and max_sec > video_state[vid]["max_sec"]:
                video_state[vid]["max_sec"] = max_sec

            if batch is not None and len(batch) > 0:
                counts_list = detector.predict_batch(batch)
                for sec, counts in zip(seconds_meta, counts_list):
                    video_state[vid]["secs"][sec].append(counts)

            if is_last:
                st = video_state.pop(vid)
                lines = finalize_video(
                    vid,
                    st["secs"],
                    st["max_sec"],
                    error=err,
                )
                for line in lines:
                    out_file.write(line + "\n")
                out_file.flush()
                total_videos_processed += 1
                if total_videos_processed % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"[Pipeline] Processed {total_videos_processed} videos ({elapsed:.1f}s elapsed)")

        for w in workers:
            w.join(timeout=2)
    else:
        # Serial / single-process execution mode
        for vid, data, err in iter_videos(parquet_files, skip=completed_videos, limit=limit):
            if err is not None:
                lines = finalize_video(vid, {}, -1, error=err)
                for line in lines:
                    out_file.write(line + "\n")
                out_file.flush()
                total_videos_processed += 1
                continue

            video_secs = collections.defaultdict(list)
            v_max_sec = -1
            v_err = None
            for chunk in process_video_frames(vid, data, k=k, chunk_size=batch_size, imgsz=imgsz):
                _, batch, seconds_meta, is_last, chunk_max_sec, chunk_err = chunk
                if chunk_max_sec is not None and chunk_max_sec > v_max_sec:
                    v_max_sec = chunk_max_sec
                if chunk_err is not None:
                    v_err = chunk_err
                if batch is not None and len(batch) > 0:
                    counts_list = detector.predict_batch(batch)
                    for sec, counts in zip(seconds_meta, counts_list):
                        video_secs[sec].append(counts)

            lines = finalize_video(vid, video_secs, v_max_sec, error=v_err)
            for line in lines:
                out_file.write(line + "\n")
            out_file.flush()
            total_videos_processed += 1
            if total_videos_processed % 10 == 0:
                elapsed = time.time() - start_time
                print(f"[Pipeline] Processed {total_videos_processed} videos ({elapsed:.1f}s elapsed)")

    out_file.close()
    elapsed = time.time() - start_time
    print(f"[Pipeline] Done! Processed {total_videos_processed} videos in {elapsed:.2f} seconds.")
