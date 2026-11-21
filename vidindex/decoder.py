"""Decoder side of the pipeline: runs inside worker processes.

A worker takes one video (mp4 bytes) at a time, decodes it, keeps only a few
frames per second, letterboxes them exactly like Ultralytics does, and sends
them to the main process in chunks of at most `chunk` frames.

Only stdlib is imported at module level.  numpy / cv2 / av are imported lazily
so that importing this module in the parent process stays cheap.
"""
import io
import math
import os
import struct
import tempfile
import time

from .common import (IMGSZ, STRIDE, bucket_of, display_matrix_rotation,
                     letterbox_params, rot90_k, short_error)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def letterbox(img, cv2, np, imgsz=IMGSZ):
    """Same resize + padding as Ultralytics LetterBox(auto=True, center=True)."""
    h, w = img.shape[:2]
    new_w, new_h, top, bottom, left, right = letterbox_params(h, w, imgsz, STRIDE)
    if (w, h) != (new_w, new_h):
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if top or bottom or left or right:
        img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img


# ---------------------------------------------------------------------------
# Rotation tag detection (several PyAV versions expose it differently)
# ---------------------------------------------------------------------------
def detect_rotation_k(stream, frame):
    """Number of 90-degree counter-clockwise turns needed to un-rotate frames."""
    candidates = []
    # 1) newer PyAV: frame.rotation (degrees, counter-clockwise)
    try:
        r = getattr(frame, "rotation", None)
        if r is not None:
            candidates.append(float(r))
    except Exception:
        pass
    # 2) DISPLAYMATRIX frame side data
    try:
        for sd in (getattr(frame, "side_data", None) or []):
            t = getattr(sd, "type", None)
            name = str(getattr(t, "name", t)).upper()
            if "DISPLAYMATRIX" in name:
                raw = bytes(sd)
                if len(raw) >= 36:
                    candidates.append(display_matrix_rotation(struct.unpack("=9i", raw[:36])))
    except Exception:
        pass
    # 3) old-style "rotate" stream tag (degrees clockwise)
    try:
        m = stream.metadata.get("rotate") if hasattr(stream, "metadata") else None
        if m:
            candidates.append(-float(m))
    except Exception:
        pass
    for angle in candidates:
        k = rot90_k(angle)
        if k:
            return k
    return 0


# ---------------------------------------------------------------------------
# Frame generators.  Both yield (second, img_or_None) for EVERY decoded frame:
# img is a full-size BGR uint8 array for selected frames and None otherwise.
# ---------------------------------------------------------------------------
def iter_frames_av(data, k, select_all=False, dec_threads=1, max_bad=50,
                   deadline=None, st=None):
    """Decode with PyAV.

    k          frames sampled per second (first frame of every 1/k-second slot)
    """
    import av
    import cv2
    import numpy as np

    bio = io.BytesIO(data)
    container = av.open(bio, mode="r")
    video_stream = None
    for s in container.streams:
        if s.type == "video":
            video_stream = s
            break
    if video_stream is None:
        container.close()
        raise ValueError("no video stream found in container")

    video_stream.thread_type = "AUTO"
    if dec_threads > 1:
        video_stream.threads = dec_threads

    time_base = float(video_stream.time_base) if video_stream.time_base else 1.0 / 30.0
    avg_fps = float(video_stream.average_rate) if video_stream.average_rate and video_stream.average_rate > 0 else 30.0

    t0 = None
    rot_k = 0
    last_bucket = None
    bad_count = 0
    frame_idx = 0

    try:
        for frame in container.decode(video_stream):
            if deadline is not None and time.time() > deadline:
                raise TimeoutError("decoding exceeded deadline")

            if frame.pts is not None:
                t = float(frame.pts * time_base)
            else:
                t = frame_idx / avg_fps
            frame_idx += 1

            if t0 is None:
                t0 = t
                rot_k = detect_rotation_k(video_stream, frame)

            t_rel = t - t0
            if t_rel < 0:
                t_rel = 0.0

            sec = int(t_rel)
            b = bucket_of(t_rel, k)

            if select_all or (b != last_bucket):
                last_bucket = b
                try:
                    img = frame.to_ndarray(format="bgr24")
                    if rot_k == 1:
                        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    elif rot_k == 2:
                        img = cv2.rotate(img, cv2.ROTATE_180)
                    elif rot_k == 3:
                        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                    bad_count = 0
                    yield sec, img
                except Exception:
                    bad_count += 1
                    if bad_count >= max_bad:
                        raise ValueError(f"too many consecutive frame decode failures ({bad_count})")
                    yield sec, None
            else:
                yield sec, None
    finally:
        container.close()


def iter_frames_cv2(data, k, select_all=False):
    """Fallback decoder using OpenCV VideoCapture."""
    import cv2
    import numpy as np

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            raise ValueError("cv2 failed to open video")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or math.isnan(fps):
            fps = 30.0

        frame_idx = 0
        last_bucket = None
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            t_rel = frame_idx / fps
            frame_idx += 1
            sec = int(t_rel)
            b = bucket_of(t_rel, k)
            if select_all or (b != last_bucket):
                last_bucket = b
                yield sec, frame
            else:
                yield sec, None
        cap.release()


def decode_video_frames(data, k=5, select_all=False, dec_threads=1):
    """Decode video bytes into frames using PyAV with fallback to cv2."""
    try:
        yield from iter_frames_av(data, k=k, select_all=select_all, dec_threads=dec_threads)
    except Exception as e_av:
        try:
            yield from iter_frames_cv2(data, k=k, select_all=select_all)
        except Exception:
            raise e_av


def process_video_frames(video_id, data, k=5, chunk_size=64, imgsz=IMGSZ):
    """Process a video's bytes: decode, letterbox, and yield chunks of frames.

    Yields:
        (video_id, batch_array_or_None, seconds_list, is_last, max_sec, error)
    """
    import cv2
    import numpy as np

    if not data:
        yield video_id, None, [], True, None, "empty video data"
        return

    frames_buf = []
    meta_buf = []
    max_sec = -1

    try:
        for sec, img in decode_video_frames(data, k=k):
            if sec > max_sec:
                max_sec = sec
            if img is not None:
                lb_img = letterbox(img, cv2, np, imgsz=imgsz)
                frames_buf.append(lb_img)
                meta_buf.append(sec)
                if len(frames_buf) >= chunk_size:
                    batch = np.stack(frames_buf, axis=0)
                    yield video_id, batch, meta_buf, False, max_sec, None
                    frames_buf = []
                    meta_buf = []

        if frames_buf:
            batch = np.stack(frames_buf, axis=0)
            yield video_id, batch, meta_buf, True, max_sec, None
        else:
            yield video_id, None, [], True, max_sec, None
    except Exception as e:
        yield video_id, None, [], True, max_sec, short_error(e)
