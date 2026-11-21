"""Common constants and utility functions for video indexing."""
import json
import math

IMGSZ = 640
STRIDE = 32
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.7

# Order strictly matches Section 1 of brief and unit test bitmasks:
# Bit 0: person, Bit 1: car, Bit 2: dog, Bit 3: laptop, Bit 4: cell phone
TARGET_CLASSES = ["person", "car", "dog", "laptop", "cell phone"]
COCO_CLASS_IDS = [0, 2, 16, 63, 67]
COCO_ID_TO_TARGET_IDX = {0: 0, 2: 1, 16: 2, 63: 3, 67: 4}


def median_floor(vals):
    """Compute the median of an integer sequence, rounded down if halfway."""
    if not vals:
        return 0
    s = sorted(vals)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) // 2


def second_result(samples):
    """Aggregate frame detections within a 1-second interval.

    samples: list of 5-tuples (person, car, dog, laptop, phone)
    Returns:
        (mask, people)
        mask: integer bitmask where bit i corresponds to TARGET_CLASSES[i]
              set if present in at least half of the sampled frames (>= ceil(N / 2)).
        people: median people count across the sampled frames, rounded down.
    """
    if not samples:
        return (0, 0)
    n = len(samples)
    mask = 0
    for idx in range(5):
        present_count = sum(1 for frame in samples if frame[idx] > 0)
        if present_count * 2 >= n:
            mask |= (1 << idx)
    people = median_floor([frame[0] for frame in samples])
    return (mask, people)


def finalize_video(video_id, secs, max_second, error=None):
    """Generate JSON Lines entries for a video.

    video_id: str ("filename:row")
    secs: dict {second_int: [counts_tuple, ...]}
    max_second: int, the highest second observed
    error: optional error string if the video failed partway or couldn't decode.
    """
    lines = []
    if max_second is not None and max_second >= 0:
        end = max_second if error is not None else max_second + 1
        for sec in range(end):
            samples = secs.get(sec, [])
            mask, people = second_result(samples)
            objs = [TARGET_CLASSES[i] for i in range(5) if (mask & (1 << i))]
            entry = {
                "video": video_id,
                "second": sec,
                "objects": objs,
                "people": people,
            }
            lines.append(json.dumps(entry))
    if error is not None:
        lines.append(json.dumps({"video": video_id, "error": str(error)}))
    return lines


def bucket_of(t, k):
    """Returns the slot index for timestamp t given k samples per second."""
    return int(t * k)


def letterbox_params(h, w, imgsz=IMGSZ, stride=STRIDE):
    """Compute LetterBox parameters identical to Ultralytics (auto=True, center=True)."""
    r = min(imgsz / h, imgsz / w)
    new_w = int(round(w * r))
    new_h = int(round(h * r))
    dw = (imgsz - new_w) % stride
    dh = (imgsz - new_h) % stride
    dw /= 2.0
    dh /= 2.0
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    return new_w, new_h, top, bottom, left, right


def display_matrix_rotation(matrix):
    """Compute rotation in degrees counter-clockwise from a 9-element 32-bit display matrix."""
    scale_x = math.hypot(matrix[0] / 65536.0, matrix[3] / 65536.0)
    scale_y = math.hypot(matrix[1] / 65536.0, matrix[4] / 65536.0)
    if scale_x == 0.0 or scale_y == 0.0:
        return 0.0
    return -math.atan2(matrix[1] / scale_y, matrix[0] / scale_x) * 180.0 / math.pi


def rot90_k(angle):
    """Number of 90-degree counter-clockwise turns for a given angle in degrees."""
    return int(round(angle / 90.0)) % 4


def short_error(e, max_len=100):
    """Format an exception into a single-line short string."""
    msg = str(e).strip().replace("\n", " ")
    if len(msg) > max_len:
        msg = msg[: max_len - 3] + "..."
    return msg
