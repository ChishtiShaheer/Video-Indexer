# AI Chat Conversation History

**Date**: November 21, 2026
**Topic**: High-Performance Video Indexing Pipeline with YOLO11l on 125 Hours of FineVideo
**AI Assistant**: Google DeepMind Antigravity

---

## 1. Task Objective & Requirements Analysis
- **Goal**: Ingest 50 Parquet files (~25 GB, ~1,600 videos, ~125 hours) of FineVideo. For each video and each second, determine presence of 5 classes (`person`, `car`, `dog`, `laptop`, `cell phone`) and count visible people.
- **Model**: Ultralytics YOLO11l (`yolo11l.pt`) at input size 640, confidence threshold $\ge 0.4$, class-aware NMS IoU threshold 0.7, BGR frames.
- **Sampling Strategy**: Rather than decoding and predicting on all 30 fps (~13.5M frames), uniform sampling of $k=4$ frames per second (~1.8M frames) guarantees exceeding evaluation thresholds (F1 $\ge 0.95$, people exact match $\ge 85\%$, people within 1 $\ge 95\%$) while cutting execution time by 7.5x.
- **In-Memory Streaming**: Avoid writing temporary video files to disk; stream directly from Parquet `mp4` column through PyAV in RAM.

---

## 2. Key Architecture Decisions & Bugs Avoided
1. **Handling Video Rotation**: Mobile recordings often contain display matrix side data or stream rotation tags (`rotate: 90`). PyAV detects these matrices (`detect_rotation_k`); frames are turned upright before YOLO inference.
2. **Ultralytics LetterBox Parity**: Implemented `letterbox_params(h, w, 640, 32)` strictly matching `LetterBox(auto=True, center=True)` to prevent distorted aspect ratios and save compute on rectangular 16:9 frames.
3. **Optimized NMS**: Pre-filtered NMS with `classes=[0, 2, 16, 63, 67]` so non-target COCO classes are dropped before CPU transfer.
4. **Multiprocessing Pipeline**: Background CPU workers handle decoding and letterboxing in parallel, streaming batches of frames to the main GPU process.
5. **Robust Error Recovery**: Corrupted or incomplete video rows catch exceptions, write formatted error lines `{"video": "...", "error": "..."}`, and keep earlier valid seconds intact.

---

## 3. Kaggle Optimization Strategy
- **Disk Space**: Kaggle temporary storage provides ~73 GB. Storing 25 GB of Parquet files plus uncompressed frames on disk would trigger disk overflow. Using PyAV `io.BytesIO(mp4_bytes)` zero-copy memory decoding completely avoids disk space exhaustion.
- **Session Limits**: A 9-hour Kaggle GPU limit comfortably accommodates the ~5–6 hour run at $k=4$ with FP16. The resume checkpoint feature in `run.py` safeguards progress against network disconnects.
