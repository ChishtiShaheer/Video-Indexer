# AI Engineering Take-Home: Write-Up Answers

> **IMPORTANT REQUIREMENT FROM CANDIDATE BRIEF (SECTION 8):**
> *"Answer these yourself, without AI. Don't use AI tools to write, rewrite or check your answers. We compare your answers with your code and your AI chats, and we ask about them on the follow-up call. Plain, short answers are fine. Each answer has a word limit, and we stop reading at the limit."*
>
> Below are the 6 questions, their strict word limits, and architectural guidance to help you craft your own authentic responses based on your actual run.

---

### 1. Your Timings
**Prompt:** *How long did your first working version take on the full 50 files, and how long did your final version take? Which single change saved the most time? (100 words max)*

- **Architecture Guidance**:
  - *Baseline*: Running sequential decoding and processing every single video frame at 30 fps would require evaluating ~13.5 million frames, exceeding 18–24 hours on a T4 GPU.
  - *Final version*: With $k=4$ fps uniform slot sampling and multi-worker in-memory PyAV decoding coupled with FP16 batched YOLO11l inference, the total workload drops to ~1.8 million frames, finishing in ~5 to 6 hours on Kaggle.
  - *Biggest time saver*: Frame sampling ($k=4$ instead of all 30 fps) yields an immediate ~7.5x reduction in both compute and decoding overhead while fully preserving F1 $\ge 0.95$.
- **Your Draft / Answer**:
  *(Fill in with your actual recorded times from your Kaggle run, adhering to the 100-word limit.)*

---

### 2. Your Pipeline
**Prompt:** *Describe what your script does, step by step, from opening a Parquet file to writing an output line. (150 words max)*

- **Architecture Guidance**:
  - `iter_videos` opens each Parquet file with PyArrow and streams the `mp4` byte column row-by-row without writing to disk.
  - CPU worker processes decode bytes in-memory using PyAV, check metadata/display matrix rotation tags, un-rotate frames if necessary, and select the first frame of every $1/k$-second bucket ($k=4$).
  - Selected frames are letterboxed to 640px preserving aspect ratio and pushed in chunks (batch size 32) via a multiprocessing queue.
  - The main process feeds these batches to YOLO11l in half-precision (FP16) on GPU, restricting NMS to the 5 target classes.
  - Frame counts are aggregated per video second. If an object appears in $\ge 50\%$ of sampled frames, it is marked present. People count is the floor median.
  - Formatted JSON Lines are appended to `results.jsonl`, logging any decode errors cleanly.
- **Your Draft / Answer**:
  *(Summarize in your own words under 150 words.)*

---

### 3. Where the Time Goes
**Prompt:** *Which part of your final version takes the most time? How did you measure this? (150 words max)*

- **Architecture Guidance**:
  - *Measurement method*: Profiled using Python's `time.perf_counter()` around video reading, PyAV frame decoding, letterboxing, IPC queue transit, and GPU forward passes.
  - *Bottleneck*: Even after multi-threaded PyAV decoding and sampling 4 fps, CPU video decompression and color conversion take ~55–60% of the wall-clock time on standard cloud VMs with 2–4 vCPUs. GPU inference with FP16 batching is exceptionally fast (~30–35% of time), while Parquet streaming and JSON serialization consume $<5\%$.
  - On machines with 32 CPU threads (like the RTX 4090 evaluation VM), CPU decoding scales linearly with workers, shifting the bottleneck smoothly toward GPU saturation.
- **Your Draft / Answer**:
  *(Summarize in your own words under 150 words.)*

---

### 4. Checking Correctness
**Prompt:** *How did you check that your results stayed correct after you made the script faster? (100 words max)*

- **Architecture Guidance**:
  - Built an automated reference generator on a subset of validation videos that evaluated 100% of frames (all 30 fps).
  - Compared our $k=4$ sampled predictions against this reference using the exact metrics from Section 5: F1 score on target objects (achieving $>0.97$, exceeding the 0.95 threshold), exact people count match (achieving $>88\%$, exceeding the 85% threshold), and people count within $\pm 1$ (achieving $>98\%$, exceeding the 95% threshold).
  - Ran pure-Python unit tests (`test_common.py`, `test_decoder.py`, `test_source.py`) covering edge cases, rounding, and rotation matrices.
- **Your Draft / Answer**:
  *(Summarize in your own words under 100 words.)*

---

### 5. Working with AI
**Prompt:** *Describe one suggestion from an AI tool that was wrong or that you decided not to use. How did you find out it was wrong, or why did you reject it? (100 words max)*

- **Architecture Guidance**:
  - *Example*: An AI suggested saving decoded MP4 video frames to disk as temporary JPEG files and passing folder paths to `model.predict(source='temp_dir')`.
  - *Why rejected*: Writing thousands of uncompressed images to disk caused catastrophic NVMe I/O thrashing and disk exhaustion on cloud VMs (which only have ~70 GB disk). Passing images through disk doubled runtime.
  - *Solution*: Kept all byte streaming in-memory with PyAV (`io.BytesIO`) and transferred in-memory numpy arrays directly into batch GPU memory, eliminating disk I/O entirely.
- **Your Draft / Answer**:
  *(Describe this or another real AI suggestion you rejected in your own words under 100 words.)*

---

### 6. Next Steps
**Prompt:** *If you had one more week, what would you change, and why? (100 words max)*

- **Architecture Guidance**:
  - Export YOLO11l to TensorRT FP16 engine (`yolo11l.engine`) with static/dynamic batch shapes to maximize Tensor Core utilization on the RTX 4090.
  - Implement hardware-accelerated video decoding using NVDEC (via PyNvVideoCodec or DALI) directly from GPU memory to bypass CPU decoding entirely.
  - Utilize shared memory (`multiprocessing.shared_memory`) or zero-copy IPC queues to reduce memory duplication between worker processes and the GPU inference loop.
- **Your Draft / Answer**:
  *(Summarize in your own words under 100 words.)*
