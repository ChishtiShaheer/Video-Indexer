#!/usr/bin/env bash
# Automation script for running the pipeline on Kaggle or a cloud GPU instance.
set -euo pipefail

echo "=== Step 1: System Info ==="
nvidia-smi || true
lscpu | grep -E "Model name|CPU\(s\):" || true
free -h
df -h .

echo "=== Step 2: Install Python Requirements ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Step 3: Run Preparation Command ==="
python run.py --prepare --batch-size 64
# Optional: if TensorRT is available, build an FP16 engine with:
# python run.py --prepare --export-trt --batch-size 64

echo "=== Step 4: Run Pure-Python Unit Tests ==="
python -m unittest discover -s tests -v

echo "=== Step 5: Download 50 FineVideo Parquet Files ==="
if [ ! -d "finevideo/data" ]; then
    echo "Downloading 50 Parquet files from HuggingFace..."
    hf download HuggingFaceFV/finevideo --repo-type dataset --include "data/train-000[0-4]*" --local-dir finevideo
fi

echo "=== Step 6: Quick Baseline Check (First 20 Videos) ==="
python run.py --input finevideo/data --output results_baseline.jsonl --limit 20 --fps 5 --batch-size 64

echo "=== Step 7: Execute Full Indexing Pipeline ==="
NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 1)
START_TIME=$(date +%s)

if [ "${NUM_GPUS}" -ge 2 ]; then
    echo "Detected ${NUM_GPUS} GPUs. Running 2 parallel shards across cuda:0 and cuda:1..."
    python run.py --input finevideo/data --output results.jsonl --shard-index 0 --num-shards 2 --device cuda:0 --workers 6 --fps 5 --batch-size 64 &
    python run.py --input finevideo/data --output results.jsonl --shard-index 1 --num-shards 2 --device cuda:1 --workers 6 --fps 5 --batch-size 64 &
    wait
    echo "Merging shard outputs into results.jsonl..."
    cat results.shard0of2.jsonl results.shard1of2.jsonl > results.jsonl
else
    echo "Running single GPU pipeline..."
    python run.py --input finevideo/data --output results.jsonl --fps 5 --batch-size 64
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "Indexing finished in ${ELAPSED} seconds."

echo "=== Step 8: Compress Results to results.jsonl.gz ==="
gzip -k -9 -f results.jsonl

echo "=== Step 9: Verify Results ==="
echo "Total lines in output:"
wc -l results.jsonl
echo "Sample first 3 lines:"
head -n 3 results.jsonl
echo "Sample last 3 lines:"
tail -n 3 results.jsonl
echo "Results compressed size:"
ls -lh results.jsonl.gz

echo "=== All Steps Complete! Download results.jsonl.gz and submit ==="
