#!/bin/bash
# Download public medical datasets for training.
# Requires: huggingface-cli, git

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data/raw"
mkdir -p "$DATA_DIR"

echo "=== Downloading medical datasets ==="

# 1. shibing624/medical (200w Chinese medical QA pairs)
echo "[1/3] Downloading shibing624/medical..."
python -c "
from datasets import load_dataset
ds = load_dataset('shibing624/medical', 'finetune', split='train')
# Save first 100k as sample
ds.select(range(min(100000, len(ds)))).to_json('$DATA_DIR/shibing624_medical.jsonl', force_ascii=False)
print(f'Saved {min(100000, len(ds))} records')
"

# 2. FreedomIntelligence/Huatuo-26M (Chinese medical QA)
echo "[2/3] Downloading Huatuo-26M sample..."
python -c "
from datasets import load_dataset
ds = load_dataset('FreedomIntelligence/Huatuo-26M', split='train')
ds.select(range(min(50000, len(ds)))).to_json('$DATA_DIR/huatuo26m_sample.jsonl', force_ascii=False)
print(f'Saved {min(50000, len(ds))} records')
"

# 3. CMB (Chinese Medical Benchmark) for evaluation
echo "[3/3] Downloading CMB..."
python -c "
from datasets import load_dataset
import json
ds = load_dataset('FreedomIntelligence/CMB', 'CMB-Exam', split='test')
with open('$DATA_DIR/cmb_test.jsonl', 'w', encoding='utf-8') as f:
    for item in ds:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
print(f'Saved {len(ds)} records')
"

echo "=== Done. Data saved to $DATA_DIR ==="
ls -lh "$DATA_DIR"
