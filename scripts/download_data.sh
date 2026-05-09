#!/bin/bash
# Download public medical datasets for training.
# Usage: bash scripts/download_data.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data/raw"
mkdir -p "$DATA_DIR"

export HF_DATASETS_TRUST_REMOTE_CODE=1

echo "=== Downloading medical datasets ==="

# 1. shibing624/medical (200w Chinese medical QA pairs, 'finetune' config)
echo "[1/2] Downloading shibing624/medical..."
python -c "
from datasets import load_dataset
ds = load_dataset('shibing624/medical', 'finetune', trust_remote_code=True, split='train')
# 取前10w条作为样本
ds.select(range(min(100000, len(ds)))).to_json('$DATA_DIR/shibing624_medical.jsonl', force_ascii=False)
print(f'Saved {min(100000, len(ds))} records')
"

# 2. FreedomIntelligence/Huatuo26M-Lite (17.8w Chinese medical QA)
# 注意: 用 Lite 版而非完整 Huatuo-26M (2600w 条太大)
echo "[2/2] Downloading Huatuo26M-Lite..."
python -c "
from datasets import load_dataset
ds = load_dataset('FreedomIntelligence/Huatuo26M-Lite', split='train')
ds.select(range(min(50000, len(ds)))).to_json('$DATA_DIR/huatuo26m_sample.jsonl', force_ascii=False)
print(f'Saved {min(50000, len(ds))} records')
"

echo "=== Done. Data saved to $DATA_DIR ==="
ls -lh "$DATA_DIR"
