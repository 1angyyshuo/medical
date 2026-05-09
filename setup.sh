#!/bin/bash
# 一键初始化
# 用法: git clone 项目后，bash setup.sh

set -e

# 1. 安装依赖
echo "=== Installing dependencies ==="
pip install -r requirements.txt

# 2. 安装 MedicalGPT 依赖
if [ -d "MedicalGPT" ]; then
    pip install -r MedicalGPT/requirements.txt
fi

echo ""
echo "=== Setup done ==="
echo ""
echo "下一步:"
echo "  1. bash scripts/download_data.sh      # 下载数据"
echo "  2. python scripts_project/02_filter_medical_data.py --no_vector_recall --recall_top_k 2000"
echo "  3. python scripts_project/03_build_sft.py --general_sft_path MedicalGPT/data/sft/sharegpt_zh_1K_format.jsonl --general_n 1000 --medical_n 1000 --safety_n 0"
echo "  4. 用 MedicalGPT 原版脚本训练: bash MedicalGPT/scripts/run_sft.sh"
echo "  5. python scripts_project/05_run_eval.py --model Qwen/Qwen2.5-3B-Instruct --peft <checkpoint路径> --name sft_raw"
