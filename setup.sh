#!/bin/bash
# 一键初始化
# 用法: git clone 项目后，bash setup.sh

set -e

# 1. 拉取 MedicalGPT submodule
echo "=== Init MedicalGPT submodule ==="
git submodule update --init --recursive

# 2. 安装依赖
echo "=== Installing dependencies ==="
pip install -r requirements.txt
pip install -r MedicalGPT/requirements.txt

echo ""
echo "=== Setup done ==="
echo ""
echo "项目结构:"
echo "  scripts_project/   数据构造 + 评测脚本"
echo "  MedicalGPT/        训练代码"
echo "  docs/              学习文档"
echo ""
echo "下一步:"
echo "  1. bash scripts/download_data.sh"
echo "  2. python scripts_project/02_filter_medical_data.py --no_vector_recall --recall_top_k 2000"
echo "  3. python scripts_project/03_build_sft.py --general_sft_path MedicalGPT/data/sft/sharegpt_zh_1K_format.jsonl --general_n 1000 --medical_n 1000 --safety_n 0"
echo "  4. 训练: bash MedicalGPT/scripts/run_sft.sh"
echo "  5. 评测: python scripts_project/05_run_eval.py --model Qwen/Qwen2.5-3B-Instruct --peft <checkpoint> --name sft_raw"
