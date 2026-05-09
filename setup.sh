#!/bin/bash
# 一键初始化 — clone 项目后只需跑这一条命令

set -e

# 克隆 MedicalGPT（如果还没有）
if [ ! -d "MedicalGPT" ]; then
    echo "Cloning MedicalGPT..."
    git clone https://github.com/shibing624/MedicalGPT.git
fi

# 安装依赖
pip install -r requirements.txt
pip install -r MedicalGPT/requirements.txt

echo "=== Setup done ==="
echo "Next: python scripts_project/01_build_anchor.py"
