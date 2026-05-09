# 数据目录

## 目录说明

```
data/
├── raw/          # 原始下载的公开数据集
├── processed/    # 过滤和筛选后的数据
└── synthetic/    # LLM 合成的偏好数据
```

## 获取数据

### 1. 下载公开医疗数据

```bash
bash scripts/download_data.sh
```

### 2. 手动获取的数据来源

| 数据集 | 来源 | 用途 |
|---|---|---|
| shibing624/medical | HuggingFace | 200w 中文医疗 QA |
| FreedomIntelligence/Huatuo-26M | HuggingFace | 中文医学知识 QA |
| C-Eval (医学子集) | lm-evaluation-harness | 医学资格评测 |
| CMMLU (医学 track) | lm-evaluation-harness | 医学知识评测 |

### 3. 数据构造流程

```bash
# Step 1: 构建锚点
python scripts_project/01_build_anchor.py

# Step 2: 筛选医疗数据
python scripts_project/02_filter_medical_data.py

# Step 3: 构建 SFT 数据集
python scripts_project/03_build_sft.py

# Step 4: 构建偏好数据
python scripts_project/04_build_preference.py
```
