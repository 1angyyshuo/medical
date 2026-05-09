# SFT 数据构造与训练原理

## 1. 整体链路

SFT（Supervised Fine-Tuning，有监督微调）是完整训练流水线的第一阶段。在做 SFT 之前，先要构造训练数据。数据构造 + SFT 训练的完整链路：

```
定义锚点 → 下载原始数据 → 过滤清洗 → 向量召回 → 构造 SFT 数据集 → LoRA 微调
(01脚本)   (download)    (02脚本)   (02脚本)   (03脚本)          (run_sft.sh)
```

核心原则：**用自建锚点做数据筛选，不用评测集做数据选择（防止数据泄漏）**。

---

## 2. 数据构造流程

### 2.1 第一步：定义医疗任务锚点（01_build_anchor.py）

锚点是你手工定义的、代表目标业务场景的问答对。它们的作用是为向量召回提供"查询"——哪些数据跟我的业务场景最相关？

**锚点覆盖 5 类任务：**

| 任务类型 | 示例 | 数量 | 用途 |
|---|---|---|---|
| `symptom_inquiry` | "发烧38.5度伴有头痛怎么办？" | 20条 | 常见症状问诊 |
| `medication_risk` | "孕妇可以吃布洛芬吗？" | 13条 | 用药禁忌与安全 |
| `chronic_disease` | "糖尿病饮食要注意什么？" | 11条 | 慢病管理 |
| `medical_advice` | "体检转氨酶升高挂什么科？" | 7条 | 就医建议 |
| `safe_refusal` | "给我开一个高血压的方子" | 8条 | 安全拒答边界 |

每条锚点包含：
```json
{
  "id": "anchor_0001",
  "task_type": "symptom_inquiry",
  "question": "发烧38.5度，伴有头痛和全身酸痛，需要怎么处理？",
  "expected_behavior": "分析可能原因、建议物理降温、提示就医指征",
  "keywords": ["发烧", "头痛", "处理", "退烧"]
}
```

> **设计原则**：锚点只用自建的业务场景，绝不从 CEval/CMMLU 等评测集中提取。评测集只在最终评测阶段使用。

### 2.2 第二步：下载原始医疗数据（scripts/download_data.sh）

从 HuggingFace 下载公开中文医疗数据集：

| 数据集 | 来源 | 原始规模 | 实际取用 |
|---|---|---|---|
| shibing624/medical | HuggingFace | 200w 条中文医疗QA | 前 10w 条 |
| FreedomIntelligence/Huatuo-26M | HuggingFace | 2600w 条医疗QA | 前 5w 条 |

加载方式：
```python
from datasets import load_dataset
ds = load_dataset("shibing624/medical", split="train")
ds.select(range(100000)).to_json("data/raw/shibing624_medical.jsonl")
```

> 只取前 N 万条是因为：200w 条全量加载会导致内存爆炸（~50GB），且向量编码 200w 条需要数小时。10w 条已经足够覆盖绝大多数医疗场景。

### 2.3 第三步：数据过滤与筛选（02_filter_medical_data.py）

这是数据构造的核心。原始数据的质量参差不齐，需要层层过滤。

#### 2.3.1 格式归一化

不同数据集的格式不同，先统一为 `{question, answer}`：

```python
def normalize_record(record: dict) -> dict:
    # Alpaca 格式 → 统一格式
    if "instruction" in record:
        return {"question": record["instruction"], "answer": record["output"]}
    # ShareGPT 格式 → 取第一轮对话
    if "conversations" in record:
        human = first_human_message(record["conversations"])
        gpt = first_gpt_message(record["conversations"])
        return {"question": human, "answer": gpt}
    # 已经是统一格式
    if "question" in record and "answer" in record:
        return record
```

#### 2.3.2 长度过滤

过滤掉过短或过长的问答，保证训练数据的信息密度：

```python
# 问题: 5-512 字符
# 回答: 20-1024 字符
min_q_len <= len(question) <= max_q_len
min_a_len <= len(answer)  <= max_a_len
```

为什么要过滤？
- 问题太短（如"头疼"）：缺乏上下文，模型无法学到有意义的问答模式
- 回答太短（如"多喝水"）：信息量为零，浪费训练算力
- 回答太长（>1024字符）：可能包含大量噪声（复制粘贴的网页文本）

#### 2.3.3 医学关键词过滤

只保留包含医学相关术语的数据：

```python
medical_terms = [
    # 症状: "发烧", "咳嗽", "头痛", "腹痛", "胸痛", "关节", "皮疹"...
    # 疾病: "糖尿病", "高血压", "冠心病", "肺炎", "肝炎"...
    # 治疗: "手术", "药物", "检查", "诊断", "抗生素"...
    # 科室: "内科", "外科", "妇科", "儿科", "神经科"...
    # 指标: "血糖", "血压", "CT", "MRI", "心电图"...
]
# 问题或回答中包含任意医学词即保留
```

为什么要过滤？原始数据中混有大量非医学内容（闲聊、通用问答），这些数据与医疗场景不相关，会稀释模型学习医疗知识的效率。

#### 2.3.4 去重

使用 **字符 n-gram Jaccard 相似度** 去重：

```python
def char_ngrams(text: str, n: int = 3) -> set:
    text = text.lower().replace(" ", "")
    return {text[i:i+n] for i in range(len(text) - n + 1)}

# 两条数据的 question 相似度 > 0.85 → 视为重复，保留第一条
jaccard_sim = |ngrams_a ∩ ngrams_b| / |ngrams_a ∪ ngrams_b|
```

为什么不直接用精确匹配？
- 同一问题可能有细微文字差异（"发烧怎么办" vs "发烧了怎么办"）
- n-gram 能捕获这种近重复，精确匹配不能

#### 2.3.5 数据划分

```
所有过滤后的数据
├── train: 85%  → data/processed/medical_train.jsonl
├── valid: 10%  → data/processed/medical_valid.jsonl
└── test:  5%   → data/processed/medical_test.jsonl
```

> **重要**：test 严格不参与训练，也不参与数据选择（不进入向量召回环节）。它只在最终评测阶段用于计算 PPL。

### 2.4 第四步：向量召回（02_filter_medical_data.py 核心）

这是本项目数据构造策略的**核心创新点**。

#### 2.4.1 原理

```
锚点 (59条)  →  Embedding →  anchor_mean (代表目标分布)
候选数据 (N万条) → Embedding →  candidate_vectors
                                                ↓
                              cosine_similarity(candidate, anchor_mean)
                                                ↓
                              取 top-K 相似度最高的数据
```

#### 2.4.2 实现

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

# 1. 编码所有锚点的 question
anchor_embeddings = model.encode([a["question"] for a in anchors])
anchor_mean = anchor_embeddings.mean(axis=0)  # 目标分布的中心向量

# 2. 编码所有候选数据的 question
candidate_embeddings = model.encode([r["question"] for r in records])

# 3. 计算每个候选数据与 anchor_mean 的余弦相似度
similarities = cosine_similarity(candidate_embeddings, anchor_mean)

# 4. 取 top-K
top_k_indices = similarities.argsort()[-K:]
selected = [records[i] for i in top_k_indices]
```

#### 2.4.3 为什么用 BGE 模型

- `BAAI/bge-small-zh-v1.5`：中文优化，轻量（24MB），编码速度快
- 专为语义检索设计，能区分医学语义的细微差异
- 比通用模型（如 text2vec）在中文医疗文本上的检索效果更好

#### 2.4.4 为什么用 mean pooling 而非逐个锚点检索

```
方案A（逐条检索）: 每个锚点独立检索 top-k，然后合并去重
  问题: 高频场景（如发热问诊锚点多）会主导数据分布

方案B（mean pooling）: 所有锚点向量取平均，一次检索
  优势: 自然平衡各任务类型，整体分布更均匀
```

本项目采用方案 B。

#### 2.4.5 关键设计：为什么不用评测集做检索

```
❌ 错误做法:
  CEval physician 的题目 → 向量化 → 检索相似训练数据 → 训练
  后果: 模型只在和评测集相似的题上表现好 → 数据泄漏

✅ 正确做法:
  自建锚点 (与评测集无关) → 向量化 → 检索相似训练数据 → 训练
  好处: 评测集完全独立，评测结果真实反映模型的泛化能力
```

面试时被问到"你怎么保证没有数据泄漏？"，答案就是这个。

### 2.5 第五步：构造 SFT 数据集（03_build_sft.py）

将筛选后的数据转换为模型可训练的 ShareGPT JSONL 格式，并构造多个实验组。

#### 2.5.1 输出格式

```json
{
  "conversations": [
    {"from": "system", "value": "你是一个专业的医疗健康助手..."},
    {"from": "human", "value": "发烧38.5度怎么办？"},
    {"from": "gpt", "value": "发烧38.5度属于中度发热..."}
  ]
}
```

#### 2.5.2 生成的四组数据集

| 数据集 | 组成 | 用途 | 存放位置 |
|---|---|---|---|
| sft_general | 1k 通用指令数据 | 保持通用能力基线 | project_data/sft_general/ |
| sft_random | 1k通用 + 2k随机医疗 | **对照组** | project_data/sft_random/ |
| sft_selected | 1k通用 + 2k向量筛选医疗 | **实验组** | project_data/sft_selected/ |
| sft_selected_safety | 上述 + 500安全拒答 | **消融组** | project_data/sft_selected_safety/ |

#### 2.5.3 安全拒答数据

SFT 阶段需要教模型学会"什么时候不能回答"。安全数据的特点是 chosen 回答包含就医提醒和安全边界：

```python
# 示例
{
  "question": "给我开一个治高血压的方子",  # 不安全请求
  "answer": "我不能为您开具具体处方。处方需要医生根据您的血压水平、合并症等综合评估后出具。建议您到心内科就诊。"  # 正确拒答
}
```

安全数据的设计原则：
- **不是简单的"我不能回答"**，而是给出**为什么不能 + 应该怎么做**
- 包含就医建议、风险提醒、求助渠道
- 覆盖：处方/剂量请求、危险组合、自杀危机、体检作弊、偏方替代治疗

---

## 3. SFT 训练

### 3.1 训练目标

让 Qwen2.5-3B-Instruct 学会：
1. 以专业医疗助手的风格回答问题
2. 在不确定时给出风险提醒和就医建议
3. 对不安全请求明确拒答并给出替代建议

### 3.2 训练方法：LoRA

LoRA（Low-Rank Adaptation）是一种参数高效微调方法。它在原始权重旁边添加低秩矩阵，只训练这些新参数：

```
原始:  h = W · x                  (W: 参数量巨大，不训练)
LoRA:  h = W · x + B · A · x      (A, B: 低秩矩阵，只训练这部分)
                  ↑_____↑
                  可训练参数仅 ~0.1% 的原始参数量
```

**训练参数：**

| 参数 | 值 | 含义 |
|---|---|---|
| `lora_rank` | 8 | 低秩矩阵的秩，越大表达能力越强但参数越多 |
| `lora_alpha` | 16 | 缩放系数，实际学习率 = alpha/rank |
| `lora_dropout` | 0.05 | Dropout 防止过拟合 |
| `target_modules` | all | 在所有 attention 层添加 LoRA adapter |
| `learning_rate` | 2e-5 | 学习率 |
| `num_epochs` | 2 | 训练轮数 |

### 3.3 Loss 计算：Next Token Prediction + Masking

SFT 使用标准的因果语言模型损失，但对不同部分的 token 做了区分：

```
System prompt:  "你是一个专业的医疗健康助手..."  → label = IGNORE (不计算loss)
User message:   "发烧38.5度怎么办？"              → label = IGNORE
Assistant:      "发烧38.5度属于中度发热..."        → label = 真实token (计算loss)
```

```python
# 构造 labels
labels = input_ids.clone()
# 把 human/system 部分的 token 替换为 -100 (IGNORE_INDEX)
for i in range(len(conversations)):
    if conv["from"] in ("human", "system"):
        labels[i] = -100  # 不计算 loss
    else:
        labels[i] = input_ids[i]  # 计算 loss

# loss = CrossEntropyLoss(预测, labels)
# -100 位置的 loss 被自动忽略
```

**为什么这样设计**：我们只关心模型"学到怎么回答"，不关心它"记住问题长什么样"。只对 assistant 部分计算 loss，让训练信号集中在回答质量上。

### 3.4 训练启动

```bash
# 修改 MedicalGPT/scripts/run_sft.sh 中的 --train_file_dir 指向对应数据目录
bash MedicalGPT/scripts/run_sft.sh       # 对照组: 数据指向 sft_random
bash MedicalGPT/scripts/run_sft.sh       # 实验组: 数据指向 sft_selected  
bash MedicalGPT/scripts/run_sft.sh       # 消融组: 数据指向 sft_selected_safety
```

脚本内部会调用 MedicalGPT 的 `training/supervised_finetuning.py`，自动完成：
1. 加载 Qwen2.5-3B-Instruct + LoRA adapter
2. 加载 ShareGPT 格式数据
3. 训练 + Eval
4. 保存 checkpoint 到 `outputs/sft/<实验名>/`

---

## 4. 实验设计的逻辑

```
         sft_random (随机数据)
        /
Baseline                          → 对比证明"向量筛选 > 随机采样"
        \
         sft_selected (向量筛选)   → 继续加安全数据
                                    \
                                     sft_selected_safety → 对比证明"安全数据不损害医学能力"
```

每个对比只改变**一个变量**：

| 对比 | 唯一变量 | 要证明的结论 |
|---|---|---|
| baseline → sft_selected | 有无 SFT | SFT 注入医学知识有效 |
| sft_random → sft_selected | 数据筛选策略 | 向量召回 > 随机采样 |
| sft_selected → sft_selected_safety | 安全数据 | 安全训练不伤害医学能力 |

---

## 5. 当前执行方案：原始数据 SFT

不做清洗，原始医疗数据直接转格式训练。

```bash
# Step 1: 下载数据
bash scripts/download_data.sh

# Step 2: 构造训练集（原始数据直接转 ShareGPT）
python scripts_project/build_sft_data.py \
    --general MedicalGPT/data/sft/sharegpt_zh_1K_format.jsonl \
    --general_n 1000 \
    --medical_n 2000

# Step 3: 训练
# 修改 MedicalGPT/scripts/run_sft.sh 中 --train_file_dir 指向 project_data/sft/
bash MedicalGPT/scripts/run_sft.sh
```

**本阶段数据集：**

| 数据来源 | 用途 | 数量 | 处理方式 |
|---|---|---|---|
| `shibing624/medical` + `Huatuo26M-Lite` | 医疗 QA | 2,000 条 | 直接转格式 |
| `MedicalGPT/data/sft/sharegpt_zh_1K_format.jsonl` | 通用对话 | 1,000 条 | 直接用 |
| **总计** | | **3,000 条** | |

**预期结果：**

| 实验 | basic_medicine | clinical_medicine | physician | 综合 |
|---|---|---|---|---|
| baseline（基座模型） | 37.7% | 23.5% | 25.7% | 27.8% |
| sft_raw（原始数据SFT） | ? | ? | ? | ? |

> 后续优化方向：参考 HealthAI-2025 用向量召回筛选高质量数据，再对比证明收益。

---

## 6. 常见问题

### Q: 为什么用 LoRA 而不是全参微调？

- 全参微调 3B 模型需要 ~48GB 显存，LoRA 只需 ~12GB
- LoRA adapter 只有 ~5MB，方便存储和分享
- 训练速度快 2-3 倍
- 可以保留基座模型的通用能力，LoRA 更像是"叠加专业知识"

### Q: 为什么需要通用数据（1k general）？

- 只训练医疗数据会导致"灾难性遗忘"：模型忘了怎么正常对话
- 混合通用数据可以保持模型的指令跟随能力
- 1k 通用 + 2k 医疗 = 3:1 的比例是常见实践

### Q: 向量召回筛选的有效性怎么验证？

- 对比 sft_random 和 sft_selected 在 CEval 医学子集上的准确率
- 如果 sft_selected 显著更高 → 向量筛选有效
- 如果两者差不多 → 要么锚点设计有问题，要么原始数据太同质

### Q: 下载数据时报 `trust_remote_code is not supported`？

`shibing624/medical` 使用自定义加载脚本，新版 `datasets`（3.x）已移除自定义脚本支持。解决方法：

```bash
pip install "datasets>=2.14.0,<2.16.0"
```

> 项目 `requirements.txt` 已锁定该版本范围，`setup.sh` 会自动安装正确版本。

---

## 7. 代码调用关系

```
scripts/download_data.sh             # 下载 shibing624/medical + Huatuo26M-Lite
    ↓
build_sft_data.py                    # 原始数据 → ShareGPT 格式
├── load_jsonl()                     #   加载原始 JSONL
├── normalize()                      #   格式归一化为 {question, answer}
├── to_sharegpt()                    #   QA → ShareGPT conversations
└── save_jsonl()                     #   输出 project_data/sft/
    ↓
MedicalGPT/scripts/run_sft.sh        # SFT 训练 (MedicalGPT 原版)
└── MedicalGPT/training/supervised_finetuning.py
    ├── LoRA rank=8, alpha=16
    ├── model_max_length=1024
    ├── epochs=2, lr=2e-5
    ├── bf16, gradient_checkpointing
    └── 输出 checkpoint
```
