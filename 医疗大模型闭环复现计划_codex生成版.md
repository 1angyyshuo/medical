# 医疗大模型闭环复现计划

## 1. 项目定位

项目名称：面向中文医疗问诊的可复现训练与评测闭环

项目目标不是简单跑通 MedicalGPT，而是复现一条完整链路：

数据构造 -> SFT / 偏好优化训练 -> 自动评测 + 人工评测 + 消融实验 -> 证明模型确实变强

建议任务边界：

- 场景：中文医疗问诊、医学常识问答、初步分诊建议、用药风险提醒。
- 不做：替代医生诊断、处方生成、急诊决策。
- 用户价值：提升小模型在医疗问答中的专业性、拒答边界、问诊澄清能力和医学资格类选择题能力。

推荐 Base Model：

- 低资源版本：`Qwen/Qwen3.5-0.8B` 或 `Qwen2.5-0.5B-Instruct`
- 简历展示版本：`Qwen2.5-1.5B` / `Qwen2.5-3B`
- 训练方式：LoRA / QLoRA，优先保证完整闭环可跑通。

## 2. 为什么是这三类仓库

### 数据构造参考

`HealthAI-2025` 的核心价值不是照搬数据，而是学习“医疗任务如何拆成可训练、可评测、可解释的数据单元”：

- 明确任务类型：问答、推理、分诊、病历摘要、安全拒答。
- 明确样本字段：输入、输出、证据、标签、难度、来源、质检结果。
- 明确数据流：原始医疗语料 -> 清洗 -> 分类 -> 合成/改写 -> 质检 -> 训练集/验证集/测试集。

说明：当前直接访问 `https://github.com/yuiai2001/HealthAI-2025` 返回 404，因此复现时应保留“数据构造思路”，不要把该仓库写成已完整依赖的代码来源。

### 训练框架

`shibing624/MedicalGPT` 已经提供完整训练入口：

- `training/supervised_finetuning.py`：SFT
- `training/reward_modeling.py`：Reward Model
- `training/ppo_training.py`：PPO
- `training/dpo_training.py`：DPO
- `training/grpo_training.py`：GRPO
- `scripts/run_sft.sh`、`scripts/run_rm.sh`、`scripts/run_ppo.sh`、`scripts/run_grpo.sh`：可改造启动脚本

本地数据格式：

- SFT：`{"conversations":[{"from":"human","value":"..."},{"from":"gpt","value":"..."}]}`
- Reward/DPO：`{"conversations":[...],"chosen":"...","rejected":"..."}`
- GRPO：`{"question":"...","answer":"..."}`

### 评估框架

`EleutherAI/lm-evaluation-harness` 用来做统一自动评测：

- 公共基准：CEval / CMMLU / MMLU / MedQA 类任务，按可用任务选择。
- 自建任务：把医疗问答、安全拒答、病历推理测试集做成 lm-eval YAML 任务。
- 指标：accuracy、exact match、F1、loglikelihood、perplexity、拒答准确率、幻觉率。

## 3. 复现路线

### Phase 0：环境与基线

目标：先得到可对比的 baseline。

产出：

- 环境文件：Python、CUDA、torch、transformers、trl、peft、lm-eval 版本。
- Baseline 模型评测结果：Base / Instruct 原模型在医疗基准和自建测试集上的分数。

命令示例：

```bash
cd MedicalGPT
pip install -r requirements.txt --upgrade
pip install "lm_eval[hf]"
```

基线评测示例：

```bash
lm_eval --model hf \
  --model_args pretrained=Qwen/Qwen2.5-1.5B-Instruct,dtype=bfloat16 \
  --tasks ceval-valid \
  --device cuda:0 \
  --batch_size auto \
  --output_path ../runs/baseline_ceval.json
```

如果 CEval 任务名和本机 lm-eval 版本不一致，先执行：

```bash
lm_eval ls tasks | grep -i ceval
```

Windows PowerShell 可用：

```powershell
lm_eval ls tasks | Select-String -Pattern "ceval"
```

### Phase 1：数据构造闭环

目标：构造自己的医疗数据，而不是直接照搬别人的数据集。

数据分层：

| 数据集 | 用途 | 建议规模 | 来源 |
|---|---:|---:|---|
| `general_sft` | 保持通用指令能力 | 1k-5k | MedicalGPT 自带 ShareGPT 样例 / 公开通用指令集 |
| `medical_sft_raw` | 医疗基础问答 | 5k-20k | shibing624/medical 子集、公开医学百科、问诊 QA |
| `medical_sft_selected` | 目标分布增强 | 2k-10k | 基于自建 dev anchor 的向量召回 |
| `medical_reasoning` | 医学推理/病情分析 | 1k-5k | LLM 合成 + 规则质检 |
| `medical_preference` | 偏好优化 | 2k-10k | chosen/rejected 合成 + 小规模人工抽检 |
| `medical_eval_private` | 自建测试集 | 300-1000 | 严格不参与训练和召回 |

关键设计：不要用 CEval 验证集直接召回训练数据。

更稳的做法：

1. 从业务场景定义 100-300 条 `dev_anchor`，例如“发热问诊”“用药禁忌”“慢病管理”“就医建议”“安全拒答”。
2. 用向量模型对 `dev_anchor` 和候选医疗数据编码。
3. 召回相似度高、覆盖多类任务的数据。
4. 对召回数据做去重、长度过滤、敏感风险过滤、答案质量规则过滤。
5. CEval / CMMLU / 自建 test 只用于评测，不进入训练选择环节。

推荐质检规则：

- 去重：question MinHash / embedding 相似度去重。
- 长度：问题 5-512 token，答案 20-1024 token。
- 医疗安全：涉及处方剂量、急症、孕妇儿童、精神危机时必须建议线下就医或医生确认。
- 答案格式：优先包含“可能原因、建议检查、风险提醒、就医建议”。
- 噪声过滤：删除明显胡说、乱码、过度确定诊断、违法违规建议。

### Phase 2：SFT 训练

目标：让模型学会中文医疗问答风格、边界意识和基础知识。

对照实验：

| 实验 | 训练数据 | 目的 |
|---|---|---|
| A0 | Base model | 原始基线 |
| A1 | general_sft 1k | 看通用指令微调是否伤害医疗能力 |
| A2 | general_sft 1k + random medical 2k | 随机医疗数据增益 |
| A3 | general_sft 1k + selected medical 2k | 证明向量筛选有效 |
| A4 | A3 + safety/refusal 500 | 证明安全数据有效 |

启动脚本从 `MedicalGPT/scripts/run_sft.sh` 改：

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 training/supervised_finetuning.py \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --train_file_dir ../project_data/sft_selected \
  --validation_file_dir ../project_data/sft_valid \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --do_train --do_eval \
  --use_peft True \
  --model_max_length 1024 \
  --num_train_epochs 1 \
  --learning_rate 2e-5 \
  --output_dir ../runs/sft_selected_lora \
  --target_modules all \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --bf16 \
  --report_to tensorboard \
  --gradient_checkpointing True
```

记录：

- train loss / eval loss
- 训练耗时
- 显存占用
- 最终 checkpoint
- 失败与调参记录

### Phase 3：偏好优化

优先建议做 DPO，再做 GRPO。PPO 工程复杂度更高，面试可以讲思路，但复现成本更大。

#### DPO 路线

偏好数据构造：

- `chosen`：结构完整、包含风险提醒、不过度诊断。
- `rejected`：过度确定、缺少就医建议、忽略禁忌、答非所问、啰嗦或幻觉。

样例：

```json
{
  "conversations": [{"from": "human", "value": "孕妇发烧 38.8 度可以吃布洛芬吗？"}],
  "chosen": "孕期发热需要谨慎处理，不建议自行服用布洛芬，尤其孕晚期更应避免。建议先物理降温、补液，并尽快联系产科或发热门诊，由医生评估感染原因和可用退热药。",
  "rejected": "可以吃布洛芬，一次两片，很快就会退烧。"
}
```

价值：

- 相比 PPO，DPO 更容易稳定复现。
- 能明确优化“更安全、更专业、更有边界”的回答偏好。

#### GRPO 路线

GRPO 数据适合做可验证问答或医学选择题推理：

- 输入：医学问题 / 选择题 / 病例推理题。
- 奖励函数：答案正确性 + 格式规范 + 安全边界。

可改造点：

- 正确性奖励：答案包含标准答案或选项。
- 推理格式奖励：包含“分析/结论/建议”。
- 安全奖励：涉及高风险问题时包含就医提醒。
- 惩罚项：直接给处方剂量、绝对化诊断、编造检查结果。

### Phase 4：评测闭环

评测至少包含三层。

#### 自动评测

| 评测集 | 指标 | 目的 |
|---|---|---|
| CEval 医学相关子集 | accuracy | 医学考试知识 |
| CMMLU medical 子集 | accuracy | 中文医学知识泛化 |
| 自建医疗 QA | EM / F1 / LLM judge | 问答质量 |
| 自建安全集 | refusal accuracy / unsafe rate | 医疗安全 |
| held-out 医疗 SFT | ppl / eval loss | 语言建模质量 |

lm-eval 示例：

```bash
lm_eval --model hf \
  --model_args pretrained=Qwen/Qwen2.5-1.5B-Instruct,peft=../runs/sft_selected_lora,dtype=bfloat16 \
  --tasks ceval-valid,cmmlu \
  --device cuda:0 \
  --batch_size auto \
  --output_path ../runs/sft_selected_eval.json
```

#### 人工评测

抽样 100 条，3 个维度 1-5 分：

- 专业性：医学事实是否准确。
- 有用性：是否给出可操作建议。
- 安全性：是否有边界提醒，是否避免危险建议。

人工评测表：

| 样本ID | Base | SFT | DPO/GRPO | 专业性 | 有用性 | 安全性 | 备注 |
|---|---|---|---|---:|---:|---:|---|

#### 消融实验

必须能回答：我的每一步有没有用？

| 实验 | 改动 | 预期证明 |
|---|---|---|
| random medical vs selected medical | 随机数据 vs 向量筛选数据 | 数据选择有效 |
| no safety vs safety mix | 去掉安全数据 | 安全拒答数据有效 |
| SFT vs SFT+DPO | 加偏好优化 | 偏好训练有效 |
| small data vs full data | 2k vs 10k | 数据规模收益 |
| one-stage vs staged | 直接 DPO vs SFT 后 DPO | 训练顺序有效 |

## 4. 结果记录模板

| 模型 | 数据 | 训练方法 | CEval-med | CMMLU-med | 自建QA | 安全拒答准确率 | PPL/Eval loss |
|---|---|---|---:|---:|---:|---:|---:|
| Base | - | - | x.x | x.x | x.x | x.x | x.x |
| SFT-random | 1k general + 2k random medical | LoRA SFT | x.x | x.x | x.x | x.x | x.x |
| SFT-selected | 1k general + 2k selected medical | LoRA SFT | x.x | x.x | x.x | x.x | x.x |
| SFT-safety | selected + safety | LoRA SFT | x.x | x.x | x.x | x.x | x.x |
| DPO | preference 5k | SFT + DPO | x.x | x.x | x.x | x.x | x.x |
| GRPO | reasoning 2k | SFT + GRPO | x.x | x.x | x.x | x.x | x.x |

## 5. 推荐最终项目结构

```text
Medicalchat/
  MedicalGPT/
  project_data/
    raw/
    anchors/
    sft_random/
    sft_selected/
    preference/
    grpo/
    eval_private/
  scripts_project/
    01_build_anchor.py
    02_filter_medical_data.py
    03_build_sft.py
    04_build_preference.py
    05_run_lm_eval.ps1
    06_collect_results.py
  runs/
    baseline/
    sft_random/
    sft_selected/
    dpo/
    grpo/
  reports/
    experiment_log.md
    result_table.csv
    case_study.md
```

## 6. 面试讲法

一句话版本：

我复现了一个中文医疗小模型训练闭环，不是只跑 MedicalGPT，而是围绕“数据构造、训练策略、评测证据”做了端到端实验：先用自建医疗任务锚点做数据筛选，再用 LoRA SFT 注入医疗问答能力，最后用 DPO/GRPO 优化安全性和推理格式，并通过 CEval/CMMLU 医学子集、自建医疗安全集和消融实验验证每个组件的贡献。

简历版本：

- 基于 MedicalGPT 搭建中文医疗大模型训练闭环，完成 Qwen/LLaMA 系列小模型的 LoRA SFT、Reward/DPO/GRPO 训练与 lm-evaluation-harness 自动评测。
- 设计医疗数据构造流程：构建问诊、用药风险、慢病管理、安全拒答等任务锚点，使用向量召回从候选医疗语料中筛选目标分布数据，并通过去重、长度、风险规则和人工抽检控制噪声。
- 构造偏好数据集，将“专业、谨慎、有边界”的回答作为 chosen，将过度诊断、缺少就医提醒、危险用药建议作为 rejected，用于 DPO/Reward Model 训练。
- 建立评测体系，覆盖 CEval/CMMLU 医学子集、自建医疗 QA、安全拒答测试集和人工评分，并通过随机数据 vs 向量筛选数据、SFT vs SFT+DPO 等消融实验验证改进来源。

答辩重点：

- 为什么不用评测集召回训练数据：会造成数据泄漏；我只用自建 dev anchors 做数据选择，CEval/CMMLU/私有测试集只在最终评测阶段使用。
- 为什么先 SFT 再 DPO/GRPO：SFT 先补领域指令和知识表达，偏好优化再调回答风格、安全边界和推理结构。
- 为什么不一开始做 PPO：PPO 对奖励模型、采样和稳定性要求更高；先用 DPO 做稳定闭环，再扩展 PPO/GRPO 更合理。
- 怎么证明有效：看公共医学基准、自建测试集、人工评分和消融实验，而不是只看训练 loss。

## 7. 最小可复现版本

如果时间紧，先完成最小闭环：

1. 选 `Qwen/Qwen3.5-0.8B` 或 `Qwen2.5-0.5B-Instruct`。
2. 构造 1k 通用 + 2k 医疗 SFT。
3. 训练 `SFT-random` 和 `SFT-selected` 两个 LoRA。
4. 用 lm-eval 跑 CEval/CMMLU 医学相关任务。
5. 自建 100 条医疗安全测试集，人工或 LLM judge 打分。
6. 写出结果表和 5 条 case study。

做到这一步，项目就已经从“跑脚本”变成“有证据的训练闭环”。

