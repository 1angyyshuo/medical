# 实验日志

## 实验记录表

| 日期 | 实验名称 | 阶段 | 模型 | 数据 | 训练参数 | 输出目录 | 状态 | 备注 |
|---|---|---|---|---|---|---|---|---|
| YYYY-MM-DD | baseline | Phase 0 | Qwen2.5-3B-Instruct | - | - | outputs/eval/baseline | ⏳ | CEval医学子集基线 |
| YYYY-MM-DD | sft_random | Phase 2 | Qwen2.5-3B-Instruct | 1k general + 2k random medical | LoRA r=8, lr=2e-5, epoch=2 | outputs/sft/sft_random | ⏳ | 随机数据对照 |
| YYYY-MM-DD | sft_selected | Phase 2 | Qwen2.5-3B-Instruct | 1k general + 2k selected medical | LoRA r=8, lr=2e-5, epoch=2 | outputs/sft/sft_selected | ⏳ | 向量筛选数据 |
| YYYY-MM-DD | sft_selected_safety | Phase 2 | Qwen2.5-3B-Instruct | selected + 500 safety | LoRA r=8, lr=2e-5, epoch=2 | outputs/sft/sft_selected_safety | ⏳ | 加安全数据 |
| YYYY-MM-DD | dpo_selected | Phase 3 | Qwen2.5-3B-Instruct | preference pairs | LoRA r=8, lr=5e-6, beta=0.1 | outputs/dpo/sft_selected | ⏳ | DPO 偏好优化 |

## 训练记录

### sft_random
- 开始时间:
- 结束时间:
- 显存占用:
- Train loss 起始/最终:
- Eval loss 起始/最终:
- 遇到的错误及解决:
  -

### sft_selected
- 开始时间:
- 结束时间:
- 显存占用:
- Train loss 起始/最终:
- Eval loss 起始/最终:
- 遇到的错误及解决:
  -

### sft_selected_safety
- 开始时间:
- 结束时间:
- 显存占用:
- Train loss 起始/最终:
- Eval loss 起始/最终:
- 遇到的错误及解决:
  -

### dpo_selected
- 开始时间:
- 结束时间:
- 显存占用:
- Train loss 起始/最终:
- Eval loss 起始/最终:
- 遇到的错误及解决:
  -
