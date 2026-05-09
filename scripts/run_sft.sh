#!/bin/bash
# SFT 训练启动脚本
# 用法:
#   bash scripts/run_sft.sh sft_random        # 通用+随机医疗数据
#   bash scripts/run_sft.sh sft_selected      # 通用+向量筛选医疗数据
#   bash scripts/run_sft.sh sft_selected_safety # 上述+安全数据

set -e

# ---- 配置 ----
EXPERIMENT="${1:-sft_selected}"
BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

TRAIN_FILE="$PROJECT_ROOT/project_data/${EXPERIMENT}/train.jsonl"
VALID_FILE="$PROJECT_ROOT/project_data/sft_valid/valid.jsonl"
OUTPUT_DIR="$PROJECT_ROOT/outputs/sft/${EXPERIMENT}"

# ---- 检查数据文件 ----
if [ ! -f "$TRAIN_FILE" ]; then
    echo "ERROR: Training data not found: $TRAIN_FILE"
    echo "Run: python3 scripts_project/03_build_sft.py"
    exit 1
fi

echo "=== SFT Training: $EXPERIMENT ==="
echo "Model:    $BASE_MODEL"
echo "Train:    $TRAIN_FILE"
echo "Output:   $OUTPUT_DIR"

# ---- SwanLab 配置 ----
SWANLAB_API_KEY="${SWANLAB_API_KEY:-}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-medical-chat}"
if [ -n "$SWANLAB_API_KEY" ]; then
    export SWANLAB_API_KEY
    echo "SwanLab enabled (project: $SWANLAB_PROJECT)"
fi

# ---- 训练参数 ----
NUM_GPUS="${NUM_GPUS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
USE_QLORA="${USE_QLORA:-false}"

# ---- 构建 torchrun 命令 ----
CMD_ARGS=(
    --model_name_or_path "$BASE_MODEL"
    --train_file_dir "$(dirname "$TRAIN_FILE")"
    --validation_file_dir "$(dirname "$VALID_FILE")"
    --per_device_train_batch_size "$BATCH_SIZE"
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --do_train
    --do_eval
    --use_peft true
    --model_max_length "$MAX_LENGTH"
    --num_train_epochs "$NUM_EPOCHS"
    --learning_rate "$LEARNING_RATE"
    --lora_rank "$LORA_RANK"
    --lora_alpha "$LORA_ALPHA"
    --lora_dropout 0.05
    --target_modules all
    --output_dir "$OUTPUT_DIR"
    --overwrite_output_dir
    --report_to swanlab
    --swanlab_project "$SWANLAB_PROJECT"
    --save_strategy epoch
    --evaluation_strategy epoch
    --logging_steps 10
    --save_total_limit 2
    --load_best_model_at_end true
    --metric_for_best_model eval_loss
    --save_safetensors true
    --gradient_checkpointing true
    --bf16
    --dataloader_num_workers 2
    --template_name qwen
)

# QLoRA 模式
if [ "$USE_QLORA" = "true" ]; then
    CMD_ARGS+=(--quantization_bit 4)
fi

# 单卡
if [ "$NUM_GPUS" -le 1 ]; then
    echo "Running single-GPU training..."
    python "$PROJECT_ROOT/MedicalGPT/training/supervised_finetuning.py" "${CMD_ARGS[@]}"
else
    echo "Running multi-GPU training with $NUM_GPUS GPUs..."
    torchrun --nproc_per_node "$NUM_GPUS" \
        "$PROJECT_ROOT/MedicalGPT/training/supervised_finetuning.py" "${CMD_ARGS[@]}"
fi

echo "=== SFT Done. Model saved to $OUTPUT_DIR ==="
