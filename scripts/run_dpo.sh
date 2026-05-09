#!/bin/bash
# DPO 偏好优化训练
# 用法:
#   bash scripts/run_dpo.sh sft_selected     # 对指定 SFT 模型做 DPO

set -e

# ---- 配置 ----
SFT_EXPERIMENT="${1:-sft_selected}"
BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# SFT checkpoint (LoRA adapter)
SFT_CHECKPOINT="$PROJECT_ROOT/outputs/sft/${SFT_EXPERIMENT}"

# Preference data
PREF_TRAIN="$PROJECT_ROOT/project_data/preference/train.jsonl"
PREF_VALID="$PROJECT_ROOT/project_data/preference/valid.jsonl"

# Output
OUTPUT_DIR="$PROJECT_ROOT/outputs/dpo/${SFT_EXPERIMENT}"

# ---- 检查前置条件 ----
if [ ! -d "$SFT_CHECKPOINT" ]; then
    echo "ERROR: SFT checkpoint not found: $SFT_CHECKPOINT"
    echo "Run SFT first: bash scripts/run_sft.sh $SFT_EXPERIMENT"
    exit 1
fi

if [ ! -f "$PREF_TRAIN" ]; then
    echo "ERROR: Preference data not found: $PREF_TRAIN"
    echo "Run: python3 scripts_project/04_build_preference.py"
    exit 1
fi

echo "=== DPO Training: $SFT_EXPERIMENT ==="
echo "SFT base: $SFT_CHECKPOINT"
echo "Pref data: $PREF_TRAIN"
echo "Output:   $OUTPUT_DIR"

# ---- SwanLab 配置 ----
SWANLAB_API_KEY="${SWANLAB_API_KEY:-}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-medical-chat}"
if [ -n "$SWANLAB_API_KEY" ]; then
    export SWANLAB_API_KEY
    echo "SwanLab enabled (project: $SWANLAB_PROJECT)"
fi

# ---- DPO 参数 ----
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
BETA="${BETA:-0.1}"  # KL penalty coefficient

CMD_ARGS=(
    --model_name_or_path "$BASE_MODEL"
    --peft_path "$SFT_CHECKPOINT"
    --train_file_dir "$(dirname "$PREF_TRAIN")"
    --validation_file_dir "$(dirname "$PREF_VALID")"
    --per_device_train_batch_size "$BATCH_SIZE"
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --do_train
    --do_eval
    --use_peft true
    --model_max_length "$MAX_LENGTH"
    --max_prompt_length 512
    --max_target_length 512
    --num_train_epochs "$NUM_EPOCHS"
    --learning_rate "$LEARNING_RATE"
    --beta "$BETA"
    --lora_rank 8
    --lora_alpha 16
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
    --template_name qwen
)

python "$PROJECT_ROOT/MedicalGPT/training/dpo_training.py" "${CMD_ARGS[@]}"

echo "=== DPO Done. Model saved to $OUTPUT_DIR ==="
