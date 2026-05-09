"""
运行模型评测 — 基于 ceval/ceval-exam 医学子集。

评测数据集 (HuggingFace: ceval/ceval-exam):
- basic_medicine     基础医学
- clinical_medicine  临床医学
- physician          医师资格考试 (执业医师)

用法:
  # Baseline 评测
  python scripts_project/05_run_eval.py --model Qwen/Qwen2.5-3B-Instruct --name baseline

  # SFT 模型评测 (LoRA)
  python scripts_project/05_run_eval.py --model Qwen/Qwen2.5-3B-Instruct --peft outputs/sft/sft_selected --name sft_selected

  # DPO 模型评测
  python scripts_project/05_run_eval.py --model Qwen/Qwen2.5-3B-Instruct --peft outputs/dpo/sft_selected --name dpo_selected
"""

import json
import os
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from loguru import logger
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eval"

# CEval 医学子集
MEDICAL_SUBSETS = [
    "basic_medicine",
    "clinical_medicine",
    "physician",
]

# CEval 选择题选项标签
OPTION_LABELS = ["A", "B", "C", "D"]


def load_ceval_medical_subsets() -> dict[str, list[dict]]:
    """Load medical subsets from ceval/ceval-exam on HuggingFace.

    Returns:
        Dict mapping subset_name -> list of {question, A, B, C, D, answer}
    """
    from datasets import load_dataset

    subsets = {}
    for subset_name in MEDICAL_SUBSETS:
        logger.info(f"Loading ceval/ceval-exam [{subset_name}]...")
        ds = load_dataset("ceval/ceval-exam", subset_name, split="test")
        items = []
        for row in ds:
            items.append({
                "id": row.get("id", ""),
                "question": row["question"].strip(),
                "A": row["A"].strip(),
                "B": row["B"].strip(),
                "C": row["C"].strip(),
                "D": row["D"].strip(),
                "answer": row["answer"].strip().upper(),
            })
        subsets[subset_name] = items
        logger.info(f"  {subset_name}: {len(items)} questions")
    return subsets


def build_prompt(question: str, options: dict[str, str]) -> str:
    """Build a multi-choice prompt for CEval medical questions.

    Uses the Qwen chat template style. The model should output the correct option letter.
    """
    lines = [
        "以下是一道医学单项选择题，请选出最正确的答案。",
        "",
        f"题目: {question}",
        "",
    ]
    for label in OPTION_LABELS:
        lines.append(f"{label}. {options[label]}")
    lines.append("")
    lines.append("答案: ")
    return "\n".join(lines)


def load_model_and_tokenizer(
    model_name: str,
    peft_path: Optional[str] = None,
    device: str = "cuda:0",
):
    """Load model and tokenizer with optional LoRA adapter."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    if peft_path:
        logger.info(f"Loading LoRA adapter: {peft_path}")
        model = PeftModel.from_pretrained(model, peft_path)
        model = model.merge_and_unload()

    return model, tokenizer


def evaluate_multichoice(
    model,
    tokenizer,
    items: list[dict],
    device: str = "cuda:0",
    max_new_tokens: int = 10,
) -> dict:
    """Evaluate multi-choice accuracy on a set of questions.

    Uses generation-based approach: prompt the model, extract the predicted
    option letter from the generated text.
    """
    correct = 0
    total = 0
    details = []

    for item in tqdm(items, desc="Evaluating"):
        prompt = build_prompt(
            item["question"],
            {label: item[label] for label in OPTION_LABELS},
        )

        # Use Qwen chat template
        messages = [
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the generated part
        generated = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        ).strip()

        # Extract predicted answer (first A/B/C/D found)
        pred = None
        for char in generated.upper():
            if char in OPTION_LABELS:
                pred = char
                break

        is_correct = (pred == item["answer"])
        if is_correct:
            correct += 1
        total += 1

        details.append({
            "id": item["id"],
            "question": item["question"][:100],
            "expected": item["answer"],
            "predicted": pred,
            "generated": generated[:100],
            "correct": is_correct,
        })

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0,
        "details": details,
    }


def print_results(subset_results: dict[str, dict]) -> None:
    """Pretty-print evaluation results."""
    print(f"\n{'='*60}")
    print("CEval 医学子集评测结果")
    print(f"{'='*60}")
    total_correct = 0
    total_questions = 0

    for subset_name in MEDICAL_SUBSETS:
        r = subset_results.get(subset_name, {})
        acc = r.get("accuracy", 0)
        correct = r.get("correct", 0)
        total = r.get("total", 0)
        total_correct += correct
        total_questions += total
        print(f"  {subset_name:25s}: {correct:4d}/{total:4d} = {acc:.4f} ({acc*100:.1f}%)")

    overall_acc = total_correct / total_questions if total_questions > 0 else 0
    print(f"  {'─'*50}")
    print(f"  {'Overall':25s}: {total_correct:4d}/{total_questions:4d} = {overall_acc:.4f} ({overall_acc*100:.1f}%)")


def main():
    parser = ArgumentParser(description="Evaluate medical LLM on CEval medical subsets")
    parser.add_argument("--model", required=True, help="Base model name or path (e.g., Qwen/Qwen2.5-3B-Instruct)")
    parser.add_argument("--peft", default=None, help="Path to LoRA adapter")
    parser.add_argument("--name", required=True, help="Experiment name for output directory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--subsets", nargs="+", default=MEDICAL_SUBSETS,
                        help=f"Subsets to evaluate (default: {MEDICAL_SUBSETS})")

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / args.name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load evaluation data ----
    logger.info("Loading CEval medical evaluation datasets...")
    all_subsets = load_ceval_medical_subsets()

    # Filter to requested subsets
    subsets_to_eval = {
        name: items
        for name, items in all_subsets.items()
        if name in args.subsets
    }

    if not subsets_to_eval:
        logger.error(f"No subsets found matching: {args.subsets}")
        logger.info(f"Available: {list(all_subsets.keys())}")
        return

    # ---- Load model ----
    model, tokenizer = load_model_and_tokenizer(args.model, args.peft, args.device)

    # ---- Evaluate each subset ----
    subset_results = {}
    for subset_name, items in subsets_to_eval.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {subset_name} ({len(items)} questions)")
        logger.info(f"{'='*60}")

        result = evaluate_multichoice(model, tokenizer, items, args.device)
        subset_results[subset_name] = {
            "total": result["total"],
            "correct": result["correct"],
            "accuracy": result["accuracy"],
        }

        # Save per-subset details
        detail_path = run_dir / f"{subset_name}_details.json"
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump(result["details"], f, ensure_ascii=False, indent=2)

    # ---- Print results ----
    print_results(subset_results)

    # ---- Save summary ----
    total_correct = sum(r["correct"] for r in subset_results.values())
    total_questions = sum(r["total"] for r in subset_results.values())

    summary = {
        "experiment": args.name,
        "model": args.model,
        "peft": args.peft,
        "timestamp": timestamp,
        "subsets": {
            name: {
                "total": r["total"],
                "correct": r["correct"],
                "accuracy": round(r["accuracy"], 6),
            }
            for name, r in subset_results.items()
        },
        "overall": {
            "total": total_questions,
            "correct": total_correct,
            "accuracy": round(total_correct / total_questions, 6) if total_questions > 0 else 0,
        },
    }

    summary_path = run_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"\nResults saved to: {run_dir}")
    logger.info(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
