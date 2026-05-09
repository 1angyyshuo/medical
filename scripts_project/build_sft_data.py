"""
将原始医疗数据转为 SFT 训练格式。

用法:
  # 只用医疗数据
  python build_sft_data.py --medical_n 2000

  # 混通用数据
  python build_sft_data.py --general MedicalGPT/data/sft/sharegpt_zh_1K_format.jsonl --medical_n 2000
"""

import json
import random
from argparse import ArgumentParser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "project_data" / "sft"

MEDICAL_SYSTEM = "你是一个专业的医疗健康助手。请注意：你提供的是健康信息参考，不能替代医生的专业诊断。遇到急症、用药调整等问题请务必建议用户就医。"


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_jsonl(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} records -> {path}")


def normalize(record: dict) -> dict:
    """把各种格式统一为 {question, answer}"""
    if "question" in record and "answer" in record:
        return record
    if "input" in record and "output" in record:
        return {"question": record["input"], "answer": record["output"]}
    if "instruction" in record and "output" in record:
        q = record["instruction"]
        if record.get("input"):
            q = q + "\n" + record["input"]
        return {"question": q, "answer": record["output"]}
    if "conversations" in record:
        convs = record["conversations"]
        human = next((c["value"] for c in convs if c.get("from") == "human"), "")
        gpt = next((c["value"] for c in convs if c.get("from") == "gpt"), "")
        return {"question": human, "answer": gpt}
    return record


def to_sharegpt(question: str, answer: str) -> dict:
    return {
        "conversations": [
            {"from": "system", "value": MEDICAL_SYSTEM},
            {"from": "human", "value": question},
            {"from": "gpt", "value": answer},
        ]
    }


def load_medical_data(n: int) -> list[dict]:
    """加载原始医疗数据并归一化"""
    all_records = []
    for f in sorted(RAW_DIR.glob("*.jsonl")):
        print(f"Loading {f.name}...")
        for r in load_jsonl(f):
            r = normalize(r)
            if r.get("question") and r.get("answer"):
                all_records.append(r)

    random.shuffle(all_records)
    selected = all_records[:n]
    print(f"Selected {len(selected)} medical QA from {len(all_records)} total")
    return selected


def main():
    parser = ArgumentParser(description="Build SFT training data")
    parser.add_argument("--general", type=Path, default=None, help="通用数据 JSONL")
    parser.add_argument("--general_n", type=int, default=1000)
    parser.add_argument("--medical_n", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # 通用数据（可选）
    general_records = []
    if args.general and args.general.exists():
        general_records = load_jsonl(args.general)
        general_records = general_records[:args.general_n]
        print(f"Loaded {len(general_records)} general records")

    # 医疗数据
    medical_records = load_medical_data(args.medical_n)
    medical_sft = [to_sharegpt(r["question"], r["answer"]) for r in medical_records]

    # 合并
    data = general_records + medical_sft
    random.shuffle(data)

    save_jsonl(data, args.output / "train.jsonl")
    print(f"Total: {len(data)} samples ({len(general_records)} general + {len(medical_sft)} medical)")


if __name__ == "__main__":
    main()
