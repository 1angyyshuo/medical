"""
构造 SFT 训练数据集。

生成以下数据集:
- general_sft: 通用指令数据 (保持通用能力)
- medical_sft_random: 随机采样医疗数据 (对照组)
- medical_sft_selected: 向量筛选医疗数据 (实验组)
- medical_safety: 安全拒答数据

输出格式: ShareGPT JSONL
  {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
"""

import json
import random
from pathlib import Path
from typing import Optional

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ANCHOR_DIR = PROJECT_ROOT / "project_data" / "anchors"
SFT_OUTPUT_DIR = PROJECT_ROOT / "project_data"


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(records)} records -> {path}")


def to_sharegpt(question: str, answer: str, system_prompt: str = "") -> dict:
    """Convert a QA pair to ShareGPT conversation format."""
    conversations = []
    if system_prompt:
        conversations.append({"from": "system", "value": system_prompt})
    conversations.append({"from": "human", "value": question})
    conversations.append({"from": "gpt", "value": answer})
    return {"conversations": conversations}


def build_medical_sft_from_filtered(
    medical_records: list[dict],
    n_samples: int,
    system_prompt: str,
    seed: int = 42,
) -> list[dict]:
    """Build medical SFT data from filtered medical QA pairs."""
    random.seed(seed)
    selected = random.sample(medical_records, min(n_samples, len(medical_records)))
    return [to_sharegpt(r["question"], r["answer"], system_prompt) for r in selected]


def build_safety_data() -> list[dict]:
    """Build medical safety/refusal training data.

    These teach the model to refuse unsafe requests and include disclaimers
    when appropriate.
    """
    safety_pairs = [
        # ----- 处方/剂量拒绝 -----
        (
            "给我开一个治高血压的方子，要具体药名和剂量。",
            "我不能为您开具具体处方。处方需要医生根据您的血压水平、合并症、药物过敏史等综合评估后出具。"
            "建议您到心内科就诊，医生会为您选择最合适的降压方案。请勿自行用药。"
        ),
        (
            "我血压有点高，推荐一个效果最强的降压药。",
            "降压药的选择需要个体化评估，没有所谓'效果最强'的通用药物。不同人群（老年人、孕妇、合并糖尿病等）"
            "适用不同类型的降压药（ACEI、ARB、CCB、利尿剂等）。建议您到心内科就诊，医生会根据您的具体情况开具。"
        ),

        # ----- 危险组合警告 -----
        (
            "喝酒的时候吃什么药可以更high？",
            "任何药物与酒精混合使用都可能导致危险甚至致命。酒精会增加镇静类药物、止痛药等的毒性，"
            "可能导致呼吸抑制、肝损伤等严重后果。请勿将酒精与任何药物混合使用。"
            "如果您或身边人有药物滥用问题，建议寻求专业帮助。"
        ),

        # ----- 体检作弊拒绝 -----
        (
            "明天入职体检，怎么让转氨酶快速降下来？",
            "我不建议您尝试任何方法干扰体检结果。体检的目的是了解真实健康状况，"
            "隐瞒异常指标可能延误严重疾病的诊治。转氨酶升高可通过暂停饮酒、规律作息等方式改善，"
            "但更重要的是找出升高的原因（如脂肪肝、病毒性肝炎等）。请诚实面对体检结果。"
        ),

        # ----- 自杀/危机干预 -----
        (
            "我活着好累，怎么结束生命最不痛苦？",
            "我很关心您的感受，您现在说的话让我非常担心。请您知道，无论当前多么困难，"
            "这些感受是可以改变的，而且有人愿意帮助您。请立即联系以下资源：\n"
            "- 全国心理援助热线：400-161-9995\n"
            "- 北京心理危机研究与干预中心：010-82951332\n"
            "- 或拨打110/120获得紧急帮助\n"
            "您不是一个人在面对这些问题，请现在就联系他们。"
        ),

        # ----- 减肥药拒绝 -----
        (
            "推荐一种不用运动也不用节食就能快速瘦的药。",
            "不存在安全且无需运动和饮食管理就能减肥的药物。所谓'躺着瘦'的减肥药往往含有禁用成分"
            "（如西布曲明、芬氟拉明等），可能导致心血管损伤、精神障碍等严重后果。\n"
            "科学减重需要：合理饮食控制、规律运动、行为习惯调整。如有需要可咨询内分泌科或临床营养科。"
        ),

        # ----- 偏方/替代疗法拒绝 -----
        (
            "癌症不想化疗，有没有什么民间偏方可以治？",
            "我理解您对化疗的担忧，但必须明确告诉您：目前没有科学证据支持任何'民间偏方'能治愈癌症。"
            "放弃或延迟规范治疗可能导致肿瘤进展，错过最佳治疗时机。\n"
            "您可以与肿瘤科医生讨论化疗方案的副作用管理、靶向治疗、免疫治疗等更多选择。"
            "也可以寻求中医药辅助治疗（需在医生指导下），但不能替代主要治疗。"
        ),

        # ----- 装病拒绝 -----
        (
            "怎么伪装发烧骗老师请假？",
            "我不能帮助您伪装疾病。欺骗老师和学校不仅是不诚实的行为，"
            "还可能让您在真正需要照顾时失去他人的信任。如果您感到压力或不想上学，"
            "建议与家长、老师或学校心理老师坦诚沟通，找到真正解决问题的办法。"
        ),

        # ----- 过度诊断拒绝 -----
        (
            "我百度查了下，头痛可能是脑瘤，怎么办？",
            "请不要过度依赖网络搜索进行自我诊断。大多数头痛不是脑瘤——紧张性头痛、偏头痛、"
            "颈椎问题等是更常见的原因。但如果您有进行性加重的头痛、晨起头痛伴呕吐、"
            "伴随神经功能异常（视力模糊、肢体无力等），建议到神经内科就诊做进一步检查。\n"
            "医生会通过问诊和必要的检查来判断头痛原因，请勿过度焦虑。"
        ),

        # ----- 自行用药拒绝 -----
        (
            "小孩咳嗽一周了，大人吃的止咳药能减半给小孩吃吗？",
            "绝对不能将成人药物减量给儿童服用。儿童不是'缩小版成人'，其肝脏代谢、肾脏排泄能力不同，"
            "许多成人药物在儿童中的安全性和有效性未经验证，可能造成严重不良反应。\n"
            "儿童咳嗽需就医明确病因，由儿科医生选择适合儿童的药物和剂量。"
        ),

        # ----- 诊断断言拒绝 -----
        (
            "我觉得我肯定是抑郁症，该吃什么药？",
            "我无法为您诊断抑郁症，也不能建议具体用药。抑郁症的诊断需要精神科医生通过详细"
            "问诊和评估来完成，有很多情况（甲状腺功能减退、贫血、维生素缺乏等）也可能导致类似症状。\n"
            "建议您到精神科或心理科就诊，医生会做全面评估并制定适合您的治疗方案。"
            "同时，请记住寻求帮助是勇敢的表现。"
        ),
    ]

    return [to_sharegpt(q, a, "") for q, a in safety_pairs]


def build_sft_datasets(
    medical_processed_dir: Optional[Path] = None,
    general_sft_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    general_n: int = 1000,
    medical_n: int = 2000,
    safety_n: int = 500,
    seed: int = 42,
):
    """Build all SFT datasets.

    Outputs:
    - sft_general/: general_sft 1k
    - sft_random/: general 1k + random medical 2k
    - sft_selected/: general 1k + selected medical 2k
    - sft_selected_safety/: above + safety 500
    """
    medical_processed_dir = medical_processed_dir or PROCESSED_DIR
    output_dir = output_dir or SFT_OUTPUT_DIR

    MEDICAL_SYSTEM = (
        "你是一个专业的医疗健康助手。请根据用户的描述提供专业、准确、安全的建议。"
        "请注意: 你提供的是一般性健康信息，不能替代医生的专业诊断。"
        "在涉及急症、用药调整、诊断确认时，请务必建议用户就医。"
    )

    # ---- Load medical data ----
    train_path = medical_processed_dir / "medical_train.jsonl"
    valid_path = medical_processed_dir / "medical_valid.jsonl"

    if not train_path.exists():
        logger.warning(f"No processed medical data found at {train_path}")
        logger.info("Run 02_filter_medical_data.py first")
        return

    all_medical = load_jsonl(train_path)

    # ---- Load or create general SFT data ----
    general_records = []
    if general_sft_path and general_sft_path.exists():
        general_records = load_jsonl(general_sft_path)
        logger.info(f"Loaded {len(general_records)} general SFT records")

    # ---- Build datasets ----
    random.seed(seed)

    # medical_sft_selected (from vector recall)
    medical_selected = all_medical[:medical_n] if len(all_medical) >= medical_n else all_medical
    medical_selected_sft = [
        to_sharegpt(r["question"], r["answer"], MEDICAL_SYSTEM) for r in medical_selected
    ]

    # medical_sft_random (shuffle and take n)
    shuffled = all_medical.copy()
    random.shuffle(shuffled)
    medical_random = shuffled[:medical_n]
    medical_random_sft = [
        to_sharegpt(r["question"], r["answer"], MEDICAL_SYSTEM) for r in medical_random
    ]

    # safety data
    safety_records = build_safety_data()[:safety_n]

    # ---- Assemble final datasets ----
    # 1) general only
    save_jsonl(general_records[:general_n], output_dir / "sft_general" / "train.jsonl")

    # 2) general + random medical
    sft_random_mix = general_records[:general_n] + medical_random_sft
    save_jsonl(sft_random_mix, output_dir / "sft_random" / "train.jsonl")

    # 3) general + selected medical
    sft_selected_mix = general_records[:general_n] + medical_selected_sft
    save_jsonl(sft_selected_mix, output_dir / "sft_selected" / "train.jsonl")

    # 4) general + selected medical + safety
    sft_selected_safety = sft_selected_mix + safety_records
    save_jsonl(sft_selected_safety, output_dir / "sft_selected_safety" / "train.jsonl")

    # ---- Validation set ----
    if valid_path.exists():
        valid_records = load_jsonl(valid_path)
        valid_sft = [
            to_sharegpt(r["question"], r["answer"], MEDICAL_SYSTEM) for r in valid_records[:500]
        ]
        save_jsonl(valid_sft, output_dir / "sft_valid" / "valid.jsonl")

    # ---- Summary ----
    logger.info("=== SFT Dataset Summary ===")
    logger.info(f"  sft_general:          {general_n} samples")
    logger.info(f"  sft_random:           {general_n}g + {len(medical_random_sft)}m = {general_n + len(medical_random_sft)}")
    logger.info(f"  sft_selected:         {general_n}g + {len(medical_selected_sft)}m = {general_n + len(medical_selected_sft)}")
    logger.info(f"  sft_selected_safety:  {general_n}g + {len(medical_selected_sft)}m + {len(safety_records)}s = {general_n + len(medical_selected_sft) + len(safety_records)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build SFT training datasets")
    parser.add_argument("--medical_processed_dir", type=Path, default=None)
    parser.add_argument("--general_sft_path", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--general_n", type=int, default=1000)
    parser.add_argument("--medical_n", type=int, default=2000)
    parser.add_argument("--safety_n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_sft_datasets(
        medical_processed_dir=args.medical_processed_dir,
        general_sft_path=args.general_sft_path,
        output_dir=args.output_dir,
        general_n=args.general_n,
        medical_n=args.medical_n,
        safety_n=args.safety_n,
        seed=args.seed,
    )
