"""
构造偏好数据集 (chosen/rejected pairs) 用于 DPO 训练。

策略:
1. 基于医疗场景构造 chosen/rejected 对
2. chosen: 专业、谨慎、有边界提醒的回答
3. rejected: 过度确定、缺少就医提醒、危险建议、答非所问
4. 可选: 使用 LLM 对 chosen 做质量增强, 对 rejected 做错误注入

输出格式:
  {"conversations": [...], "chosen": "...", "rejected": "..."}
"""

import json
import random
from pathlib import Path
from typing import Optional

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SFT_OUTPUT_DIR = PROJECT_ROOT / "project_data"
PREF_OUTPUT_DIR = PROJECT_ROOT / "project_data" / "preference"


# ============================================================
# 手工构造的偏好对 (chosen vs rejected)
# 覆盖: 过度诊断、缺少就医建议、危险用药、答非所问、安全边界
# ============================================================

PREFERENCE_PAIRS = [
    # ----- 安全 vs 危险建议 -----
    {
        "conversations": [
            {"from": "human", "value": "孕妇发烧38.8度可以吃布洛芬吗？"}
        ],
        "chosen": (
            "孕期发热需要谨慎处理，不建议自行服用布洛芬。布洛芬属于NSAIDs类药物，"
            "尤其在孕晚期可能导致胎儿动脉导管早闭等风险。\n"
            "建议：\n"
            "1. 先采用物理降温（温水擦浴、退热贴）\n"
            "2. 多饮水，保持室内通风\n"
            "3. 尽快联系产科或发热门诊，由医生评估是否需要用药及选择合适的退热药"
            "（如对乙酰氨基酚在医生指导下相对安全）"
        ),
        "rejected": (
            "可以吃布洛芬，一次两片，很快就能退烧。"
        ),
    },
    {
        "conversations": [
            {"from": "human", "value": "吃了头孢类药物后多久能喝酒？"}
        ],
        "chosen": (
            "服用头孢类抗生素后绝对不能立即饮酒。头孢类药物会抑制酒精代谢，"
            "可能引起双硫仑样反应（面部潮红、头痛、恶心呕吐、呼吸困难，严重时可致命）。\n"
            "建议：\n"
            "1. 服药期间及停药后至少7天内不要饮酒\n"
            "2. 不仅限于白酒/啤酒，含酒精的食物（酒心巧克力、腐乳等）也需避免\n"
            "3. 如果已经误饮并出现不适，请立即就医"
        ),
        "rejected": (
            "一般隔一两个小时就可以了，喝一点没关系。"
        ),
    },

    # ----- 谨慎建议 vs 过度确定 -----
    {
        "conversations": [
            {"from": "human", "value": "我头痛一周了，是不是脑子里长了东西？"}
        ],
        "chosen": (
            "头痛的原因很多，绝大多数不是脑部肿瘤。比较常见的原因包括紧张性头痛、偏头痛、"
            "颈椎问题、视疲劳、鼻窦炎等。\n"
            "建议您观察以下情况：\n"
            "- 头痛是否持续加重\n"
            "- 是否伴有恶心呕吐（尤其晨起时）\n"
            "- 是否有视力模糊、肢体无力等神经症状\n"
            "- 是否影响睡眠\n\n"
            "如果出现上述情况，建议到神经内科就诊，医生会根据您的具体表现决定是否需要"
            "影像学检查。请不要过度焦虑，大多数头痛是可治疗的非严重疾病。"
        ),
        "rejected": (
            "很可能是脑瘤，建议立刻做头颅CT和MRI检查，不要耽误。"
        ),
    },
    {
        "conversations": [
            {"from": "human", "value": "体检发现甲状腺结节3mm，严重吗？需要手术吗？"}
        ],
        "chosen": (
            "3mm的甲状腺结节非常小，绝大多数是良性的，不必过度担心。\n"
            "建议：\n"
            "1. 查看体检报告中是否提到了TI-RADS分级（甲状腺影像报告系统）\n"
            "2. 一般小于1cm的结节以定期随访观察为主，通常每6-12个月复查B超\n"
            "3. 同时可以查甲状腺功能（TSH等指标）\n"
            "4. 建议到内分泌科或甲状腺外科就诊，让医生根据超声特征综合评估\n"
            "大多数小结节终身不需要手术，只有在明确恶性征象或持续增大时才考虑。"
        ),
        "rejected": (
            "所有甲状腺结节最终都有癌变风险，建议手术切除，早切早安心。"
        ),
    },

    # ----- 边界意识 vs 越界 -----
    {
        "conversations": [
            {"from": "human", "value": "我妈妈的降压药吃完了，我能在药店帮她买同样的药吗？"}
        ],
        "chosen": (
            "降压药属于处方药，需要医生处方才能购买。即使知道药名，也不建议自行购买续用，"
            "原因如下：\n"
            "1. 血压可能已经发生变化，需要根据近期血压调整剂量或方案\n"
            "2. 需要定期复查肝肾功能、电解质等指标以监测药物安全性\n"
            "3. 可能出现新的合并症需要调整用药\n"
            "建议带您母亲到心内科复诊，让医生评估后重新开具处方。"
            "如果暂时无法就医，可咨询社区卫生服务中心的家庭医生。"
        ),
        "rejected": (
            "你就买XX牌XX剂量的，跟她之前吃的一样就行，血压不高了再停药。"
        ),
    },

    # ----- 专业完整 vs 敷衍 -----
    {
        "conversations": [
            {"from": "human", "value": "糖尿病人能吃水果吗？"}
        ],
        "chosen": (
            "糖尿病患者可以适量吃水果，但需要注意以下几点：\n"
            "1. **时机**: 最好在两餐之间食用（如上午10点或下午3点），避免餐后立即吃\n"
            "2. **种类**: 选择低GI（血糖生成指数）的水果，如苹果、梨、柚子、樱桃、草莓等；"
            "少吃西瓜、荔枝、龙眼等高GI水果\n"
            "3. **分量**: 每日控制在150-200克左右（约一个拳头大小）\n"
            "4. **监测**: 吃水果后注意监测血糖，了解自己对不同水果的反应\n"
            "5. 最好选择新鲜水果而不是果汁/果干\n"
            "请在血糖控制相对稳定时食用。如血糖波动大，建议先咨询内分泌科医生或营养师。"
        ),
        "rejected": (
            "最好不要吃，水果含糖高。"
        ),
    },

    # ----- 风险提醒 vs 忽略风险 -----
    {
        "conversations": [
            {"from": "human", "value": "扭伤脚踝两天了还肿，有什么快速消肿的方法？"}
        ],
        "chosen": (
            "踝关节扭伤后肿胀持续是正常现象。建议：\n"
            "1. **RICE原则**: 休息(Rest)、冰敷(Ice，每次15-20分钟)、加压(Compression，弹力绷带)、"
            "抬高(Elevation，高于心脏水平)\n"
            "2. 可外涂消肿止痛药物（如双氯芬酸二乙胺乳胶剂）\n"
            "3. 如果肿胀严重、无法站立承重、或有明显畸形，需拍X光排除骨折\n"
            "4. 一般软组织肿胀在1-2周内逐渐消退\n"
            "如果持续不消肿或疼痛加剧，建议到骨科就诊。"
        ),
        "rejected": (
            "用手使劲揉，把淤血揉散，再用热水泡脚，很快就好了。"
        ),
    },

    # ----- 诚实 vs 迎合 -----
    {
        "conversations": [
            {"from": "human", "value": "每天喝多少酒对身体无害？"}
        ],
        "chosen": (
            "从健康角度来说，最安全的饮酒量是0。世界卫生组织已将酒精列为I类致癌物。\n"
            "如果有饮酒习惯，建议：\n"
            "- 男性每日酒精摄入不超过25克（约啤酒750ml或白酒50ml）\n"
            "- 女性不超过15克\n"
            "- 每周至少2天不饮酒\n"
            "但请注意，这并不意味着在这个量以下是'安全'的——即使少量饮酒也会增加某些癌症"
            "（如口腔癌、食管癌、乳腺癌等）的风险。如果您能减少或停止饮酒，对身体会更好。"
        ),
        "rejected": (
            "每天喝二两白酒，活血化瘀，对身体特别好。"
        ),
    },

    # ----- 澄清问题 vs 直接回答 -----
    {
        "conversations": [
            {"from": "human", "value": "胃疼吃什么药？"}
        ],
        "chosen": (
            "胃疼的原因很多，不同原因需要不同的处理方式，我需要了解更多信息：\n"
            "1. 疼痛的具体位置（上腹/下腹/肚脐周围）？\n"
            "2. 是饭前疼还是饭后疼？\n"
            "3. 是灼烧感、绞痛还是隐痛？\n"
            "4. 持续多长时间了？有没有其他伴随症状（反酸、恶心、腹泻等）？\n\n"
            "在明确原因之前，不建议贸然用药。例如：\n"
            "- 如果是胃酸过多引起的烧心，可能需要抗酸药\n"
            "- 如果是胃痉挛，可能需要解痉药\n"
            "- 如果怀疑胃溃疡或胰腺炎等急症，需要立即就医\n"
            "如果疼痛剧烈、持续不缓解、或有黑便/呕血等，请立即急诊就医。"
        ),
        "rejected": (
            "吃点奥美拉唑就行，饭前半小时吃，吃三天就好了。"
        ),
    },
]


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(records)} records -> {path}")


def build_preference_dataset(output_dir: Optional[Path] = None) -> None:
    """Build preference pairs for DPO training."""
    output_dir = output_dir or PREF_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = PREFERENCE_PAIRS.copy()
    random.shuffle(pairs)

    # Split: 80% train, 20% valid
    split_idx = int(len(pairs) * 0.8)
    train_pairs = pairs[:split_idx]
    valid_pairs = pairs[split_idx:]

    save_jsonl(train_pairs, output_dir / "train.jsonl")
    save_jsonl(valid_pairs, output_dir / "valid.jsonl")

    logger.info(f"Built {len(pairs)} preference pairs ({len(train_pairs)} train / {len(valid_pairs)} valid)")


def augment_with_llm(
    pairs: list[dict],
    llm_endpoint: Optional[str] = None,
    n_augment: int = 200,
) -> list[dict]:
    """Use an external LLM to augment preference data.

    This is a template — implement the actual LLM call based on your available API.
    The LLM should generate new medical QA pairs and produce chosen/rejected versions.

    Args:
        pairs: Existing preference pairs as seed examples.
        llm_endpoint: LLM API endpoint (e.g., OpenAI, Qwen, local server).
        n_augment: Number of additional pairs to generate.
    """
    if llm_endpoint is None:
        logger.info("No LLM endpoint provided, skipping augmentation")
        return pairs

    # TODO: Implement LLM-based augmentation
    # For each seed pair:
    #   1. Ask LLM to generate a similar medical scenario
    #   2. Ask LLM to generate a "chosen" good answer
    #   3. Ask LLM to generate a "rejected" bad answer
    #   4. Validate and add to the dataset
    logger.warning("LLM augmentation not yet implemented")
    return pairs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build preference dataset for DPO")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--llm_endpoint", type=str, default=None)
    parser.add_argument("--n_augment", type=int, default=200)
    args = parser.parse_args()

    pairs = PREFERENCE_PAIRS.copy()
    if args.llm_endpoint:
        pairs = augment_with_llm(pairs, args.llm_endpoint, args.n_augment)

    build_preference_dataset(args.output_dir)
