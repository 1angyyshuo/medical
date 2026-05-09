"""
从原始医疗数据中筛选高质量子集。

流程:
1. 加载原始医疗数据 (shibing624/medical 或其他来源)
2. 使用向量模型编码锚点和候选数据
3. 基于余弦相似度召回与锚点匹配的数据
4. 去重、长度过滤、质量过滤
5. 划分 train/valid/test

不做: 用 CEval/CMMLU 验证集做数据选择 (那会导致数据泄漏)。
"""

import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ANCHOR_DIR = PROJECT_ROOT / "project_data" / "anchors"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_anchors(anchor_path: Optional[Path] = None) -> list[dict]:
    """Load anchor questions from 01_build_anchor.py output."""
    if anchor_path is None:
        anchor_path = ANCHOR_DIR / "medical_anchors.jsonl"
    anchors = []
    with open(anchor_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                anchors.append(json.loads(line))
    logger.info(f"Loaded {len(anchors)} anchors")
    return anchors


def load_raw_data(data_path: Optional[Path] = None) -> list[dict]:
    """Load raw medical QA data.

    Supports multiple formats:
    - JSONL with 'input'/'output' keys
    - JSONL with 'question'/'answer' keys
    - JSONL with 'conversations' (ShareGPT format)
    """
    if data_path is None:
        data_path = RAW_DIR

    records = []
    jsonl_files = list(data_path.glob("*.jsonl"))
    if not jsonl_files:
        logger.warning(f"No JSONL files found in {data_path}")
        return records

    for jsonl_file in jsonl_files:
        logger.info(f"Loading {jsonl_file}")
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    logger.info(f"Loaded {len(records)} raw records")
    return records


def normalize_record(record: dict) -> dict:
    """Normalize diverse formats into a unified {question, answer} dict."""
    # 已经是统一格式
    if "question" in record and "answer" in record:
        return record

    # Alpaca格式
    if "instruction" in record and "output" in record:
        question = record["instruction"]
        if record.get("input"):
            question = question + "\n" + record["input"]
        return {"question": question, "answer": record["output"]}

    # ShareGPT格式 (取第一轮对话)
    if "conversations" in record:
        convs = record["conversations"]
        human_msgs = [c["value"] for c in convs if c.get("from") == "human"]
        gpt_msgs = [c["value"] for c in convs if c.get("from") == "gpt"]
        if human_msgs and gpt_msgs:
            return {"question": human_msgs[0], "answer": gpt_msgs[0]}

    # input/output 格式
    if "input" in record and "output" in record:
        return {"question": record["input"], "answer": record["output"]}

    return record


def length_filter(
    records: list[dict],
    min_q_len: int = 5,
    max_q_len: int = 512,
    min_a_len: int = 20,
    max_a_len: int = 1024,
) -> list[dict]:
    """Filter by question and answer character length."""
    filtered = []
    for r in records:
        q = r.get("question", "")
        a = r.get("answer", "")
        if min_q_len <= len(q) <= max_q_len and min_a_len <= len(a) <= max_a_len:
            filtered.append(r)
    logger.info(f"Length filter: {len(records)} -> {len(filtered)}")
    return filtered


def medical_keyword_filter(records: list[dict]) -> list[dict]:
    """Keep records that contain at least some medical relevance.

    Uses a simple keyword-based approach. In production you might use a
    trained classifier or LLM-based quality scoring.
    """
    medical_terms = [
        # 症状
        "发烧", "咳嗽", "头痛", "腹痛", "胸痛", "关节", "皮疹", "失眠", "头晕",
        "恶心", "呕吐", "腹泻", "便秘", "乏力", "发热", "疼痛", "肿胀", "出血",
        # 疾病
        "糖尿病", "高血压", "冠心病", "哮喘", "肺炎", "肝炎", "肾炎", "胃炎",
        "甲亢", "甲减", "贫血", "白血病", "肿瘤", "癌症", "脑梗", "心梗",
        "慢阻肺", "乙肝", "结核", "艾滋", "痛风", "骨质疏松", "抑郁症",
        # 治疗
        "手术", "药物", "治疗", "检查", "诊断", "预防", "康复", "疫苗",
        "抗生素", "降压药", "降糖药", "化疗", "放疗", "透析",
        # 科室
        "内科", "外科", "妇科", "儿科", "眼科", "耳鼻喉", "皮肤科", "神经科",
        "消化科", "呼吸科", "心内科", "内分泌", "骨科", "泌尿",
        # 身体部位
        "心脏", "肝脏", "肾脏", "肺部", "胃", "肠道", "血管", "神经",
        "骨骼", "肌肉", "皮肤", "眼睛", "耳朵", "口腔", "甲状腺",
        # 检查/指标
        "血糖", "血压", "血脂", "尿酸", "转氨酶", "肌酐", "CT", "MRI",
        "B超", "心电图", "血常规", "尿常规",
    ]
    filtered = []
    for r in records:
        text = r.get("question", "") + r.get("answer", "")
        if any(term in text for term in medical_terms):
            filtered.append(r)
    logger.info(f"Medical keyword filter: {len(records)} -> {len(filtered)}")
    return filtered


def deduplicate_by_question(records: list[dict], threshold: float = 0.85) -> list[dict]:
    """Remove near-duplicate questions using simple character Jaccard similarity.

    For serious dedup, use MinHash or embedding similarity.
    """
    def char_ngrams(text: str, n: int = 3) -> set:
        text = text.lower().replace(" ", "")
        return {text[i:i + n] for i in range(len(text) - n + 1)}

    ngrams_list = [char_ngrams(r.get("question", "")) for r in records]
    keep_indices = []
    seen = set()

    for i, ng in enumerate(ngrams_list):
        if i in seen:
            continue
        keep_indices.append(i)
        for j in range(i + 1, len(ngrams_list)):
            if j in seen:
                continue
            intersection = len(ng & ngrams_list[j])
            union = len(ng | ngrams_list[j])
            if union > 0 and intersection / union > threshold:
                seen.add(j)

    filtered = [records[i] for i in keep_indices]
    logger.info(f"Deduplication: {len(records)} -> {len(filtered)}")
    return filtered


def vector_recall(
    records: list[dict],
    anchors: list[dict],
    top_k: int = 5000,
    model_name: str = "BAAI/bge-small-zh-v1.5",
    similarity_threshold: float = 0.5,
) -> list[dict]:
    """Use embedding model to recall data similar to anchors.

    This is the core of the "data distribution alignment" strategy.
    Instead of using eval sets for recall (which would be cheating),
    we use self-defined anchors that represent the desired task distribution.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.warning("sentence_transformers not available, falling back to random sampling")
        return random.sample(records, min(top_k, len(records)))

    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    # Encode anchors
    anchor_texts = [a["question"] for a in anchors]
    logger.info(f"Encoding {len(anchor_texts)} anchors...")
    anchor_embeddings = model.encode(anchor_texts, show_progress_bar=True)
    anchor_mean = anchor_embeddings.mean(axis=0, keepdims=True)

    # Encode candidate questions
    candidate_texts = [r["question"] for r in records]
    logger.info(f"Encoding {len(candidate_texts)} candidates...")
    candidate_embeddings = model.encode(candidate_texts, show_progress_bar=True)

    # Compute cosine similarity to anchor mean
    anchor_norm = anchor_mean / np.linalg.norm(anchor_mean, axis=1, keepdims=True)
    candidate_norm = candidate_embeddings / np.linalg.norm(
        candidate_embeddings, axis=1, keepdims=True
    )
    similarities = np.dot(candidate_norm, anchor_norm.T).flatten()

    # Select top-k above threshold
    qualified = np.where(similarities >= similarity_threshold)[0]
    logger.info(f"{len(qualified)} records above similarity threshold {similarity_threshold}")

    if len(qualified) <= top_k:
        selected_indices = qualified.tolist()
    else:
        selected_indices = qualified[
            np.argsort(similarities[qualified])[-top_k:]
        ].tolist()

    recalled = [records[i] for i in selected_indices]
    logger.info(f"Vector recall: {len(records)} -> {len(recalled)}")
    return recalled


def split_train_valid_test(
    records: list[dict],
    train_ratio: float = 0.85,
    valid_ratio: float = 0.10,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Split records into train/valid/test sets."""
    random.seed(seed)
    shuffled = records.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    valid_end = train_end + int(n * valid_ratio)

    splits = {
        "train": shuffled[:train_end],
        "valid": shuffled[train_end:valid_end],
        "test": shuffled[valid_end:],
    }
    for name, subset in splits.items():
        logger.info(f"  {name}: {len(subset)} records")
    return splits


def save_splits(splits: dict[str, list[dict]], output_dir: Path) -> None:
    """Save splits to JSONL files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, records in splits.items():
        path = output_dir / f"medical_{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info(f"Saved {path} ({len(records)} records)")


def main(
    raw_data_dir: Optional[Path] = None,
    anchor_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    use_vector_recall: bool = True,
    recall_top_k: int = 10000,
):
    raw_data_dir = raw_data_dir or RAW_DIR
    output_dir = output_dir or PROCESSED_DIR

    # 1. Load anchors
    anchors = load_anchors(anchor_path)

    # 2. Load raw data
    records = load_raw_data(raw_data_dir)
    if not records:
        logger.error("No raw data found. Please download data to data/raw/ first.")
        logger.info("Suggested: shibing624/medical on HuggingFace")
        return

    # 3. Normalize format
    records = [normalize_record(r) for r in records]
    valid = [r for r in records if r.get("question") and r.get("answer")]
    logger.info(f"After normalization: {len(valid)} valid records")

    # 4. Quality filtering
    filtered = length_filter(valid)
    filtered = medical_keyword_filter(filtered)
    filtered = deduplicate_by_question(filtered)

    # 5. Vector recall (distribution alignment)
    if use_vector_recall:
        filtered = vector_recall(filtered, anchors, top_k=recall_top_k)

    # 6. Split and save
    splits = split_train_valid_test(filtered)
    save_splits(splits, output_dir)

    logger.info("Done. Processed data saved to {}", output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filter and select medical QA data")
    parser.add_argument("--raw_data_dir", type=Path, default=None)
    parser.add_argument("--anchor_path", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--no_vector_recall", action="store_true")
    parser.add_argument("--recall_top_k", type=int, default=10000)
    args = parser.parse_args()

    main(
        raw_data_dir=args.raw_data_dir,
        anchor_path=args.anchor_path,
        output_dir=args.output_dir,
        use_vector_recall=not args.no_vector_recall,
        recall_top_k=args.recall_top_k,
    )
