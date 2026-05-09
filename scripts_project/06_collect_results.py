"""
收集和对比所有实验的 CEval 医学子集评测结果。

评估维度:
- basic_medicine (基础医学) accuracy
- clinical_medicine (临床医学) accuracy
- physician (医师资格) accuracy
- overall (综合) accuracy

生成 Markdown 对比报告 + 消融分析。
"""

import json
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "outputs" / "eval"
REPORTS_DIR = PROJECT_ROOT / "reports"

MEDICAL_SUBSETS = ["basic_medicine", "clinical_medicine", "physician"]


# ---- 实验配置说明 ----
EXPERIMENT_INFO = {
    "baseline": {
        "model": "Qwen2.5-3B-Instruct",
        "data": "-",
        "method": "-",
        "description": "基座模型原始分数",
    },
    "sft_random": {
        "model": "Qwen2.5-3B-Instruct",
        "data": "1k general + 2k random medical",
        "method": "LoRA SFT",
        "description": "随机采样医疗数据 (对照组)",
    },
    "sft_selected": {
        "model": "Qwen2.5-3B-Instruct",
        "data": "1k general + 2k vector-selected medical",
        "method": "LoRA SFT",
        "description": "向量召回筛选医疗数据 (实验组)",
    },
    "sft_selected_safety": {
        "model": "Qwen2.5-3B-Instruct",
        "data": "selected + 500 safety refusals",
        "method": "LoRA SFT",
        "description": "加安全拒答数据 (消融组)",
    },
    "dpo_selected": {
        "model": "Qwen2.5-3B-Instruct",
        "data": "preference pairs (~50)",
        "method": "SFT + DPO",
        "description": "偏好优化后 (DPO)",
    },
}


def find_latest_eval(experiment_name: str) -> Optional[Path]:
    """Find the most recent evaluation directory for an experiment."""
    exp_dir = EVAL_DIR / experiment_name
    if not exp_dir.exists():
        return None
    subdirs = sorted([d for d in exp_dir.iterdir() if d.is_dir()], reverse=True)
    return subdirs[0] if subdirs else None


def load_summary(experiment_name: str) -> Optional[dict]:
    """Load the summary.json from the latest eval of an experiment."""
    latest = find_latest_eval(experiment_name)
    if not latest:
        return None

    summary_path = latest / "summary.json"
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            return json.load(f)
    return None


def format_pct(value: Optional[float]) -> str:
    """Format accuracy as percentage string."""
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def generate_report(experiments: list[str]) -> str:
    """Generate a complete Markdown evaluation report."""
    lines = []
    lines.append("# 医疗大模型 CEval 医学子集评测报告")
    lines.append(f"\n> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"> 评测数据集: [ceval/ceval-exam](https://huggingface.co/datasets/ceval/ceval-exam)")
    lines.append(f"> 医学子集: {', '.join(MEDICAL_SUBSETS)}")
    lines.append("")

    # ---- 1. 主结果表 ----
    lines.append("## 1. 评测结果总览\n")
    lines.append("| 实验 | basic_medicine | clinical_medicine | physician | **综合** |")
    lines.append("|---|---:|---:|---:|---:|")

    all_summaries = {}
    for exp_name in experiments:
        summary = load_summary(exp_name)
        all_summaries[exp_name] = summary

        if summary is None:
            lines.append(f"| {exp_name} | - | - | - | - |")
            continue

        subsets = summary.get("subsets", {})
        overall = summary.get("overall", {})

        bm = format_pct(subsets.get("basic_medicine", {}).get("accuracy"))
        cm = format_pct(subsets.get("clinical_medicine", {}).get("accuracy"))
        ph = format_pct(subsets.get("physician", {}).get("accuracy"))
        ov = format_pct(overall.get("accuracy"))

        lines.append(f"| {exp_name} | {bm} | {cm} | {ph} | **{ov}** |")

    lines.append("")

    # ---- 2. 详细数据表 ----
    lines.append("## 2. 详细数据 (correct/total)\n")
    lines.append("| 实验 | basic_medicine | clinical_medicine | physician | 综合 |")
    lines.append("|---|---|---|---|---|")

    for exp_name in experiments:
        summary = all_summaries.get(exp_name)
        if summary is None:
            lines.append(f"| {exp_name} | - | - | - | - |")
            continue

        subsets = summary.get("subsets", {})
        overall = summary.get("overall", {})

        def fmt_detail(subset_name: str) -> str:
            s = subsets.get(subset_name, {})
            return f"{s.get('correct', 0)}/{s.get('total', 0)}"

        bm = fmt_detail("basic_medicine")
        cm = fmt_detail("clinical_medicine")
        ph = fmt_detail("physician")
        ov = f"{overall.get('correct', 0)}/{overall.get('total', 0)}"

        lines.append(f"| {exp_name} | {bm} | {cm} | {ph} | {ov} |")

    lines.append("")

    # ---- 3. 消融实验分析 ----
    lines.append("## 3. 消融实验分析\n")

    ablation_pairs = [
        ("baseline", "sft_selected", "SFT 微调有效性: 基座模型 vs SFT 后"),
        ("sft_random", "sft_selected", "数据筛选策略: 随机采样 vs 向量召回"),
        ("sft_selected", "sft_selected_safety", "安全数据贡献: 有无安全拒答训练"),
        ("sft_selected", "dpo_selected", "偏好优化收益: SFT vs SFT+DPO"),
    ]

    for exp_a, exp_b, description in ablation_pairs:
        lines.append(f"### {description}\n")
        sum_a = all_summaries.get(exp_a)
        sum_b = all_summaries.get(exp_b)

        if sum_a is None or sum_b is None:
            lines.append("(数据不足，无法对比)\n")
            continue

        lines.append("| 指标 | {} | {} | 变化 |".format(exp_a, exp_b))
        lines.append("|---|---|---|---|")

        for subset_name in MEDICAL_SUBSETS:
            acc_a = sum_a.get("subsets", {}).get(subset_name, {}).get("accuracy")
            acc_b = sum_b.get("subsets", {}).get(subset_name, {}).get("accuracy")

            if acc_a is not None and acc_b is not None:
                delta = acc_b - acc_a
                direction = "+" if delta > 0 else ""
                lines.append(
                    f"| {subset_name} | {format_pct(acc_a)} | {format_pct(acc_b)} "
                    f"| {direction}{delta*100:.1f}% |"
                )

        # Overall
        ov_a = sum_a.get("overall", {}).get("accuracy")
        ov_b = sum_b.get("overall", {}).get("accuracy")
        if ov_a is not None and ov_b is not None:
            delta = ov_b - ov_a
            direction = "+" if delta > 0 else ""
            lines.append(
                f"| **综合** | **{format_pct(ov_a)}** | **{format_pct(ov_b)}** "
                f"| **{direction}{delta*100:.1f}%** |"
            )

        lines.append("")

    # ---- 4. 实验配置 ----
    lines.append("## 4. 实验配置\n")
    lines.append("| 实验 | 模型 | 训练数据 | 训练方法 | 说明 |")
    lines.append("|---|---|---|---|---|")
    for exp_name in experiments:
        info = EXPERIMENT_INFO.get(exp_name, {})
        lines.append(
            f"| {exp_name} | {info.get('model', '-')} | {info.get('data', '-')} "
            f"| {info.get('method', '-')} | {info.get('description', '-')} |"
        )

    lines.append("")
    lines.append("---")
    lines.append(f"*报告由 `scripts_project/06_collect_results.py` 自动生成*")

    return "\n".join(lines)


def main():
    parser = ArgumentParser(description="Collect and compare CEval medical evaluation results")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=[
            "baseline",
            "sft_random",
            "sft_selected",
            "sft_selected_safety",
            "dpo_selected",
        ],
        help="Experiment names to include in the report",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output_path = args.output or REPORTS_DIR / "eval_report.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = generate_report(args.experiments)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
