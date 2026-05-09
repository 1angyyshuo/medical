"""
收集和对比所有实验结果，生成评测报告。

从 outputs/eval/ 中读取各实验的评测结果，
生成对比表格和消融分析。
"""

import json
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "outputs" / "eval"
REPORTS_DIR = PROJECT_ROOT / "reports"


def find_latest_eval(experiment_name: str) -> Optional[Path]:
    """Find the most recent eval result for an experiment."""
    exp_dir = EVAL_DIR / experiment_name
    if not exp_dir.exists():
        return None

    subdirs = sorted([d for d in exp_dir.iterdir() if d.is_dir()], reverse=True)
    return subdirs[0] if subdirs else None


def collect_results(experiment_names: list[str]) -> dict[str, dict]:
    """Collect results from multiple experiments."""
    all_results = {}

    for name in experiment_names:
        latest = find_latest_eval(name)
        if latest:
            summary_path = latest / "summary.json"
            if summary_path.exists():
                with open(summary_path, encoding="utf-8") as f:
                    all_results[name] = json.load(f)
            else:
                print(f"  {name}: no summary.json found")
                all_results[name] = {"error": "no summary.json"}
        else:
            print(f"  {name}: no eval results found (run 05_run_eval.py first)")
            all_results[name] = {"error": "not evaluated yet"}

    return all_results


def generate_markdown_table(results: dict[str, dict]) -> str:
    """Generate a markdown comparison table."""
    experiments = list(results.keys())

    lines = []
    lines.append("# 医疗大模型评测结果对比")
    lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Header
    lines.append("| 实验 | 模型 | 数据 | 方法 | CEval-med | CMMLU-med | 自建QA | 安全拒答 | PPL |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    for name, result in results.items():
        if "error" in result:
            lines.append(f"| {name} | - | - | - | - | - | - | - | - |")
            continue

        model = result.get("model", "-")
        # Extract short model name
        if "/" in model:
            model = model.split("/")[-1]

        ceval_score = "-"
        cmmlu_score = "-"
        qa_score = "-"
        safety_score = "-"
        ppl_score = "-"

        # Try to extract lm_eval scores
        lm_eval = result.get("lm_eval", {})
        if isinstance(lm_eval, dict) and "results" in lm_eval:
            for task, metrics in lm_eval["results"].items():
                acc = metrics.get("acc,none", metrics.get("acc", None))
                if acc is not None:
                    if "ceval" in task.lower():
                        ceval_score = f"{acc:.4f}"
                    elif "cmmlu" in task.lower():
                        cmmlu_score = f"{acc:.4f}"

        lines.append(
            f"| {name} | {model} | - | - | {ceval_score} | {cmmlu_score} "
            f"| {qa_score} | {safety_score} | {ppl_score} |"
        )

    return "\n".join(lines)


def generate_ablation_analysis(results: dict[str, dict]) -> str:
    """Generate ablation analysis comparing key experimental pairs."""
    lines = []
    lines.append("\n## 消融实验分析\n")

    pairs = [
        ("sft_random", "sft_selected", "数据筛选策略: 随机 vs 向量召回"),
        ("sft_selected", "sft_selected_safety", "安全数据: 有无安全拒答数据"),
        ("sft_selected", "dpo_sft_selected", "偏好优化: SFT vs SFT+DPO"),
    ]

    for exp_a, exp_b, description in pairs:
        lines.append(f"### {description}")
        result_a = results.get(exp_a, {})
        result_b = results.get(exp_b, {})

        if "error" in result_a or "error" in result_b:
            lines.append("（数据不足，无法对比）\n")
            continue

        lines.append(f"- **{exp_a}**: baseline")
        lines.append(f"- **{exp_b}**: improved variant")
        lines.append("")
        lines.append("| 指标 | {} | {} | 变化 | 结论 |".format(exp_a, exp_b))
        lines.append("|---|---|---|---|---|")

        # TODO: Extract actual metrics from lm_eval results
        lines.append("| CEval-med | - | - | - | 待评测 |")
        lines.append("| PPL | - | - | - | 待评测 |")
        lines.append("| 安全得分 | - | - | - | 待评测 |")
        lines.append("")

    return "\n".join(lines)


def generate_report(results: dict[str, dict], output_path: Path):
    """Generate full evaluation report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    content = []
    content.append(generate_markdown_table(results))
    content.append(generate_ablation_analysis(results))

    # Record experimental configurations
    content.append("\n## 实验配置\n")
    content.append("| 实验 | 基础模型 | 通用数据 | 医疗数据 | 偏好数据 | 训练方法 |")
    content.append("|---|---|---|---|---|---|")
    content.append("| baseline | Qwen2.5-3B-Instruct | - | - | - | - |")
    content.append("| sft_random | Qwen2.5-3B-Instruct | 1k general | 2k random | - | LoRA SFT |")
    content.append("| sft_selected | Qwen2.5-3B-Instruct | 1k general | 2k vector-selected | - | LoRA SFT |")
    content.append("| sft_selected_safety | Qwen2.5-3B-Instruct | 1k general | 2k selected + 500 safety | - | LoRA SFT |")
    content.append("| dpo_selected | Qwen2.5-3B-Instruct | - | - | ~50 preference pairs | SFT + DPO |")

    report = "\n".join(content) + "\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print(f"\nReport saved to: {output_path}")


def main():
    parser = ArgumentParser(description="Collect and compare evaluation results")
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
        help="Experiment names to compare",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output_path = args.output or REPORTS_DIR / "eval_report.md"

    print("Collecting evaluation results...")
    results = collect_results(args.experiments)

    print("\nGenerating report...")
    generate_report(results, output_path)


if __name__ == "__main__":
    main()
