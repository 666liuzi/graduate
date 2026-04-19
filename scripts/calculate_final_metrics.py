import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.pipeline import canonicalize_label, load_food101_classes, load_json


def calculate_all_metrics(
    fusion_path="./results/final_fusion_results.json",
    baseline2_path="./results/baseline2_results.json",
    data_dir="./data",
):
    classes_list = load_food101_classes(data_dir=data_dir)
    fusion_data = load_json(fusion_path)
    baseline2_data = load_json(baseline2_path) if os.path.exists(baseline2_path) else {}

    total_samples = len(fusion_data)
    correct_exp1 = 0
    correct_exp2 = 0
    correct_exp3 = 0
    broken_locks = 0

    t_threshold = 0.8
    for img_path, sample in fusion_data.items():
        true_idx = int(sample["true_label"])
        true_name = classes_list[true_idx]

        exp1_pred_idx = sample["edge_pred"] if float(sample["edge_conf"]) >= t_threshold else sample["cloud_pred"]
        if int(exp1_pred_idx) == true_idx:
            correct_exp1 += 1

        if baseline2_data:
            exp2_pred_name = canonicalize_label(
                str(baseline2_data.get(img_path, {}).get("baseline2_pred", "unknown")).strip()
            )
            if exp2_pred_name == true_name:
                correct_exp2 += 1

        final_label = canonicalize_label(
            str(sample.get("final_label") or sample.get("final_fusion_pred", "unknown"))
        )
        if final_label == true_name:
            correct_exp3 += 1

        if sample.get("lockable") and final_label != classes_list[int(sample["cloud_pred"])]:
            broken_locks += 1

    acc_exp1 = (correct_exp1 / total_samples) * 100 if total_samples else 0.0
    acc_exp2 = (correct_exp2 / total_samples) * 100 if total_samples and baseline2_data else 0.0
    acc_exp3 = (correct_exp3 / total_samples) * 100 if total_samples else 0.0

    print("\n" + "=" * 48)
    print("最终答辩核心指标对比")
    print("=" * 48)
    print(f"实验(1) [纯视觉协同, T={t_threshold}]: {acc_exp1:.2f}%")
    if baseline2_data:
        print(f"实验(2) [纯多模态+LLM 基线]: {acc_exp2:.2f}%")
    else:
        print("实验(2) [纯多模态+LLM 基线]: 未提供 baseline2_results.json")
    print(f"实验(3) [闭集保守融合]: {acc_exp3:.2f}%")
    print(f"锁定样本被改坏数量: {broken_locks}")
    print("=" * 48)


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate final metrics for the closed-set fusion pipeline.")
    parser.add_argument("--fusion", default="./results/final_fusion_results.json")
    parser.add_argument("--baseline2", default="./results/baseline2_results.json")
    parser.add_argument("--data_dir", default="./data")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    calculate_all_metrics(
        fusion_path=cli_args.fusion,
        baseline2_path=cli_args.baseline2,
        data_dir=cli_args.data_dir,
    )
