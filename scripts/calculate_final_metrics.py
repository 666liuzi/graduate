import argparse
import os
import sys
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from utils.pipeline import (
    canonicalize_label,
    infer_routing_bucket,
    load_food101_classes,
    load_json,
    normalize_result_key,
    resolve_candidate_index,
)


def calculate_all_metrics(
    fusion_path="./results/final_fusion_results.json",
    baseline2_path="./results/baseline2_results.json",
    data_dir="./data",
    logger=None,
):
    if logger is None:
        logger = setup_logger("calculate_final_metrics")

    fusion_data = load_json(fusion_path)
    classes_list = _load_or_infer_classes(fusion_data, data_dir)
    baseline2_raw = load_json(baseline2_path) if os.path.exists(baseline2_path) else {}
    baseline2_data = {normalize_result_key(path): value for path, value in baseline2_raw.items()}

    total_samples = len(fusion_data)
    correct_exp1 = 0
    correct_exp2 = 0
    correct_exp3 = 0
    correct_cloud_only = 0
    broken_locks = 0
    baseline2_overlap = 0
    baseline2_missing = 0
    final_source_counter = Counter()
    routing_bucket_counter = Counter()
    qwen_invalid_counter = Counter()
    deepseek_invalid_counter = Counter()
    reasoner_decision_counter = Counter()
    edge_override_counter = Counter()
    reasoner_called_counter = Counter()
    bucket_metrics = {}

    t_threshold = 0.8
    for img_path, sample in fusion_data.items():
        normalized_path = normalize_result_key(img_path)
        true_idx = int(sample["true_label"])
        true_name = classes_list[true_idx]

        exp1_pred_idx = sample["edge_pred"] if float(sample["edge_conf"]) >= t_threshold else sample["cloud_pred"]
        if int(exp1_pred_idx) == true_idx:
            correct_exp1 += 1

        cloud_only_label = classes_list[int(sample["cloud_pred"])]
        if cloud_only_label == true_name:
            correct_cloud_only += 1

        baseline2_sample = baseline2_data.get(normalized_path)
        if baseline2_sample is not None:
            baseline2_overlap += 1
            exp2_pred_name = canonicalize_label(str(baseline2_sample.get("baseline2_pred", "unknown")).strip())
            if exp2_pred_name == true_name:
                correct_exp2 += 1
        else:
            baseline2_missing += 1

        final_pred_index = sample.get("final_pred_index")
        if final_pred_index is not None:
            final_label = classes_list[int(final_pred_index)]
        else:
            final_label = canonicalize_label(
                str(sample.get("final_label") or sample.get("final_fusion_pred", "unknown"))
            )
        if final_label == true_name:
            correct_exp3 += 1

        if sample.get("lockable") and int(sample.get("final_pred_index", sample["cloud_pred"])) != int(sample["cloud_pred"]):
            broken_locks += 1

        qwen_pred_index = resolve_candidate_index(
            sample.get("candidate_entries", []),
            sample.get("qwen_choice"),
            fallback_index=sample["cloud_pred"],
        )
        routing_bucket = str(
            sample.get(
                "routing_bucket",
                infer_routing_bucket(sample["edge_pred"], sample["cloud_pred"], qwen_pred_index),
            )
        )
        bucket_record = bucket_metrics.setdefault(
            routing_bucket,
            {"total": 0, "edge": 0, "cloud": 0, "qwen": 0, "final": 0},
        )
        bucket_record["total"] += 1
        bucket_record["edge"] += int(int(sample["edge_pred"]) == true_idx)
        bucket_record["cloud"] += int(int(sample["cloud_pred"]) == true_idx)
        bucket_record["qwen"] += int(int(qwen_pred_index) == true_idx)
        bucket_record["final"] += int(final_label == true_name)

        final_source_counter[str(sample.get("final_source", "unknown"))] += 1
        routing_bucket_counter[routing_bucket] += 1
        if sample.get("qwen_invalid"):
            qwen_invalid_counter[str(sample.get("qwen_invalid_reason", "unknown"))] += 1
        if sample.get("deepseek_invalid"):
            deepseek_invalid_counter[str(sample.get("deepseek_invalid_reason", "unknown"))] += 1
        if sample.get("edge_override_applied"):
            edge_override_counter["true"] += 1
        if sample.get("deepseek_api_called"):
            reasoner_called_counter["true"] += 1
        reasoner_decision_counter[str(sample.get("deepseek_decision", "SKIPPED"))] += 1

    acc_exp1 = (correct_exp1 / total_samples) * 100 if total_samples else 0.0
    acc_exp2 = (correct_exp2 / total_samples) * 100 if total_samples else 0.0
    acc_exp3 = (correct_exp3 / total_samples) * 100 if total_samples else 0.0
    acc_cloud_only = (correct_cloud_only / total_samples) * 100 if total_samples else 0.0

    lines = [
        "",
        "=" * 56,
        "最终答辩核心指标对比",
        "=" * 56,
        f"cloud-only 基线: {acc_cloud_only:.2f}%",
        f"实验(1) [纯视觉协同, T={t_threshold}]: {acc_exp1:.2f}%",
        f"实验(2) [纯多模态+LLM 基线]: {acc_exp2:.2f}%",
        f"实验(3) [当前融合结果]: {acc_exp3:.2f}%",
        f"锁定样本被改坏数量: {broken_locks}",
        f"baseline2_key_overlap: {baseline2_overlap}/{total_samples}",
        f"baseline2_missing_count: {baseline2_missing}",
        f"final_source 分布: {dict(final_source_counter)}",
        f"routing_bucket 分布: {dict(routing_bucket_counter)}",
        f"qwen_invalid 分布: {dict(qwen_invalid_counter)}",
        f"deepseek_invalid 分布: {dict(deepseek_invalid_counter)}",
        f"edge_override_count: {edge_override_counter.get('true', 0)}",
        f"deepseek_reasoner_called_count: {reasoner_called_counter.get('true', 0)}",
        f"deepseek_reasoner_decision 分布: {dict(reasoner_decision_counter)}",
        "=" * 56,
    ]

    for bucket_name in ("agreement", "qwen_eq_cloud", "qwen_eq_edge", "all_different"):
        bucket_record = bucket_metrics.get(bucket_name)
        if not bucket_record or not bucket_record["total"]:
            continue
        total = bucket_record["total"]
        lines.append(
            (
                f"bucket={bucket_name} | total={total} | "
                f"edge={bucket_record['edge'] / total * 100:.2f}% | "
                f"cloud={bucket_record['cloud'] / total * 100:.2f}% | "
                f"qwen={bucket_record['qwen'] / total * 100:.2f}% | "
                f"final={bucket_record['final'] / total * 100:.2f}%"
            )
        )

    for line in lines:
        print(line)
    logger.info("\n".join(lines[1:]))


def _load_or_infer_classes(fusion_data, data_dir):
    try:
        return load_food101_classes(data_dir=data_dir)
    except Exception:
        inferred = {}
        for img_path, sample in fusion_data.items():
            normalized_path = normalize_result_key(img_path)
            class_name = canonicalize_label(normalized_path.split("/")[-2])
            inferred.setdefault(int(sample["true_label"]), class_name)

        if not inferred:
            raise

        max_index = max(inferred)
        return [inferred[idx] for idx in range(max_index + 1)]


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate final metrics for the fusion pipeline.")
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
