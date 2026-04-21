import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from utils.pipeline import load_json, normalize_result_key, save_json


FUSION_RESET_FIELDS = [
    "final_label",
    "final_pred_index",
    "final_source",
    "deepseek_choice",
    "deepseek_choice_label",
    "deepseek_invalid",
    "deepseek_invalid_reason",
    "deepseek_raw_output",
    "forced_closed_set",
]


def prepare_retry_cache_reset(
    qwen_input="./results/qwen_rerank_results_v2.json",
    fusion_input="./results/final_fusion_results_v2.json",
    qwen_output="./results/qwen_rerank_results_v2_retry2838.json",
    fusion_output="./results/final_fusion_results_v2_retry2838.json",
    retry_keys_output="./results/retry_keys_2838.txt",
    invalid_reason="candidate_out_of_range",
    limit=None,
    dry_run=False,
    logger=None,
):
    if logger is None:
        logger = setup_logger("prepare_retry_cache_reset")

    qwen_results = load_json(qwen_input)
    fusion_results = load_json(fusion_input)

    retry_keys = _select_retry_keys(qwen_results, invalid_reason=invalid_reason, limit=limit)
    retry_key_set = {normalize_result_key(key) for key in retry_keys}
    fusion_missing = []

    qwen_retry_results = {}
    qwen_reset_count = 0
    for raw_key, sample in qwen_results.items():
        normalized_key = normalize_result_key(raw_key)
        updated_sample = dict(sample)
        if normalized_key in retry_key_set:
            updated_sample["qwen_completed"] = False
            qwen_reset_count += 1
        qwen_retry_results[raw_key] = updated_sample

    fusion_retry_results = {}
    fusion_reset_count = 0
    for raw_key, sample in fusion_results.items():
        normalized_key = normalize_result_key(raw_key)
        updated_sample = dict(sample)
        if normalized_key in retry_key_set:
            for field in FUSION_RESET_FIELDS:
                updated_sample.pop(field, None)
            fusion_reset_count += 1
        fusion_retry_results[raw_key] = updated_sample

    fusion_key_set = {normalize_result_key(key) for key in fusion_results}
    for retry_key in retry_keys:
        if normalize_result_key(retry_key) not in fusion_key_set:
            fusion_missing.append(retry_key)

    logger.info(
        f"Retry cache reset prepared | invalid_reason={invalid_reason} | selected={len(retry_keys)} | "
        f"qwen_reset={qwen_reset_count} | fusion_reset={fusion_reset_count} | fusion_missing={len(fusion_missing)} | "
        f"dry_run={dry_run}"
    )

    if retry_keys:
        preview = ", ".join(retry_keys[:3])
        logger.info(f"Retry key preview: {preview}")

    if fusion_missing:
        logger.info(f"Fusion missing preview: {', '.join(fusion_missing[:3])}")

    if dry_run:
        return {
            "retry_keys": retry_keys,
            "qwen_reset_count": qwen_reset_count,
            "fusion_reset_count": fusion_reset_count,
            "fusion_missing": fusion_missing,
        }

    save_json(qwen_retry_results, qwen_output)
    save_json(fusion_retry_results, fusion_output)
    _save_retry_keys(retry_keys, retry_keys_output)

    logger.info(f"Saved qwen retry file to {qwen_output}")
    logger.info(f"Saved fusion retry file to {fusion_output}")
    logger.info(f"Saved retry keys to {retry_keys_output}")

    return {
        "retry_keys": retry_keys,
        "qwen_reset_count": qwen_reset_count,
        "fusion_reset_count": fusion_reset_count,
        "fusion_missing": fusion_missing,
    }


def _select_retry_keys(qwen_results, invalid_reason, limit=None):
    retry_keys = [
        raw_key
        for raw_key, sample in qwen_results.items()
        if str(sample.get("qwen_invalid_reason", "")) == invalid_reason
    ]
    if limit is not None:
        retry_keys = retry_keys[:limit]
    return retry_keys


def _save_retry_keys(retry_keys, path):
    output_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for key in retry_keys:
            handle.write(f"{normalize_result_key(key)}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare retry copies of qwen/fusion result files by clearing cache fields."
    )
    parser.add_argument("--qwen_input", default="./results/qwen_rerank_results_v2.json")
    parser.add_argument("--fusion_input", default="./results/final_fusion_results_v2.json")
    parser.add_argument("--qwen_output", default="./results/qwen_rerank_results_v2_retry2838.json")
    parser.add_argument("--fusion_output", default="./results/final_fusion_results_v2_retry2838.json")
    parser.add_argument("--retry_keys_output", default="./results/retry_keys_2838.txt")
    parser.add_argument("--invalid_reason", default="candidate_out_of_range")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    prepare_retry_cache_reset(
        qwen_input=cli_args.qwen_input,
        fusion_input=cli_args.fusion_input,
        qwen_output=cli_args.qwen_output,
        fusion_output=cli_args.fusion_output,
        retry_keys_output=cli_args.retry_keys_output,
        invalid_reason=cli_args.invalid_reason,
        limit=cli_args.limit,
        dry_run=cli_args.dry_run,
    )
