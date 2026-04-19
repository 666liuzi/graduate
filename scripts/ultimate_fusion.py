import argparse
import os
import sys

from openai import OpenAI
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.pipeline import (
    candidate_map_by_key,
    extract_json_object,
    normalize_candidate_choice,
    load_json,
    save_json,
    should_flag_ood,
)


DEFAULT_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")


def run_ultimate_fusion(
    rerank_path="./results/qwen_rerank_results.json",
    output_path="./results/final_fusion_results.json",
    api_key=None,
    base_url=DEFAULT_BASE_URL,
):
    client = None
    if api_key:
        client = OpenAI(api_key=api_key, base_url=base_url)

    all_results = load_json(rerank_path)
    saved_results = load_json(output_path) if os.path.exists(output_path) else {}

    merged_results = {}
    for img_path, sample in tqdm(all_results.items()):
        cached = saved_results.get(img_path)
        if cached and "final_label" in cached:
            merged_results[img_path] = cached
            continue

        merged_results[img_path] = _finalize_sample(sample, client)

        if len(merged_results) % 500 == 0:
            save_json(merged_results, output_path)

    save_json(merged_results, output_path)
    return merged_results


def _finalize_sample(sample, client):
    candidate_entries = sample.get("candidate_entries", [])
    candidate_by_key = candidate_map_by_key(candidate_entries)
    cloud_key = sample.get("cloud_candidate_key")
    if not candidate_entries or cloud_key not in candidate_by_key:
        cloud_label = sample.get("locked_label") or sample.get("qwen_choice_label") or ""
        return {
            **sample,
            "deepseek_choice": cloud_key or "",
            "deepseek_choice_label": cloud_label,
            "deepseek_invalid": True,
            "deepseek_invalid_reason": "candidate_entries_missing",
            "deepseek_raw_output": "",
            "final_label": cloud_label,
            "final_pred_index": sample.get("cloud_pred"),
            "final_source": "cloud_fallback_invalid_candidates",
            "forced_closed_set": bool(sample.get("ood_flag", False)),
        }

    cloud_entry = candidate_by_key[cloud_key]
    cloud_label = cloud_entry["label"]
    cloud_index = int(cloud_entry["index"])

    result = dict(sample)
    result.setdefault("qwen_choice", cloud_key)
    result.setdefault("qwen_choice_label", cloud_label)
    result.setdefault("ood_score", 0)
    result["ood_flag"] = bool(sample.get("ood_flag", should_flag_ood(sample.get("ood_score", 0))))

    if sample.get("lockable") or sample.get("edge_pred") == sample.get("cloud_pred"):
        result.update(
            {
                "deepseek_choice": cloud_key,
                "deepseek_choice_label": cloud_label,
                "deepseek_invalid": False,
                "deepseek_invalid_reason": "",
                "deepseek_raw_output": "",
                "final_label": cloud_label,
                "final_pred_index": cloud_index,
                "final_source": "visual_lock",
                "forced_closed_set": False,
            }
        )
        return result

    deepseek_choice, deepseek_invalid, deepseek_invalid_reason, deepseek_raw_output = _resolve_deepseek_choice(
        sample=sample,
        candidate_by_key=candidate_by_key,
        cloud_key=cloud_key,
        client=client,
    )

    chosen_entry = candidate_by_key[deepseek_choice]
    final_label = cloud_label
    final_pred_index = cloud_index
    final_source = "cloud_fallback"

    qwen_choice = sample.get("qwen_choice")
    qwen_best_support = int(sample.get("qwen_best_support", 0))
    qwen_cloud_support = int(sample.get("qwen_cloud_support", 0))
    ood_score = int(sample.get("ood_score", 0))

    if (
        not sample.get("qwen_invalid", False)
        and not deepseek_invalid
        and qwen_choice == deepseek_choice
        and qwen_choice != cloud_key
        and qwen_best_support >= 75
        and (qwen_best_support - qwen_cloud_support) >= 15
        and ood_score <= 1
    ):
        final_label = chosen_entry["label"]
        final_pred_index = int(chosen_entry["index"])
        final_source = "qwen_deepseek_agreement"

    forced_closed_set = bool(result["ood_flag"] and final_label == cloud_label)
    result.update(
        {
            "deepseek_choice": deepseek_choice,
            "deepseek_choice_label": chosen_entry["label"],
            "deepseek_invalid": deepseek_invalid,
            "deepseek_invalid_reason": deepseek_invalid_reason,
            "deepseek_raw_output": deepseek_raw_output,
            "final_label": final_label,
            "final_pred_index": final_pred_index,
            "final_source": final_source,
            "forced_closed_set": forced_closed_set,
        }
    )
    return result


def _resolve_deepseek_choice(sample, candidate_by_key, cloud_key, client):
    if sample.get("qwen_invalid", False):
        return cloud_key, True, "qwen_invalid", ""

    qwen_choice = sample.get("qwen_choice")
    qwen_best_support = int(sample.get("qwen_best_support", 0))
    qwen_cloud_support = int(sample.get("qwen_cloud_support", 0))
    ood_score = int(sample.get("ood_score", 0))

    if qwen_choice == cloud_key:
        return cloud_key, False, "", ""
    if qwen_best_support < 75:
        return cloud_key, False, "qwen_support_below_threshold", ""
    if (qwen_best_support - qwen_cloud_support) < 15:
        return cloud_key, False, "qwen_margin_below_threshold", ""
    if ood_score > 1:
        return cloud_key, False, "ood_guard_triggered", ""
    if client is None:
        return cloud_key, True, "api_key_missing", ""

    prompt = _build_deepseek_prompt(sample, candidate_by_key, cloud_key)
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个只输出候选 JSON 的闭集仲裁器。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=64,
            stream=False,
        )
        raw_output = response.choices[0].message.content.strip()
    except Exception as exc:
        return cloud_key, True, str(exc), ""

    parsed_json = extract_json_object(raw_output)
    if parsed_json is not None:
        deepseek_choice = normalize_candidate_choice(parsed_json.get("best_candidate"))
    else:
        deepseek_choice = normalize_candidate_choice(raw_output)

    if deepseek_choice not in candidate_by_key:
        return cloud_key, True, "candidate_out_of_range", raw_output

    return deepseek_choice, False, "", raw_output


def _build_deepseek_prompt(sample, candidate_by_key, cloud_key):
    candidate_lines = []
    for key, entry in candidate_by_key.items():
        candidate_lines.append(
            f"{key}: {entry['label']} (cloud_prob={entry['cloud_prob']:.4f}, edge_prob={entry['edge_prob']:.4f})"
        )
    candidate_block = "\n".join(candidate_lines)

    return f"""你是 Food-101 闭集候选仲裁器。你只能从给定候选中选择一个 best_candidate。

候选集合：
{candidate_block}

默认保底候选（cloud_top1）：
{cloud_key}

Qwen 结构化证据：
- best_candidate: {sample.get('qwen_choice')}
- qwen_best_support: {sample.get('qwen_best_support', 0)}
- qwen_cloud_support: {sample.get('qwen_cloud_support', 0)}
- ood_score: {sample.get('ood_score', 0)}
- open_description: {sample.get('open_description', '')}
- visual_evidence: {sample.get('visual_evidence', '')}
- candidate_support: {sample.get('candidate_support', {})}

决策要求：
1. 只能输出候选 JSON，不允许输出新类别。
2. 如果证据不够强，应保持 cloud_top1。
3. 如果 ood_score > 1，应保持 cloud_top1。

只输出 JSON：
{{"best_candidate":"A"}}"""


def parse_args():
    parser = argparse.ArgumentParser(description="Conservative closed-set DeepSeek fusion.")
    parser.add_argument("--rerank", default="./results/qwen_rerank_results.json")
    parser.add_argument("--output", default="./results/final_fusion_results.json")
    parser.add_argument("--api_key", default=os.environ.get("DEEPSEEK_API_KEY"))
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    run_ultimate_fusion(
        rerank_path=cli_args.rerank,
        output_path=cli_args.output,
        api_key=cli_args.api_key,
        base_url=cli_args.base_url,
    )
