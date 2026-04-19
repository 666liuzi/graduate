import argparse
import os
import sys

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen_vl_utils import process_vision_info
from utils.logger import setup_logger
from utils.pipeline import (
    build_candidate_entries,
    candidate_label_list,
    extract_json_object,
    find_candidate_key,
    get_candidate_support,
    load_food101_classes,
    load_json,
    normalize_candidate_choice,
    resolve_image_path,
    resize_image_to_max_pixels,
    save_json,
    should_flag_ood,
)


DEFAULT_MODEL_DIR = os.environ.get(
    "QWEN_MODEL_DIR", "/root/autodl-tmp/models/zpeng1989/Chinese_Food_Qwen25vl_3B_Model"
)


def run_qwen_candidate_rerank(
    baseline_path="./results/baseline_visual_results.json",
    prototype_path="./results/prototype_bank.json",
    output_path="./results/qwen_rerank_results.json",
    model_dir=DEFAULT_MODEL_DIR,
    data_dir="./data",
    max_samples=None,
    logger=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if logger is None:
        logger = setup_logger("qwen_candidate_rerank")

    classes_list = load_food101_classes(data_dir=data_dir)
    baseline_results = load_json(baseline_path)
    prototype_bank = load_json(prototype_path)
    saved_results = load_json(output_path) if os.path.exists(output_path) else {}

    logger.info("加载 Qwen2.5-VL-3B，用于闭集候选重排...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_dir)

    all_results = {}
    processed = 0
    for raw_img_path, sample in tqdm(baseline_results.items()):
        cached = saved_results.get(raw_img_path)
        if cached and cached.get("qwen_completed"):
            all_results[raw_img_path] = cached
            continue

        candidate_entries = build_candidate_entries(sample, classes_list, max_candidates=6)
        cloud_candidate_key = find_candidate_key(candidate_entries, sample["cloud_pred"])
        if cloud_candidate_key is None:
            all_results[raw_img_path] = _build_fallback_record(
                sample=sample,
                raw_img_path=raw_img_path,
                candidate_entries=candidate_entries,
                cloud_candidate_key="A",
                invalid_reason="cloud_candidate_missing",
                data_dir=data_dir,
            )
            continue

        record = {
            **sample,
            "resolved_path": resolve_image_path(raw_img_path, data_dir=data_dir),
            "candidate_entries": candidate_entries,
            "candidate_labels": candidate_label_list(candidate_entries),
            "cloud_candidate_key": cloud_candidate_key,
        }

        if sample.get("lockable") or sample.get("edge_pred") == sample.get("cloud_pred"):
            record.update(
                {
                    "qwen_completed": True,
                    "qwen_skipped": True,
                    "qwen_invalid": False,
                    "qwen_invalid_reason": "",
                    "qwen_choice": cloud_candidate_key,
                    "qwen_choice_label": classes_list[sample["cloud_pred"]],
                    "candidate_support": {
                        entry["key"]: 100 if entry["key"] == cloud_candidate_key else 0
                        for entry in candidate_entries
                    },
                    "qwen_best_support": 100,
                    "qwen_cloud_support": 100,
                    "ood_score": 0,
                    "ood_flag": False,
                    "open_description": "",
                    "visual_evidence": "edge/cloud top1 agree; rerank skipped",
                    "qwen_raw_output": "",
                }
            )
            all_results[raw_img_path] = record
            processed += 1
            if max_samples is not None and processed >= max_samples:
                break
            continue

        inference_result = _infer_qwen_choice(
            model=model,
            processor=processor,
            raw_img_path=raw_img_path,
            sample=sample,
            candidate_entries=candidate_entries,
            prototype_bank=prototype_bank,
            data_dir=data_dir,
        )
        record.update(inference_result)
        all_results[raw_img_path] = record
        processed += 1

        if processed % 200 == 0:
            save_json(all_results, output_path)

        if max_samples is not None and processed >= max_samples:
            break

    save_json(all_results, output_path)
    logger.info(f"Qwen 候选重排结果已保存至 {output_path}")
    return all_results


def _infer_qwen_choice(model, processor, raw_img_path, sample, candidate_entries, prototype_bank, data_dir):
    cloud_candidate_key = find_candidate_key(candidate_entries, sample["cloud_pred"])
    cloud_label = candidate_entries[ord(cloud_candidate_key) - ord("A")]["label"]

    try:
        target_image = _load_resized_image(resolve_image_path(raw_img_path, data_dir=data_dir), 768 * 28 * 28)
        content = [
            {"type": "text", "text": _build_qwen_prompt(candidate_entries)},
            {"type": "image", "image": target_image},
            {"type": "text", "text": "上图是目标图。下面依次提供每个候选类别的参考图。"},
        ]

        for entry in candidate_entries:
            content.append(
                {
                    "type": "text",
                    "text": f"候选 {entry['key']} = {entry['label']}。下面是该候选的 2 张参考图。",
                }
            )
            prototypes = prototype_bank[entry["label"]]["prototypes"][:2]
            for prototype in prototypes:
                prototype_image = _load_resized_image(prototype["path"], 256 * 28 * 28)
                content.append({"type": "image", "image": prototype_image})

        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=96,
                do_sample=False,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
    except Exception as exc:
        return _build_invalid_qwen_result(candidate_entries, cloud_candidate_key, cloud_label, str(exc))

    parsed_json = extract_json_object(output_text)
    if parsed_json is None:
        return _build_invalid_qwen_result(
            candidate_entries, cloud_candidate_key, cloud_label, "json_parse_failed", output_text
        )

    best_candidate = normalize_candidate_choice(parsed_json.get("best_candidate"))
    valid_candidate_keys = {entry["key"] for entry in candidate_entries}
    if best_candidate not in valid_candidate_keys:
        return _build_invalid_qwen_result(
            candidate_entries, cloud_candidate_key, cloud_label, "candidate_out_of_range", output_text
        )

    candidate_support = {}
    raw_support = parsed_json.get("candidate_support", {})
    for entry in candidate_entries:
        raw_value = raw_support.get(entry["key"], 0) if isinstance(raw_support, dict) else 0
        try:
            candidate_support[entry["key"]] = max(0, min(100, int(raw_value)))
        except (TypeError, ValueError):
            candidate_support[entry["key"]] = 0

    qwen_choice_label = next(entry["label"] for entry in candidate_entries if entry["key"] == best_candidate)
    ood_score = _normalize_ood_score(parsed_json.get("ood_score", 0))

    return {
        "qwen_completed": True,
        "qwen_skipped": False,
        "qwen_invalid": False,
        "qwen_invalid_reason": "",
        "qwen_choice": best_candidate,
        "qwen_choice_label": qwen_choice_label,
        "candidate_support": candidate_support,
        "qwen_best_support": get_candidate_support(candidate_support, best_candidate),
        "qwen_cloud_support": get_candidate_support(candidate_support, cloud_candidate_key),
        "ood_score": ood_score,
        "ood_flag": should_flag_ood(ood_score),
        "open_description": str(parsed_json.get("open_description", "")).strip(),
        "visual_evidence": str(parsed_json.get("visual_evidence", "")).strip(),
        "qwen_raw_output": output_text,
    }


def _build_qwen_prompt(candidate_entries):
    candidate_lines = [
        f"{entry['key']}: {entry['label']} (cloud_prob={entry['cloud_prob']:.4f}, edge_prob={entry['edge_prob']:.4f})"
        for entry in candidate_entries
    ]
    candidate_block = "\n".join(candidate_lines)

    return f"""你是 Food-101 闭集候选重排器。你会看到 1 张目标图和若干候选类别的参考图。

候选类别如下：
{candidate_block}

任务要求：
1. 你只能从候选 A-F 中选择 1 个最佳候选，不能输出新类别。
2. 你必须给出 candidate_support，为每个候选打 0-100 的整数分。
3. 你必须给出 ood_score：
   0 = 明显属于 Food-101 闭集且候选中有明显匹配
   1 = 属于 Food-101 闭集但视觉上较难区分
   2 = 可能不属于当前候选集合或有越界风险
   3 = 高度怀疑不属于 Food-101 闭集
4. open_description 用一句中文概括你认为目标图像里真正像什么；visual_evidence 用一句中文说明主要视觉证据。

只输出 JSON，不要输出解释、Markdown 或代码块。
输出格式固定为：
{{
  "best_candidate": "A",
  "candidate_support": {{"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0}},
  "ood_score": 0,
  "open_description": "",
  "visual_evidence": ""
}}"""


def _load_resized_image(path, max_pixels):
    image = Image.open(path).convert("RGB")
    return resize_image_to_max_pixels(image, max_pixels=max_pixels)


def _normalize_ood_score(value):
    try:
        return max(0, min(3, int(value)))
    except (TypeError, ValueError):
        return 3


def _build_invalid_qwen_result(
    candidate_entries, cloud_candidate_key, cloud_label, invalid_reason, raw_output=""
):
    return {
        "qwen_completed": True,
        "qwen_skipped": False,
        "qwen_invalid": True,
        "qwen_invalid_reason": invalid_reason,
        "qwen_choice": cloud_candidate_key,
        "qwen_choice_label": cloud_label,
        "candidate_support": {entry["key"]: 0 for entry in candidate_entries},
        "qwen_best_support": 0,
        "qwen_cloud_support": 0,
        "ood_score": 3,
        "ood_flag": True,
        "open_description": "",
        "visual_evidence": "",
        "qwen_raw_output": raw_output,
    }


def _build_fallback_record(
    sample, raw_img_path, candidate_entries, cloud_candidate_key, invalid_reason, data_dir
):
    cloud_label = candidate_entries[0]["label"] if candidate_entries else ""
    return {
        **sample,
        "resolved_path": resolve_image_path(raw_img_path, data_dir=data_dir),
        "candidate_entries": candidate_entries,
        "candidate_labels": candidate_label_list(candidate_entries),
        "cloud_candidate_key": cloud_candidate_key,
        **_build_invalid_qwen_result(candidate_entries, cloud_candidate_key, cloud_label, invalid_reason),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run closed-set Qwen reranking on ambiguous samples.")
    parser.add_argument("--baseline", default="./results/baseline_visual_results.json")
    parser.add_argument("--prototype_bank", default="./results/prototype_bank.json")
    parser.add_argument("--output", default="./results/qwen_rerank_results.json")
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    run_qwen_candidate_rerank(
        baseline_path=cli_args.baseline,
        prototype_path=cli_args.prototype_bank,
        output_path=cli_args.output,
        model_dir=cli_args.model_dir,
        data_dir=cli_args.data_dir,
        max_samples=cli_args.max_samples,
    )
