import argparse
import os
import sys
from collections import Counter

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
    normalize_result_key,
    resolve_image_path,
    resize_image_to_max_pixels,
    salvage_qwen_json_fields,
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
    if logger is None:
        logger = setup_logger("qwen_candidate_rerank")

    classes_list = load_food101_classes(data_dir=data_dir)
    baseline_results = load_json(baseline_path)
    prototype_bank = load_json(prototype_path)
    saved_results = load_json(output_path) if os.path.exists(output_path) else {}
    normalized_saved = {normalize_result_key(path): value for path, value in saved_results.items()}

    all_results = {}
    rerank_queue = []
    summary = Counter()

    for raw_img_path, sample in baseline_results.items():
        normalized_path = normalize_result_key(raw_img_path)
        cached = normalized_saved.get(normalized_path)
        if cached and cached.get("qwen_completed"):
            all_results[normalized_path] = cached
            summary["cached"] += 1
            continue

        candidate_entries = build_candidate_entries(sample, classes_list, max_candidates=6)
        cloud_candidate_key = find_candidate_key(candidate_entries, sample["cloud_pred"])
        if cloud_candidate_key is None:
            all_results[normalized_path] = _build_fallback_record(
                sample=sample,
                raw_img_path=raw_img_path,
                candidate_entries=candidate_entries,
                cloud_candidate_key="A",
                invalid_reason="cloud_candidate_missing",
                data_dir=data_dir,
            )
            summary["fallback_missing_cloud_candidate"] += 1
            continue

        record = {
            **sample,
            "resolved_path": resolve_image_path(raw_img_path, data_dir=data_dir),
            "candidate_entries": candidate_entries,
            "candidate_labels": candidate_label_list(candidate_entries),
            "cloud_candidate_key": cloud_candidate_key,
        }

        if sample.get("lockable") or sample.get("edge_pred") == sample.get("cloud_pred"):
            record.update(_build_locked_result(candidate_entries, cloud_candidate_key, classes_list[sample["cloud_pred"]]))
            all_results[normalized_path] = record
            summary["locked_skipped"] += 1
            continue

        rerank_queue.append((normalized_path, record))

    logger.info(
        f"Qwen 重排准备完成 | total={len(baseline_results)} | locked_skipped={summary['locked_skipped']} | "
        f"cached={summary['cached']} | rerank_queue={len(rerank_queue)}"
    )

    if rerank_queue:
        logger.info("加载 Qwen2.5-VL-3B，用于闭集候选重排...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(model_dir)

        processed = 0
        invalid_counter = Counter()
        parse_mode_counter = Counter()
        status_counter = Counter()
        for normalized_path, record in tqdm(rerank_queue, total=len(rerank_queue), desc="Qwen rerank"):
            inference_result = _infer_qwen_choice(
                model=model,
                processor=processor,
                sample=record,
                candidate_entries=record["candidate_entries"],
                prototype_bank=prototype_bank,
            )
            record.update(inference_result)
            all_results[normalized_path] = record
            processed += 1

            parse_mode_counter[str(record.get("qwen_parse_mode", "unknown"))] += 1
            if record.get("qwen_invalid"):
                invalid_counter[str(record.get("qwen_invalid_reason", "unknown"))] += 1
            elif record.get("qwen_parse_mode") == "strict_json":
                status_counter["strict_json_success"] += 1
            elif record.get("qwen_parse_mode") == "regex_salvage":
                status_counter["regex_salvage_success"] += 1

            if processed % 200 == 0:
                save_json(all_results, output_path)

            if max_samples is not None and processed >= max_samples:
                break

        logger.info(
            f"Qwen 重排完成 | processed={processed} | parse_modes={dict(parse_mode_counter)} | "
            f"success={dict(status_counter)} | invalid={dict(invalid_counter)}"
        )

    save_json(all_results, output_path)
    logger.info(f"Qwen 候选重排结果已保存至 {output_path}")
    return all_results


def _infer_qwen_choice(model, processor, sample, candidate_entries, prototype_bank):
    cloud_candidate_key = sample["cloud_candidate_key"]
    cloud_label = next(entry["label"] for entry in candidate_entries if entry["key"] == cloud_candidate_key)

    try:
        target_image = _load_resized_image(sample["resolved_path"], 768 * 28 * 28)
        content = [
            {"type": "text", "text": _build_qwen_prompt(candidate_entries)},
            {"type": "image", "image": target_image},
            {"type": "text", "text": "Target image. Prototype images follow by candidate order."},
        ]

        for entry in candidate_entries:
            content.append(
                {
                    "type": "text",
                    "text": f"Candidate {entry['key']} = {entry['label']}. Two prototype images follow.",
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
                max_new_tokens=160,
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
        return _build_invalid_qwen_result(
            candidate_entries,
            cloud_candidate_key,
            cloud_label,
            invalid_reason=str(exc),
            raw_output="",
            parse_mode="exception_fallback",
        )

    parsed_json = extract_json_object(output_text)
    parse_mode = "strict_json"
    valid_candidate_keys = {entry["key"] for entry in candidate_entries}
    if parsed_json is None:
        parsed_json = salvage_qwen_json_fields(output_text, valid_candidate_keys=valid_candidate_keys)
        parse_mode = "regex_salvage" if parsed_json is not None else "fallback"

    if parsed_json is None:
        return _build_invalid_qwen_result(
            candidate_entries,
            cloud_candidate_key,
            cloud_label,
            invalid_reason="json_parse_failed",
            raw_output=output_text,
            parse_mode=parse_mode,
        )

    best_candidate = normalize_candidate_choice(parsed_json.get("best_candidate"))
    if best_candidate is None:
        return _build_invalid_qwen_result(
            candidate_entries,
            cloud_candidate_key,
            cloud_label,
            invalid_reason="top_level_json_missing_best_candidate",
            raw_output=output_text,
            parse_mode=parse_mode,
        )

    if best_candidate not in valid_candidate_keys:
        return _build_invalid_qwen_result(
            candidate_entries,
            cloud_candidate_key,
            cloud_label,
            invalid_reason="true_candidate_out_of_range",
            raw_output=output_text,
            parse_mode=parse_mode,
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
        "qwen_parse_mode": parse_mode,
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
        f"{entry['key']}={entry['label']}|cloud={entry['cloud_prob']:.4f}|edge={entry['edge_prob']:.4f}"
        for entry in candidate_entries
    ]
    candidate_block = ";".join(candidate_lines)

    return (
        "You are a Food-101 closed-set reranker. "
        "You must choose exactly one best candidate from the provided candidates only. "
        "Return a single-line JSON object with keys best_candidate, candidate_support, ood_score, "
        "open_description, visual_evidence. "
        "candidate_support must be a dictionary from candidate letter to integer 0-100. "
        "ood_score must be one integer: 0,1,2,3. "
        "No markdown. No code fences. No explanations. "
        f"Candidates:{candidate_block}. "
        'Output example: {"best_candidate":"A","candidate_support":{"A":80,"B":10,"C":10},"ood_score":0,"open_description":"...","visual_evidence":"..."}'
    )


def _build_locked_result(candidate_entries, cloud_candidate_key, cloud_label):
    return {
        "qwen_completed": True,
        "qwen_skipped": True,
        "qwen_invalid": False,
        "qwen_invalid_reason": "",
        "qwen_parse_mode": "locked_skip",
        "qwen_choice": cloud_candidate_key,
        "qwen_choice_label": cloud_label,
        "candidate_support": {
            entry["key"]: 100 if entry["key"] == cloud_candidate_key else 0 for entry in candidate_entries
        },
        "qwen_best_support": 100,
        "qwen_cloud_support": 100,
        "ood_score": 0,
        "ood_flag": False,
        "open_description": "",
        "visual_evidence": "edge/cloud top1 agree; rerank skipped",
        "qwen_raw_output": "",
    }


def _load_resized_image(path, max_pixels):
    image = Image.open(path).convert("RGB")
    return resize_image_to_max_pixels(image, max_pixels=max_pixels)


def _normalize_ood_score(value):
    try:
        return max(0, min(3, int(value)))
    except (TypeError, ValueError):
        return 3


def _build_invalid_qwen_result(
    candidate_entries,
    cloud_candidate_key,
    cloud_label,
    invalid_reason,
    raw_output="",
    parse_mode="fallback",
):
    return {
        "qwen_completed": True,
        "qwen_skipped": False,
        "qwen_invalid": True,
        "qwen_invalid_reason": invalid_reason,
        "qwen_parse_mode": parse_mode,
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
        **_build_invalid_qwen_result(
            candidate_entries,
            cloud_candidate_key,
            cloud_label,
            invalid_reason=invalid_reason,
            parse_mode="fallback",
        ),
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
