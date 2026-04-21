import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from utils.pipeline import (
    candidate_map_by_key,
    extract_json_object,
    infer_edge_support_case,
    infer_routing_bucket,
    load_json,
    normalize_result_key,
    resolve_candidate_index,
    resolve_candidate_label,
    save_json,
)


DEFAULT_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
BEST_RULE_EDGE_CONF_THRESHOLD = 0.8
BEST_RULE_CLOUD_CONF_THRESHOLD = 0.4


def run_ultimate_fusion(
    rerank_path="./results/qwen_rerank_results.json",
    output_path="./results/final_fusion_results.json",
    api_key=None,
    base_url=DEFAULT_BASE_URL,
    decision_mode="best_rule_v1",
    deepseek_model="deepseek-reasoner",
    subset_mode="hard_buckets",
    max_reasoner_calls=None,
    debug_dump_path=None,
    logger=None,
):
    if logger is None:
        logger = setup_logger("ultimate_fusion")

    use_reasoner = decision_mode == "reasoner_hard_buckets"
    client = None
    if use_reasoner and api_key:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url)

    rerank_raw = load_json(rerank_path)
    rerank_results = {normalize_result_key(path): value for path, value in rerank_raw.items()}
    saved_raw = {}
    if os.path.exists(output_path):
        try:
            saved_raw = load_json(output_path)
        except Exception:
            if logger is not None:
                logger.info(f"Ignore unreadable cached output: {output_path}")
            saved_raw = {}
    saved_results = {normalize_result_key(path): value for path, value in saved_raw.items()}

    merged_results = {}
    summary = Counter()
    reasoner_gate = _build_reasoner_gate(max_reasoner_calls)

    for img_path, sample in tqdm(rerank_results.items(), total=len(rerank_results), desc="Fusion"):
        cached = saved_results.get(img_path)
        if cached and "final_label" in cached:
            merged_results[img_path] = cached
            summary["cached"] += 1
            continue

        finalized = _finalize_sample(
            img_path=img_path,
            sample=sample,
            client=client,
            decision_mode=decision_mode,
            deepseek_model=deepseek_model,
            subset_mode=subset_mode,
            reasoner_gate=reasoner_gate,
            debug_dump_path=debug_dump_path,
        )
        merged_results[img_path] = finalized

        summary[f"source::{finalized.get('final_source', 'unknown')}"] += 1
        summary[f"bucket::{finalized.get('routing_bucket', 'unknown')}"] += 1
        summary[f"edge_support::{finalized.get('edge_support_case', 'unknown')}"] += 1
        summary[f"reasoner_decision::{finalized.get('deepseek_decision', 'SKIPPED')}"] += 1
        summary[f"reasoner_parse::{finalized.get('deepseek_parse_source', 'none')}"] += 1
        if finalized.get("edge_override_applied"):
            summary["edge_override_applied"] += 1
        if finalized.get("deepseek_api_called"):
            summary["deepseek_reasoner_called"] += 1
        if finalized.get("deepseek_invalid"):
            summary[f"deepseek_invalid::{finalized.get('deepseek_invalid_reason', 'unknown')}"] += 1

        if len(merged_results) % 500 == 0:
            save_json(merged_results, output_path)

    save_json(merged_results, output_path)
    logger.info(
        f"Fusion complete | total={len(rerank_results)} | cached={summary['cached']} | "
        f"edge_override_applied={summary['edge_override_applied']} | "
        f"deepseek_reasoner_called={summary['deepseek_reasoner_called']} | "
        f"sources={_subset_counter(summary, 'source::')} | "
        f"buckets={_subset_counter(summary, 'bucket::')} | "
        f"edge_support={_subset_counter(summary, 'edge_support::')} | "
        f"reasoner_decision={_subset_counter(summary, 'reasoner_decision::')} | "
        f"reasoner_parse={_subset_counter(summary, 'reasoner_parse::')} | "
        f"deepseek_invalid={_subset_counter(summary, 'deepseek_invalid::')}"
    )
    return merged_results


def _finalize_sample(
    img_path,
    sample,
    client,
    decision_mode,
    deepseek_model,
    subset_mode,
    reasoner_gate,
    debug_dump_path,
):
    result = dict(sample)
    candidate_entries = sample.get("candidate_entries", [])
    candidate_by_key = candidate_map_by_key(candidate_entries)
    cloud_key = sample.get("cloud_candidate_key")
    cloud_idx = int(sample.get("cloud_pred", -1))
    edge_idx = int(sample.get("edge_pred", -1))
    cloud_label = resolve_candidate_label(candidate_entries, cloud_key, fallback_label=sample.get("locked_label", ""))

    if not candidate_entries or cloud_key not in candidate_by_key:
        result.update(
            {
                "routing_bucket": "invalid_candidates",
                "edge_support_case": "edge_not_strong",
                "edge_override_applied": False,
                "deepseek_decision": "SKIPPED",
                "deepseek_reason": "candidate_entries_missing",
                "deepseek_policy": decision_mode,
                "deepseek_model": deepseek_model if decision_mode == "reasoner_hard_buckets" else "",
                "deepseek_subset_hit": False,
                "deepseek_api_called": False,
                "deepseek_choice": cloud_key or "",
                "deepseek_choice_label": cloud_label,
                "deepseek_invalid": True,
                "deepseek_invalid_reason": "candidate_entries_missing",
                "deepseek_raw_output": "",
                "deepseek_parse_source": "no_text_found",
                "deepseek_raw_response_preview": "",
                "final_label": cloud_label,
                "final_pred_index": cloud_idx,
                "final_source": "cloud_fallback_invalid_candidates",
                "forced_closed_set": False,
            }
        )
        return result

    qwen_choice = sample.get("qwen_choice") or cloud_key
    qwen_idx = resolve_candidate_index(candidate_entries, qwen_choice, fallback_index=cloud_idx)
    qwen_label = resolve_candidate_label(
        candidate_entries, qwen_choice, fallback_label=sample.get("qwen_choice_label", cloud_label)
    )
    routing_bucket = infer_routing_bucket(edge_idx, cloud_idx, qwen_idx)
    edge_support_case = infer_edge_support_case(
        edge_idx,
        qwen_idx,
        sample.get("edge_conf", 0.0),
        sample.get("cloud_conf", 0.0),
    )

    result.update(
        {
            "routing_bucket": routing_bucket,
            "edge_support_case": edge_support_case,
            "qwen_choice": qwen_choice,
            "qwen_choice_label": qwen_label,
        }
    )

    base_decision = _apply_best_rule(sample, candidate_entries, qwen_idx, qwen_label, cloud_key, cloud_idx, cloud_label)
    result.update(
        {
            "edge_override_applied": base_decision["edge_override_applied"],
            "deepseek_policy": decision_mode,
            "deepseek_model": deepseek_model if decision_mode == "reasoner_hard_buckets" else "",
            "deepseek_subset_hit": False,
            "deepseek_api_called": False,
            "deepseek_decision": "SKIPPED",
            "deepseek_reason": "best_rule_v1_default",
            "deepseek_choice": base_decision["final_key"],
            "deepseek_choice_label": base_decision["final_label"],
            "deepseek_invalid": False,
            "deepseek_invalid_reason": "",
            "deepseek_raw_output": "",
            "deepseek_parse_source": "no_text_found",
            "deepseek_raw_response_preview": "",
            "final_label": base_decision["final_label"],
            "final_pred_index": base_decision["final_pred_index"],
            "final_source": base_decision["final_source"],
            "forced_closed_set": False,
        }
    )

    if decision_mode != "reasoner_hard_buckets":
        return result

    subset_hit = _should_use_reasoner(sample, routing_bucket, base_decision["edge_override_applied"], subset_mode)
    result["deepseek_subset_hit"] = subset_hit
    if not subset_hit:
        result["deepseek_reason"] = "subset_skip"
        return result

    if client is None:
        result.update(
            {
                "deepseek_invalid": True,
                "deepseek_invalid_reason": "api_key_missing",
                "deepseek_reason": "api_key_missing",
                "deepseek_parse_source": "no_text_found",
            }
        )
        return result

    gate_allowed = _allow_reasoner_call(routing_bucket, reasoner_gate)
    if not gate_allowed:
        result["deepseek_reason"] = "reasoner_budget_skip"
        return result

    reasoner_result = _run_reasoner(
        img_path=img_path,
        sample=sample,
        candidate_entries=candidate_entries,
        routing_bucket=routing_bucket,
        edge_support_case=edge_support_case,
        client=client,
        deepseek_model=deepseek_model,
        cloud_key=cloud_key,
        qwen_choice=qwen_choice,
        debug_dump_path=debug_dump_path,
    )
    result.update(reasoner_result)

    if reasoner_result["deepseek_invalid"]:
        return result

    final_key = _resolve_reasoner_final_key(
        routing_bucket=routing_bucket,
        reasoner_decision=reasoner_result["deepseek_decision"],
        sample=sample,
        qwen_choice=qwen_choice,
        cloud_key=cloud_key,
    )
    if final_key is None:
        result.update(
            {
                "deepseek_invalid": True,
                "deepseek_invalid_reason": "reasoner_decision_invalid",
                "deepseek_reason": "reasoner_decision_invalid",
            }
        )
        return result

    final_entry = candidate_by_key.get(final_key)
    if final_entry is None:
        result.update(
            {
                "deepseek_invalid": True,
                "deepseek_invalid_reason": "reasoner_candidate_missing",
                "deepseek_reason": "reasoner_candidate_missing",
            }
        )
        return result

    result.update(
        {
            "deepseek_choice": final_key,
            "deepseek_choice_label": final_entry["label"],
            "final_label": final_entry["label"],
            "final_pred_index": int(final_entry["index"]),
            "final_source": _reasoner_final_source(routing_bucket, reasoner_result["deepseek_decision"]),
        }
    )
    return result


def _apply_best_rule(sample, candidate_entries, qwen_idx, qwen_label, cloud_key, cloud_idx, cloud_label):
    edge_idx = int(sample["edge_pred"])
    edge_conf = float(sample.get("edge_conf", 0.0))
    cloud_conf = float(sample.get("cloud_conf", 0.0))
    routing_bucket = infer_routing_bucket(edge_idx, cloud_idx, qwen_idx)

    if routing_bucket == "agreement":
        return {
            "final_key": cloud_key,
            "final_label": cloud_label,
            "final_pred_index": cloud_idx,
            "final_source": "visual_lock",
            "edge_override_applied": False,
        }

    use_edge_override = (
        routing_bucket in {"qwen_eq_cloud", "all_different"}
        and edge_conf >= BEST_RULE_EDGE_CONF_THRESHOLD
        and cloud_conf <= BEST_RULE_CLOUD_CONF_THRESHOLD
    )
    if use_edge_override:
        edge_key = _find_candidate_key(candidate_entries, edge_idx)
        edge_label = resolve_candidate_label(
            candidate_entries, edge_key, fallback_label=sample.get("edge_label", "")
        )
        return {
            "final_key": edge_key or "",
            "final_label": edge_label,
            "final_pred_index": edge_idx,
            "final_source": "edge_override",
            "edge_override_applied": True,
        }

    if routing_bucket == "qwen_eq_cloud":
        return {
            "final_key": cloud_key,
            "final_label": cloud_label,
            "final_pred_index": cloud_idx,
            "final_source": "qwen_eq_cloud_keep_cloud",
            "edge_override_applied": False,
        }

    if routing_bucket == "qwen_eq_edge":
        return {
            "final_key": sample.get("qwen_choice") or cloud_key,
            "final_label": qwen_label,
            "final_pred_index": qwen_idx,
            "final_source": "qwen_eq_edge_use_qwen",
            "edge_override_applied": False,
        }

    return {
        "final_key": sample.get("qwen_choice") or cloud_key,
        "final_label": qwen_label,
        "final_pred_index": qwen_idx,
        "final_source": "all_different_use_qwen",
        "edge_override_applied": False,
    }


def _should_use_reasoner(sample, routing_bucket, edge_override_applied, subset_mode):
    if sample.get("qwen_invalid", False):
        return False
    if routing_bucket == "agreement":
        return False
    if edge_override_applied:
        return False

    if subset_mode == "hard_buckets":
        return routing_bucket in {"qwen_eq_edge", "all_different"}
    if subset_mode == "qwen_noncloud_only":
        return routing_bucket in {"qwen_eq_edge", "all_different"}
    if subset_mode == "qwen_eq_edge_only":
        return routing_bucket == "qwen_eq_edge"
    if subset_mode == "all_disagreements":
        return routing_bucket != "agreement"
    return False


def _run_reasoner(
    img_path,
    sample,
    candidate_entries,
    routing_bucket,
    edge_support_case,
    client,
    deepseek_model,
    cloud_key,
    qwen_choice,
    debug_dump_path,
):
    prompt = (
        _build_qwen_edge_prompt(sample, candidate_entries, edge_support_case, cloud_key, qwen_choice)
        if routing_bucket == "qwen_eq_edge"
        else _build_all_different_prompt(sample, candidate_entries, edge_support_case, cloud_key, qwen_choice)
    )
    try:
        response = client.chat.completions.create(
            model=deepseek_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict Food-101 fusion judge. "
                        "Return only one-line JSON. No markdown. No code fences."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=96,
            stream=False,
        )
        raw_output, parse_source, raw_preview = _extract_reasoner_output(response)
    except Exception as exc:
        preview = str(exc)
        _append_debug_dump(
            debug_dump_path,
            {
                "img_path": img_path,
                "routing_bucket": routing_bucket,
                "parse_source": "exception",
                "raw_output": preview,
                "raw_response_preview": preview[:1200],
                "decision": "SKIPPED",
                "invalid": True,
                "invalid_reason": str(exc),
            },
        )
        return {
            "deepseek_api_called": True,
            "deepseek_decision": "SKIPPED",
            "deepseek_reason": str(exc),
            "deepseek_invalid": True,
            "deepseek_invalid_reason": str(exc),
            "deepseek_raw_output": preview,
            "deepseek_parse_source": "no_text_found",
            "deepseek_raw_response_preview": preview[:1200],
        }

    parsed_json = extract_json_object(raw_output)
    decision = ""
    reason = ""
    if parsed_json is not None:
        decision = str(parsed_json.get("decision", "")).strip().upper()
        reason = str(parsed_json.get("reason", "")).strip()
    else:
        decision = _extract_reasoner_decision(raw_output)

    valid_decisions = (
        {"USE_QWEN_EDGE", "KEEP_CLOUD"}
        if routing_bucket == "qwen_eq_edge"
        else {"USE_QWEN", "USE_CLOUD", "USE_EDGE"}
    )

    if decision not in valid_decisions:
        serialized_choice = _serialize_object(response.choices[0] if getattr(response, "choices", None) else None)
        serialized_response = _serialize_object(response)
        choice_decision = _extract_reasoner_decision(serialized_choice)
        response_decision = _extract_reasoner_decision(serialized_response)
        if choice_decision in valid_decisions:
            decision = choice_decision
            raw_output = serialized_choice
            parse_source = "serialized_choice"
        elif response_decision in valid_decisions:
            decision = response_decision
            raw_output = serialized_response
            parse_source = "serialized_response"
        if not raw_preview:
            raw_preview = (serialized_choice or serialized_response)[:1200]

    invalid = decision not in valid_decisions
    if not raw_output:
        raw_output = raw_preview or "NO_TEXT_FOUND"
        parse_source = "no_text_found"

    result = {
        "deepseek_api_called": True,
        "deepseek_decision": decision if decision else "SKIPPED",
        "deepseek_reason": reason if reason else ("invalid_reasoner_output" if invalid else ""),
        "deepseek_invalid": invalid,
        "deepseek_invalid_reason": "invalid_reasoner_output" if invalid else "",
        "deepseek_raw_output": raw_output,
        "deepseek_parse_source": parse_source,
        "deepseek_raw_response_preview": raw_preview[:1200],
    }
    _append_debug_dump(
        debug_dump_path,
        {
            "img_path": img_path,
            "routing_bucket": routing_bucket,
            "parse_source": parse_source,
            "raw_output": raw_output,
            "raw_response_preview": raw_preview[:1200],
            "decision": result["deepseek_decision"],
            "reason": result["deepseek_reason"],
            "invalid": invalid,
            "invalid_reason": result["deepseek_invalid_reason"],
        },
    )
    return result


def _build_reasoner_gate(max_reasoner_calls):
    if not max_reasoner_calls:
        return None
    qwen_eq_edge_limit = (int(max_reasoner_calls) + 1) // 2
    all_different_limit = int(max_reasoner_calls) // 2
    return {
        "max_total": int(max_reasoner_calls),
        "total_called": 0,
        "bucket_limits": {
            "qwen_eq_edge": qwen_eq_edge_limit,
            "all_different": all_different_limit,
        },
        "bucket_counts": Counter(),
    }


def _allow_reasoner_call(routing_bucket, reasoner_gate):
    if reasoner_gate is None:
        return True
    if reasoner_gate["total_called"] >= reasoner_gate["max_total"]:
        return False
    limit = reasoner_gate["bucket_limits"].get(routing_bucket, 0)
    if reasoner_gate["bucket_counts"][routing_bucket] >= limit:
        return False
    reasoner_gate["bucket_counts"][routing_bucket] += 1
    reasoner_gate["total_called"] += 1
    return True


def _extract_reasoner_output(response):
    serialized_response = _serialize_object(response)
    raw_preview = serialized_response[:1200]
    choices = getattr(response, "choices", None) or []
    first_choice = choices[0] if choices else None
    message = getattr(first_choice, "message", None) if first_choice is not None else None

    message_content = getattr(message, "content", None)
    if isinstance(message_content, str) and message_content.strip():
        return message_content.strip(), "message_content", raw_preview

    blocks_text = _flatten_text_blocks(message_content)
    if blocks_text:
        return blocks_text, "message_content_blocks", raw_preview

    reasoning_content = getattr(message, "reasoning_content", None)
    reasoning_text = _flatten_text_blocks(reasoning_content)
    if reasoning_text:
        return reasoning_text, "reasoning_content", raw_preview

    serialized_choice = _serialize_object(first_choice)
    choice_token = _extract_reasoner_decision(serialized_choice)
    if choice_token:
        return serialized_choice, "serialized_choice", raw_preview

    response_token = _extract_reasoner_decision(serialized_response)
    if response_token:
        return serialized_response, "serialized_response", raw_preview

    return "", "no_text_found", raw_preview


def _flatten_text_blocks(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        text = value.get("text") or value.get("content") or value.get("value")
        return str(text).strip() if text else ""
    if isinstance(value, (list, tuple)):
        chunks = []
        for block in value:
            block_text = _flatten_text_blocks(block)
            if block_text:
                chunks.append(block_text)
        return "\n".join(chunks).strip()

    block_type = getattr(value, "type", "")
    if block_type and str(block_type).lower() not in {"text", "output_text"}:
        return ""

    for attr in ("text", "content", "value"):
        attr_value = getattr(value, attr, None)
        if attr_value:
            flattened = _flatten_text_blocks(attr_value)
            if flattened:
                return flattened
    return ""


def _serialize_object(value):
    if value is None:
        return ""
    try:
        if hasattr(value, "model_dump_json"):
            dumped = value.model_dump_json()
            return dumped if isinstance(dumped, str) else str(dumped)
        if hasattr(value, "model_dump"):
            return json.dumps(value.model_dump(), ensure_ascii=False, default=str)
        if isinstance(value, (dict, list, tuple, str, int, float, bool)):
            if isinstance(value, str):
                return value
            return json.dumps(value, ensure_ascii=False, default=str)
        if hasattr(value, "__dict__"):
            return json.dumps(value.__dict__, ensure_ascii=False, default=str)
    except Exception:
        pass
    return str(value)


def _append_debug_dump(debug_dump_path, payload):
    if not debug_dump_path:
        return
    debug_path = Path(debug_dump_path)
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    with open(debug_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _resolve_reasoner_final_key(routing_bucket, reasoner_decision, sample, qwen_choice, cloud_key):
    if routing_bucket == "qwen_eq_edge":
        if reasoner_decision == "USE_QWEN_EDGE":
            return qwen_choice
        if reasoner_decision == "KEEP_CLOUD":
            return cloud_key
        return None

    if reasoner_decision == "USE_QWEN":
        return qwen_choice
    if reasoner_decision == "USE_CLOUD":
        return cloud_key
    if reasoner_decision == "USE_EDGE":
        return _find_candidate_key(sample.get("candidate_entries", []), int(sample["edge_pred"]))
    return None


def _reasoner_final_source(routing_bucket, decision):
    if routing_bucket == "qwen_eq_edge":
        if decision == "USE_QWEN_EDGE":
            return "reasoner_use_qwen_edge"
        return "reasoner_keep_cloud"
    if decision == "USE_QWEN":
        return "reasoner_use_qwen"
    if decision == "USE_EDGE":
        return "reasoner_use_edge"
    return "reasoner_keep_cloud"


def _build_qwen_edge_prompt(sample, candidate_entries, edge_support_case, cloud_key, qwen_choice):
    candidate_block = _format_candidate_entries(candidate_entries)
    qwen_label = resolve_candidate_label(candidate_entries, qwen_choice, sample.get("qwen_choice_label", ""))
    cloud_label = resolve_candidate_label(candidate_entries, cloud_key, sample.get("locked_label", ""))
    return (
        "Food-101 hard-bucket decision.\n"
        "Bucket: qwen_eq_edge.\n"
        "Choose exactly one decision: USE_QWEN_EDGE or KEEP_CLOUD.\n"
        f"edge_label={sample.get('edge_label', '')}, edge_conf={float(sample.get('edge_conf', 0.0)):.4f}\n"
        f"cloud_label={sample.get('cloud_label', '')}, cloud_conf={float(sample.get('cloud_conf', 0.0)):.4f}\n"
        f"qwen_label={qwen_label}\n"
        f"cloud_default_key={cloud_key}, cloud_default_label={cloud_label}\n"
        f"qwen_choice_key={qwen_choice}, qwen_best_support={int(sample.get('qwen_best_support', 0))}, "
        f"qwen_cloud_support={int(sample.get('qwen_cloud_support', 0))}\n"
        f"edge_support_case={edge_support_case}\n"
        f"open_description={sample.get('open_description', '')}\n"
        f"visual_evidence={sample.get('visual_evidence', '')}\n"
        f"candidate_entries={candidate_block}\n"
        'Return one-line JSON like {"decision":"USE_QWEN_EDGE","reason":"..."}.\n'
        "If you cannot output JSON, output exactly one token from: USE_QWEN_EDGE KEEP_CLOUD."
    )


def _build_all_different_prompt(sample, candidate_entries, edge_support_case, cloud_key, qwen_choice):
    candidate_block = _format_candidate_entries(candidate_entries)
    qwen_label = resolve_candidate_label(candidate_entries, qwen_choice, sample.get("qwen_choice_label", ""))
    cloud_label = resolve_candidate_label(candidate_entries, cloud_key, sample.get("locked_label", ""))
    edge_key = _find_candidate_key(candidate_entries, int(sample["edge_pred"]))
    edge_label = resolve_candidate_label(candidate_entries, edge_key, sample.get("edge_label", ""))
    return (
        "Food-101 hard-bucket decision.\n"
        "Bucket: all_different.\n"
        "Choose exactly one decision: USE_QWEN, USE_CLOUD, or USE_EDGE.\n"
        f"edge_key={edge_key}, edge_label={edge_label}, edge_conf={float(sample.get('edge_conf', 0.0)):.4f}\n"
        f"cloud_key={cloud_key}, cloud_label={cloud_label}, cloud_conf={float(sample.get('cloud_conf', 0.0)):.4f}\n"
        f"qwen_key={qwen_choice}, qwen_label={qwen_label}, qwen_best_support={int(sample.get('qwen_best_support', 0))}, "
        f"qwen_cloud_support={int(sample.get('qwen_cloud_support', 0))}\n"
        f"edge_support_case={edge_support_case}\n"
        f"open_description={sample.get('open_description', '')}\n"
        f"visual_evidence={sample.get('visual_evidence', '')}\n"
        f"candidate_entries={candidate_block}\n"
        'Return one-line JSON like {"decision":"USE_QWEN","reason":"..."}.\n'
        "If you cannot output JSON, output exactly one token from: USE_QWEN USE_CLOUD USE_EDGE."
    )


def _format_candidate_entries(candidate_entries):
    return [
        {
            "key": entry["key"],
            "label": entry["label"],
            "index": int(entry["index"]),
            "cloud_prob": round(float(entry.get("cloud_prob", 0.0)), 4),
            "edge_prob": round(float(entry.get("edge_prob", 0.0)), 4),
        }
        for entry in candidate_entries
    ]


def _find_candidate_key(candidate_entries, class_index):
    for entry in candidate_entries:
        if int(entry["index"]) == int(class_index):
            return entry["key"]
    return None


def _extract_reasoner_decision(text):
    text_upper = str(text).upper()
    for token in ("USE_QWEN_EDGE", "KEEP_CLOUD", "USE_QWEN", "USE_CLOUD", "USE_EDGE"):
        if re.search(rf"\b{token}\b", text_upper):
            return token
    return ""


def _subset_counter(counter, prefix):
    return {key[len(prefix) :]: value for key, value in counter.items() if key.startswith(prefix)}


def parse_args():
    parser = argparse.ArgumentParser(description="High-accuracy fusion with deterministic routing and optional DeepSeek reasoner.")
    parser.add_argument("--rerank", default="./results/qwen_rerank_results.json")
    parser.add_argument("--output", default="./results/final_fusion_results.json")
    parser.add_argument("--api_key", default=os.environ.get("DEEPSEEK_API_KEY"))
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL)
    parser.add_argument("--decision_mode", default="best_rule_v1", choices=["best_rule_v1", "reasoner_hard_buckets"])
    parser.add_argument("--deepseek_model", default="deepseek-reasoner")
    parser.add_argument(
        "--subset_mode",
        default="hard_buckets",
        choices=["hard_buckets", "qwen_noncloud_only", "qwen_eq_edge_only", "all_disagreements"],
    )
    parser.add_argument("--max_reasoner_calls", type=int, default=None)
    parser.add_argument("--debug_dump_path", default="")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    run_ultimate_fusion(
        rerank_path=cli_args.rerank,
        output_path=cli_args.output,
        api_key=cli_args.api_key,
        base_url=cli_args.base_url,
        decision_mode=cli_args.decision_mode,
        deepseek_model=cli_args.deepseek_model,
        subset_mode=cli_args.subset_mode,
        max_reasoner_calls=cli_args.max_reasoner_calls,
        debug_dump_path=cli_args.debug_dump_path or None,
    )
