import json
import math
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_CLASSES_PATH = DEFAULT_DATA_DIR / "food-101" / "meta" / "classes.txt"


def canonicalize_label(label):
    return label.strip().lower().replace(" ", "_")


def normalize_result_key(path):
    return str(path).replace("\\", "/").strip()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data, path):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4, ensure_ascii=False)


def load_food101_classes(classes_path=None, data_dir=None):
    resolved_classes_path = Path(classes_path) if classes_path else DEFAULT_CLASSES_PATH
    if resolved_classes_path.exists():
        with open(resolved_classes_path, "r", encoding="utf-8") as handle:
            return [canonicalize_label(line) for line in handle.readlines() if line.strip()]

    resolved_data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    try:
        from torchvision.datasets import Food101

        dataset = Food101(root=str(resolved_data_dir), split="test", download=False)
        return [canonicalize_label(label) for label in dataset.classes]
    except Exception as exc:
        raise FileNotFoundError(
            f"Unable to load Food-101 class names from {resolved_classes_path}"
        ) from exc


def resolve_image_path(raw_img_path, project_root=None, data_dir=None):
    normalized = str(raw_img_path).replace("\\", "/")
    candidate = Path(normalized)
    if candidate.is_absolute() and candidate.exists():
        return str(candidate.resolve())

    base_root = Path(project_root) if project_root else PROJECT_ROOT
    from_project = base_root / normalized
    if from_project.exists():
        return str(from_project.resolve())

    images_root = (Path(data_dir) if data_dir else DEFAULT_DATA_DIR) / "food-101" / "images"
    parts = normalized.split("/")
    if len(parts) >= 2:
        fallback = images_root / parts[-2] / parts[-1]
        if fallback.exists():
            return str(fallback.resolve())
        return str(fallback)

    return str(from_project)


def extract_json_object(text):
    if not text:
        return None

    cleaned = text.strip()
    outer_fence_match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if outer_fence_match:
        cleaned = outer_fence_match.group(1).strip()

    json_candidate = _extract_balanced_json_object(cleaned)
    if not json_candidate:
        return None

    try:
        return json.loads(json_candidate)
    except json.JSONDecodeError:
        return None


def salvage_qwen_json_fields(text, valid_candidate_keys=None):
    if not text:
        return None

    valid_keys = set(valid_candidate_keys or [])
    choice = None
    choice_match = re.search(r'"best_candidate"\s*:\s*"?(?P<choice>[A-F])"?', text, re.IGNORECASE)
    if choice_match:
        choice = normalize_candidate_choice(choice_match.group("choice"))

    if valid_keys and choice not in valid_keys:
        return None

    if choice is None:
        return None

    ood_match = re.search(r'"ood_score"\s*:\s*"?(?P<score>[0-3])"?', text, re.IGNORECASE)
    open_desc_match = re.search(
        r'"open_description"\s*:\s*"(?P<value>(?:[^"\\]|\\.)*)"', text, re.IGNORECASE
    )
    visual_match = re.search(
        r'"visual_evidence"\s*:\s*"(?P<value>(?:[^"\\]|\\.)*)"', text, re.IGNORECASE
    )

    support_map = {}
    for key in valid_keys:
        support_match = re.search(
            rf'"{re.escape(key)}"\s*:\s*"?(?P<score>\d{{1,3}})"?', text, re.IGNORECASE
        )
        if support_match:
            support_map[key] = max(0, min(100, int(support_match.group("score"))))

    return {
        "best_candidate": choice,
        "candidate_support": support_map,
        "ood_score": int(ood_match.group("score")) if ood_match else 3,
        "open_description": _decode_json_fragment(open_desc_match.group("value")) if open_desc_match else "",
        "visual_evidence": _decode_json_fragment(visual_match.group("value")) if visual_match else "",
    }


def normalize_candidate_choice(choice):
    if not choice:
        return None

    cleaned = str(choice).strip().upper()
    match = re.search(r"[A-F]", cleaned)
    return match.group(0) if match else None


def prepare_topk_predictions(logits, classes_list, topk=3):
    import torch
    import torch.nn.functional as F

    probabilities = F.softmax(logits, dim=1)
    top_probs, top_indices = torch.topk(probabilities, k=min(topk, probabilities.shape[1]), dim=1)

    batch_predictions = []
    for sample_indices, sample_probs in zip(top_indices, top_probs):
        sample_records = []
        for idx_tensor, prob_tensor in zip(sample_indices, sample_probs):
            class_index = int(idx_tensor.item())
            sample_records.append(
                {
                    "index": class_index,
                    "label": classes_list[class_index],
                    "prob": round(float(prob_tensor.item()), 4),
                }
            )
        batch_predictions.append(sample_records)

    return batch_predictions


def compute_topk_margin(topk_predictions):
    if not topk_predictions:
        return 0.0
    if len(topk_predictions) == 1:
        return round(float(topk_predictions[0]["prob"]), 4)
    margin = topk_predictions[0]["prob"] - topk_predictions[1]["prob"]
    return round(float(margin), 4)


def compute_lock_metadata(edge_topk, cloud_topk):
    edge_top1 = edge_topk[0] if edge_topk else None
    cloud_top1 = cloud_topk[0] if cloud_topk else None
    is_locked = bool(edge_top1 and cloud_top1 and edge_top1["index"] == cloud_top1["index"])

    return {
        "lockable": is_locked,
        "lock_reason": "visual_agreement" if is_locked else "",
        "locked_label": cloud_top1["label"] if is_locked else "",
        "locked_index": cloud_top1["index"] if is_locked else None,
    }


def build_candidate_entries(sample, classes_list, max_candidates=6):
    candidate_records = []
    seen_indices = set()

    for source_name in ("cloud_topk", "edge_topk"):
        for record in normalize_topk_records(sample, source_name, classes_list):
            class_index = int(record["index"])
            if class_index in seen_indices:
                continue
            seen_indices.add(class_index)
            candidate_records.append(
                {
                    "key": "",
                    "index": class_index,
                    "label": classes_list[class_index],
                    "cloud_prob": round(
                        float(
                            _lookup_probability(
                                normalize_topk_records(sample, "cloud_topk", classes_list),
                                class_index,
                            )
                        ),
                        4,
                    ),
                    "edge_prob": round(
                        float(
                            _lookup_probability(
                                normalize_topk_records(sample, "edge_topk", classes_list),
                                class_index,
                            )
                        ),
                        4,
                    ),
                    "sources": _lookup_sources(sample, class_index),
                }
            )
            if len(candidate_records) >= max_candidates:
                break
        if len(candidate_records) >= max_candidates:
            break

    for offset, record in enumerate(candidate_records):
        record["key"] = chr(ord("A") + offset)

    return candidate_records


def candidate_map_by_key(candidate_entries):
    return {entry["key"]: entry for entry in candidate_entries}


def candidate_label_list(candidate_entries):
    return [entry["label"] for entry in candidate_entries]


def resolve_candidate_entry(candidate_entries, candidate_key):
    for entry in candidate_entries:
        if entry["key"] == candidate_key:
            return entry
    return None


def resolve_candidate_index(candidate_entries, candidate_key, fallback_index=None):
    entry = resolve_candidate_entry(candidate_entries, candidate_key)
    if entry is not None:
        return int(entry["index"])
    return None if fallback_index is None else int(fallback_index)


def resolve_candidate_label(candidate_entries, candidate_key, fallback_label=""):
    entry = resolve_candidate_entry(candidate_entries, candidate_key)
    if entry is not None:
        return str(entry["label"])
    return fallback_label


def infer_routing_bucket(edge_index, cloud_index, qwen_index):
    edge_index = int(edge_index)
    cloud_index = int(cloud_index)
    qwen_index = int(qwen_index)

    if edge_index == cloud_index:
        return "agreement"
    if qwen_index == cloud_index:
        return "qwen_eq_cloud"
    if qwen_index == edge_index:
        return "qwen_eq_edge"
    return "all_different"


def infer_edge_support_case(
    edge_index,
    qwen_index,
    edge_conf,
    cloud_conf,
    strong_edge_threshold=0.85,
    weak_cloud_threshold=0.5,
):
    if float(edge_conf) < strong_edge_threshold or float(cloud_conf) > weak_cloud_threshold:
        return "edge_not_strong"
    if int(edge_index) == int(qwen_index):
        return "edge_aligns_with_qwen"
    return "edge_conflicts_with_qwen"


def find_candidate_key(candidate_entries, class_index):
    for entry in candidate_entries:
        if int(entry["index"]) == int(class_index):
            return entry["key"]
    return None


def get_candidate_support(candidate_support, candidate_key):
    if not candidate_key:
        return 0
    value = candidate_support.get(candidate_key, 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def should_flag_ood(ood_score):
    try:
        return int(ood_score) >= 2
    except (TypeError, ValueError):
        return False


def resize_image_to_max_pixels(image, max_pixels):
    width, height = image.size
    current_pixels = width * height
    if current_pixels <= max_pixels:
        return image

    scale = math.sqrt(max_pixels / float(current_pixels))
    new_width = max(28, int(width * scale))
    new_height = max(28, int(height * scale))
    return image.resize((new_width, new_height))


def _lookup_probability(topk_records, class_index):
    for record in topk_records:
        if int(record["index"]) == int(class_index):
            return float(record["prob"])
    return 0.0


def _lookup_sources(sample, class_index):
    sources = []
    if any(
        int(record["index"]) == int(class_index)
        for record in normalize_topk_records(sample, "cloud_topk", None)
    ):
        sources.append("cloud")
    if any(
        int(record["index"]) == int(class_index)
        for record in normalize_topk_records(sample, "edge_topk", None)
    ):
        sources.append("edge")
    return sources


def normalize_topk_records(sample, topk_key, classes_list):
    topk_records = sample.get(topk_key, [])
    if topk_records:
        return topk_records

    if topk_key == "edge_topk" and "edge_pred" in sample:
        class_index = int(sample["edge_pred"])
        label = classes_list[class_index] if classes_list else sample.get("edge_label", "")
        return [{"index": class_index, "label": label, "prob": float(sample.get("edge_conf", 0.0))}]

    if topk_key == "cloud_topk" and "cloud_pred" in sample:
        class_index = int(sample["cloud_pred"])
        label = classes_list[class_index] if classes_list else sample.get("cloud_label", "")
        return [
            {"index": class_index, "label": label, "prob": float(sample.get("cloud_conf", 0.0))}
        ]

    return []


def _decode_json_fragment(value):
    try:
        return json.loads(f'"{value}"')
    except json.JSONDecodeError:
        return value.replace('\\"', '"')


def _extract_balanced_json_object(text):
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return None
