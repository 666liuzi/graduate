import json
import math
import re
from pathlib import Path

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_CLASSES_PATH = DEFAULT_DATA_DIR / "food-101" / "meta" / "classes.txt"


def canonicalize_label(label):
    return label.strip().lower().replace(" ", "_")


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
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1)

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None


def normalize_candidate_choice(choice):
    if not choice:
        return None

    cleaned = str(choice).strip().upper()
    match = re.search(r"[A-F]", cleaned)
    return match.group(0) if match else None


def prepare_topk_predictions(logits, classes_list, topk=3):
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
