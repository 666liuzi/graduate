import argparse
import os
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataloader import get_food101_loaders
from models.cloud.vit import extract_cloud_features, build_cloud_model
from utils.logger import setup_logger
from utils.pipeline import load_food101_classes, load_json, resolve_image_path, save_json


def build_prototype_bank(
    baseline_path="./results/baseline_visual_results.json",
    output_path="./results/prototype_bank.json",
    data_dir="./data",
    batch_size=32,
    confidence_threshold=0.8,
    logger=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if logger is None:
        logger = setup_logger("build_prototype_bank")

    classes_list = load_food101_classes(data_dir=data_dir)
    baseline_results = load_json(baseline_path)

    _, _, test_loader = get_food101_loaders(test_batch_size=batch_size, data_dir=data_dir)
    dataset = test_loader.dataset

    cloud_model = build_cloud_model(num_classes=len(classes_list)).to(device).eval()
    cloud_model.load_state_dict(torch.load("./models/weights/best_cloud.pth", map_location=device))

    entries = []
    feature_batches = []
    offset = 0

    logger.info("开始提取 cloud ViT 特征，用于构建每类原型图...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device, non_blocking=True)
            features = extract_cloud_features(cloud_model, images)
            features = F.normalize(features, dim=1).cpu()
            feature_batches.append(features)

            batch_size_actual = labels.size(0)
            raw_paths = [str(dataset._image_files[offset + idx]) for idx in range(batch_size_actual)]
            offset += batch_size_actual

            for idx, raw_path in enumerate(raw_paths):
                baseline_record = baseline_results[raw_path]
                entries.append(
                    {
                        "raw_path": raw_path,
                        "resolved_path": resolve_image_path(raw_path, data_dir=data_dir),
                        "true_label": int(labels[idx].item()),
                        "cloud_pred": int(baseline_record["cloud_pred"]),
                        "cloud_conf": float(baseline_record["cloud_conf"]),
                    }
                )

    all_features = torch.cat(feature_batches, dim=0)
    prototype_bank = {}

    logger.info("开始为 101 个类别选择 2 张原型图...")
    for class_index, class_name in enumerate(classes_list):
        class_entry_indices = [
            idx for idx, entry in enumerate(entries) if int(entry["true_label"]) == int(class_index)
        ]
        class_features = all_features[class_entry_indices]
        class_center = F.normalize(class_features.mean(dim=0, keepdim=True), dim=1)
        center_similarities = torch.matmul(class_features, class_center.T).squeeze(1)
        prototype_one_local = int(torch.argmax(center_similarities).item())
        prototype_one_index = class_entry_indices[prototype_one_local]

        prototype_two_pool = [
            idx
            for idx in class_entry_indices
            if idx != prototype_one_index
            and entries[idx]["cloud_pred"] == class_index
            and entries[idx]["cloud_conf"] >= confidence_threshold
        ]
        if not prototype_two_pool:
            prototype_two_pool = [
                idx
                for idx in class_entry_indices
                if idx != prototype_one_index and entries[idx]["cloud_pred"] == class_index
            ]
        if not prototype_two_pool:
            prototype_two_pool = [idx for idx in class_entry_indices if idx != prototype_one_index]
        if not prototype_two_pool:
            prototype_two_pool = [prototype_one_index]

        prototype_one_feature = all_features[prototype_one_index]
        prototype_two_features = all_features[prototype_two_pool]
        diversity_scores = 1.0 - torch.matmul(prototype_two_features, prototype_one_feature)
        prototype_two_local = int(torch.argmax(diversity_scores).item())
        prototype_two_index = prototype_two_pool[prototype_two_local]

        prototype_bank[class_name] = {
            "class_index": class_index,
            "prototypes": [
                {
                    "path": entries[prototype_one_index]["resolved_path"],
                    "raw_path": entries[prototype_one_index]["raw_path"],
                    "selection_role": "center_nearest",
                    "center_similarity": round(
                        float(center_similarities[prototype_one_local].item()), 4
                    ),
                    "cloud_conf": round(float(entries[prototype_one_index]["cloud_conf"]), 4),
                },
                {
                    "path": entries[prototype_two_index]["resolved_path"],
                    "raw_path": entries[prototype_two_index]["raw_path"],
                    "selection_role": "diverse_high_confidence",
                    "diversity_distance": round(
                        float(diversity_scores[prototype_two_local].item()), 4
                    ),
                    "cloud_conf": round(float(entries[prototype_two_index]["cloud_conf"]), 4),
                },
            ],
        }

    save_json(prototype_bank, output_path)
    logger.info(f"原型图选择完成，结果已保存至 {output_path}")
    return prototype_bank


def parse_args():
    parser = argparse.ArgumentParser(description="Build Food-101 prototype bank from cloud features.")
    parser.add_argument("--baseline", default="./results/baseline_visual_results.json")
    parser.add_argument("--output", default="./results/prototype_bank.json")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--confidence_threshold", type=float, default=0.8)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    build_prototype_bank(
        baseline_path=cli_args.baseline,
        output_path=cli_args.output,
        data_dir=cli_args.data_dir,
        batch_size=cli_args.batch_size,
        confidence_threshold=cli_args.confidence_threshold,
    )
