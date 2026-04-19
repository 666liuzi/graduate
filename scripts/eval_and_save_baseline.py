import argparse
import os
import sys

import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataloader import get_food101_loaders
from models.cloud.vit import build_cloud_model
from models.edge.mobilenetv3 import build_edge_model
from utils.logger import setup_logger
from utils.pipeline import (
    compute_lock_metadata,
    compute_topk_margin,
    load_food101_classes,
    prepare_topk_predictions,
    save_json,
)


def save_baseline_results(
    output_path="./results/baseline_visual_results.json",
    data_dir="./data",
    batch_size=32,
    topk=3,
    max_samples=None,
    logger=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if logger is None:
        logger = setup_logger("save_baseline")

    logger.info(f"--- 开始收集视觉端云模型基线数据 | 设备: {device} ---")

    classes_list = load_food101_classes(data_dir=data_dir)
    _, _, test_loader = get_food101_loaders(test_batch_size=batch_size, data_dir=data_dir)
    dataset = test_loader.dataset

    edge_model = build_edge_model(num_classes=len(classes_list)).to(device).eval()
    cloud_model = build_cloud_model(num_classes=len(classes_list)).to(device).eval()

    edge_model.load_state_dict(torch.load("./models/weights/best_edge.pth", map_location=device))
    cloud_model.load_state_dict(torch.load("./models/weights/best_cloud.pth", map_location=device))

    results_dict = {}
    logger.info("开始遍历测试集，提取 top-k、margin 和锁定信息...")

    processed = 0
    offset = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            edge_logits = edge_model(images)
            cloud_logits = cloud_model(images)

            edge_topk_batch = prepare_topk_predictions(edge_logits, classes_list, topk=topk)
            cloud_topk_batch = prepare_topk_predictions(cloud_logits, classes_list, topk=topk)

            batch_size_actual = labels.size(0)
            raw_paths = [str(dataset._image_files[offset + idx]) for idx in range(batch_size_actual)]
            offset += batch_size_actual

            for idx, raw_path in enumerate(raw_paths):
                edge_topk = edge_topk_batch[idx]
                cloud_topk = cloud_topk_batch[idx]
                lock_metadata = compute_lock_metadata(edge_topk, cloud_topk)

                sample_record = {
                    "true_label": int(labels[idx].item()),
                    "edge_pred": int(edge_topk[0]["index"]),
                    "edge_conf": edge_topk[0]["prob"],
                    "edge_topk": edge_topk,
                    "edge_margin": compute_topk_margin(edge_topk),
                    "cloud_pred": int(cloud_topk[0]["index"]),
                    "cloud_conf": cloud_topk[0]["prob"],
                    "cloud_topk": cloud_topk,
                    "cloud_margin": compute_topk_margin(cloud_topk),
                    **lock_metadata,
                }

                results_dict[raw_path] = sample_record
                processed += 1

                if max_samples is not None and processed >= max_samples:
                    save_json(results_dict, output_path)
                    logger.info(f"已导出 {processed} 条基线记录至 {output_path}")
                    return results_dict

    save_json(results_dict, output_path)
    logger.info(f"所有预测结果已成功保存至 {output_path}")
    return results_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Export edge/cloud top-k baseline results.")
    parser.add_argument("--output", default="./results/baseline_visual_results.json")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    save_baseline_results(
        output_path=cli_args.output,
        data_dir=cli_args.data_dir,
        batch_size=cli_args.batch_size,
        topk=cli_args.topk,
        max_samples=cli_args.max_samples,
    )
