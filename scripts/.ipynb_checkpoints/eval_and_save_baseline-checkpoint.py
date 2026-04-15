import sys
import os
import time
import torch
import json
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataloader import get_food101_loaders
from models.edge.mobilenetv3 import build_edge_model
from models.cloud.vit import build_cloud_model
from core.router import calculate_confidence
from utils.logger import setup_logger

def save_baseline_results(logger=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if logger is None:
        logger = setup_logger("save_baseline")
        
    logger.info(f"--- 开始收集视觉端云模型基线数据 | 设备: {device} ---")

    # 获取测试集（batch_size=1）
    _, _, test_loader = get_food101_loaders(test_batch_size=1)
    
    # 实例化并加载最佳权重
    edge_model = build_edge_model(num_classes=101).to(device).eval()
    cloud_model = build_cloud_model(num_classes=101).to(device).eval()
    
    edge_model.load_state_dict(torch.load('./models/weights/best_edge.pth'))
    cloud_model.load_state_dict(torch.load('./models/weights/best_cloud.pth'))

    results_dict = {}
    
    logger.info("开始遍历测试集，提取并保存所有视觉模型的预测结果...")
    
    with torch.no_grad():
        for idx, (images, labels) in enumerate(tqdm(test_loader)):
            images = images.to(device)
            true_label = labels.item()
            
            # 获取当前图片的原始路径（便于后续大模型读取同一张图）
            # Food101 dataset 将图片路径存在 _image_files 中
            img_path = str(test_loader.dataset._image_files[idx])
            
            # 1. 端侧全量推理
            edge_logits = edge_model(images)
            edge_pred, edge_conf = calculate_confidence(edge_logits)
            
            # 2. 云端全量推理（为了给后续大模型提供最全面的信息，这里无论阈值多少，都跑一次云端记录下来）
            cloud_logits = cloud_model(images)
            cloud_pred, cloud_conf = calculate_confidence(cloud_logits)
            
            results_dict[img_path] = {
                "true_label": true_label,
                "edge_pred": edge_pred,
                "edge_conf": round(edge_conf, 4),
                "cloud_pred": cloud_pred,
                "cloud_conf": round(cloud_conf, 4)
            }

    # 将结果保存为 JSON
    save_dir = './results'
    os.makedirs(save_dir, exist_ok=True)
    out_file = os.path.join(save_dir, 'baseline_visual_results.json')
    
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=4, ensure_ascii=False)
        
    logger.info(f"所有预测结果已成功保存至 {out_file}")
    return results_dict

if __name__ == "__main__":
    save_baseline_results()