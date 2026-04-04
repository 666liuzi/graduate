import sys
import os
import time
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataloader import get_food101_loaders
from models.edge.mobilenetv3 import build_edge_model
from models.cloud.vit import build_cloud_model
from core.router import calculate_confidence, requires_cloud_fallback
from utils.logger import setup_logger

def run_collaborative_inference(threshold=0.8, logger=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if logger is None:
        logger = setup_logger("eval_collab")
        
    logger.info(f"--- 开启端云协同实验 | 当前阈值: {threshold} | 设备: {device} ---")

    # 加载测试数据（协同推理时 batch_size 必须为 1，模拟实时单样本流入）
    _, _, test_loader = get_food101_loaders(test_batch_size=1)
    
    # 实例化模型并加载微调后的权重
    edge_model = build_edge_model(num_classes=101).to(device).eval()
    cloud_model = build_cloud_model(num_classes=101).to(device).eval()
    
    edge_weight_path = './models/weights/best_edge.pth'
    cloud_weight_path = './models/weights/best_cloud.pth'
    
    if os.path.exists(edge_weight_path) and os.path.exists(cloud_weight_path):
        edge_model.load_state_dict(torch.load(edge_weight_path))
        cloud_model.load_state_dict(torch.load(cloud_weight_path))
    else:
        logger.warning("警告：未找到预训练权重，模型将使用初始随机权重进行推理！请先运行 train_baseline.py")

    total_samples = 0
    correct_predictions = 0
    cloud_calls = 0  
    total_time = 0.0 

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            total_samples += 1
            
            start_time = time.time()

            # 端侧初筛
            edge_logits = edge_model(images)
            pred_class, confidence = calculate_confidence(edge_logits)

            # 路由决策
            if requires_cloud_fallback(confidence, threshold):
                cloud_calls += 1
                cloud_logits = cloud_model(images)
                pred_class, _ = calculate_confidence(cloud_logits)
            
            total_time += (time.time() - start_time)
            if pred_class == labels.item():
                correct_predictions += 1

    accuracy = correct_predictions / total_samples
    cloud_offload_rate = cloud_calls / total_samples
    avg_latency = (total_time / total_samples) * 1000 

    logger.info("=== 协同推理结果 ===")
    logger.info(f"设定阈值: {threshold}")
    logger.info(f"总样本数: {total_samples}")
    logger.info(f"系统最终准确率: {accuracy * 100:.2f}%")
    logger.info(f"云端求助率(Offload Rate): {cloud_offload_rate * 100:.2f}%")
    logger.info(f"平均单样本延迟: {avg_latency:.2f} ms")
    logger.info("==================\n")

    return accuracy, cloud_offload_rate, avg_latency

if __name__ == "__main__":
    # 使用统一的 logger 记录多次测试
    main_logger = setup_logger("eval_collab_batch")
    
    # 批量测试多个阈值，收集绘制 Trade-off 曲线的数据
    thresholds = [0.0, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0]
    main_logger.info(f"准备测试的阈值列表: {thresholds}")
    
    for t in thresholds:
        run_collaborative_inference(threshold=t, logger=main_logger)