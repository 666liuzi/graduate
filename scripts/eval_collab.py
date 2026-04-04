import sys
import os
import time
import torch

# 确保能导入根目录下的自定义模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataloader import get_cifar100_loaders
from models.edge.mobilenetv3 import build_edge_model
from models.cloud.vit import build_cloud_model
from core.router import calculate_confidence, requires_cloud_fallback

def run_collaborative_inference(threshold=0.8):
    """
    运行大小模型协同推理实验
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 开启端云协同实验 | 当前阈值: {threshold} | 设备: {device} ---")

    # 1. 加载数据与模型
    test_loader = get_cifar100_loaders(batch_size=1)
    edge_model = build_edge_model().to(device).eval()
    cloud_model = build_cloud_model().to(device).eval()

    # 指标记录器
    total_samples = 0
    correct_predictions = 0
    cloud_calls = 0  # 记录传给云端的次数
    total_time = 0.0 # 记录总推理延迟

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            total_samples += 1
            
            start_time = time.time()

            # 第一步：端侧计算 (边缘设备实时处理)
            edge_logits = edge_model(images)
            pred_class, confidence = calculate_confidence(edge_logits)

            # 第二步：路由决策判定 (是否向云端求助)
            if requires_cloud_fallback(confidence, threshold):
                # 触发云端大模型回退
                cloud_calls += 1
                cloud_logits = cloud_model(images)
                # 使用云端更准确的结果
                pred_class, _ = calculate_confidence(cloud_logits)
            
            # 记录时间与准确率
            total_time += (time.time() - start_time)
            if pred_class == labels.item():
                correct_predictions += 1

            # 打印前 5 个样本的详细处理过程（展示用）
            if total_samples <= 5:
                route = "云端" if requires_cloud_fallback(confidence, threshold) else "端侧"
                print(f"[样本 {total_samples}] 端侧置信度: {confidence:.3f} -> 采纳 {route} 结果")

    # 3. 计算最终评价指标
    accuracy = correct_predictions / total_samples
    cloud_offload_rate = cloud_calls / total_samples
    avg_latency = (total_time / total_samples) * 1000 # 转换为毫秒

    print("\n=== 协同推理结果 ===")
    print(f"总样本数: {total_samples}")
    print(f"系统最终准确率: {accuracy * 100:.2f}%")
    print(f"云端求助率: {cloud_offload_rate * 100:.2f}% (数字越小，节省带宽越多)")
    print(f"平均单样本延迟: {avg_latency:.2f} ms")
    print("==================\n")

    return accuracy, cloud_offload_rate, avg_latency

if __name__ == "__main__":
    # 你可以通过修改阈值，来画出 Trade-off（权衡）曲线
    # 阈值 0.0 = 全部用端侧跑 (低延迟，低精度)
    # 阈值 1.0 = 全部发云端跑 (高延迟，高精度)
    run_collaborative_inference(threshold=0.8)