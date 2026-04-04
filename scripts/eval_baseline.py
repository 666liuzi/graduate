import torch
import time
import sys
import os

# 1. 必须先将项目根目录加入路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 2. 然后再导入根目录下的本地库
import pytorch_cifar_models  
from data.dataloader import get_cifar100_test_loader

def evaluate_model(model, dataloader, device, num_samples=1000):
    model.eval()
    correct = 0
    total = 0
    
    # CUDA Warmup (预热)
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 32, 32).to(device)
        for _ in range(10): 
            _ = model(dummy_input)
            
    # 正式计时推理
    start_time = time.time()
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if total >= num_samples: 
                break
                
    end_time = time.time()
    accuracy = 100 * correct / total
    avg_latency = (end_time - start_time) / total
    return accuracy, avg_latency

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的计算设备: {device}")
    
    loader = get_cifar100_test_loader(batch_size=1)
    
    print("\n--- 加载并评估小模型 (ResNet20 - CIFAR100) ---")
    small_model = pytorch_cifar_models.cifar100_resnet20(pretrained=True).to(device)
    acc_small, lat_small = evaluate_model(small_model, loader, device)
    print(f"准确率: {acc_small:.2f}%, 单样本平均延迟: {lat_small*1000:.2f} ms")
    
    print("\n--- 加载并评估大模型 (ResNet110 - CIFAR100) ---")
    big_model = pytorch_cifar_models.cifar100_resnet110(pretrained=True).to(device)
    acc_big, lat_big = evaluate_model(big_model, loader, device)
    print(f"准确率: {acc_big:.2f}%, 单样本平均延迟: {lat_big*1000:.2f} ms")