import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm # 进度条库

# 确保能导入根目录下的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataloader import get_cifar100_loaders
from models.edge.mobilenetv3 import build_edge_model
from models.cloud.vit import build_cloud_model

def train_model(model_type='edge', epochs=15, batch_size=32, lr=1e-3):
    """
    训练基准模型的函数
    :param model_type: 'edge' 或 'cloud'
    :param epochs: 训练轮数
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== 开始训练 {model_type} 模型 | 设备: {device} ===")

    # 1. 准备数据
    train_loader, test_loader = get_cifar100_loaders(
        train_batch_size=batch_size, 
        test_batch_size=32 # 验证时稍微大点可以测得快，协同推理时再设为1
    )

    # 2. 构建模型
    if model_type == 'edge':
        model = build_edge_model(num_classes=100)
    elif model_type == 'cloud':
        model = build_cloud_model(num_classes=100)
    else:
        raise ValueError("model_type 必须是 'edge' 或 'cloud'")
    
    model = model.to(device)

    # 3. 定义损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    # 使用 AdamW 优化器，它在 Fine-tune 预训练模型时表现非常好
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # 4. 创建保存权重的目录
    save_dir = './models/weights'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'best_{model_type}.pth')

    best_acc = 0.0 # 记录最高准确率

    # 5. 开始训练循环
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        
        # 使用 tqdm 包装 train_loader 产生进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"}) # 在进度条后显示Loss

        # 每个 Epoch 结束后，在测试集上验证准确率
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_acc = 100 * correct / total
        print(f"--> Epoch {epoch} 验证准确率: {epoch_acc:.2f}% | 平均Loss: {running_loss/len(train_loader):.4f}")

        # 如果准确率破纪录，则保存模型权重
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), save_path)
            print(f"    [★] 发现最佳模型！已保存至 {save_path}")

    print(f"=== {model_type} 训练完成！历史最高准确率: {best_acc:.2f}% ===")

if __name__ == "__main__":
    # 第一步：先训练端侧小模型 (大概需要十几分钟到半小时)
    #train_model(model_type='edge', epochs=15, batch_size=32, lr=1e-3)
    
    # 等端侧训练完，你可以把上面一行注释掉，取消下面这行的注释来训练云端大模型
    # 注意：ViT 模型较大，如果显存不够(Out of Memory)，请把 batch_size 改为 16 或 8，学习率 lr 降为 1e-4
    train_model(model_type='cloud', epochs=15, batch_size=16, lr=1e-4)