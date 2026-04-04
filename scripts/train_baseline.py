import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataloader import get_food101_loaders
from models.edge.mobilenetv3 import build_edge_model
from models.cloud.vit import build_cloud_model
from utils.logger import setup_logger

def train_model(model_type='cloud', total_epochs=20, batch_size=16, base_lr=3e-4):
    logger = setup_logger(f"train_{model_type}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"=== 两阶段严谨微调 (含混合精度) {model_type} 模型 | 设备: {device} | Batch: {batch_size} ===")

    train_loader, val_loader, test_loader = get_food101_loaders(
        train_batch_size=batch_size, test_batch_size=32)

    if model_type == 'edge':
        model = build_edge_model(num_classes=101).to(device)
    else:
        model = build_cloud_model(num_classes=101).to(device)

    criterion = nn.CrossEntropyLoss()
    save_dir = './models/weights'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'best_{model_type}.pth')
    best_val_acc = 0.0

    # 【新增】初始化梯度缩放器，用于混合精度训练防下溢
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # ================= 第一阶段：Linear Probing (仅训练分类头) =================
    if model_type == 'cloud':
        warmup_epochs = 5
        logger.info(f"\n[Stage 1] 开始冻结骨干网络，仅训练分类头 {warmup_epochs} 轮...")
        
        for param in model.parameters():
            param.requires_grad = False
        for param in model.heads.head.parameters():
            param.requires_grad = True

        head_optimizer = optim.AdamW(model.heads.head.parameters(), lr=1e-3)

        for epoch in range(1, warmup_epochs + 1):
            model.train()
            running_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Stage 1 Epoch {epoch}/{warmup_epochs}")
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                head_optimizer.zero_grad()
                
                # 【新增】使用 autocast 开启前向传播的混合精度
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                # 【修改】使用 scaler 缩放 loss 并反向传播
                scaler.scale(loss).backward()
                scaler.step(head_optimizer)
                scaler.update()
                
                running_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # Stage 1 的验证集评估
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    # 推理时同样可以使用混合精度加速
                    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                        outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_acc = 100 * correct / total
            logger.info(f"--> Stage 1 Epoch {epoch} 验证集准确率: {val_acc:.2f}% | 平均Loss: {running_loss/len(train_loader):.4f}")

        for param in model.parameters():
            param.requires_grad = True
        logger.info("[Stage 1 完成] 骨干网络已解冻，准备进入全量微调阶段。\n")
    else:
        warmup_epochs = 0 

    # ================= 第二阶段：全量 Fine-Tuning =================
    ft_epochs = total_epochs - warmup_epochs
    logger.info(f"[Stage 2] 开始全量微调，共 {ft_epochs} 轮...")
    
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ft_epochs, eta_min=1e-6)

    for epoch in range(1, ft_epochs + 1):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Stage 2 Epoch {epoch}/{ft_epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # 【新增】混合精度前向传播
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, labels)

            # 【修改】混合精度反向传播与梯度裁剪的兼容处理
            scaler.scale(loss).backward()
            
            # 必须先 unscale 才能正确裁剪梯度
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{current_lr:.2e}"})
        
        scheduler.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        avg_loss = running_loss/len(train_loader)
        logger.info(f"--> Stage 2 Epoch {epoch} 验证集准确率: {val_acc:.2f}% | 平均Loss: {avg_loss:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            logger.info(f"    [★] 验证集表现提升，已保存最佳模型！(Acc: {best_val_acc:.2f}%)")

    # ================= 最终测试集评估 =================
    logger.info(f"\n=== 开始进行 {model_type} 最终测试集评估 ===")
    model.load_state_dict(torch.load(save_path))
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
    test_acc = 100 * test_correct / test_total
    logger.info(f"!!! 最终盲测 - 测试集准确率(Test Acc): {test_acc:.2f}% !!!\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练 CIFAR-100 端云协同模型")
    parser.add_argument('--model', type=str, choices=['edge', 'cloud'], required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)

    args = parser.parse_args()
    train_model(model_type=args.model, total_epochs=args.epochs, batch_size=args.batch_size, base_lr=args.lr)