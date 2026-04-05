import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

def get_food101_loaders(train_batch_size=32, test_batch_size=1, data_dir='./data', val_split=0.1):
    """
    Food-101 数据加载器：包含 Train, Val, Test 三重划分与高分辨率强增强
    """
    # 训练集：强力数据增强 (对抗 ViT 过拟合)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), 
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),             
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 测试/验证集：标准 ImageNet 预处理 (保证评估严谨性)
    test_transform = transforms.Compose([
        transforms.Resize(256),                    
        transforms.CenterCrop(224),                
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 注意：Food101 使用 split 参数 ('train' 或 'test')
    full_train_dataset_with_aug = torchvision.datasets.Food101(
        root=data_dir, split='train', download=True, transform=train_transform)
    
    full_train_dataset_no_aug = torchvision.datasets.Food101(
        root=data_dir, split='train', download=True, transform=test_transform)

    # 划分 Train 和 Val
    total_size = len(full_train_dataset_with_aug) # Food-101 训练集共 75750 张
    train_size = int((1 - val_split) * total_size)
    
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(total_size, generator=generator).tolist()
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_subset = Subset(full_train_dataset_with_aug, train_indices)
    val_subset = Subset(full_train_dataset_no_aug, val_indices)

    # 独立的测试集 (25250 张)
    test_dataset = torchvision.datasets.Food101(
        root=data_dir, split='test', download=True, transform=test_transform)

    train_loader = DataLoader(train_subset, batch_size=train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=test_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader