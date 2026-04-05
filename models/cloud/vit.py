import torch.nn as nn
from torchvision import models

def build_cloud_model(num_classes=101):
    """云端高性能大模型：ViT-Base"""
    # 加载预训练权重
    weights = models.ViT_B_16_Weights.DEFAULT
    model = models.vit_b_16(weights=weights)
    
    # 替换分类头 (1000 -> 100)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    
    return model