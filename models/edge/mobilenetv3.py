import torch.nn as nn
from torchvision import models

def build_edge_model(num_classes=100):
    """端侧轻量小模型：MobileNetV3-Small"""
    # 加载预训练权重
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    
    # 替换分类头 (1000 -> 100)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    
    return model