import torch
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


def extract_cloud_features(model, images):
    """Return the CLS embedding before the classification head."""
    x = model._process_input(images)
    batch_size = x.shape[0]

    class_token = model.class_token.expand(batch_size, -1, -1)
    x = torch.cat([class_token, x], dim=1)
    x = model.encoder(x)
    return x[:, 0]


def forward_cloud_with_features(model, images):
    features = extract_cloud_features(model, images)
    logits = model.heads(features)
    return logits, features
