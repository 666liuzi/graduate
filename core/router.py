import torch
import torch.nn.functional as F

def calculate_confidence(logits):
    """
    计算模型的置信度 (Maximum Softmax Probability)
    输入: logits (未经过 softmax 的输出)
    输出: 预测的类别索引, 对应的置信度 (0~1之间)
    """
    probs = F.softmax(logits, dim=1)
    confidence, predicted_class = torch.max(probs, dim=1)
    return predicted_class.item(), confidence.item()

def requires_cloud_fallback(confidence, threshold):
    """
    路由判定算法：
    如果端侧置信度低于设定的阈值(threshold)，则认为端侧"算不准"，需要求助云端。
    """
    return confidence < threshold