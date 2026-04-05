# 基于大小模型协同的泛在感知任务执行
**Ubiquitous Perception Task Execution Based on Big-Small Model Collaboration**

本项目为端云协同推理系统（Edge-Cloud Collaborative Inference System）的仿真实现，旨在解决泛在感知场景下边缘设备算力受限与深度学习大模型高资源消耗之间的矛盾。系统通过构建“端侧初筛+云端兜底”的动态协同架构，实现推理精度与系统延迟的最优权衡（Trade-off）。

## 🌟 核心特性 (Features)

- **高效端侧初筛 (Edge Inference)**：采用轻量级 `MobileNetV3-Small`，部署于资源受限的边缘端，拦截并处理高置信度的“简单样本”，极大地降低系统平均延迟。
- **高精度云端兜底 (Cloud Fallback)**：采用大视觉模型 `ViT-Base` 作为云端专家模型，处理长尾与困难样本。
- **动态协同路由 (Adaptive Routing)**：基于最大 Softmax 概率（MSP, Maximum Softmax Probability）设计自适应阈值路由算法，实现数据的智能分流。
- **前沿微调策略 (Advanced Fine-tuning)**：
  - 支持 **AMP (自动混合精度)** 训练，大幅降低显存开销。
  - 针对 ViT 创新的**两阶段微调机制**（Linear Probing -> Full Fine-Tuning）。
  - **差异化参数分组学习率**（主干 `1e-5`，分类头 `1e-4`），彻底解决大模型在下游细粒度数据集微调时的“灾难性遗忘”问题。

## 📊 当前阶段实验结果 (Results)

数据集：**Food-101** (101类细粒度图像，包含强数据增强预处理)

| 模型角色 | 网络架构 | 参数量 | 测试集盲测准确率 (Test Acc) | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| **端侧 (Edge)** | MobileNetV3-Small | ~2.5M | **76.23%** | 20轮全量微调，快速推理 |
| **云端 (Cloud)** | ViT-Base | ~86M | **86.36%** | 两阶段微调 + 差异化学习率 |
| **协同 (Collab)**| MobileNetV3 + ViT | - | **待测** | 性能阶梯 10.13%，具备完美协同潜力 |

## 🛠️ 环境依赖 (Requirements)

建议使用 Python 3.8+ 及支持 CUDA 的 GPU 运行本系统。

```bash
pip install torch torchvision tqdm
```

## 🚀 快速开始 (Quick Start)

由于项目代码采用模块化结构，**请务必在项目根目录下执行所有脚本**，以避免路径引用的模块未找到错误。

### 1. 数据准备
系统会自动下载并解压 Food-101 数据集到 `./data` 目录下。

### 2. 训练端侧小模型 (Edge Model)
推荐使用较大的 Batch Size（如 128）以充分利用显卡算力：
```bash
python scripts/train_baseline.py --model edge --epochs 20 --batch_size 128
```
*最优权重将自动保存至 `./models/weights/best_edge.pth`*

### 3. 训练云端大模型 (Cloud Model)
云端 ViT 模型显存占用较大，推荐 Batch Size 设为 16。脚本内部已自动配置好两阶段训练与差异化学习率：
```bash
python scripts/train_baseline.py --model cloud --epochs 20 --batch_size 16
```
*最优权重将自动保存至 `./models/weights/best_cloud.pth`*

### 4. 运行端云协同仿真 (Collaborative Evaluation)
当端云模型的预训练权重均准备完毕后，运行协同脚本。该脚本将遍历多个置信度阈值，评估系统的综合准确率与云端卸载率（Offload Rate）：
```bash
python scripts/eval_collab.py
```

## 📂 目录结构 (Directory Structure)

```text
graduate/
├── core/
│   └── router.py            # 基于 MSP 的核心协同路由判定算法
├── data/
│   └── dataloader.py        # Food-101 数据集加载器（包含数据增强策略）
├── models/
│   ├── edge/
│   │   └── mobilenetv3.py   # 端侧轻量级模型架构定义
│   ├── cloud/
│   │   └── vit.py           # 云端 Vision Transformer 架构定义
│   └── weights/             # 训练产出的最佳模型权重保存目录 (git ignored)
├── scripts/
│   ├── train_baseline.py    # 统一的模型微调训练主脚本（支持 edge/cloud）
│   └── eval_collab.py       # 端云协同推理仿真与评估脚本
├── utils/
│   └── logger.py            # 终端与文件双向日志记录器
└── results/
    └── logs/                # 历史训练日志存档 (git ignored)
```

## 📅 下一步计划 (TODO)
- [x] 重构数据管线，接入更复杂的 Food-101 数据集。
- [x] 完成双模型微调，实现 >10% 的理想性能阶梯。
- [ ] 在 `eval_collab.py` 中注入真实的端云网络通信延迟（Network Latency）模拟机制。
- [ ] 量化分析双模型的计算量（FLOPs）。
- [ ] 绘制并分析精度-延迟权衡（Accuracy-Latency Trade-off）Pareto 边界曲线。
```
