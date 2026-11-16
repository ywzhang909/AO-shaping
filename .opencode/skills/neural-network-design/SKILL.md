---
name: neural-network-design
description: 设计和构建各种架构的神经网络，包括CNNs、RNNs、Transformers和注意力机制，使用PyTorch和TensorFlow。
---

# 神经网络设计

## 概览

这项技能涵盖了设计和实现包括CNNs、RNNs、Transformers和ResNets在内的神经网络架构，使用PyTorch和TensorFlow，重点关注架构选择、层组成和优化技术。

## 使用场景

- 为计算机视觉任务设计自定义神经网络架构，如图像分类或目标检测
- 构建序列模型用于时间序列预测、自然语言处理或视频分析
- 实施基于Transformer的模型用于语言理解或生成任务
- 创建结合CNNs、RNNs和注意力机制的混合架构
- 优化网络深度、宽度和跳跃连接以获得更好的训练和性能
- 选择适当的激活函数、归一化层和正则化技术

## 核心架构类型

- **前馈网络（MLPs）**：全连接层
- **卷积网络（CNNs）**：图像处理
- **递归网络（RNNs、LSTMs、GRUs）**：序列处理
- **Transformers**：基于自注意力的架构
- **混合模型**：结合多种架构类型

## 网络设计原则

- **深度与宽度**：层数和单元数之间的权衡
- **跳跃连接**：残差网络用于更深层次的训练
- **归一化**：批量归一化、层归一化以增加稳定性
- **正则化**：Dropout、L1/L2防止过拟合
- **激活函数**：ReLU、GELU、Swish提供非线性

## PyTorch和TensorFlow实现

```python
import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# 1. 前馈神经网络（MLP）
print("=== 1. 前馈神经网络 ===")

class MLPPyTorch(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

mlp = MLPPyTorch(input_size=784, hidden_sizes=[512, 256, 128], output_size=10)
print(f"MLP 参数： {sum(p.numel() for p in mlp.parameters()):,}")

# 2. 卷积神经网络（CNN）
print("\n=== 2. 卷积神经网络 ===")

class CNNPyTorch(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv blocks
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # 全连接层
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

cnn = CNNPyTorch()
print(f"CNN 参数： {sum(p.numel() for p in cnn.parameters()):,}")

# 3. 递归神经网络（LSTM）
print("\n=== 3. LSTM 网络 ===")

class LSTMPyTorch(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]
        output = self.fc(last_hidden)
        return output

lstm = LSTMPyTorch(input_size=100, hidden_size=128, num_layers=2, output_size=10)
print(f"LSTM 参数： {sum(p.numel() for p in lstm.parameters()):,}")

# 4. Transformer 块
print("\n=== 4. Transformer 架构 ===")

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # 自注意力
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # 前馈
        ff_out = self.feedforward(x)
        x = self.norm2(x + ff_out)
        return x

class TransformerPyTorch(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, 10)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = x.mean(dim=1)  # 全局平均池化
        x = self.fc(x)
        return x

transformer = TransformerPyTorch(vocab_size=1000, d_model=256, num_heads=8,
                                 num_layers=3, d_ff=512)
print(f"Transformer 参数： {sum(p.numel() for p in transformer.parameters()):,}")

# 5. 残差网络（ResNet）
print("\n=== 5. 残差网络 ===")

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class ResNetPyTorch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 10)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

resnet = ResNetPyTorch()
print(f"ResNet 参数： {sum(p.numel() for p in resnet.parameters()):,}")

# 6. TensorFlow Keras模型与自定义层
print("\n=== 6. TensorFlow Keras 模型 ===")

tf_model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.GlobalAveragePooling2D(),

    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])

print(f"TensorFlow 模型参数： {tf_model.count_params():,}")
tf_model.summary()

# 7. 模型比较
models_info = {
    'MLP': mlp,
    'CNN': cnn,
    'LSTM': lstm,
    'Transformer': transformer,
    'ResNet': resnet,
}

param_counts = {name: sum(p.numel() for p in model.parameters())
                for name, model in models_info.items()}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 参数计数
axes[0].barh(list(param_counts.keys()), list(param_counts.values()), color='steelblue')
axes[0].set_xlabel('参数数量')
axes[0].set_title('模型复杂度比较')
axes[0].set_xscale('log')

# 架构比较表
architectures = {
    'MLP': '前馈、密集层',
    'CNN': '卷积层、池化',
    'LSTM': '递归、长期记忆',
    'Transformer': '自注意力、并行处理',
    'ResNet': '残差连接、跳跃路径'
}

y_pos = np.arange(len(architectures))
axes[1].axis('off')
table_data = [[name, architectures[name]] for name in architectures.keys()]
table = axes[1].table(cellText=table_data, colLabels=['模型', '架构'],
                      cellLoc='left', loc='center', bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

plt.tight_layout()
plt.savefig('neural_network_architectures.png', dpi=100, bbox_inches='tight')
print("\n可视化保存为 'neural_network_architectures.png'")

print("\n神经网络设计分析完成！")
```

## 架构选择指南

- **MLP**：表格数据、简单分类
- **CNN**：图像分类、目标检测
- **LSTM/GRU**：时间序列、序列数据
- **Transformer**：NLP、长距离依赖
- **ResNet**：非常深的网络、图像任务

## 关键设计考虑因素

- 输入/输出形状兼容性
- CNNs的感受野大小
- RNNs的序列长度
- Transformers的注意力头数
- ResNets的跳跃连接放置

## 交付物

- 网络架构定义
- 参数计数分析
- 逐层描述
- 数据流向图
- 性能基准测试
- 部署要求
