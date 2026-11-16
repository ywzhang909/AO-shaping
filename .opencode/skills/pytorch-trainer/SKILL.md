---
name: pytorch-trainer
description: PyTorch模型训练技能，包含自定义训练循环、梯度管理和GPU优化。
allowed-tools:
  - Read
  - Write
  - Bash
  - Glob
  - Grep
---

# pytorch-trainer

## 概述

PyTorch模型训练技能，包含自定义训练循环、梯度管理、GPU优化以及与实验跟踪系统的集成。

## 能力

- 自定义训练循环执行
- 学习率调度（StepLR、CosineAnnealing、OneCycleLR等）
- 梯度裁剪和累积
- 混合精度训练（AMP）
- 检查点管理和恢复
- DataLoader优化
- 多GPU训练（DataParallel、DistributedDataParallel）
- 带耐心值的早停机制

## 目标流程

- 带实验跟踪的模型训练流水线
- 分布式训练编排
- AutoML流水线编排

## 工具和库

- PyTorch
- PyTorch Lightning（可选）
- torchvision, torchaudio, torchtext
- CUDA工具包

## 输入模式

```json
{
  "type": "object",
  "required": ["modelPath", "dataConfig", "trainingConfig"],
  "properties": {
    "modelPath": {
      "type": "string",
      "description": "模型定义文件的路径"
    },
    "dataConfig": {
      "type": "object",
      "properties": {
        "trainPath": { "type": "string" },
        "valPath": { "type": "string" },
        "batchSize": { "type": "integer" },
        "numWorkers": { "type": "integer" }
      }
    },
    "trainingConfig": {
      "type": "object",
      "properties": {
        "epochs": { "type": "integer" },
        "learningRate": { "type": "number" },
        "optimizer": { "type": "string" },
        "scheduler": { "type": "string" },
        "mixedPrecision": { "type": "boolean" },
        "gradientClipping": { "type": "number" },
        "gradientAccumulation": { "type": "integer" }
      }
    },
    "checkpointConfig": {
      "type": "object",
      "properties": {
        "saveDir": { "type": "string" },
        "saveEvery": { "type": "integer" },
        "resumeFrom": { "type": "string" }
      }
    }
  }
}
```

## 输出模式

```json
{
  "type": "object",
  "required": ["status", "metrics", "checkpointPath"],
  "properties": {
    "status": {
      "type": "string",
      "enum": ["success", "error", "early_stopped"]
    },
    "metrics": {
      "type": "object",
      "properties": {
        "trainLoss": { "type": "number" },
        "valLoss": { "type": "number" },
        "trainAccuracy": { "type": "number" },
        "valAccuracy": { "type": "number" },
        "epochsTrained": { "type": "integer" },
        "trainingTime": { "type": "number" }
      }
    },
    "checkpointPath": {
      "type": "string"
    },
    "learningCurve": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "epoch": { "type": "integer" },
          "trainLoss": { "type": "number" },
          "valLoss": { "type": "number" }
        }
      }
    }
  }
}
```

## 使用示例

```javascript
{
  kind: 'skill',
  title: '训练PyTorch模型',
  skill: {
    name: 'pytorch-trainer',
    context: {
      modelPath: 'models/resnet.py',
      dataConfig: {
        trainPath: 'data/train',
        valPath: 'data/val',
        batchSize: 32,
        numWorkers: 4
      },
      trainingConfig: {
        epochs: 100,
        learningRate: 0.001,
        optimizer: 'AdamW',
        scheduler: 'cosine',
        mixedPrecision: true,
        gradientClipping: 1.0
      }
    }
  }
}
```
