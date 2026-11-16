---
name: model-training
description: 使用PyTorch进行机器学习模型训练工作流程的全面指南，涵盖数据准备、训练循环、超参数调优、实验跟踪、检查点管理、早停、学习率调度、分布式训练和模型评估。
---

# 模型训练

## 概述

模型训练是教导机器学习模型基于数据做出预测或决策的过程。此技能涵盖全面的训练工作流程，包括管道设计、数据准备、训练循环、超参数调优、实验跟踪、检查点管理、早停、学习率调度、分布式训练和模型评估。

## 前提条件

- 理解PyTorch和深度学习基础知识
- 神经网络架构知识
- 熟悉数据预处理和增强
- 理解损失函数和优化器
- 机器学习指标基础知识

## 关键概念

### 训练管道架构

- **模块化设计**: 数据、模型、优化器和训练逻辑的分离
- **配置管理**: 基于YAML的配置以实现可重复性
- **检查点管理**: 保存和加载模型状态
- **早停**: 通过早停防止过拟合
- **实验跟踪**: 记录指标和超参数

### 数据准备

- **训练/验证/测试分割**: 适当的模型评估数据划分
- **自定义数据集**: 实现PyTorch Dataset类
- **数据加载器**: 高效的数据加载，支持批处理和洗牌
- **数据增强**: 增加数据多样性以提高泛化能力

### 训练循环模式

- **基本训练循环**: 标准前向/后向传播
- **混合精度训练**: 使用FP16加速训练
- **梯度累积**: 模拟更大的批大小
- **分布式训练**: 多GPU和多节点训练

### 超参数调优

- **网格搜索**: 参数空间中的穷举搜索
- **随机搜索**: 参数的随机采样
- **贝叶斯优化**: 使用Optuna的智能参数探索

### 学习率调度

- **StepLR**: 周期性学习率衰减
- **CosineAnnealingLR**: 余弦退火调度
- **ReduceLROnPlateau**: 基于指标的自适应学习率
- **OneCycleLR**: 单周期学习率策略

## 实现指南

### 训练管道设计

#### 管道架构

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import yaml
from typing import Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingPipeline:
    """具有模块化的完整训练管道。"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

    def setup_data(self):
        """设置数据加载器。"""
        raise NotImplementedError

    def setup_model(self):
        """设置模型架构。"""
        raise NotImplementedError

    def setup_optimizer(self):
        """设置优化器和调度器。"""
        raise NotImplementedError

    def train_epoch(self, epoch: int):
        """训练一个轮次。"""
        raise NotImplementedError

    def validate(self, epoch: int):
        """验证模型。"""
        raise NotImplementedError

    def train(self):
        """主训练循环。"""
        logger.info(f"Starting training on {self.device}")

        best_metric = float('inf') if self.config.get('minimize_metric', True) else 0.0
        patience_counter = 0

        for epoch in range(self.config['epochs']):
            # 训练
            train_metrics = self.train_epoch(epoch)

            # 验证
            val_metrics = self.validate(epoch)

            # 记录指标
            self._log_metrics(epoch, train_metrics, val_metrics)

            # 学习率调度
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics[self.config['monitor_metric']])
                else:
                    self.scheduler.step()

            # 保存检查点
            self._save_checkpoint(epoch, val_metrics)

            # 早停
            current_metric = val_metrics[self.config['monitor_metric']]
            is_better = (current_metric < best_metric) if self.config.get('minimize_metric', True) \
                       else (current_metric > best_metric)

            if is_better:
                best_metric = current_metric
                patience_counter = 0
                self._save_best_model(epoch, val_metrics)
            else:
                patience_counter += 1
                if patience_counter >= self.config.get('early_stopping_patience', 10):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

    def _log_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """记录训练和验证指标。"""
        logger.info(f"Epoch {epoch}: Train {train_metrics} | Val {val_metrics}")

    def _save_checkpoint(self, epoch: int, metrics: Dict):
        """保存训练检查点。"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config
        }

        path = Path(self.config['checkpoint_dir']) / f"checkpoint_epoch_{epoch}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

    def _save_best_model(self, epoch: int, metrics: Dict):
        """保存最佳模型。"""
        path = Path(self.config['checkpoint_dir']) / "best_model.pt"
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'epoch': epoch,
            'metrics': metrics
        }, path)

    def load_checkpoint(self, checkpoint_path: str):
        """加载训练检查点。"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint['metrics']
```

#### 配置管理

```python
# config.yaml
model:
  name: "resnet50"
  num_classes: 10
  pretrained: true

data:
  data_dir: "./data"
  batch_size: 32
  num_workers: 4
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1

training:
  epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001
  momentum: 0.9

optimizer:
  type: "Adam"
  betas: [0.9, 0.999]

scheduler:
  type: "CosineAnnealingLR"
  T_max: 100
  eta_min: 0.00001

early_stopping:
  patience: 10
  monitor_metric: "val_loss"
  minimize_metric: true

checkpoint_dir: "./checkpoints"
log_dir: "./logs"
```

```python
import yaml

def load_config(config_path: str) -> Dict:
    """从YAML文件加载配置。"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# 使用
config = load_config("config.yaml")
pipeline = TrainingPipeline(config)
```

### 数据准备

#### 训练/验证/测试分割

```python
from torch.utils.data import random_split, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

def create_splits(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """创建训练/验证/测试分割。"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例总和必须为1"

    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    return train_dataset, val_dataset, test_dataset

def stratified_split(dataset, labels, train_ratio=0.8, val_ratio=0.1, seed=42):
    """为分类创建分层分割。"""
    train_indices, temp_indices = train_test_split(
        np.arange(len(dataset)),
        test_size=(1 - train_ratio),
        stratify=labels,
        random_state=seed
    )

    val_ratio_adjusted = val_ratio / (val_ratio + (1 - train_ratio - val_ratio))
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=(1 - val_ratio_adjusted),
        stratify=[labels[i] for i in temp_indices],
        random_state=seed
    )

    return train_indices, val_indices, test_indices
```

#### 自定义数据集

```python
from torch.utils.data import Dataset
from PIL import Image
import json
from pathlib import Path

class CustomImageDataset(Dataset):
    """自定义图像数据集。"""

    def __init__(self, data_dir, transform=None, split='train'):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split

        # 加载注释
        annotation_file = self.data_dir / f"{split}_annotations.json"
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)

        self.image_paths = list(self.annotations.keys())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 加载图像
        image_path = self.data_dir / self.split / "images" / self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        # 获取标签
        label = self.annotations[self.image_paths[idx]]['label']

        # 应用变换
        if self.transform:
            image = self.transform(image)

        return image, label

class CustomTextDataset(Dataset):
    """自定义文本数据集。"""

    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
```

#### 数据加载器

```python
from torchvision import transforms

def create_data_loaders(config):
    """使用变换创建数据加载器。"""

    # 定义变换
    train_transform = transforms.Compose([
        transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集
    full_dataset = CustomImageDataset(
        config['data']['data_dir'],
        transform=None
    )

    train_dataset, val_dataset, test_dataset = create_splits(full_dataset)

    # 应用变换
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
```

#### 数据增强

```python
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

# PyTorch变换
advanced_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Albumentations变换
albumentations_transform = A.Compose([
    A.Resize(256, 256),
    A.RandomCrop(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.Rotate(limit=30, p=0.5),
    A.OneOf([
        A.GaussNoise(p=1.0),
        A.ISONoise(p=1.0),
    ], p=0.2),
    A.OneOf([
        A.MotionBlur(p=1.0),
        A.MedianBlur(p=1.0),
        A.GaussianBlur(p=1.0),
    ], p=0.2),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.3),
    A.Cutout(num_holes=8, max_h_size=16, max_w_size=16, p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

class AlbumentationsDataset(Dataset):
    """使用Albumentations变换的数据集。"""

    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image=image)['image']

        return image, label
```

### 训练循环模式

#### 基本训练循环

```python
import torch
import torch.nn as nn
from tqdm import tqdm

def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个轮次。"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="训练")
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 后向传播
        loss.backward()
        optimizer.step()

        # 指标
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # 更新进度条
        pbar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total

    return {'loss': avg_loss, 'accuracy': accuracy}
```

#### 验证循环

```python
def validate(model, val_loader, criterion, device):
    """验证模型。"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="验证"):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total

    return {'loss': avg_loss, 'accuracy': accuracy}
```

#### 使用混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

def train_epoch_amp(model, train_loader, criterion, optimizer, device, scaler=None):
    """使用自动混合精度训练。"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    if scaler is None:
        scaler = GradScaler()

    for inputs, targets in tqdm(train_loader, desc="训练(AMP)"):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total

    return {'loss': avg_loss, 'accuracy': accuracy}
```

#### 使用梯度累积训练

```python
def train_epoch_accumulation(model, train_loader, criterion, optimizer, device, accumulation_steps=4):
    """使用梯度累积训练以获取更大的有效批大小。"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad()

    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc="训练")):
        inputs, targets = inputs.to(device), targets.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets) / accumulation_steps

        # 后向传播
        loss.backward()

        # 累积梯度
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # 指标
        total_loss += loss.item() * accumulation_steps
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    # 处理剩余批次
    if (batch_idx + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total

    return {'loss': avg_loss, 'accuracy': accuracy}
```

### 超参数调优

#### 网格搜索

```python
import itertools
from copy import deepcopy

def grid_search(model_class, train_loader, val_loader, param_grid, device):
    """在超参数上执行网格搜索。"""
    results = []

    # 生成所有组合
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for params in combinations:
        print(f"\n测试参数: {params}")

        # 使用当前参数创建模型
        model = model_class(**params)
        model = model.to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=params.get('learning_rate', 0.001)
        )

        criterion = nn.CrossEntropyLoss()

        # 训练几个轮次
        val_acc = 0
        for epoch in range(5):  # 快速训练
            train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = validate(model, val_loader, criterion, device)
            val_acc = val_metrics['accuracy']

        results.append({
            'params': params,
            'val_accuracy': val_acc
        })

    # 按验证准确率排序
    results.sort(key=lambda x: x['val_accuracy'], reverse=True)

    return results

# 使用
param_grid = {
    'learning_rate': [0.001, 0.0001, 0.00001],
    'dropout': [0.1, 0.3, 0.5],
    'hidden_size': [128, 256, 512]
}

results = grid_search(MyModel, train_loader, val_loader, param_grid, device)
print(f"最佳参数: {results[0]['params']}")
```

#### 随机搜索

```python
import random

def random_search(model_class, train_loader, val_loader, param_ranges, n_iterations=20, device='cuda'):
    """在超参数上执行随机搜索。"""
    results = []

    for i in range(n_iterations):
        # 采样随机参数
        params = {}
        for key, (min_val, max_val, is_log) in param_ranges.items():
            if is_log:
                params[key] = 10 ** random.uniform(min_val, max_val)
            else:
                params[key] = random.uniform(min_val, max_val)

        print(f"\n迭代 {i+1}/{n_iterations}: {params}")

        # 训练和评估
        model = model_class(**params)
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
        criterion = nn.CrossEntropyLoss()

        val_acc = 0
        for epoch in range(5):
            train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = validate(model, val_loader, criterion, device)
            val_acc = val_metrics['accuracy']

        results.append({
            'params': params,
            'val_accuracy': val_acc
        })

    results.sort(key=lambda x: x['val_accuracy'], reverse=True)
    return results

# 使用
param_ranges = {
    'learning_rate': (-4, -2, True),  # 对数尺度: 10^-4 到 10^-2
    'dropout': (0.1, 0.5, False),
    'hidden_size': (128, 512, False)
}
```

#### 使用Optuna进行贝叶斯优化

```python
import optuna

def objective(trial, model_class, train_loader, val_loader, device):
    """Optuna目标函数。"""

    # 建议超参数
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    hidden_size = trial.suggest_categorical('hidden_size', [128, 256, 512])

    # 创建模型
    model = model_class(dropout=dropout, hidden_size=hidden_size)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # 训练
    best_val_acc = 0
    for epoch in range(10):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)
        val_acc = val_metrics['accuracy']

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        # 剪枝不理想的试验
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val_acc

def run_optuna_study(model_class, train_loader, val_loader, device, n_trials=50):
    """运行Optuna研究。"""
    study = optuna.create_study(direction='maximize')

    study.optimize(
        lambda trial: objective(trial, model_class, train_loader, val_loader, device),
        n_trials=n_trials,
        timeout=None,
        show_progress_bar=True
    )

    print("\n最佳试验:")
    trial = study.best_trial
    print(f"  值: {trial.value}")
    print(f"  参数: {trial.params}")

    return study

# 使用
study = run_optuna_study(MyModel, train_loader, val_loader, device, n_trials=50)
```

### 实验跟踪

#### MLflow集成

```python
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

class MLflowTracker:
    """MLflow实验跟踪器。"""

    def __init__(self, experiment_name, tracking_uri=None):
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        self.experiment = mlflow.set_experiment(experiment_name)
        self.run = None

    def start_run(self, run_name=None, params=None):
        """开始MLflow运行。"""
        self.run = mlflow.start_run(run_name=run_name)

        if params:
            mlflow.log_params(params)

    def log_metrics(self, metrics, step=None):
        """记录指标。"""
        mlflow.log_metrics(metrics, step=step)

    def log_model(self, model, model_name="model"):
        """记录PyTorch模型。"""
        mlflow.pytorch.log_model(model, model_name)

    def log_artifact(self, file_path):
        """记录工件。"""
        mlflow.log_artifact(file_path)

    def end_run(self, status="FINISHED"):
        """结束MLflow运行。"""
        mlflow.end_run(status=status)

# 使用
tracker = MLflowTracker("my_experiment")

tracker.start_run(run_name="experiment_1", params={
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100
})

for epoch in range(100):
    train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
    val_metrics = validate(model, val_loader, criterion, device)

    tracker.log_metrics({f"train_{k}": v for k, v in train_metrics.items()}, step=epoch)
    tracker.log_metrics({f"val_{k}": v for k, v in val_metrics.items()}, step=epoch)

tracker.log_model(model)
tracker.end_run()
```

#### Weights & Biases集成

```python
import wandb

class WandBTracker:
    """Weights & Biases跟踪器。"""

    def __init__(self, project_name, config=None):
        wandb.init(project=project_name, config=config)
        self.config = wandb.config

    def log_metrics(self, metrics, step=None):
        """记录指标。"""
        wandb.log(metrics, step=step)

    def log_model(self, model, model_name="model"):
        """记录模型。"""
        torch.save(model.state_dict(), f"{model_name}.pt")
        wandb.save(f"{model_name}.pt")

    def log_image(self, image, caption):
        """记录图像。"""
        wandb.log({caption: wandb.Image(image)})

    def finish(self):
        """完成W&B运行。"""
        wandb.finish()

# 使用
tracker = WandBTracker("my_project", config={
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100
})

for epoch in range(100):
    train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
    val_metrics = validate(model, val_loader, criterion, device)

    tracker.log_metrics({**{f"train_{k}": v for k, v in train_metrics.items()},
                        **{f"val_{k}": v for k, v in val_metrics.items()}}, step=epoch)

tracker.log_model(model)
tracker.finish()
```

#### TensorBoard集成

```python
from torch.utils.tensorboard import SummaryWriter
import torchvision

class TensorBoardTracker:
    """TensorBoard跟踪器。"""

    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def log_metrics(self, metrics, step, prefix=""):
        """记录指标。"""
        for key, value in metrics.items():
            self.writer.add_scalar(f"{prefix}/{key}", value, step)

    def log_images(self, images, step, tag="images"):
        """记录图像。"""
        grid = torchvision.utils.make_grid(images)
        self.writer.add_image(tag, grid, step)

    def log_model_graph(self, model, inputs):
        """记录模型图。"""
        self.writer.add_graph(model, inputs)

    def log_histograms(self, model, step):
        """记录参数直方图。"""
        for name, param in model.named_parameters():
            self.writer.add_histogram(name, param, step)

    def close(self):
        """关闭写入器。"""
        self.writer.close()

# 使用
tracker = TensorBoardTracker("./logs")

for epoch in range(100):
    train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
    val_metrics = validate(model, val_loader, criterion, device)

    tracker.log_metrics(train_metrics, epoch, "train")
    tracker.log_metrics(val_metrics, epoch, "val")
    tracker.log_histograms(model, epoch)

tracker.close()
```

### 检查点管理

#### 检查点保存和加载

```python
import os
from pathlib import Path
import shutil

class CheckpointManager:
    """管理模型检查点。"""

    def __init__(self, checkpoint_dir, max_to_keep=5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep
        self.checkpoints = []

    def save_checkpoint(self, model, optimizer, scheduler, epoch, metrics, filename=None):
        """保存检查点。"""
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pt"

        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics
        }

        torch.save(checkpoint, checkpoint_path)
        self.checkpoints.append(checkpoint_path)

        # 仅保留max_to_keep个检查点
        if len(self.checkpoints) > self.max_to_keep:
            oldest = self.checkpoints.pop(0)
            if oldest.exists():
                oldest.unlink()

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path, model, optimizer=None, scheduler=None):
        """加载检查点。"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return checkpoint['epoch'], checkpoint.get('metrics', {})

    def get_latest_checkpoint(self):
        """获取最新检查点路径。"""
        if not self.checkpoints:
            return None
        return max(self.checkpoints, key=os.path.getctime)

    def get_best_checkpoint(self, metric_name, minimize=True):
        """获取具有最佳指标的检查点。"""
        best_checkpoint = None
        best_value = float('inf') if minimize else float('-inf')

        for checkpoint_path in self.checkpoints:
            checkpoint = torch.load(checkpoint_path)
            value = checkpoint['metrics'].get(metric_name)

            if value is not None:
                if (minimize and value < best_value) or (not minimize and value > best_value):
                    best_value = value
                    best_checkpoint = checkpoint_path

        return best_checkpoint
```

### 早停

```python
class EarlyStopping:
    """早停，当验证指标停止改善时停止训练。"""

    def __init__(self, patience=10, min_delta=0, mode='min', verbose=True):
        """
        参数:
            patience: 停止前等待的轮次数
            min_delta: 被视为改进的最小变化
            mode: 'min'用于最小化指标，'max'用于最大化指标
            verbose: 打印消息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, metric):
        """检查是否应停止训练。"""
        if self.best_score is None:
            self.best_score = metric
        elif self._is_improvement(metric):
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def _is_improvement(self, metric):
        """检查指标是否改进。"""
        if self.mode == 'min':
            return metric < self.best_score - self.min_delta
        else:
            return metric > self.best_score + self.min_delta

# 使用
early_stopping = EarlyStopping(patience=10, mode='min')

for epoch in range(100):
    train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
    val_metrics = validate(model, val_loader, criterion, device)

    if early_stopping(val_metrics['loss']):
        print(f'Early stopping at epoch {epoch}')
        break
```

### 学习率调度

#### 常见调度器

```python
import torch.optim as optim

# StepLR - 每step_size轮次按gamma衰减LR
scheduler_step = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=30,
    gamma=0.1
)

# ExponentialLR - 每轮次按gamma衰减LR
scheduler_exp = optim.lr_scheduler.ExponentialLR(
    optimizer,
    gamma=0.95
)

# CosineAnnealingLR - 余弦退火
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,
    eta_min=1e-6
)

# ReduceLROnPlateau - 当指标平台时减少LR
scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.1,
    patience=5,
    verbose=True
)

# OneCycleLR - 单周期策略
scheduler_onecycle = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.01,
    total_steps=1000,
    pct_start=0.3,
    anneal_strategy='cos'
)

# CosineAnnealingWarmRestarts
scheduler_warm = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,
    T_mult=2
)
```

#### 自定义调度器

```python
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np

class WarmupCosineScheduler(_LRScheduler):
    """具有预热和余弦退火的学习率调度器。"""

    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=0, max_lr=None):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.max_lr = max_lr if max_lr else optimizer.param_groups[0]['lr']
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # 线性预热
            return [self.max_lr * (self.last_epoch + 1) / self.warmup_epochs
                    for _ in self.base_lrs]
        else:
            # 余弦退火
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
            return [self.min_lr + (self.max_lr - self.min_lr) * cosine_factor
                    for _ in self.base_lrs]

# 使用
scheduler = WarmupCosineScheduler(
    optimizer,
    warmup_epochs=10,
    max_epochs=100,
    min_lr=1e-6
)
```

### 分布式训练

#### DataParallel（单节点，多GPU）

```python
import torch.nn as nn

# 使用DataParallel包装模型
if torch.cuda.device_count() > 1:
    print(f"使用 {torch.cuda.device_count()} 个GPU")
    model = nn.DataParallel(model)

model = model.to(device)

# 训练循环保持不变
for epoch in range(epochs):
    train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
    val_metrics = validate(model, val_loader, criterion, device)
```

#### DistributedDataParallel（多节点，多GPU）

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup_distributed():
    """设置分布式训练。"""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_distributed():
    """清理分布式训练。"""
    dist.destroy_process_group()

def create_dataloader_distributed(dataset, batch_size, num_workers, rank, world_size):
    """创建分布式数据加载器。"""
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader, sampler

# 训练脚本
def train_distributed(rank, world_size, config):
    """分布式训练函数。"""
    local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # 创建模型
    model = MyModel().to(device)
    model = DDP(model, device_ids=[local_rank])

    # 创建数据加载器
    train_dataset = CustomDataset(...)
    train_loader, train_sampler = create_dataloader_distributed(
        train_dataset, config['batch_size'], config['num_workers'], rank, world_size
    )

    # 训练循环
    for epoch in range(config['epochs']):
        train_sampler.set_epoch(epoch)

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    cleanup_distributed()

# 使用torchrun启动
# torchrun --nproc_per_node=4 train_script.py
```

### 模型评估

#### 分类指标

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import numpy as np

def evaluate_classification(model, dataloader, device, num_classes):
    """评估分类模型。"""
    model.eval()

    all_predictions = []
    all_targets = []
    all_probabilities = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = outputs.max(1)

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    # 转换为numpy数组
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    probabilities = np.array(all_probabilities)

    # 计算指标
    metrics = {
        'accuracy': accuracy_score(targets, predictions),
        'precision': precision_score(targets, predictions, average='weighted'),
        'recall': recall_score(targets, predictions, average='weighted'),
        'f1': f1_score(targets, predictions, average='weighted'),
        'confusion_matrix': confusion_matrix(targets, predictions).tolist(),
        'classification_report': classification_report(targets, predictions, output_dict=True)
    }

    return metrics, predictions, probabilities
```

#### 对象检测指标

```python
from collections import defaultdict
import numpy as np

def calculate_iou(box1, box2):
    """计算两个框之间的IoU。"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

def calculate_ap(predictions, targets, iou_threshold=0.5, num_classes=80):
    """计算对象检测的平均精度。"""
    ap_per_class = []

    for class_id in range(num_classes):
        # 过滤此类的预测和目标
        class_preds = [p for p in predictions if p['class_id'] == class_id]
        class_targets = [t for t in targets if t['class_id'] == class_id]

        if len(class_targets) == 0:
            continue

        # 按置信度排序预测
        class_preds.sort(key=lambda x: x['confidence'], reverse=True)

        # 计算TP和FP
        tp = np.zeros(len(class_preds))
        fp = np.zeros(len(class_preds))
        matched_targets = set()

        for i, pred in enumerate(class_preds):
            best_iou = 0
            best_target_idx = -1

            for j, target in enumerate(class_targets):
                if j in matched_targets:
                    continue

                iou = calculate_iou(pred['bbox'], target['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_target_idx = j

            if best_iou >= iou_threshold:
                tp[i] = 1
                matched_targets.add(best_target_idx)
            else:
                fp[i] = 1

        # 计算精确率和召回率
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recalls = tp_cumsum / len(class_targets)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)

        # 使用11点插值计算AP
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            mask = recalls >= t
            if np.any(mask):
                p = np.max(precisions[mask])
                ap += p
        ap_per_class.append(ap / 11.0)

    return float(np.mean(ap_per_class)) if ap_per_class else 0.0
```
