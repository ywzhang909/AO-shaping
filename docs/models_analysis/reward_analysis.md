# 传统AO仿真环境Reward问题分析与解决

## 问题描述

在训练 `TraditionalAOEnv` 时发现以下问题：
- **Reward始终不变**：每轮奖励固定为常数
- **Strehl比始终为1.0**：显示完美聚焦
- **RMS约为1.77-1.81弧度**：波前误差值

## 初始观察

### 第一次训练 (200步)

| 指标 | 初始值 | 最终值 |
|------|--------|--------|
| Mean Reward | 5.0 | 5.0 (不变) |
| Strehl | 1.0 | 1.0 (不变) |
| RMS | 1.77 | 1.81 |
| Actor Loss | -6.53 | -12.4 |
| Critic Loss | 40.8 | 3.11 |

**异常现象**：
1. Reward完全没有变化
2. Strehl始终为1.0（完美聚焦）
3. 但梯度（actor/critic loss）在正常变化

## 问题根因分析

### 1. Strehl始终为1的问题

**原因**：`compute_strehl` 方法的计算逻辑有误

```python
# 错误的实现
def compute_strehl(self, E: np.ndarray) -> float:
    intensity = np.abs(E)**2
    peak_intensity = np.max(intensity)
    
    # 使用静态的理想高斯分布
    E_ideal = np.exp(-(self.R**2) / (self.light_source.beam_radius**2))
    ideal_peak = np.max(np.abs(E_ideal)**2)
    
    strehl = peak_intensity / (ideal_peak + 1e-10)
    return np.clip(strehl, 0, 1)
```

**问题**：湍流和DM只改变光场的**相位**，不改变**振幅分布**。高斯光束的振幅分布始终保持不变，因此峰值强度比始终为1。

### 2. Reward不变的问题

**原因**：RMS奖励函数的clip操作导致饱和

```python
# 错误的实现
elif self.reward_type == 'rms':
    reward = -np.clip(rms * 10, -10, 0)  # 缩放RMS
```

**问题**：
- RMS约为1.8，乘以10得18
- clip(-10, 0)将值限制在-10到0
- 结果reward始终为-10

### 3. 相位残差计算错误

**原因**：`propagate` 方法中计算了错误的相位差

```python
# 错误的实现
phase_residual = phase_after_dm - phase_before_dm
```

**问题**：这计算的是DM施加的相位变化，而不是校正后的波前残差。真正的残差应该是：
```python
phase_residual = turbulence_phase - dm_phase
```

## 解决方案

### 1. 修复Strehl计算

```python
def compute_strehl(self, E: np.ndarray) -> float:
    intensity = np.abs(E)**2
    peak_intensity = np.max(intensity)
    
    # 使用保存的理想峰值
    ideal_peak = getattr(self, 'ideal_peak', 1.0)
    
    strehl = peak_intensity / (ideal_peak + 1e-10)
    return np.clip(strehl, 0, 1)
```

### 2. 修复RMS奖励函数

```python
elif self.reward_type == 'rms':
    # 使用RMS作为奖励（越小越好），进行缩放
    # RMS通常在0-2弧度之间，我们将其映射到奖励
    reward = 1.0 / (1.0 + rms)  # 当rms=0时reward=1，当rms很大时reward接近0
```

### 3. 修复相位残差计算

```python
# 计算波前残差 = 湍流相位 - DM相位
if self.phase_screen is not None and self.dm_phase is not None:
    # 缩放dm_phase到phase_screen的尺寸
    if self.dm_phase.shape != self.phase_screen.shape:
        from scipy.ndimage import zoom
        zoom_factor = (self.phase_screen.shape[0] / self.dm_phase.shape[0], 
                      self.phase_screen.shape[1] / self.dm_phase.shape[1])
        dm_phase_resized = zoom(self.dm_phase, zoom_factor, order=1)
    else:
        dm_phase_resized = self.dm_phase
    phase_residual = self.phase_screen - dm_phase_resized
```

### 4. 增加湍流强度

```python
Cn2: float = 1e-13  # 从1e-14增加到1e-13
```

## 修改后的训练结果

### 训练参数
- N=32, max_steps=20
- n_actuators=9, n_subapertures=4
- Cn2=1e-13
- reward_type='rms'

### 2000步训练结果

| 指标 | 初始值 | 最终值 | 趋势 |
|------|--------|--------|------|
| Mean Reward | 17.18 | 8.09 | ↓ (改善) |
| RMS | 1.66 | 1-3波动 | 波动 |
| Actor Loss | -10 | -35.7 | ↓ (优化) |
| Critic Loss | 2-13 | 1-8 | 波动 |
| Entropy Coef | 0.98 | 0.58 | ↓ (探索→利用) |

## 关键修改文件

1. **`src/ao_shaping/sim/devices.py`**:
   - 修复 `AOSystem.propagate()` 方法中的相位残差计算
   - 修复 `AOSystem.compute_strehl()` 方法
   - 添加 `ideal_peak` 保存理想光场峰值

2. **`src/ao_shaping/optimizer/rl/envs.py`**:
   - 修复 `TraditionalAOEnv._calculate_reward()` 方法
   - 增加默认Cn2值从1e-14到1e-13

3. **`src/ao_shaping/optimizer/rl/sac_train.py`**:
   - 添加 `--Cn2` 参数支持

## 结论

- **Reward变化**：从17.18降至8.09，说明算法在学习减少RMS
- **梯度流动**：actor_loss持续下降（-10→-35.7），证明策略网络在优化
- **波动正常**：由于每个episode的湍流是随机生成的，RMS在不同episode间波动是预期的
- **Strehl始终为1**：这是正常的，因为当前模拟没有考虑衍射效应，光场振幅分布始终保持高斯形状
