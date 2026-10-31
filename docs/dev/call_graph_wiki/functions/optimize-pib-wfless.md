---
title: optimize_pib（DM 电压无波前传感 PIB 优化）
created: 2026-10-13
updated: 2026-10-13
type: function
tags: [optimizer,pib,wfless]
source: src/ao_shaping/optimizer/wfless/pib.py:451
---

## 签名

```python
optimize_pib(center, epochs, r_bucket, delta, lr, exposure_time_ms, cam_type, shrink_iter, shrink_ratio, cam_id, show, init_v, … (+25 个))
```

## docstring（原文）

```
优化PIB（Power in Bucket）

    Args:
        center (str or tuple): 中心位置。
        epochs (int): 迭代次数。
        r_bucket (float): 桶半径。如果设置为0，则根据功率半径自动调整。
        delta (float): 分布参数。
        lr (float): 学习率。如果设置为0，则根据功率半径自动调整。
        exposure_time_ms (int): 曝光时间（毫秒）。如果设置为0，则自动曝光。
        shrink_iter (int): 收缩迭代次数。如果设置为0，则不进行收缩。
        shrink_ratio (float): 收缩比例。
        cam_id (int): 相机ID。
        show (bool): 是否显示图像。
        init_v (list): 初始电压。
        cam_size (int): 相机图像大小。
        target_max_brightness (float): 目标最大亮度。如果设置为0，则迭代过程中不自动调整曝光时间。
        dm_unit_mask (list): DM单元掩码。
        dm_neibor_diff (float): DM邻居电压差。
        dm_max_voltage (float): DM最大电压。
        dm_min_voltage (float): DM最小电压。
        dm (DM | None): DM实例。若为None则创建NlightDM（向后兼容）。
        optimizer_type (str): 梯度阶段使用的优化器类型，支持 Adam/AdaMOD/SGD/Muno 系列。
        enable_adaptive_search (bool): 是否启用局部最优后的自适应邻域搜索。
        search_interval (int): 邻域搜索触发间隔。
        search_warmup (int): 启动邻域搜索前的最小迭代数。
        search_patience (int): 最佳 PIB 无提升时，等待多少轮后触发搜索。
        search_samples (int): 每次邻域搜索评估的候选解数量。
        search_radius (float | None): 初始邻域搜索半径。
        search_min_radius (float | None): 邻域搜索最小半径。
        search_max_radius (float | None): 邻域搜索最大半径。
        search_expand_ratio (float): 搜索失败时的邻域扩张系数。
        search_shrink_ratio (float): 搜索成功时的邻域收缩系数。
        search_improvement_tol (float): 判定改进的最小 PIB 增益。
        tabu_memory_size (int): 禁忌表容量。
        tabu_quantization (float): 电压量化步长，用于禁忌表去重。
        search_anchor (str): 邻域搜索起点，可选 "best" 或 "current"。
        random_seed (int | None): 随机种子。
        objective (str): 优化目标函数，可选 'pib'(最大化), 'radiu'(最小化半径), 'avg_radiu'(最大化平均半径)。
        **kwargs: 其他参数。
```

## 直接调用（出边）

- `ao_shaping.drivers.ccd.common.get_camera_exposure_ms`
- `_extract_optimizer_momentum`
- [[functions/spots|`radius`]]
- `learning_schedule`
- `_generate_search_candidates`
- `_objective_to_min`
- `_reset_optimizer_state`
- `_should_trigger_adaptive_search`
- `ao_shaping.algorithm.goal_functions.target_func.ImageTargetFunc` *(未解析/外部)*
- `ao_shaping.algorithm.tabu.tabu_search.AdaptiveSearchState` *(未解析/外部)*
- `ao_shaping.algorithm.tabu.tabu_search.TabuMemory` *(未解析/外部)*
- `ao_shaping.drivers.ccd.common.resample_on_saturation`
- `ao_shaping.drivers.dm._registry.get_dm_registry`
- `ao_shaping.optimizer.constants.create_optimizer`
- `ao_shaping.optimizer.spid_pib_loop.open_pib_bench`
- `ao_shaping.optimizer.spid_pib_loop.spid_pib_epoch`
- `ao_shaping.utils.ImageVoltagesDisplay` *(未解析/外部)*
- `ao_shaping.utils.Recorder` *(未解析/外部)*
- `ao_shaping.utils.objective_direction.objective_direction`

## 谁调用它（入边）

- [[commands/pib|`run`]]
- [[commands/pipeline|`run`]]

## 相似函数（自动）

与本页函数结构/职责相近的函数（相似度得分，来自 `build_similarity.py`；含义与阈值见 [[relations/similar-functions|相似函数总览]]）：

- 0.384 — [[functions/optimize-pib-combined|`optimize_pib`]]（同名 `optimize_pib`；共享 9 个被调函数（调用结构高度相似））


## 相关页面

- [[entry/cli]]
