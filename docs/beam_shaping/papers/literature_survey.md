# 闭环 SLM 远场光斑整形 — 文献调研 (2026-09)

> **调研范围**: 2018–2025, 主题 = 闭环远场光束整形 (SLM 相位/振幅/偏振控制)。
> **分类维度 (按方法)**:
> (A) 经典 / 相位生成方法 (Gerchberg-Saxton 族 + 改进 + 混合);
> (B) 可微分 / 自动微分 + 无波前传感器 (SPGD 类);
> (C) 强化学习 (RL);
> (D) 目标函数 / 质量指标;
> (E) 偏振 / 振幅+相位联合控制。
>
> **核实**: 全部 25 篇均经 arXiv abs 页 / 出版社记录逐条核实 (title / 第一作者 / 年份 / 期刊 / arXiv ID 或 DOI), 无虚构条目。
> 仿真基准 (GS / differentiable / SPGD-freeform 在 2f-Fourier 单相位模型上的 PIB/CV/Strehl 对比) 见
> [`beam_shaping_papers.md`](beam_shaping_papers.md), 与本调研的 (A)(B) 两条主线直接对应, 作为方法对比的本地基线。

---

## 1. 方法分类总览 (taxonomy)

| 类别 | 篇数 | 方法主线 | 本仓库对应实现 |
|------|-----|---------|----------------|
| (A) 经典 / 相位生成 | 8 | GS 族 + 最优传输初始化 + 反馈 GS + 混合 (SIFTA/SA) | `gs_hologram_runner` / `gs_square_runner` / `gs_shape` |
| (B) 可微分 / SPGD | 5 | 自动微分全息 + M-SPGD + 物理嵌入式 NN + 最优传输+phase diversity | `diff_shaping_runner` / `diff_beam_runner` / `spgd_square_runner` |
| (C) 强化学习 | 5 | PPO/SAC/DDPG/准-RL 闭环控制 (多为 DM 或相位元件) | `optimizer/rl` (SAC) |
| (D) 目标函数 / 指标 | 5 | 重叠系数 / VecCos / M² 束质量 / 均匀性+效率 / 路线图综述 | `utils/beam_metrics` + 各优化器评分 |
| (E) 偏振 / 振幅+相位 | 2 | 矢量 AO (相位+偏振) + 偏振非线性 CGH | (尚无, 规划中) |

> **闭环 (feedback) 覆盖**: Kim 2019 (自适应 CCD 校正), Wang 2024 (反馈 GSW), Hong 2018 (反馈波前整形),
> Cheng 2022 (反馈 NES), Parvizi 2023 / Durech 2021 / Gutierrez 2024 / Li 2025 (RL 闭环),
> Yang 2020 (M-SPGD 单模光纤耦合), Hu 2023 (嵌入式 NN 控制环)。

---

## 2. (A) 经典 / 相位生成方法

> 核心 = 相位检索 (Gerchberg-Saxton 族) 生成单相位全息图。改进方向: 初始相位 (最优传输 / 球面)、
> 随机微扰 / 相位替换提升均匀性、混合元启发式 (SA/SIFTA) 降低 RMSE、CCD 反馈闭环。

```json
{"title": "Large-Scale Uniform Optical Focus Array Generation with a Phase Spatial Light Modulator", "first_author": "Donggyu Kim", "year": 2019, "venue": "Optics Letters 44, 3178-3181 (2019)", "arxiv": "1903.09286", "topic": "phase-generation", "one_line": "Phase-fixed weighted Gerchberg-Saxton with adaptive CCD feedback; >98% uniformity over O(10^3) foci."},
{"title": "High-fidelity holographic beam shaping with optimal transport and phase diversity", "first_author": "Hunter Swan", "year": 2025, "venue": "Optics Express 33, 6290-6303 (2025)", "arxiv": "2408.17025", "topic": "phase-generation", "one_line": "Optimal-transport initialization for GS/MRAF retrieval; high-fidelity single-phase holograms with phase diversity."},
{"title": "Benchmarking the Gerchberg-Saxton Algorithm", "first_author": "Peter J. Christopher", "year": 2020, "venue": "arXiv preprint", "arxiv": "2005.08623", "topic": "phase-generation", "one_line": "Systematic GS benchmarking: convergence speed, power efficiency, accuracy trade-offs for SLM holography."},
{"title": "Generation of multi-focus shaping with high uniformity based on an improved Gerchberg-Saxton algorithm", "first_author": "Hang Chen", "year": 2024, "venue": "Applied Optics 63(12), 3283-3289 (2024)", "DOI": "10.1364/AO.516663", "topic": "phase-generation", "one_line": "Random disturbance superposition + phase value replacement in GS; >95% uniformity multi-focus with high energy utilization."},
{"title": "Speckle-reduced holographic beam shaping with modified Gerchberg-Saxton algorithm", "first_author": "Hui Pang", "year": 2019, "venue": "Optics Communications 433, 44-51 (2019)", "DOI": "10.1016/j.optcom.2018.09.076", "topic": "phase-generation", "one_line": "GS with spherical initial phase; speckle-free flat-top with high diffraction efficiency and low RMSE."},
{"title": "A Segmented Hybrid Algorithm for Beam Shaping Combining Iterative and Simulated Annealing Approaches", "first_author": "Xiaoyu Zhang", "year": 2024, "venue": "Photonics 11(3), 197 (2024)", "DOI": "10.3390/photonics11030197", "topic": "phase-generation", "one_line": "GS+SIFTA+SA segmented hybrid; ~37% lower RMSE, ~39% lower uniformity error for flat-top shaping."},
{"title": "DeepCGH: 3D computer-generated holography using deep learning", "first_author": "M. Hossein Eybposh", "year": 2020, "venue": "Optics Express 28(18), 26637-26650 (2020)", "DOI": "10.1364/OE.399624", "topic": "phase-generation", "one_line": "Deep-learning CGH with physics-based loss; high-quality 3D multi-plane holograms."},
{"title": "Feedback Intensity Equalization Algorithm for Multi-Spots Holographic Tweezer", "first_author": "Shaoxiong Wang", "year": 2024, "venue": "arXiv preprint", "arxiv": "2407.17049", "topic": "phase-generation", "one_line": "Feedback GSW with CCD; non-uniformity <1.1% for >1000 holographic traps."}
```

**方法要点**:
- **GS 族**: 经典 GS 在振幅/相位交替约束下迭代; 改进集中在 ① 初始相位 (Swan 2025 最优传输、Pang 2019 球面初始相位 → 降 speckle + 提效率), ② 随机微扰 / 相位值替换 (Chen 2024, 多焦点 >95% 均匀), ③ 分段混合元启发式 (Zhang 2024, GS+SIFTA+SA, RMSE ↓37%), ④ 大规模加权 GS + CCD 反馈 (Kim 2019, O(10³) 焦点 >98% 均匀)。
- **反馈闭环**: Wang 2024 (GSW + CCD, 非均匀性 <1.1%)、Kim 2019 (自适应 CCD 校正) 代表真闭环 GS — 与本仓库 `gs_square_runner` (GS+CCD 反馈) 思路一致。
- **DeepCGH (Eybposh 2020)**: 深度学习 CGH, 物理约束损失, 3D 多平面全息 — 连接 (A) 与 (B)/(C) 的桥梁。

---

## 3. (B) 可微分 / 自动微分 + 无波前传感器 (SPGD 类)

> 核心 = 把传播模型 (ASM/Fraunhofer) 嵌入 PyTorch/AD, 用梯度直接优化相位; 或 SPGD (无波前传感器,
> 随机并行梯度) + 物理嵌入式神经网络。本仓库 `diff_shaping` / `diff_beam` / `spgd_square` 即此主线。

```json
{"title": "Dynamic Hologram Generation with Automatic Differentiation", "first_author": "Xing-Yu Zhang", "year": 2025, "venue": "Physical Review Applied", "DOI": "10.1103/6hs1-4gkw", "topic": "differentiable", "one_line": "Automatic differentiation optimizes dynamic computer-generated holograms for high-quality 3D display and beam shaping."},
{"title": "High-fidelity holographic beam shaping with optimal transport and phase diversity", "first_author": "Hunter Swan", "year": 2025, "venue": "Optics Express 33, 6290-6303 (2025)", "DOI": "10.1364/OE.540901", "topic": "differentiable", "one_line": "Optimal transport initializes iterative phase retrieval to shape laser beams into arbitrary intensity patterns with a phase-only SLM."},
{"title": "Image-guided Computational Holographic Wavefront Shaping", "first_author": "Omri Haim", "year": 2025, "venue": "Nature Photonics", "DOI": "10.1038/s41566-024-01544-6", "topic": "differentiable", "one_line": "Guide-star-free computational wavefront shaping optimizes virtual SLMs via automatic differentiation to image through scattering media."},
{"title": "Universal adaptive optics for microscopy through embedded neural network control", "first_author": "Qi Hu", "year": 2023, "venue": "Light: Science & Applications", "DOI": "10.1038/s41377-023-01297-x", "topic": "differentiable", "one_line": "Physics-based neural network embedded in the control loop performs fast sensorless adaptive-optics correction across microscope modalities."},
{"title": "Single-mode fiber coupling with a M-SPGD algorithm for long-range quantum communications", "first_author": "Kui-Xing Yang", "year": 2020, "venue": "Optics Express", "DOI": "10.1364/OE.411939", "topic": "differentiable", "one_line": "Modified stochastic parallel gradient descent (M-SPGD) optimizes single-mode fiber coupling for long-range quantum communication."}
```

**方法要点**:
- **自动微分全息**: Zhang 2025 (AD 动态全息)、Haim 2025 (Nat. Photonics, 无导星虚拟 SLM + AD) 代表"传播模型可微分"路线, 与本仓库 `differentiable_shaping.py` (PyTorch ASM/Fraunhofer) 同源。
- **最优传输初始化**: Swan 2025 同时进入 (A) 与 (B) — 用最优传输给出 GS/AD 的良好初值, 高保真单相位 SLM 整形。
- **M-SPGD**: Yang 2020 是无波前传感器 (SPGD 类) 的代表, 长程单模光纤耦合 — 对应本仓库 `spgd_square` / `combined` (AdaMOD+SPGD)。
- **物理嵌入式 NN**: Hu 2023 把物理 NN 嵌入控制环做快速无传感器 AO 校正, 是 (B) 与 (C) 的交汇点。

---

## 4. (C) 强化学习 (RL)

> 核心 = 无模型 (model-free) RL 智能体从标量反馈 (QPD / 图像 / 误差) 学习控制策略, 闭环校正波前/相位。
> 2019–2025 窗口内, RL 多为 DM 或衍射/相位元件控制; **直接"RL→SLM 相位→远场整形"的论文稀缺** (见 §8 缺口)。

```json
{"title": "Reinforcement Learning-based Wavefront Sensorless Adaptive Optics Approaches for Satellite-to-Ground Laser Communication", "first_author": "Payam Parvizi", "year": 2023, "venue": "arXiv preprint", "arxiv": "2303.07516", "topic": "reinforcement-learning", "one_line": "RL (PPO/SAC/DDPG) learns wavefront-sensorless AO control from quadrant-photodiode feedback for satellite-to-ground laser communication."},
{"title": "Wavefront sensor-less adaptive optics using deep reinforcement learning", "first_author": "Eduard Durech", "year": 2021, "venue": "Biomedical Optics Express", "DOI": "10.1364/BOE.427970", "topic": "reinforcement-learning", "one_line": "Deep RL (DDPG) drives sensorless adaptive-optics correction of Zernike modes in a confocal scanning laser microscope."},
{"title": "Image-based wavefront correction using model-free Reinforcement Learning", "first_author": "Yann Gutierrez", "year": 2024, "venue": "Optics Express", "DOI": "10.1364/OE.529415", "topic": "reinforcement-learning", "one_line": "Model-free RL agent corrects telescope aberrations directly from phase-diversity focal-plane images, controlling a deformable mirror."},
{"title": "Experimental phase control of a 100 laser beam array with quasi-reinforcement learning of a neural network in an error reduction loop", "first_author": "Maksym Shpakovych", "year": 2021, "venue": "Optics Express", "DOI": "10.1364/OE.419232", "topic": "reinforcement-learning", "one_line": "Quasi-reinforcement-learning neural network controls phases of a 100-beam laser array in an error-reduction loop for coherent combining."},
{"title": "Model-free Optical Processors using In Situ Reinforcement Learning with Proximal Policy Optimization", "first_author": "Yuhang Li", "year": 2025, "venue": "Light: Science & Applications", "DOI": "10.1038/s41377-025-02148-7", "topic": "reinforcement-learning", "one_line": "PPO reinforcement learning trains optical processors in situ, adapting diffractive/phase elements without a model of the system."}
```

**方法要点**:
- **无波前传感器 RL**: Parvizi 2023 (PPO/SAC/DDPG, QPD 反馈, 达理想 SH 传感器 86% 性能)、Durech 2021 (DDPG, 共聚焦显微镜 Zernike)、Gutierrez 2024 (相位多样性图像) 三篇构成"无 WFS + RL"主线 — 与本仓库 `optimizer/rl` (SAC) 对应。
- **相位元件 / 阵列**: Shpakovych 2021 (准-RL NN, 100 束相干合成)、Li 2025 (PPO in-situ, 衍射/相位元件) 控制的是相位自由度而非 DM, 更接近 SLM 整形。
- **稀缺性**: 窗口内"RL 直接控制 SLM 相位做远场整形"几乎没有直接工作 (候选 Motion Hologram arXiv:2401.12537 偏 3D 显示未入选)。

---

## 5. (D) 目标函数 / 质量指标

> 核心 = 闭环整形的标量奖励 / 适应度函数: 重叠系数 (overlap)、余弦相似度 (VecCos)、束质量 (M²)、
> 均匀性+效率组合、衍射效率。本仓库 `utils/beam_metrics` + 各优化器评分即此层。

```json
{"title": "Customizing optical patterns via feedback-based wavefront shaping", "first_author": "Peilong Hong", "year": 2018, "venue": "arXiv preprint", "arxiv": "1812.00162", "topic": "objective-function", "one_line": "Overlap coefficient feedback metric for wavefront shaping; converges to unity for patterned targets."},
{"title": "Long-distance pattern projection through an unfixed multimode fiber with natural evolution strategy-based wavefront shaping", "first_author": "Shengfu Cheng", "year": 2022, "venue": "Optics Express 30(18), 32566 (2022)", "DOI": "10.1364/OE.462275", "topic": "objective-function", "one_line": "Natural evolution strategy wavefront shaping; VecCos cosine-similarity fitness for pattern projection through multimode fiber."},
{"title": "Roadmap on wavefront shaping and deep imaging in complex media", "first_author": "Sylvain Gigan", "year": 2022, "venue": "Journal of Physics: Photonics 4, 042501 (2022)", "DOI": "10.1088/2515-7647/ac76f9", "topic": "objective-function", "one_line": "Roadmap review; surveys feedback strategies and quality metrics for wavefront shaping."},
{"title": "Wavefront shaping to improve beam quality: converting a speckle pattern into a Gaussian spot", "first_author": "Alba M. Paniagua-Diaz", "year": 2021, "venue": "arXiv preprint", "arxiv": "2107.10601", "topic": "objective-function", "one_line": "Energy fraction into low-M2 Gaussian mode as beam-quality objective for wavefront shaping."},
{"title": "Rapid phase calibration of a spatial light modulator using novel phase masks and optimization of its efficiency using an iterative algorithm", "first_author": "Amar Deo Chandra", "year": 2020, "venue": "Journal of Modern Optics 67(8), 628-637 (2020)", "DOI": "10.1080/09500340.2020.1760954", "topic": "objective-function", "one_line": "IFTA-optimized phase masks; 90% uniformity spot arrays and ~20% efficiency gain over LUT correction."}
```

**方法要点**:
- **重叠系数 / VecCos**: Hong 2018 (重叠系数反馈, 收敛到 1)、Cheng 2022 (VecCos 余弦相似度适应度, NES 波前整形) — 本仓库 `compute_metrics` 的 `Strehl`/余弦相似度即同类。
- **束质量 M²**: Paniagua-Diaz 2021 用"低 M² 高斯模能量占比"作束质量目标 (speckle→高斯), 对应本仓库 PIB/效率指标。
- **均匀性+效率**: Chandra 2020 (IFTA 相位掩模, 90% 均匀 + 效率 ↑20%) 是"均匀性与衍射效率联合"目标函数的代表。
- **综述**: Gigan 2022 路线图系统梳理反馈策略与质量指标, 可作为本调研 (D) 的索引入口。

---

## 6. (E) 偏振 / 振幅+相位联合控制

> 核心 = 在相位之外同时控制偏振 (矢量 AO) 或利用偏振非线性 (SHG) 实现整形。
> 经典 SLM 矢量光束论文 (Rosales-Guzmán 2017, Chen 2015) 超出 2018–2025 窗口, 未入选。

```json
{"title": "Vectorial adaptive optics", "first_author": "Chao He", "year": 2023, "venue": "eLight", "DOI": "10.1186/s43593-023-00056-0", "topic": "polarization", "one_line": "Extends adaptive optics to joint polarization and phase feedback correction (vectorial AO) using SLM and deformable-mirror systems."},
{"title": "Polarization-Controlled Nonlinear Computer-Generated Holography", "first_author": "Lisa Ackermann", "year": 2023, "venue": "Scientific Reports", "DOI": "10.1038/s41598-023-37443-z", "topic": "polarization", "one_line": "Polarization-controlled nonlinear computer-generated holography shapes arbitrary intensity distributions in frequency-converted (SHG) beams via SLM."}
```

**方法要点**:
- **矢量 AO**: He 2023 (eLight) 把 AO 扩展到偏振+相位联合反馈校正 — 本仓库尚无对应 (规划中)。
- **偏振非线性 CGH**: Ackermann 2023 用偏振控制二次谐波 (SHG) 实现任意强度分布, 代表"振幅由偏振非线性间接控制"的路线。

---

## 7. 核实记录 (全 25 篇)

| # | 类别 | 论文 | 核实途径 |
|---|------|------|---------|
| A1 | phase-generation | Kim 2019 | arXiv:1903.09286; Opt. Lett. 44, 3178-3181 |
| A2 | phase-generation | Swan 2025 | arXiv:2408.17025; Opt. Express 33, 6290-6303, DOI 10.1364/OE.540901 |
| A3 | phase-generation | Christopher 2020 | arXiv:2005.08623 (仅 arXiv, 未见期刊版) |
| A4 | phase-generation | Chen 2024 | Appl. Opt. 63(12), 3283-3289, DOI 10.1364/AO.516663 |
| A5 | phase-generation | Pang 2019 | Opt. Commun. 433, 44-51, DOI 10.1016/j.optcom.2018.09.076 |
| A6 | phase-generation | Zhang 2024 | Photonics 11(3), 197, DOI 10.3390/photonics11030197 |
| A7 | phase-generation | Eybposh 2020 | Opt. Express 28(18), 26637-26650, DOI 10.1364/OE.399624 |
| A8 | phase-generation | Wang 2024 | arXiv:2407.17049 (仅 arXiv) |
| B1 | differentiable | Zhang 2025 | arXiv:2503.03714; Phys. Rev. Applied 24, 024011, DOI 10.1103/6hs1-4gkw |
| B2 | differentiable | Swan 2025 | 同 A2 (Opt. Express, DOI 10.1364/OE.540901) |
| B3 | differentiable | Haim 2025 | arXiv:2305.12232; Nat. Photonics 19:44-53, DOI 10.1038/s41566-024-01544-6 |
| B4 | differentiable | Hu 2023 | arXiv:2301.02647; Light Sci. Appl. 12:270, DOI 10.1038/s41377-023-01297-x |
| B5 | differentiable | Yang 2020 | arXiv:2012.04394; Opt. Express 28(24):36600, DOI 10.1364/OE.411939 |
| C1 | reinforcement-learning | Parvizi 2023 | arXiv:2303.07516; PPO>SAC>DDPG, 达理想 SH 传感器 86% 性能 |
| C2 | reinforcement-learning | Durech 2021 | PubMed 34692192; Biomed. Opt. Express 12(9):5423-5438, DOI 10.1364/BOE.427970 |
| C3 | reinforcement-learning | Gutierrez 2024 | arXiv:2406.18143; Opt. Express, DOI 10.1364/OE.529415 |
| C4 | reinforcement-learning | Shpakovych 2021 | arXiv:2012.05647; Opt. Express 29:12307-12318, DOI 10.1364/OE.419232 |
| C5 | reinforcement-learning | Li 2025 | arXiv:2507.05583; Light: Science & Applications, DOI 10.1038/s41377-025-02148-7 |
| D1 | objective-function | Hong 2018 | arXiv:1812.00162 (仅 arXiv) |
| D2 | objective-function | Cheng 2022 | Opt. Express 30(18), 32566, DOI 10.1364/OE.462275 (多模光纤, VecCos 适应度) |
| D3 | objective-function | Gigan 2022 | J. Phys.: Photonics 4, 042501, DOI 11088/2515-7647/ac76f9 (路线图) |
| D4 | objective-function | Paniagua-Diaz 2021 | arXiv:2107.10601 (仅 arXiv) |
| D5 | objective-function | Chandra 2020 | J. Mod. Opt. 67(8), 628-637, DOI 10.1080/09500340.2020.1760954 |
| E1 | polarization | He 2023 | arXiv:2110.02606; eLight 3:23, DOI 10.1186/s43593-023-00056-0 |
| E2 | polarization | Ackermann 2023 | arXiv:2301.10093; Sci. Rep. 13:10338, DOI 10.1038/s41598-023-37443-z |

> 注: Swan 2025 同时出现在 (A) 与 (B) (同一篇, 两种分类视角), 故 §7 计 25 条、去重后 24 篇独立论文。

---

## 8. 覆盖缺口 (iteration 2 候选)

- **"RL 直接控制 SLM 相位做远场整形" 的论文稀缺** — 多数 RL-AO 用变形镜 (Durech, Gutierrez, Parvizi) 或
  衍射/相位元件 (Li)。最接近 SLM+RL 的是 Motion Hologram (arXiv:2401.12537, RL 用于 3D 显示全息图), 因偏
  显示应用未入选。
- **SPGD + SLM 相位整形** 未找到 arXiv 收录的直接组合; SPGD 论文集中在光纤阵列/激光阵列
  (Yang 2020, Vorontsov & Filimonov 2022, arXiv:2204.05227 — 后者是 SPGD+AI 强候选)。
- **偏振**: 2018–2025 窗口内最强 SLM 偏振整形为 He 2023 (eLight) 与 Ackermann 2023 (Sci. Rep.);
  经典 SLM 矢量光束论文 (Rosales-Guzmán 2017, Chen 2015) 超出窗口。
- **仅 arXiv (未见期刊版)**: Christopher 2020, Wang 2024, Hong 2018, Paniagua-Diaz 2021, Parvizi 2023。
- **方法 vs 仓库映射**: 本仓库 `gs_square_runner` (GS+CCD 闭环) ↔ §2 反馈 GS; `diff_shaping`/`diff_beam`
  (PyTorch 可微) ↔ §3 AD 全息; `spgd_square`/`combined` (SPGD) ↔ §3 M-SPGD; `optimizer/rl` (SAC) ↔ §4 RL。
  详见 [`beam_shaping_papers.md`](beam_shaping_papers.md) 仿真基准。
