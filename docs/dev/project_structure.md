# 项目包结构（逐文件详解）

> 外迁自根 `AGENTS.md` 的 **Project Structure** 详解（逐文件注释版）。
> 根 `AGENTS.md` 只保留精简结构树 + 指向本文；`runners/` 子包因 CI 守卫
> `tests/ao_shaping/test_runners_doc_paths.py::test_project_tree_covers_runner_subpackages`
> 钉住其子树，故仍在根树中展开。

```
AO-shaping/
├── src/
│   ├── ao_shaping/          # Main package
│   │   ├── main.py              # CLI entry point (Click-based)
│   │   ├── runners/             # Runner scripts package
│   │   │   ├── __init__.py          # 惰性 re-export (_LAZY_RUNNERS / R-27, PEP-562)
│   │   │   ├── runner_common.py     # 共享辅助 (resolve_dm / patch_sim_*_shaping)
│   │   │   ├── matrix_runner.py     # dm-matrix + hadamard-matrix (DM 响应矩阵标定)
│   │   │   ├── zernike_search_runner.py  # ga-zernike
│   │   │   ├── nlight_dm/           # DM 家族 runners
│   │   │   │   ├── wf_runner.py         # Wavefront RMS optimizer
│   │   │   │   ├── axis_beam_runner.py  # PIB optimizer
│   │   │   │   ├── pipeline_runner.py   # Serial WF→PIB pipeline
│   │   │   │   └── combined_runner.py   # AdaMOD+SPGD 混合 PIB (legacy)
│   │   │   ├── micro_drive/         # R50Power (Micro-DM) 电压驱动
│   │   │   │   └── voltage_runner.py    # alt-voltage / full-voltage
│   │   │   └── slm/                 # SLM 光路 runners
│   │   │       ├── slm_shaping_runner.py    # slm-pib 命令组 (spgd / heuristic / square)
│   │   │       ├── gsnet_runner.py          # slm-gsnet
│   │   │       ├── gs_refine_runner.py      # slm-gs-refine
│   │   │       ├── model_in_loop_runner.py  # slm-model-in-loop
│   │   │       ├── model_in_loop_session.py # 硬件 session 预检 (入射定位 / 曝光)
│   │   │       ├── rms_zernike_runner.py    # rms-zernike
│   │   │       ├── zernike_matrix_runner.py # zernike-matrix + closed-loop
│   │   │       └── offset_runner.py         # 未注册 (计划迁 tools/slm/)
│   │   ├── algorithm/            # Optimization algorithms (Adam, SGD, etc.)
│   │   ├── drivers/              # Hardware drivers (see drivers/AGENTS.md)
│   │   │   ├── ccd/              # Cameras (Daheng, MiiCam)
│   │   │   ├── dm/               # Deformable Mirrors (NLight)
│   │   │   ├── slm/              # Spatial Light Modulators (Santec, WavefrontCorrection)
│   │   │   ├── wfs/              # Wavefront Sensors (Thorlabs)
│   │   │   ├── tm/               # Timing modules (Serial/FSM)
│   │   │   ├── sim/              # Digital twin simulation
│   │   │   └── mock_devices.py   # Mock devices for testing
│   │   ├── optimizer/            # High-level optimizers
│   │   │   ├── wf/               # Wavefront-based (RMS)
│   │   │   ├── wfless/           # Wavefront-sensorless (PIB)
│   │   │   │   └── slm_square_shaping.py  # SLM方形光斑 SPGD 整形优化器
│   │   │   └── rl/               # Reinforcement learning (SAC)
│   │   ├── utils/                # Utilities (spots_calc, wavefront_calc, zernike_calc, zernike_utils, wfs_utils)
│   │   ├── ml/                  # Machine learning (U-Net+GAN, training, models) — NOTE: lives at src/ml/ as a separate standalone package
│   │   │   ├── gsnet/         # GSNet: GS + SPGD 整形训练
│   │   │   ├── gsnet_debug/   # GSNet 调试/离线评估
│   │   │   ├── phase/         # U-Net+GAN 相位预测 (dataset, discriminator, trainer, unet)
│   │   │   ├── hwdataset/     # 硬件相位→相机图像 DataLoader (index/transforms/records/dataset/cache/inspect/zernike_dataset)
│   │   │   ├── zernike/       # Zernike forward/inverse model (models, forward_model, train_coeff, train_amp, losses, metrics, augment, dataset, inverse_design, eval_stats)
│   │   │   ├── train.py       # ML 训练入口
│   │   │   └── wandb_logger.py  # WandB integration
│   │   ├── tools/                # Standalone tools (SLM phase capture, Micro-DM image collection, slm_train_data_collect)
│   │   ├── display/              # Visualization (Windows, frames for GUI)
│   │   └── gui/                  # GUI components (Streamlit), 按设备域分包:
│   │       ├── r50/              #   R50Power 控制器 UI (r50_controller_ui.py 入口)
│   │       ├── dm/               #   变形镜 UI (micro_dm_ui.py, ceramic_viewer.py)
│   │       ├── slm/              #   SLM UI (multi_slm_controller.py, slm_calibration_ui.py)
│   │       ├── zernike/          #   Zernike UI (response matrix 校准/调试查看)
│   │       └── ccd/              #   CCD 图像分析 UI (ccd_analyzer.py)
│   ├── calculators/               # Cython extensions (standalone)
│   └── optical_ui/                # [DEPRECATED] Empty package
├── tests/ao_shaping/              # Tests (mirrors src structure)
├── libs/                          # Third-party SDK binaries (gxipy, Drv_UDPST)
└── scripts/                       # Utility scripts, grouped by what they process
    ├── slm/ zernike_bench/ dm/ wfs/    # device domains (SLM bench / Zernike calib / DM / WFS)
    ├── ml/ rl/ oopao/ build/            # data+model / RL+turbulence / OOPAO backend / build
    └── _common/ call_graph/ dm_sim/ tuning_devices/   # shared helpers + pipelines + MATLAB
```
