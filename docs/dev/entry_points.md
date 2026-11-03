# Entry Points — CLI 注册与未注册 Runner

> 外迁自根 `AGENTS.md` 的 Entry Points 节。CLI 命令全集与 flag 详见 [docs/user/cli-reference.md](../user/cli-reference.md)。

## 常用命令

```bash
python src/ao_shaping/main.py wf
python src/ao_shaping/main.py pib
python src/ao_shaping/main.py pipeline
python src/ao_shaping/main.py zernike-matrix
python src/ao_shaping/main.py rms-zernike
python src/ao_shaping/main.py ga-zernike
python src/ao_shaping/main.py slm-pib spgd          # SLM Zernike PIB (SLM + CCD)
python src/ao_shaping/main.py slm-pib square        # 方形远场整形 (slm-pib 的子命令；无 spgd-square 顶层命令)
python src/ao_shaping/main.py combined              # legacy AdaMOD+SPGD 混合 PIB
python src/ao_shaping/main.py hadamard-matrix       # Hadamard 响应矩阵
```

**Console script** (`pyproject.toml [project.scripts]`)：`ao-train = ml.train:cli` —— 唯一随包安装的入口，指向独立包 `src/ml/`（不在 `main.py` 的 click 组里）。

## CLI Structure (main.py 注册关系)

```
main (click.group)                                  # 20 个命令，逐个由 cli.add_command 注册
├── wf                ← runners.nlight_dm.wf_runner
├── pib               ← runners.nlight_dm.axis_beam_runner
├── pipeline          ← runners.nlight_dm.pipeline_runner
├── zernike-matrix    ← runners.slm.zernike_matrix_runner      (attr `run`)
├── closed-loop       ← runners.slm.zernike_matrix_runner      (attr `closed_loop_run`)
├── rms-zernike       ← runners.slm.rms_zernike_runner
├── ga-zernike        ← runners.zernike_search_runner          (attr `ga_zernike_run`)
├── greedy-zernike    ← runners.zernike_search_runner          (attr `greedy_zernike_run`)
├── dm-matrix         ← runners.matrix_runner                  (attr `dm_matrix_run`)
├── hadamard-matrix   ← runners.matrix_runner                  (attr `hadamard_matrix_run`)
├── alt-voltage       ← runners.micro_drive.voltage_runner     (同模块两命令，靠 attr 区分)
├── full-voltage      ← runners.micro_drive.voltage_runner
├── combined          ← runners.nlight_dm.combined_runner      [legacy]
├── slm-pib           ← runners.slm.slm_shaping_runner         (click 组: spgd / heuristic / square)
├── slm-gsnet         ← runners.slm.gsnet_runner
├── slm-gs-refine     ← runners.slm.gs_refine_runner
├── slm-model-in-loop ← runners.slm.model_in_loop_runner
├── slm-transfer-shapes ← runners.slm.transfer_shapes_runner  (物理预测模型多形状闭环整形: 从零训练 + 逐形状整形)
├── slm-lut           ← tools.slm.calibration.lut_runner       (来自 tools/，不经 _LAZY_RUNNERS)
└── slm-diagnose      ← tools.slm.panel.diagnose               (同上)
```

`main (click.group)` # 20 个命令，逐个由 cli.add_command 注册

- 一个模块导出多个命令时必须显式给出 attr（`_IMPORT_ATTRS`），否则落到默认 attr `run` 会取回错误的命令。
- `combined` 命令仍注册于 main.py 且功能可用，作为 legacy 保留。`pipeline_runner.py` 是推荐的 WF→PIB 串行方案。

## 未注册到 main.py 的独立 Runner

需直接运行 `python -m ao_shaping.runners.<name>` 或 standalone 脚本：

- `runners/slm/offset_runner.py` (计划迁移至 `tools/slm/`)。
- ⚠️ `hadamard_matrix_runner` 曾列于此并标注「正在注册」—— 2026-10-25 实测**已注册**（cli.add_command(hadamard_matrix_run, name="hadamard-matrix")，已进入 `EXPECTED_COMMANDS` 与 help golden），故移出本行。
- ⚠️ `slm_shaping_runner` 曾列于此 (2026-10-05 前未注册), 现以 `slm-pib` **命令组**注册（其 `square` 子命令即方形整形）。

## Refactoring Notes

- All runner scripts now use centralized config from `config.py` (DM_N_ACTUATORS, PATHS, DEFAULTS)
- Common CLI helpers moved to `utils/cli_helpers.py` (parse_tuple, setup_coredumpy)
- Duplicate code eliminated across runner files
