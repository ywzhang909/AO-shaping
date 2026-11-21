# 2π 灰度来源决策（2026-11-19）

## 背景

用户要求：`two_pi_grayscale: 1023` 改为从 SLM 设备读取的 2π 灰度值；项目不再冻结
1064 nm 的 2π 工作值 993，也改为从设备读取。

核查后发现仓库里有两个都叫「2π 灰度」但含义不同的量，之前的混用正是这次问题的根因。

## 一、两个概念严格区分

| 概念 | 来源 | 用途 |
|------|------|------|
| **编码标度** `max_grayscale` | 配置 `[slm] two_pi_grayscale`，缺省 1023 | `gray = φ / 2π × max_grayscale`，把弧度编码成灰度 |
| **物理 2π 灰度** | Santec 相位表设备读数（1064 nm 典型 993、532 nm 典型 998） | 只作诊断与上报，不参与弧度→灰度编码 |

权威说明在 `src/ao_shaping/utils/slm/gray_scale.py` 顶部 docstring。
`Santec.get_wavelength_info() -> (wavelength_nm, max_grayscale_for_2pi)` 是设备动态查询入口
（内部经 `SLM_Ctrl_ReadWL` 读 phase_pi 再算 `2 / phase_pi × 1023`），返回的就是上表第二项。
993 / 998 只是这两个波长的**典型读数**，不是冻结常量。

## 二、已定决策

1. **标准光路记录 `optical_path_record.json` 的 `devices.slm.two_pi_grayscale` 字段改为设备读数。**
   - `scripts/slm/probe_optical_path_validity.py::_collect_record_devices` 调
     `slm.get_wavelength_info()[1]` 取物理 2π 灰度；
   - 设备不可读或读数非正时写 `null`，并在 `reasons` 里登记原因，由
     `build_optical_path_record` 收进 `missing`，**绝不**回落成 1023 冒充设备值；
   - 读取按现有 try/except 风格容错，只读字段失败不阻断记录落盘。
2. **仿真侧的 993 保留数值，明确标注为「仿真默认值，非设备读数」。**
   - `src/ao_shaping/drivers/sim/slm_pib_sim.py::TWO_PI_GRAY`、
     `src/ao_shaping/drivers/sim/nonideal.py::SLM_TWO_PI_GRAY`：仿真没有真实 SLM，
     无法查询相位表；993 是历史设备回报的典型值，固定它保证仿真重现性，改动会破坏既有仿真结果。
3. **文档里冻结的「2π=993」改为描述性表述**：设备读取，1064 nm 典型读数 993，
   不作为权威常量。
4. **编码标度信息若需保留，用不同字段名**（如 `encoding_max_grayscale`），避免与设备读数混淆。

## 三、为何历史记录不改

`data/debug/optical_path_probe/.../optical_path_record.json` 中已有的
`two_pi_grayscale: 1023` 属于历史记录：它记录的是当时脚本写入的编码标度，是既成事实，
改动历史数据会破坏审计链与既有数据的可复现性。本决策只修正**新记录**的写入逻辑，
历史数据原样保留；凡引用这些历史记录时，须注意该字段当时记的是编码标度而非设备读数。

## 四、影响范围

- 改动：`scripts/slm/probe_optical_path_validity.py`、两处仿真常量注释、三份 AGENTS.md、本决策记录。
- 未改动：`gray_scale.py` 的编码标度解析逻辑、`Santec.get_wavelength_info()` 本身、
  已预注册冻结的评分脚本（validate / replay / run_window / run_issue238 / _window / protocol）。
- 硬件操作：本次修改未连接设备、未执行任何硬件命令。
