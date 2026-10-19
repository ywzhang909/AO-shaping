# 整形前的聚焦、彗差校正与 CCD 中心定位

相机位置改变后，先校正光斑弥散和不对称，再定位 CCD 光斑中心，可以减少
环纹、拖尾对中心估计的影响。自动入口为
`python -m ao_shaping.tools.slm.panel.alignment_probe`。
它调用项目已有的 `focus_probe.sweep_focus`，不重新实现离焦扫描或 Zernike 数学。

## 操作步骤

1. 释放 GUI 或官方软件中的设备连接，保持本次测量期间的光路配置。
2. 用全板入射定位获得 **SLM 面板坐标**，不要把 CCD 坐标当作面板坐标。
3. 指定当前照明区域的中心、半径，运行聚焦及彗差扫描。例如：

   ```powershell
   $env:PYTHONPATH="${PWD}/src;${PWD}/libs"
   python -m ao_shaping.tools.slm.panel.alignment_probe --output data/debug/alignment_probe/session_001 --center-x 960 --center-y 720 --radius 450 --focus-points 9 --diopter-range 0.5 --coma-points 5
   ```

4. 检查 `focus.json` 的焦度、相对平相位峰值/曝光增益和半曝光线性复测。
   未得到可用焦点或线性比不在 `0.8–1.2` 内时，中止彗差扫描。
5. 检查 `result.json` 的两个彗差系数、校正前后上下/左右镜像误差及 RMS 半径。
   最终基线复测和候选复测均支持改善时才保留彗差校正，否则回退聚焦相位。
6. 使用校正后的中心进行后续 CCD 定位；继续整形时显式叠加
   `correction_phase.npy`，并重新验证当前几何和独立探针预测。该文件为完整面板的
   **未包裹弧度**，只能在驱动的规范转换入口包裹。探测退出时恢复平相位，
   不会隐式保留校正。

输出目录必须是新目录。原始数据包含逐次相位、原始 CCD 帧、曝光读回值、
UTC 微秒时间戳及相位文件对应关系；`manifest.json` 每次采集后更新。
`coma_measurements.json` 每次候选测量后更新，便于中途失败时诊断。
曝光范围由设备读出；`--max-exposure-ms 0` 使用设备范围，正值限定本次探测，
Daheng 还遵循本次环境变量 `AO_DAHENG_MAX_EXPOSURE_MS`。不继承历史光路的曝光数值。

固定曝光独立复测命令如下。它读取原扫描的设备配置及已验证曝光，
按正序/逆序交替重放 3 组，输出中心重复性与不对称度的均值和样本标准差：

```powershell
python -m ao_shaping.tools.slm.panel.alignment_probe --verify-directory data/debug/alignment_probe/session_001 --output data/debug/alignment_probe/session_001_verify
```

当前 model-in-loop 可通过现有 `--phase-disturbance-file` 接口显式加载完整面板
校正相位；用于校正时该参数只是叠加通道，不代表又施加了随机干扰。
所有训练、留出、基线及后续重放都必须保留该文件。该诊断入口目前独立运行，
尚未自动插入每一次会话预检。尤其不能把未叠加校正时的历史几何默认为校正后有效；
2026-10-19 的实验已观察到聚焦和对称性改善后，沿用旧几何仍无法通过独立留出验证。

## 估计与判据的边界

单幅光强的宽度不能唯一确定离焦正负。扫描用正负二次相位确认符号，并在最优点
位于边界时扩展搜索，最后局部细化。输出焦度是施加在 SLM 上的等效校正焦度；
缺少完整光路传递矩阵时，不能把它直接当作相机轴向位移。

彗差用规范生成器的 `(3,-1)` 与 `(3,1)` 两个模式，系数单位为 rad。
两者对应面板轴，CCD 的上下/左右由实测响应决定，不预设相机旋转或镜像关系。
扫描以两个方向镜像误差之和为目标，逐轴搜索并重复验证；不把单幅不对称度直接
换算成唯一的物理像差系数。像散、遮挡、相干杂散光同样可能造成不对称。

指标在最亮光斑附近重新估计局部强度质心，再做镜像比较，以消除平移的影响。
仅减去局部边界中位数，它不等同于真实暗场标定。使用局部窗口会遗漏窗口之外的
能量，需根据实际光斑大小设置 `--half-width`，窗口触及传感器边界时拒绝测量。
RMS 半径不得超过聚焦基线的 `1.15` 倍，防止通过弥散降低不对称度；
近似相同的镜像误差优先选择半径较小的候选。

聚焦和对称性改善不等于目标整形成功。后续仍需分别报告独立探针相关性、
目标区域 CV、EE、形状 IoU，以及固定曝光下的重复测量。

## 校正保持叠加时重新测量几何

几何测量脚本支持 `--phase-offset-file`。必须在 `sweep` 阶段传入组合相位，
使曝光探测和所有标定探针处于同一校正状态。`calibrate` 阶段读取保存的扫描数据：

```powershell
python scripts/slm/model_in_loop_hw_runbook.py --stage sweep --out data/debug/model_in_loop_hw_runbook/session_corrected --pupil-center 960,720 --region 128 --far-field-size 1024 --collect-disc 450 --zernike-radius 450 --exposure-ms 0.1 --phase-offset-file data/debug/alignment_probe/session_001/correction_phase.npy
python scripts/slm/model_in_loop_hw_runbook.py --stage calibrate --out data/debug/model_in_loop_hw_runbook/session_corrected --region 128 --far-field-size 1024 --collect-disc 450 --zernike-radius 450
```

命令中的中心、半径和曝光均为示例，需来自当前会话测量。
扫描目录另存 `phase_offset.npy`，Recorder 中 `_phase` 为已叠加后的实际显示总相位。
自动会话的几何步骤也会传递配置中的显式相位叠加文件；入射定位与照明范围的
测量状态仍需单独核对，不能将这次几何接线当作全部预检均已自动适配。
