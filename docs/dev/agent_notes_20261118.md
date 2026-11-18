# Agent 注意事项（2026-11-18 当日修改）

> 按日追加；过期条目迁入 `docs/dev/` 对应专题。

## 上机：当前 epoch 绝对强度标定（含 SLM 假死 + 物理重启）

- **设备占用**：Santec SLM #1（1064nm，memory 模式）+ Daheng CCD #0。全程只此一进程占用，无并行设备进程。
- 第一次采集（13:40:47）：CCD 打开成功，**SLM #1 `SLM_Ctrl_Open` 返回 -10002（未找到 USB 设备）** —— 与历史
  `ReadSU=-1000` 驱动板假死同源；`open` 阶段即失败，未写任何相位；`finally` 于 13:41:03 释放 SLM #1 + CCD，
  device_log 只 2 条。记录：`data/debug/absolute_calib_current_epoch_20261118/_run_execute.log`、`device_log.json`
  （失败那次）。**未重复开关设备**。
- 用户物理断电重启 SLM #1 控制器并等自检后，第二次采集（13:58:46）成功：`成功打开SLM #1`、状态正常、
  driveboard 22030108，12 个确定性相位全程无 USB 错误；自动曝光生效 **0.063 ms**（8-bit，峰值未饱和）、
  定心帧 peak 193/255（<0.98 饱和线），argmax=(1227,1107) 与当前 epoch zernike 采集 spot≈(1225,1106) 一致。
  13:59:36 `成功关闭SLM #1` + 释放 CCD，device_log 30 条，正常关闭。
- 数据：`data/debug/absolute_calib_current_epoch_20261118/`（summary/plan/device_log + 12×model + 12×frames + phase）。
  该次用的是探针默认 checkpoint（11/07 M3，grid=128/fft512/phase_sign=-1），**不是**本轮 11/18 候选
  （`axis_mismatch_validation_20261118/fit`，grid=320）—— 实测帧是纯硬件数据可复用，模型列对应旧 checkpoint。
- **复用注意**：grid=320 checkpoint 的锚定相位 (128,128) 不兼容 `absolute_fidelity_probe_20261116.py` 的默认相位集；
  11/18 fit 必须先降采样或另建 grid-128 相位集才能跑该探针。本次为抢设备窗口，用 grid-128 默认集采了实测帧。

## 遗留 / 判据边界

- 这批 absolute_calib 的探针原生指标：CV_pred 中位 3.69、CV_measured 5.17、强度比中位 0.30 —— 旧 checkpoint
  的绝对保真度仍不合格，不能作放行证据。补算的 relL2/corr 口径与 hybrid 拟合脚本不同，只作旁证不直接比较。
- 当前 epoch Zernike 方向数据 `data/debug/slm_transfer/current_epoch_zernike_direction_20261118/` 末段也因
  SLM -10032 USB 错误中断（amp=0.10 不完整，弃用）；amp=0.05 的 14 帧 ±Z5–Z11 + 9 标定行完整，可用于方向门。
- 双门未出结果前**不进锚定闭环**。离线双门评估见后台会话与 `report/shape_trials/offline_refit_20261118_*`。

## 离线双门评估结论（2026-11-18，已出结果）

- 后台 explorer 会话用当前 epoch fresh 单模 Zernike 屏（7 对 ±Z5..Z11，amp=0.05）对 8 候选跑方向门 + 零步绝对强度门回放。
  **结果：不存在同时过双门的 checkpoint，本轮不进物理闭环验收**（全程未开设备）。
- 方向门（每模式 7 对，既有 10/14 标准需 2 passes×7 modes，本轮单 pass 物理不可达；按比例折算 5/7）：
  11/18 fit 中位 **0.6856**、**4/7** 达 0.50（Z5 .877、Z8 .843、Z10 .791、Z11 .686 过；Z6 .362、Z9 .347、Z7 .026 未过）；
  唯一达 5/7 的是诊断变体 `hybrid320_fresh_all`（无残差项、非完整 checkpoint）。其余 ≤3/7。
- 强度门（in-sample 零步、无 held-out，仅作跨 epoch 对照）：11/18 fit relL2 **0.355**、CV 差 0.310、corr 0.931（8 候选中最好）；
  两个 hybrid refit relL2 ≈0.70、CV 差 1.34–1.39（明显更差）。对照 11/18 held-out 基线 relL2 0.529。
- 判据边界：方向门 7 对不足以达成字面 10/14；强度门无 held-out（不能当放行证据）；11/18 fit 短板集中在 Z6/Z7/Z9。
- 报告 `report/shape_trials/offline_refit_20261118_freshep/offline_refit_20261118_freshep.md`；产物 `data/debug/offline_refit_20261118_freshep/`
  （direction_gate_results.json、intensity_summary.json + 每候选 result.json；两驱动脚本 exit 0）。报告已登记 provenance，sync --check 干净、
  自身 4 项 provenance 测试通过（全量 provenance pytest 30 条失败均为其他会话未登记报告的既有断链，与本线无关）。

## 方向门失败机制归因：主导响应轴符号错配（2026-11-18 晚，离线）

- 目标：定位为什么冻结模型类里方向门过不了。全程只读归档 npz，未占用 SLM/CCD/激光。报告已补 §4。
- **排除项**（产物均在 `data/debug/offline_refit_20261118_freshep/`）：
  - 消融 14 参数块 + 7 残差留一（ablate_direction_failure_20261118.py）：21 变体无一超 5/7；去全部残差到 5/7 但 Z11 .222。
  - 噪声预算（noise_budget_direction_20261118.py）：逐模式 SNR 2.26–5.04、噪声相关上限 0.914–0.981 ≫ 实测相关 → 非噪声主导。
  - 模型×实测 7×7 交叉匹配（cross_mode_match_20261118.py）：对角线 6/7 行占优，无 sina/cos 整体互换（仅弱 Z7↔Z9 +0.50）→ 非索引错位。
  - 五条外生参数扫描：标尺 s（scale_sweep，Pearson 完全不敏感）、窗口 128→512（window_sweep，128 最好、加宽单调恶化）、
    相机模糊 sigma 0→8（sigma_sweep，零残差 Z7 .306→.428 但仍不过）、离焦 -0.009→0.18（defocus_sweep，Z7 模型平移恒 ~0.10 px vs 实测 1.04 px）。
    **共同特征：通过数封顶 5/7，只是失败模式集合轮换。**
  - 差分域重拟合（diff_domain_refit）：放开 s+残差后 s 被推到 17–40、留出塌到 ~0，是共线退化，未采用。
- **根因（可证伪核对）**：response_rank_20261118.py 的 7×7 差分 Gram 谱——实测参与比 2.52、PC1 占 0.583；模型(fit11) PC1 占 0.460；
  实测在模型张成空间的投影 0.870（不是装不下，而是方向对不上）。pc1_sign_check_20261118.py：
  **fit11 的"PC1 符号一致集合" {Z5,Z8,Z10,Z11} 与其方向门过门集合完全相同（4/7），零拟合参数。**
  残差把 Z6/Z7/Z9 的 PC1 符号翻反，删残差纠正这三档却把 Z11 翻错 → 解释"通过数永远 5/7、失败集合轮换"的饱和现象。
- **双门严格互斥**（新补测 eval_intensity_no_residual_20261118.py）：无残差候选方向门升到 5/7，但强度门 relL2 0.355→0.582、corr 0.931→0.804。
  注：`PhysicalForwardFit` 把 residual_modes 硬编码 range(5,12)，选择性删残差在 fit 管线里不可表示（KeyError），只能诊断。
- **决定**：冻结模型类内继续搜参数无意义；要过双门须换模型类（数据驱动局部 Jacobian / 可学习像差-平移耦合）或修订门口径，**均需用户先定夺**。
  满足双门前不进锚定闭环。报告 §4 已写入全部数字与复现命令。
