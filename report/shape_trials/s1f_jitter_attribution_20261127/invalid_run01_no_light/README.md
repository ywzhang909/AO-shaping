# 作废判读产物（run_01 无光）

本目录 `s1f_jitter_attribution_judgment.json` / `_tables.md` 是对
`data/experiments/attribution_20261123/s1/s1f_jitter_attribution_20261127/run_01`
的判读输出，**该次采集全场无有效光强**（flat 峰 41 counts，S1e 同日同曝光为 2524 counts），
帧内只有噪声。

因此判读得出的 `A.rms_px ≈ 978 px`、`pairwise_max ≈ 2676 px`、`write_screen_rms ≈ 69 px`
均为「主瓣选取在不同噪声峰之间跳变」的假象，`V1` 裁定与 `比例 93.4% / 6.6%` **全部无效**，
不得引用、不得作为 S1f 结论。

失效根因与处置见 `data/experiments/attribution_20261123/s1/s1f_jitter_attribution_20261127/run_01/INVALID_no_light.md`；
复采为 `run_02`，采集脚本已加光强预检（预注册 §9）。
