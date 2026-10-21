# Beam-Shaping Benchmark (simulation)

<!-- provenance:start -->
> **生成脚本**: [`scripts/slm/generate_beam_shaping_benchmark_report.py`](../../../scripts/slm/generate_beam_shaping_benchmark_report.py)
> **复现命令**: `python scripts/slm/generate_beam_shaping_benchmark_report.py`
> **运行环境**: 离线
> **说明**: 9 单元权威网格 (3 算法 x 3 target)；gif/ 为 6 个演化动画
<!-- provenance:end -->

**本轮结论**（3 算法 × 3 目标权威网格）：面积达标（`area_met`）是首要判据，未达标行的均匀度/填充率仅为噪声。结果：**GS 3/3 全达标**（填充率 0.876–0.965，CV 0.15–0.21，耗时 <1.3 s），为 device-less 整形的默认选择；**backprop 1/3**（仅 square 达标，CV 0.006 极优，circle/gaussian 失败），属硬件级算法族；**spgd-sim 0/3**——实测面积全部坍缩至约 1 px（CV 16–34），为已记录的失效模式而非回归，其围阀能量 0.998 是坍缩伪影。另注：`slm_shaping_bench.strehl()` 为去均值余弦相似度而非物理 Strehl，故本表不设 Strehl 列。

> **Read `area_met` before comparing algorithms.** It is the only column that says whether the algorithm produced the requested target *at all*; `uniformity_cv` and `fill_ratio` on a row that failed it describe noise.
> Area check passed, per algorithm: backprop 1/3, gs 3/3, spgd-sim 0/3.
> `spgd-sim` currently fails every shape (measured area collapses to ~1 px, CV 16-34), so its uniformity numbers are meaningless device-less -- the documented failure, not a regression. `backprop` is hardware-class: it passes square and fails the other two.
>
> ⚠️ `slm_shaping_bench.strehl()` is a mean-removed cosine similarity, not a physical Strehl ratio. No Strehl column appears here for that reason.
>
- Grid: 9 cells (3 algorithms x 3 shapes)
- Regenerate: `python scripts/slm/generate_beam_shaping_benchmark_report.py` (fully offline, ~100 s)
> The sibling `.csv` is written too but is **not committed** -- the repo has a global `*.csv` ignore rule and no CSV under `docs/` is tracked. Re-run to get it; this markdown is the tracked artefact.

| algorithm | shape | requested | measured | area_met | fill_ratio | uniformity_cv | encircled_energy | elapsed_s |
|---|---|---|---|---|---|---|---|---|
| backprop | circle | 1160 | 579 | False | 0.499 | 0.444 | 0.750 | 8.851 |
| backprop | gaussian | 392 | 160 | False | 0.408 | 0.991 | 0.023 | 1.723 |
| backprop | square | 256 | 258 | True | 1.008 | 0.006 | 0.720 | 1.683 |
| gs | circle | 1160 | 1016 | True | 0.876 | 0.154 | 0.952 | 0.805 |
| gs | gaussian | 392 | 351 | True | 0.895 | 0.210 | 0.479 | 0.795 |
| gs | square | 256 | 247 | True | 0.965 | 0.169 | 0.951 | 0.775 |
| spgd-sim | circle | 1160 | 1 | False | 0.001 | 34.038 | 0.998 | 0.829 |
| spgd-sim | gaussian | 392 | 1 | False | 0.003 | 19.773 | 0.998 | 0.816 |
| spgd-sim | square | 256 | 1 | False | 0.004 | 15.968 | 0.998 | 0.813 |
