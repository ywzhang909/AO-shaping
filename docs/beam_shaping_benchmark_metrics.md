# Beam-Shaping Benchmark (simulation)

> ⚠️ **2026-10-01 复核：这不是权威基准结果，勿据此判断算法优劣。**
>
> - 本文件只有 **1 行**（`Grid: 1 cells`），是一次 smoke 残留。
>   权威套件是 **9 行网格**（3 算法 × 3 形状），由
>   `scripts/run_device_less_full.py:4-5,32-34` 调用
>   `run_benchmark_suite(output_dir=docs/benchmarks/device_less_full)` 产生，
>   写出器是 `algorithm/signal_processing/beam_shaping_benchmark.py:562`。
> - 该权威产物目录被 `.gitignore` 排除（`.gitignore:97`
>   `docs/benchmarks/device_less**`），本 checkout 下**两个版本都不存在** ——
>   根 `README.md:1813` 却把本文件当权威链接指向它。
> - 复现：`python scripts/run_device_less_full.py`
> - ⚠️ 报告生成写在 `algorithm/` 层违反 `AGENTS.md` 的反模式红线
>   （report generation MUST live in `scripts/`），尚未修，见 `TODO.md`。
>
> ⚠️ 表中的 `Strehl` 一类指标请注意：`slm_shaping_bench.strehl()` 是**去均值余弦相似度**，
> 不是物理 Strehl 比（见 `docs/zotero_objectives/README.md` §命名冲突）。

- Grid: 1 cells

| algorithm | shape | requested | measured | area_met | fill_ratio | uniformity_cv | encircled_energy | elapsed_s |
|---|---|---|---|---|---|---|---|---|
| gs | circle | 76 | 74 | True | 0.974 | 0.188 | 0.937 | 0.018 |
