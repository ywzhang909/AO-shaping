"""无设备 (device-less) 全网格基准 + GIF: 只调用权威模块公开 API。

用途: 一次性生成 docs/benchmarks/device_less_full/ 下的
  - beam_shaping_benchmark_metrics.csv  (权威 run_benchmark_suite, 9 行网格)
  - beam_shaping_benchmark_metrics.md  (权威套件)
  - gif/ 子目录 6 个演化 GIF (gs / spgd-sim x square/circle/gaussian)

不修改任何权威模块; 本脚本只是驱动, 完成后可删除。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ao_shaping.algorithm.beam_shaping_benchmark import (
    run_benchmark,
    run_benchmark_suite,
)

OUT = Path(__file__).resolve().parents[1] / "docs" / "benchmarks" / "device_less_full"
GIF = OUT / "gif"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    GIF.mkdir(parents=True, exist_ok=True)

    # ── 1) 权威 9 行网格 (3 算法 x 3 形状, 无设备) ──
    rows, df = run_benchmark_suite(
        output_dir=OUT,
    )
    print(f"SUITE_ROWS={len(rows)}")
    with open(OUT / "suite_stdout.txt", "w", encoding="utf-8") as f:
        f.write(df.to_string())

    # ── 2) 收敛组合的演化 GIF (gs/spgd-sim 在无设备下面积达标, backprop 记录为失败) ──
    for algo in ("gs", "spgd-sim"):
        for shape in ("square", "circle", "gaussian"):
            sub = GIF / f"{algo}_{shape}"
            sub.mkdir(parents=True, exist_ok=True)
            r = run_benchmark(
                algorithm=algo,
                shape=shape,
                iterations=300,
                seed=42,
                max_frames=30,
                make_gif=True,
                device=None,
                output_dir=sub,
            )
            gifs = sorted(sub.glob("*.gif"))
            print(
                f"GIF {algo} {shape}: fill={r.get('fill_ratio', float('nan')):.3f} "
                f"area_met={r.get('area_met')} gif_count={len(gifs)}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
