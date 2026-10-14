"""WFS-only 隔离探针 - 定位 0xC0000374 堆损坏源头 (2026-09-16).

背景: n=5 zernike-matrix 实机 3/3 崩溃 0xC0000374, faulthandler 检测点在
thorlab_wfs.py L1493 (get_zernike 内下一次 np.empty 堆分配)。检测点 != 源头,
源头必在更早的原生调用 (take_image / get_spot_deviation / get_zernike /
get_wavefront)。本探针不开 SLM, 仅 WFS 循环镜像 _capture_wfs_full_state 调用序列:

    wfs.take_image()
    wfs.get_spot_deviation(cancel_tile=False)
    wfs.get_zernike(zernike_order=10)
    wfs.get_wavefront(cancel_tile=False)

- 崩溃 → WFS DLL 内部越界是源头 → 需对 WFS 所有输出缓冲做防御性 padding。
- 干净 (默认 80 iter ≈ 4x n=5 工作量) → WFS DLL 单独运行无腐蚀 → 嫌疑转向
  SLM 写入路径 (4.6MB dat 缓冲生命周期 / Santec DLL 内部), 下一阶段探针加 SLM。

用法:
    $env:PYTHONPATH="src"; python -X faulthandler C:/Users/zhangh/AppData/Local/Temp/opencode/wfs_probe.py --iters 80 --mla 512 --exp 4.0 *> data/slm_corrections/wfs_probe.log

作者: Sisyphus (崩溃勘察工作流)
"""
from __future__ import annotations

import argparse
import faulthandler
import sys
import time

import numpy as np

from ao_shaping.drivers.wfs import MlaRes, ThorlabWFS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WFS-only 堆损坏隔离探针")
    p.add_argument("--iters", type=int, default=80, help="循环次数 (默认 80)")
    p.add_argument("--mla", type=int, default=512, help="MLA 分辨率 (默认 512)")
    p.add_argument("--exp", type=float, default=4.0, help="WFS 曝光 ms (默认 4.0)")
    p.add_argument("--high-speed", action="store_true", help="启用高速模式")
    p.add_argument("--use-custom-ref", action="store_true", help="使用自定义参考")
    p.add_argument("--cancel-tile", action="store_true", help="去除 tip/tilt")
    p.add_argument("--zernike-order", type=int, default=10, help="Zernike 阶数")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    faulthandler.enable()
    print(f"[probe] WFS-only 隔离探针 iters={args.iters} mla={args.mla} "
          f"exp={args.exp}ms high_speed={args.high_speed} "
          f"custom_ref={args.use_custom_ref} cancel_tile={args.cancel_tile}",
          flush=True)

    mla_enum = {
        320: MlaRes.Res320, 512: MlaRes.Res512, 768: MlaRes.Res768,
        1024: MlaRes.Res1024, 1280: MlaRes.Res1280,
    }.get(args.mla)
    if mla_enum is None:
        print(f"[probe] 未知 MLA: {args.mla}", flush=True)
        return 2

    wfs = ThorlabWFS(
        mla_index=mla_enum,
        exposure_time=args.exp,
        high_speed=args.high_speed,
        use_custom_ref=args.use_custom_ref,
    )
    try:
        wfs.open()
    except Exception:
        print("[probe] WFS open 失败", flush=True)
        raise

    # 镜像 runner L1081-1082: 启动时 take_image + optimize_pupil
    print("[probe] open OK, 开始 optimize_pupil ...", flush=True)
    t0 = time.perf_counter()
    wfs.take_image(n_sample=1, dynamicNoiseCut=True)
    cx, cy, dx, dy = wfs.pupil = wfs.optimize_pupil()
    print(f"[probe] pupil=({cx:.3f},{cy:.3f})mm d=({dx:.3f},{dy:.3f})mm "
          f"({time.perf_counter()-t0:.1f}s)", flush=True)

    # 镜像 runner _capture_wfs_full_state (L278-282), 结果累积模拟 runner 内存行为
    snapshots: list[tuple] = []
    t_start = time.perf_counter()
    for i in range(1, args.iters + 1):
        it_t0 = time.perf_counter()
        print(f"[probe] iter {i}/{args.iters} ...", end=" ", flush=True)
        try:
            wfs.take_image()
            dev_x, dev_y = wfs.get_spot_deviation(cancel_tile=args.cancel_tile)
            z = wfs.get_zernike(zernike_order=args.zernike_order)
            wf, stats = wfs.get_wavefront(cancel_tile=args.cancel_tile)
        except Exception:
            print(f"ERROR iter {i}, elapsed={time.perf_counter()-t_start:.1f}s",
                  flush=True)
            raise
        snapshots.append((dev_x, dev_y, z, wf))
        rms = float(stats.get("rms", np.nan)) if stats else np.nan
        print(f"OK z[0..5]={np.round(z[0:6],3)} rms={rms:.4f} "
              f"({time.perf_counter()-it_t0:.2f}s)", flush=True)
        # 每 20 迭代强制一次 GC, 制造堆块腾挪, 提高小型越界命中概率
        if i % 20 == 0:
            import gc
            gc.collect()

    el = time.perf_counter() - t_start
    n_kept = sum(len(np.asarray(s[2])) for s in snapshots)
    print(f"[probe] PASS: {args.iters} iters 无崩溃, elapsed={el:.1f}s "
          f"({el/args.iters:.2f}s/iter), 累积 zernike 元素={n_kept}", flush=True)

    try:
        wfs.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())