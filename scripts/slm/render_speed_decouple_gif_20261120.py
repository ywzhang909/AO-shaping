"""把已渲染的 ``speed_decouple_actual_ccd_20261120`` MP4 转成便于 push 的 GIF。

为什么要单独一个脚本
--------------------
原视频是 1296x972 / 24 fps CFR / H.264, 时长约 173 s, 1440 KiB 左右, 本身已经很适合
仓库; 但 GitHub/PR 预览里 MP4 常常不能内联播放, GIF 则可以直接显示。这个脚本把**已经
渲染好的 MP4** (只认默认路径下、已忽略 git 的两个版本) 用 ffmpeg 的 palettegen /
paletteuse 转成 GIF:

* 只有一张全片统一的调色板 (``palettegen=max_colors=32``), 不逐帧自适应;
* 不做帧间插值或图像融合, 只按真实时间戳零阶保持, 与 MP4 的语义一致;
* 默认 3 fps / 720 px 宽 (实测采集 ~2.3 fps, 3 fps 不会丢采样), 32 色无抖动时
  8-9 MiB, 落在仓库既有 GIF 产物体积区间;
* 输入只认 ``speed_decouple_actual_ccd_20261120*.mp4`` 这个文件名模式, 避免把别的
  mp4 误转; 输出路径固定为 ``report/shape_trials/videos/<stem>.gif``, 同样带
  ``speed_decouple_actual_ccd_20261120`` 前缀, 不会覆盖任何现有文件。

不打开任何硬件、不联网、不修改 .mp4, 不 git add/commit。

用法::

    .venv\\Scripts\\python.exe scripts/slm/render_speed_decouple_gif_20261120.py
    .venv\\Scripts\\python.exe scripts/slm/render_speed_decouple_gif_20261120.py --gamma 0.45 --limit 2 --out data/debug/_smoke.gif

依赖: ffmpeg(1) 需在 PATH 上 (palettegen / paletteuse 滤镜)。
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from loguru import logger

#: 与 render_speed_decouple_actual_video_20261120.py 的输出约定保持一致。
DEFAULT_SRC = Path("report/shape_trials/videos/speed_decouple_actual_ccd_20261120.mp4")
DEFAULT_GAMMA_SRC = Path("report/shape_trials/videos/speed_decouple_actual_ccd_20261120_gamma045.mp4")
DEFAULT_OUT = Path("report/shape_trials/videos/speed_decouple_actual_ccd_20261120.gif")
NAME_PREFIX = "speed_decouple_actual_ccd_20261120"
GIF_FPS = 3
GIF_WIDTH = 720
GIF_COLORS = 32


def _ffmpeg_gif_command(
    ffmpeg: str, src: Path, out: Path, *, fps: int, width: int, colors: int
) -> list[str]:
    """palettegen / paletteuse 转 GIF; 一张全局调色板, 无抖动, 保留透明度关闭。"""
    vf = (
        f"fps={fps},scale={width}:-1:flags=lanczos,split[s0][s1];"
        f"[s0]palettegen=max_colors={colors}:reserve_transparent=0[pal];"
        f"[s1][pal]paletteuse=dither=none:diff_mode=rectangle"
    )
    return [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(src),
        "-vf", vf,
        "-loop", "0",
        str(out),
    ]


def convert(
    src: Path, out: Path, *, fps: int, width: int, colors: int
) -> int:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("PATH 上找不到 ffmpeg.exe")

    if not src.is_file():
        raise FileNotFoundError(f"源 MP4 不存在: {src.resolve()}")
    if not src.name.startswith(NAME_PREFIX) or src.suffix.lower() != ".mp4":
        raise ValueError(f"源文件名不符合约定 ({NAME_PREFIX}*.mp4): {src.name}")
    if not out.name.startswith(NAME_PREFIX) or out.suffix.lower() != ".gif":
        raise ValueError(f"输出文件名不符合约定 ({NAME_PREFIX}*.gif): {out.name}")
    if out.resolve() == src.resolve():
        raise ValueError("输出 GIF 不能覆盖源 MP4")

    cmd = _ffmpeg_gif_command(ffmpeg, src, out, fps=fps, width=width, colors=colors)
    logger.info("ffmpeg: " + " ".join(cmd))
    out.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        logger.error(f"ffmpeg 退出码 {result.returncode}: {result.stderr}")
        return 1
    if result.stderr:
        logger.warning(f"ffmpeg stderr: {result.stderr}")

    mb = out.stat().st_size / (1024 * 1024)
    logger.success(f"写出: {out.resolve()}")
    print("=" * 68)
    print(f"源 MP4                 : {src.resolve()}")
    print(f"输出 GIF               : {out.resolve()}")
    print(f"GIF 参数               : {width} px 宽, {fps} fps, {colors} 色全局调色板, 无抖动, 无限循环")
    print(f"文件大小               : {mb:.2f} MiB")
    print("=" * 68)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="把已渲染的 speed-decouple 实测 CCD MP4 转成便于 push 的 GIF (不碰硬件)。",
    )
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC, help=f"源 MP4 (默认 {DEFAULT_SRC})")
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="选 gamma 版源 MP4 (0.45 -> speed_decouple_actual_ccd_20261120_gamma045.mp4); "
             "显式 --src 时忽略",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"输出 GIF (默认 {DEFAULT_OUT})")
    parser.add_argument("--fps", type=int, default=GIF_FPS, help=f"GIF 输出帧率 (默认 {GIF_FPS})")
    parser.add_argument("--width", type=int, default=GIF_WIDTH, help=f"GIF 宽度 (默认 {GIF_WIDTH})")
    parser.add_argument("--colors", type=int, default=GIF_COLORS, help=f"调色板颜色数 (默认 {GIF_COLORS})")
    args = parser.parse_args(argv)

    src = args.src
    if args.gamma is not None and args.src == DEFAULT_SRC:
        # 视频脚本把 gamma 0.45 写成 `gamma045` (去掉小数点); 保持一致。
        token = f"{args.gamma:g}".replace(".", "")
        src = DEFAULT_SRC.with_name(f"{DEFAULT_SRC.stem}_gamma{token}{DEFAULT_SRC.suffix}")
        if args.out == DEFAULT_OUT:
            args.out = src.with_suffix(".gif")
        if not src.is_file():
            raise FileNotFoundError(
                f"gamma={args.gamma:g} 对应 MP4 不存在: {src.resolve()}; "
                "先用 render_speed_decouple_actual_video_20261120.py --gamma ... 生成"
            )
    return convert(src, args.out, fps=args.fps, width=args.width, colors=args.colors)


if __name__ == "__main__":
    sys.exit(main())
