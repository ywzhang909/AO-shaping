"""把 ``speed_decouple_capture_20261120_v2`` 的**实测 CCD 帧**渲染成一条可回放的视频。

只读已落盘的 :file:`frames.npy` / :file:`frames_meta.json` / :file:`run_summary.json`,
不打开任何相机、SLM 或网络连接, 也不修改源目录里的任何文件。

为什么需要这个脚本
------------------
那次采集的原始节奏是 "每次 SLM 换图后拍一张 CCD", 实测只有约 2.3 fps, 而
:file:`frames.npy` 是 (52, 1944, 2592) 的预分配缓冲, 只有 ``records`` 里那几张是
真的拍到的。直接把 npy 当成 52 帧放出来会凭空多出 38 帧没有数据的黑图, 也会
把"两次实测之间到底停在哪一帧"讲错。所以这里严格按 ``records[i]['t_rel_s']``
做**零阶保持** (zero-order hold):

* 每个输出时刻显示**最新的实测帧**, 直到下一条实测记录的时间戳才切换;
* 一条 transition 的最后一张实测帧一直保持到该 transition 的窗口结束
  (标称 5.0 s; 若末帧时间戳已越过 5.0 s —— 实测 18/30 条落在 5.05-5.24 s ——
  窗口改为 "末帧 t_rel_s + 一个采样间隔", 否则零阶保持会给末帧 0 s 显示时间);
* 相邻 transition 之间插入 0.5 s 的 settle 段, 只显示上一条的末帧;
* 30 条按 :file:`run_summary.json` 的数组顺序串成一条 24 fps CFR 的 MP4。

播放语义上这**不是** CCD 原生 10 fps, 也**不做**插值 / 帧间融合 / 逐帧自动归一化:
灰度一律走同一张固定查找表 (默认线性 ``0..4095 -> 0..255``, 12-bit 满量程 4095),
同一帧的亮度在整条视频里只由它的原始计数决定。``--gamma`` 可换一张全片统一的
幂律 LUT (仍是全片一张表, 不是逐帧归一化), 用于把 halo 结构从 2-12/255 的暗端
拉到可见范围; 默认 1.0 即线性, 与原始计数一一对应。

用法::

    .venv\\Scripts\\python.exe scripts/slm/render_speed_decouple_actual_video_20261120.py

    # 只渲前两条, 用来快速确认排版 / 编码参数
    .venv\\Scripts\\python.exe scripts/slm/render_speed_decouple_actual_video_20261120.py \\
        --limit 2 --out data/debug/_smoke.mp4

依赖: Pillow (uint16 -> 8bit + 叠加文字)、ffmpeg(1) (H.264 编码, 需在 PATH 上)。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image, ImageDraw, ImageFont

#: 源数据根目录 (相对仓库根; 也可用 ``--src`` 覆盖)。
DEFAULT_SRC = Path("data/debug/slm_transfer/speed_decouple_capture_20261120_v2")
#: 视频落点。
DEFAULT_OUT = Path("report/shape_trials/videos/speed_decouple_actual_ccd_20261120.mp4")

#: 每条 transition 的标称时间窗 (采集时的计划值)。
CONTENT_WINDOW_S = 5.0
#: 两条 transition 之间保持上一条末帧的时长 (采集时的 settle 计划值)。
SETTLE_S = 0.5
#: 输出固定帧率。
OUT_FPS = 24
#: 输出分辨率: 源图 2592x1944 的 1/2, 同宽高比, 无 letterbox。
OUT_SIZE = (1296, 972)
#: 12-bit 相机的有效满量程; 固定映射到 8-bit 的上界。
FULL_SCALE = 4095.0

#: 优先尝试的中文字体 (Pillow 直接吃 ttf/ttc 文件路径)。
CJK_FONT_CANDIDATES = (
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\msyhbd.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\Deng.ttf"),
)
#: 找不到中文字体时的 ASCII 字体。
ASCII_FONT_CANDIDATES = (
    Path(r"C:\Windows\Fonts\consola.ttf"),
    Path(r"C:\Windows\Fonts\arial.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
)


@dataclass(frozen=True)
class Transition:
    """一条 transition 的只读视图 (已通过完整性校验)。"""

    index: int
    label: str
    directory: Path
    t_rel_s: tuple[float, ...]
    frame_index: tuple[int, ...]
    exposure_ms: float
    duration_s: float  # 该 transition 在视频里占用多久
    achieved_fps: float | None


@dataclass(frozen=True)
class Segment:
    """视频时间轴上的一段: 从 ``start_s`` 到 ``end_s`` 显示同一张图同一套文字。"""

    start_s: float
    end_s: float
    transition: Transition
    record_pos: int  # records 下标; settle 段沿用最后一条
    settle: bool

    @property
    def key(self) -> tuple[int, int, bool]:
        return (self.transition.index, self.record_pos, self.settle)


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def load_transitions(src: Path, *, limit: int = 0) -> list[Transition]:
    """读 ``run_summary.json`` 及其引用的每个子目录, 逐条做完整性校验。

    校验项 (任一不满足就抛 ``ValueError``, 绝不"尽力而为"地猜):

    * 每个 label 都有子目录, 且子目录里有 ``frames.npy`` / ``frames_meta.json``;
    * ``n_measured == len(records)``;
    * ``len(records) <= frames.shape[0]`` (capacity 只是缓冲上限, 不是有效帧数);
    * ``records[k]['i'] == k``, 即数组下标与记录顺序一一对应;
    * ``t_rel_s`` 与 ``monotonic_ns`` 都严格单调递增。

    Args:
        src: 数据根目录。
        limit: 只取前 N 条 (0 = 全部); 仅用于冒烟测试。

    Returns:
        Transition 列表, 顺序与 ``run_summary.json`` 一致。
    """
    summary_path = src / "run_summary.json"
    if not summary_path.is_file():
        raise ValueError(f"缺少 run_summary.json: {summary_path}")
    summary = _load_json(summary_path)
    if not isinstance(summary, list) or not summary:
        raise ValueError(f"run_summary.json 不是非空数组: {summary_path}")

    labels = [str(entry["label"]) for entry in summary]
    dirs = sorted(p.name for p in src.iterdir() if p.is_dir())
    missing = sorted(set(labels) - set(dirs))
    if missing:
        raise ValueError(f"run_summary 引用了不存在的子目录: {missing}")
    extra = sorted(set(dirs) - set(labels))
    if extra:
        raise ValueError(f"子目录不在 run_summary 里 (顺序不可判定): {extra}")

    entries = summary[:limit] if limit > 0 else summary
    transitions: list[Transition] = []
    for pos, entry in enumerate(entries):
        label = str(entry["label"])
        directory = src / label
        frames_path = directory / "frames.npy"
        meta_path = directory / "frames_meta.json"
        for path in (frames_path, meta_path):
            if not path.is_file():
                raise ValueError(f"{label}: 缺少 {path.name}")

        meta = _load_json(meta_path)
        if not isinstance(meta, dict):
            raise ValueError(f"{label}: frames_meta.json 不是对象")
        records = meta.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError(f"{label}: records 为空")
        if int(meta["n_measured"]) != len(records):
            raise ValueError(
                f"{label}: n_measured={meta['n_measured']} != len(records)={len(records)}"
            )

        shape = tuple(int(v) for v in meta["shape"])  # 单帧空间尺寸 (H, W)
        arr_shape = tuple(int(v) for v in np.load(frames_path, mmap_mode="r").shape)
        if arr_shape[1:] != shape:
            raise ValueError(f"{label}: frames.npy 单帧形状 {arr_shape[1:]} != meta.shape {shape}")
        n_alloc = arr_shape[0]
        if len(records) > n_alloc:
            raise ValueError(f"{label}: len(records)={len(records)} > frames 容量 {n_alloc}")

        t_rel: list[float] = []
        frame_index: list[int] = []
        ns: list[float] = []
        for k, rec in enumerate(records):
            if int(rec["i"]) != k:
                raise ValueError(f"{label}: records[{k}]['i']={rec['i']} 与数组下标不符")
            frame_index.append(int(rec["i"]))
            t_rel.append(float(rec["t_rel_s"]))
            ns.append(float(rec["monotonic_ns"]))
        if any(b <= a for a, b in zip(t_rel, t_rel[1:])):
            raise ValueError(f"{label}: t_rel_s 不是严格递增: {t_rel}")
        if any(b <= a for a, b in zip(ns, ns[1:])):
            raise ValueError(f"{label}: monotonic_ns 不是严格递增")

        # 末帧必须真的有显示时间: 若末帧时间戳已经越过标称窗口 (实测有 18/30 条
        # 在 5.05-5.24 s 之间), 窗口取 "末帧 + 一个典型采样间隔", 否则零阶保持
        # 会给最后一帧分配 0 s, 那张实测图根本不会出现在视频里。
        gaps = sorted(b - a for a, b in zip(t_rel, t_rel[1:]))
        tail_hold = gaps[len(gaps) // 2] if gaps else 1.0 / OUT_FPS
        if t_rel[-1] > CONTENT_WINDOW_S:
            duration = t_rel[-1] + tail_hold
        else:
            duration = CONTENT_WINDOW_S
        achieved = entry.get("achieved_measure_fps")
        transitions.append(
            Transition(
                index=pos,
                label=label,
                directory=directory,
                t_rel_s=tuple(t_rel),
                frame_index=tuple(frame_index),
                exposure_ms=float(meta["exposure_ms"]),
                duration_s=duration,
                achieved_fps=float(achieved) if achieved is not None else None,
            )
        )
    return transitions


def build_segments(transitions: list[Transition], *, settle_s: float) -> list[Segment]:
    """把 transition 列表摊平成视频时间轴上的连续段 (零阶保持的切换点)。"""
    segments: list[Segment] = []
    cursor = 0.0
    for ti, trans in enumerate(transitions):
        t_rel = trans.t_rel_s
        for k, t_start in enumerate(t_rel):
            t_next = t_rel[k + 1] if k + 1 < len(t_rel) else trans.duration_s
            segments.append(
                Segment(
                    start_s=cursor + t_start,
                    end_s=cursor + t_next,
                    transition=trans,
                    record_pos=k,
                    settle=False,
                )
            )
        cursor += trans.duration_s
        if ti + 1 < len(transitions) and settle_s > 0.0:
            segments.append(
                Segment(
                    start_s=cursor,
                    end_s=cursor + settle_s,
                    transition=trans,
                    record_pos=len(t_rel) - 1,
                    settle=True,
                )
            )
            cursor += settle_s
    return segments


def _pick_font(candidates: tuple[Path, ...]) -> Path | None:
    for path in candidates:
        if path.is_file():
            return path
    return None


def _fmt_s(value: float) -> str:
    return f"{value:.2f}"


class FramePainter:
    """把一帧 uint16 CCD 图按固定映射转成 8-bit RGB 并叠加小字说明。

    灰度映射在整条视频里恒定 (一张 12-bit -> 8-bit LUT, 默认线性
    ``v * 255 / 4095``), 不随帧变化。
    """

    def __init__(self, *, size: tuple[int, int] = OUT_SIZE, gamma: float = 1.0) -> None:
        if gamma <= 0.0:
            raise ValueError(f"gamma 必须为正: {gamma}")
        self.size = size
        self.gamma = float(gamma)
        # 全片共用一张 12-bit -> 8-bit 查找表: 映射只由 gamma 决定, 与帧无关。
        levels = np.linspace(0.0, 1.0, int(FULL_SCALE) + 1, dtype=np.float64) ** self.gamma
        self.lut = np.clip(np.rint(levels * 255.0), 0.0, 255.0).astype(np.uint8)
        cjk_path = _pick_font(CJK_FONT_CANDIDATES)
        self.cjk = cjk_path is not None
        font_path = cjk_path or _pick_font(ASCII_FONT_CANDIDATES)
        if font_path is None:
            raise RuntimeError(f"找不到可用字体: 既没有 {CJK_FONT_CANDIDATES} 也没有 {ASCII_FONT_CANDIDATES}")
        self.font_path = font_path
        self.font_title = ImageFont.truetype(str(font_path), 20)
        self.font_body = ImageFont.truetype(str(font_path), 16)
        self.font_note = ImageFont.truetype(str(font_path), 13)

    # -- 文本 ---------------------------------------------------------------
    def _lines(self, seg: Segment, total: int, fps_mean: float, settle_s: float) -> tuple[str, str, str, str]:
        trans = seg.transition
        pos = seg.record_pos
        elapsed = trans.t_rel_s[pos]
        progress = min(1.0, elapsed / trans.duration_s)
        head = f"{trans.index + 1:02d}/{total:02d}"
        gamma_tag = "" if self.gamma == 1.0 else f" gamma {self.gamma:g}"
        if self.cjk:
            title = f"过渡 {head}  {trans.label}"
            if seg.settle:
                body = f"settle {_fmt_s(settle_s)} s，保持上一条末帧 (elapsed {_fmt_s(elapsed)} s)"
            else:
                body = (
                    f"elapsed {_fmt_s(elapsed)} s / {_fmt_s(trans.duration_s)} s   "
                    f"progress {progress * 100:3.0f}%   CCD 帧 {pos + 1}/{len(trans.t_rel_s)}"
                )
            note = (
                f"CCD 实测 ≈{fps_mean:.1f} fps · 视频 {OUT_FPS} fps CFR，按实测时戳零阶保持 (不插值)"
            )
            foot = (
                f"固定映射 0..{FULL_SCALE:.0f} → 0..255{gamma_tag} · "
                f"曝光 {trans.exposure_ms:g} ms · 12-bit"
            )
        else:
            title = f"Transition {head}  {trans.label}"
            if seg.settle:
                body = f"settle {_fmt_s(settle_s)} s, previous frame held (elapsed {_fmt_s(elapsed)} s)"
            else:
                body = (
                    f"elapsed {_fmt_s(elapsed)} s / {_fmt_s(trans.duration_s)} s   "
                    f"progress {progress * 100:3.0f}%   CCD frame {pos + 1}/{len(trans.t_rel_s)}"
                )
            note = (
                f"CCD measured ~{fps_mean:.1f} fps / video {OUT_FPS} fps CFR, "
                "zero-order hold on real timestamps (no interpolation)"
            )
            foot = (
                f"fixed map 0..{FULL_SCALE:.0f} -> 0..255{gamma_tag} / "
                f"exposure {trans.exposure_ms:g} ms / 12-bit"
            )
        return title, body, note, foot

    # -- 绘制 ---------------------------------------------------------------
    def paint(
        self, ccd: np.ndarray, seg: Segment, total: int, fps_mean: float, settle_s: float
    ) -> Image.Image:
        """渲染一帧。

        Args:
            ccd: uint16 CCD 帧, 任意分辨率; 只走固定灰度 LUT + BOX 下采样。
            seg: 该帧所属的时间段 (决定叠加文字)。
            total: transition 总数。
            fps_mean: 全片实测平均帧率, 仅用于文字说明。
            settle_s: transition 之间的保持时长, 仅用于文字说明。

        Returns:
            RGB 图像 (尺寸 = :attr:`size`)。
        """
        if ccd.dtype != np.uint16:
            raise TypeError(f"期望 uint16 CCD 帧, 实际 {ccd.dtype}")
        # 超出满量程的像素按满量程截断 (12-bit 有效值, 4095 已是白)。
        gray = self.lut[np.minimum(ccd, int(FULL_SCALE))]
        img = Image.fromarray(gray, mode="L")
        if img.size != self.size:
            img = img.resize(self.size, resample=Image.Resampling.BOX)
        base = img.convert("RGBA")

        title, body, note, foot = self._lines(seg, total, fps_mean, settle_s)
        pad = 9
        gap = 4
        line_h = [self.font_title.size + 2, self.font_body.size + 2, self.font_note.size + 2, self.font_note.size + 2]
        widths = [
            self._text_width(title, self.font_title),
            self._text_width(body, self.font_body),
            self._text_width(note, self.font_note),
            self._text_width(foot, self.font_note),
        ]
        bar_w = max(widths)
        panel_w = max(widths) + 2 * pad
        panel_h = sum(line_h) + gap * 3 + 2 * pad + 6
        x0, y0 = 14, self.size[1] - panel_h - 14
        x1, y1 = x0 + panel_w, y0 + panel_h

        layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        draw.rounded_rectangle([x0, y0, x1, y1], radius=6, fill=(0, 0, 0, 170))
        y = y0 + pad
        draw.text((x0 + pad, y), title, font=self.font_title, fill=(255, 255, 255, 240))
        y += line_h[0] + gap
        draw.text((x0 + pad, y), body, font=self.font_body, fill=(225, 235, 255, 235))
        y += line_h[1] + gap
        draw.text((x0 + pad, y), note, font=self.font_note, fill=(205, 205, 205, 225))
        y += line_h[2] + gap
        draw.text((x0 + pad, y), foot, font=self.font_note, fill=(175, 175, 175, 210))
        y += line_h[3] + 3

        elapsed = seg.transition.t_rel_s[seg.record_pos]
        progress = min(1.0, elapsed / seg.transition.duration_s)
        draw.rectangle(
            [x0 + pad, y, x0 + pad + bar_w, y + 3], fill=(90, 90, 90, 200)
        )
        draw.rectangle(
            [x0 + pad, y, x0 + pad + int(bar_w * progress), y + 3], fill=(120, 200, 255, 235)
        )
        return Image.alpha_composite(base, layer).convert("RGB")

    def _text_width(self, text: str, font: ImageFont.FreeTypeFont) -> int:
        left, _, right, _ = font.getbbox(text)
        return right - left


def _ffmpeg_command(ffmpeg: str, out: Path, *, fps: int, size: tuple[int, int], crf: int, preset: str) -> list[str]:
    """构造编码命令 (同时打印出来, 便于复现)。

    用 ``-f rawvideo -pix_fmt rgb24`` 从 stdin 吃帧, 输出 H.264 (libx264) +
    ``yuv420p``, ``+faststart`` 便于拖动; ``-fflags/-flags +bitexact`` 与
    ``-map_metadata -1`` 去掉时间戳元数据, 使重复执行得到同一份码流。
    """
    return [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{size[0]}x{size[1]}",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-threads", "4",
        "-map_metadata", "-1",
        "-fflags", "+bitexact",
        "-flags:v", "+bitexact",
        "-movflags", "+faststart",
        str(out),
    ]


def _fps_stats(transitions: list[Transition]) -> tuple[float, float, float, float, int]:
    """返回 (均值, 最小, 最大, 中位 dt, 实测记录总数), 均值取采集时记录的 achieved fps。"""
    fps = [t.achieved_fps for t in transitions if t.achieved_fps]
    dts: list[float] = []
    n_records = 0
    for trans in transitions:
        n_records += len(trans.t_rel_s)
        dts.extend(b - a for a, b in zip(trans.t_rel_s, trans.t_rel_s[1:]))
    if not fps:
        raise ValueError("run_summary 里没有任何 achieved_measure_fps")
    mean_fps = sum(fps) / len(fps)
    dts_sorted = sorted(dts)
    median_dt = dts_sorted[len(dts_sorted) // 2] if dts_sorted else 0.0
    return mean_fps, min(fps), max(fps), median_dt, n_records


def render(
    src: Path,
    out: Path,
    *,
    limit: int = 0,
    fps: int = OUT_FPS,
    settle_s: float = SETTLE_S,
    crf: int = 20,
    preset: str = "medium",
    gamma: float = 1.0,
) -> int:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("PATH 上找不到 ffmpeg.exe")

    transitions = load_transitions(src, limit=limit)
    segments = build_segments(transitions, settle_s=settle_s)
    total_s = segments[-1].end_s
    n_out = int(round(total_s * fps))
    mean_fps, min_fps, max_fps, median_dt, n_records = _fps_stats(transitions)

    logger.info(f"源目录: {src.resolve()}")
    logger.info(f"transition 数: {len(transitions)}  有效 CCD 帧: {n_records}")
    logger.info(
        f"实测帧率: 均值 {mean_fps:.3f} fps, 区间 [{min_fps:.3f}, {max_fps:.3f}] fps, "
        f"采样间隔中位数 {median_dt * 1000:.0f} ms"
    )
    extended = [t for t in transitions if t.duration_s > CONTENT_WINDOW_S + 1e-9]
    if extended:
        over = sum(t.duration_s - CONTENT_WINDOW_S for t in extended)
        logger.warning(
            f"{len(extended)} 条 transition 的末帧时间戳超过 {CONTENT_WINDOW_S:g} s "
            f"(窗口共延长 +{over:.2f} s): 这些条按 末帧 t_rel_s + 一个采样间隔 结束, "
            "以便末帧有真实显示时间, 不会被零阶保持吞掉"
        )
    content_s = sum(t.duration_s for t in transitions)
    logger.info(f"时间轴: 内容合计 {content_s:.2f} s ({len(transitions)} 条, 标称 {CONTENT_WINDOW_S:g} s/条) + "
                f"{len(transitions) - 1} x {settle_s:g} s settle = {total_s:.2f} s, "
                f"{n_out} 帧 @ {fps} fps")

    painter = FramePainter(gamma=gamma)
    logger.info(f"字体: {painter.font_path} (中文可用: {painter.cjk})")

    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = _ffmpeg_command(ffmpeg, out, fps=fps, size=painter.size, crf=crf, preset=preset)
    logger.info("ffmpeg: " + " ".join(cmd))

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None and proc.stderr is not None

    frames: dict[int, np.ndarray] = {}  # 当前 transition 的 mmap, 只保留一份
    last_key: tuple[int, int, bool] | None = None
    last_bytes = b""
    seg_i = 0
    rendered = 0
    peak = 0
    try:
        for j in range(n_out):
            t = j / fps
            while seg_i + 1 < len(segments) and t >= segments[seg_i].end_s:
                seg_i += 1
            seg = segments[seg_i]
            if seg.key != last_key:
                trans = seg.transition
                if trans.index not in frames:
                    frames.clear()
                    frames[trans.index] = np.load(trans.directory / "frames.npy", mmap_mode="r")
                arr = frames[trans.index][trans.frame_index[seg.record_pos]]
                peak = max(peak, int(arr.max()))
                img = painter.paint(arr, seg, len(transitions), mean_fps, settle_s)
                last_bytes = img.tobytes()
                last_key = seg.key
                if not seg.settle:
                    rendered += 1
                    if rendered % 25 == 0 or rendered == n_records:
                        logger.info(f"渲染 {rendered}/{n_records} 张实测帧 ({seg.transition.label})")
            proc.stdin.write(last_bytes)
        proc.stdin.close()
    except BrokenPipeError:
        logger.error("ffmpeg 提前退出, 编码失败")
    finally:
        ferr = proc.stderr.read().decode("utf-8", errors="replace").strip()
        code = proc.wait()
    if code != 0:
        logger.error(f"ffmpeg 退出码 {code}: {ferr}")
        return 1
    if ferr:
        logger.warning(f"ffmpeg stderr: {ferr}")

    size_mb = out.stat().st_size / (1024 * 1024)
    logger.success(f"写出: {out.resolve()}")

    print("=" * 68)
    print(f"transition 数          : {len(transitions)}")
    print(f"有效 CCD 帧数          : {n_records} (records 数 == n_measured)")
    print(f"实测 fps (采集记录)    : mean {mean_fps:.3f} / min {min_fps:.3f} / max {max_fps:.3f}")
    print(f"实测采样间隔中位数     : {median_dt * 1000:.0f} ms (约 {1.0 / median_dt:.2f} fps)")
    print(f"视频参数               : {painter.size[0]}x{painter.size[1]}, {fps} fps CFR, H.264 (libx264), "
          f"yuv420p, preset {preset}, crf {crf}, threads 4")
    print(f"时间轴                 : 内容合计 {content_s:.2f} s (标称 {CONTENT_WINDOW_S:g}s/条) + "
          f"{len(transitions) - 1}x{settle_s:g}s settle = {total_s:.2f} s ({n_out} 帧)")
    print(f"灰度映射               : 固定 0..{FULL_SCALE:.0f} -> 0..255, gamma {gamma:g} "
          f"(全片一张 LUT, 无逐帧归一化), 全片峰值 {peak}")
    print(f"文件                   : {out.resolve()}")
    print(f"文件大小               : {size_mb:.2f} MiB")
    print("=" * 68)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="把已落盘的 speed-decouple 实测 CCD 帧渲染成零阶保持的 MP4 (不碰硬件)。",
    )
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC, help="采集根目录 (含 run_summary.json)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="输出 MP4 路径")
    parser.add_argument("--limit", type=int, default=0, help="只渲前 N 条 transition (0 = 全部, 仅用于冒烟)")
    parser.add_argument("--fps", type=int, default=OUT_FPS, help=f"输出帧率 (默认 {OUT_FPS})")
    parser.add_argument("--settle", type=float, default=SETTLE_S, help=f"transition 间保持时长 (默认 {SETTLE_S})")
    parser.add_argument("--crf", type=int, default=20, help="libx264 CRF (默认 20)")
    parser.add_argument("--preset", default="medium", help="libx264 preset (默认 medium)")
    parser.add_argument(
        "--gamma", type=float, default=1.0,
        help="全片统一的灰度 gamma (默认 1.0 = 线性 0..4095 -> 0..255; 仍是一张固定 LUT, 不是逐帧归一化)",
    )
    args = parser.parse_args(argv)

    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<level>{level: <7}</level> {message}")
    try:
        return render(
            args.src, args.out, limit=args.limit, fps=args.fps,
            settle_s=args.settle, crf=args.crf, preset=args.preset, gamma=args.gamma,
        )
    except (ValueError, RuntimeError, TypeError) as exc:
        logger.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
