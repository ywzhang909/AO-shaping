"""Micro-DM 差分分析 UI (Streamlit)

包装 ``scripts/md_img_pipeline.py`` 的完整 diff→overlay→GIF→combined 流水线:

1. 运行模式: 选择数据集 → 调整流水线参数 → 后台线程运行完整流水线,
   实时显示进度 (每 IP 的 diff/overlay/gif 状态 + 全局 overlay + combined GIF),
   可随时停止。
2. 浏览模式: 无需重新运行, 直接查看已有处理结果 (逐 IP 的 diff 图像、
   质心表、overlay、动画 GIF, 以及全局 overlay 与 combined GIF)。

使用方式:
    streamlit run src/ao_shaping/gui/dm/md_img_analysis_ui.py
"""

from __future__ import annotations

import re
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from loguru import logger
from PIL import Image

from ao_shaping.utils.file import ROOT_DIR as PROJECT_ROOT

# =============================================================================
# Constants
# =============================================================================

DATA_ROOT = PROJECT_ROOT / "data" / "md_test"

# 流水线默认参数 (与 scripts/md_img_pipeline.py 保持一致)
DEFAULT_THRESHOLD = 15.0
DEFAULT_CMAP = "jet"
DEFAULT_SCALE = 0.25
DEFAULT_FPS = 8

# 运行中轮询刷新间隔 (s)
POLL_INTERVAL = 0.5

# 从文件名解析通道号: 192.168.0.101-000.png -> 0
_CHANNEL_RE = re.compile(r"-(\d{3})")

# =============================================================================
# 后台线程进度上报 (模块级, 加锁; 工作线程绝不触碰 st.session_state)
# =============================================================================

_PROGRESS: dict[str, dict] = {}
_PROGRESS_LOCK = threading.Lock()


def _update_progress(run_id: str, **fields: object) -> None:
    """更新指定 run_id 的进度条目 (线程安全)。"""
    with _PROGRESS_LOCK:
        entry = _PROGRESS.setdefault(run_id, {})
        entry.update(fields)


def _get_progress(run_id: str) -> dict:
    """读取指定 run_id 的进度快照 (线程安全)。"""
    with _PROGRESS_LOCK:
        return dict(_PROGRESS.get(run_id, {}))


# =============================================================================
# Session State Initialization
# =============================================================================

def _initialize_state() -> None:
    """初始化所有 session_state 变量。"""
    st.session_state.setdefault("mdimg_dataset", "")
    st.session_state.setdefault("mdimg_custom_input", "")
    st.session_state.setdefault("mdimg_threshold", DEFAULT_THRESHOLD)
    st.session_state.setdefault("mdimg_cmap", DEFAULT_CMAP)
    st.session_state.setdefault("mdimg_vmax", 0.0)  # 0 = 自动 (per-image max)
    st.session_state.setdefault("mdimg_notch", True)
    st.session_state.setdefault("mdimg_scale", DEFAULT_SCALE)
    st.session_state.setdefault("mdimg_fps", DEFAULT_FPS)
    st.session_state.setdefault("mdimg_skip_diff", False)
    st.session_state.setdefault("mdimg_run_id", None)
    st.session_state.setdefault("mdimg_stop_event", None)
    st.session_state.setdefault("mdimg_feedback", "")
    st.session_state.setdefault("mdimg_feedback_type", "")
    st.session_state.setdefault("mdimg_browse_ip", "")
    st.session_state.setdefault("mdimg_browse_ch", 0)
    st.session_state.setdefault("mdimg_load_download", False)


# =============================================================================
# Feedback Helpers
# =============================================================================

def set_feedback(message: str, msg_type: str = "info") -> None:
    """设置反馈信息。"""
    st.session_state.mdimg_feedback = message
    st.session_state.mdimg_feedback_type = msg_type


def show_and_clear_feedback() -> None:
    """显示反馈并清除。"""
    message = st.session_state.get("mdimg_feedback", "")
    msg_type = st.session_state.get("mdimg_feedback_type", "")
    if not message:
        return
    if msg_type == "success":
        st.success(message)
    elif msg_type == "error":
        st.error(message)
    elif msg_type == "warning":
        st.warning(message)
    else:
        st.info(message)
    st.session_state.mdimg_feedback = ""
    st.session_state.mdimg_feedback_type = ""


# =============================================================================
# 数据集 / 路径 Helpers
# =============================================================================

def _list_datasets() -> list[str]:
    """扫描 data/md_test/ 下的数据集目录 (md_img-*)。

    排除 *_processed / *_gif 输出目录, 避免把处理结果当作输入数据集。
    """
    if not DATA_ROOT.is_dir():
        return []
    names = [
        p.name
        for p in DATA_ROOT.iterdir()
        if p.is_dir()
        and p.name.startswith("md_img-")
        and not p.name.endswith("_processed")
        and not p.name.endswith("_gif")
    ]
    # 最新修改的排在最前
    names.sort(key=lambda n: (DATA_ROOT / n).stat().st_mtime, reverse=True)
    return names


def _resolve_input_dir() -> Path | None:
    """解析当前输入目录: 自定义路径优先, 否则使用所选数据集。"""
    custom = st.session_state.mdimg_custom_input.strip()
    if custom:
        p = Path(custom)
        return p if p.is_dir() else None
    ds = st.session_state.mdimg_dataset
    if not ds:
        return None
    p = DATA_ROOT / ds
    return p if p.is_dir() else None


def _derive_output_dir(input_dir: Path) -> Path:
    """输出目录: <input>_processed (与 pipeline 默认一致)。"""
    return input_dir.parent / f"{input_dir.name}_processed"


def _channel_of_name(name: str) -> int | None:
    """从文件名解析通道号, 解析失败返回 None。"""
    m = _CHANNEL_RE.search(name)
    return int(m.group(1)) if m else None


# =============================================================================
# 后台流水线线程
# =============================================================================

def run_pipeline(snapshot: dict, stop_event: threading.Event) -> None:
    """后台线程: 运行完整 diff→overlay→GIF→combined 流水线。

    绝不触碰 st.session_state; 进度通过模块级 _PROGRESS 字典 (加锁) 上报。
    参数以快照 dict 传入, 与 UI 线程解耦。
    """
    run_id = snapshot["run_id"]
    input_dir = Path(snapshot["input_dir"])
    output_dir = Path(snapshot["output_dir"])
    threshold = float(snapshot["threshold"])
    cmap = str(snapshot["cmap"])
    vmax = snapshot["vmax"]  # float | None
    notch = bool(snapshot["notch"])
    scale = float(snapshot["scale"])
    fps = int(snapshot["fps"])
    skip_diff = bool(snapshot["skip_diff"])

    try:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        import md_img_pipeline as pipe  # 延迟导入, 避免拖慢 UI 启动
    except Exception as e:
        logger.exception("无法导入 md_img_pipeline: {}", e)
        _update_progress(run_id, status="error", message=f"无法导入 md_img_pipeline: {e}")
        return

    try:
        if not input_dir.is_dir():
            raise FileNotFoundError(f"输入目录不存在: {input_dir}")

        diff_root = output_dir / "diff"
        overlay_dir = output_dir / "overlay"
        gif_dir = output_dir / "gif"

        ip_dirs = sorted(p for p in input_dir.iterdir() if p.is_dir())
        n_ips = len(ip_dirs)
        if n_ips == 0:
            raise FileNotFoundError(f"输入目录下没有控制器 IP 子目录: {input_dir}")

        # 进度单位: 每 IP 计 1 单位 + 全局 overlay + combined GIF
        total_units = n_ips + 2
        units_done = 0
        _update_progress(
            run_id, status="running", n_ips=n_ips, total_units=total_units,
            units_done=0, message=f"发现 {n_ips} 个控制器 IP",
        )

        # 共享参考: 第一个控制器的 channel-000 (与 pipeline 默认一致)
        shared_ref: np.ndarray | None = None
        if not skip_diff and ip_dirs:
            ref_path = ip_dirs[0] / f"{ip_dirs[0].name}-000.png"
            if ref_path.is_file():
                shared_ref = pipe.load_gray(ref_path)
                if notch:
                    shared_ref = pipe.notch_fft(shared_ref)
                logger.info("共享参考: {}", ref_path.name)
            else:
                logger.warning("共享参考不存在: {}", ref_path)

        for ip_idx, ip_dir in enumerate(ip_dirs):
            if stop_event.is_set():
                _update_progress(run_id, status="stopped", message="已停止")
                return
            ip_name = ip_dir.name
            base = f"[{ip_idx + 1}/{n_ips}] {ip_name}"

            # Step 1: diff 计算
            n_diff = 0
            n_valid = 0
            if not skip_diff:
                try:
                    if shared_ref is None:
                        raise FileNotFoundError("无可用共享参考")
                    results = pipe.process_ip_diff(
                        ip_dir, diff_root, shared_ref, threshold, cmap, vmax, notch
                    )
                    n_diff = len(results)
                    n_valid = sum(1 for _, _, c in results if c is not None)
                    _update_progress(run_id, message=f"{base}: diff {n_diff} 张 ({n_valid} 有效质心)")
                    logger.info("{}: diff {} 张", base, n_diff)
                except Exception as e:
                    logger.exception("{}: diff 失败: {}", base, e)
                    _update_progress(run_id, message=f"{base}: diff 失败: {e}")
                    continue

            # Step 2: overlay
            ip_diff_dir = diff_root / ip_name
            coverage = 0.0
            if ip_diff_dir.is_dir():
                try:
                    overlay_path = overlay_dir / f"{ip_name}_overlay.png"
                    coverage = pipe.compute_ip_overlay(ip_diff_dir, overlay_path)
                    _update_progress(run_id, message=f"{base}: overlay {coverage:.1f}%")
                except Exception as e:
                    logger.exception("{}: overlay 失败: {}", base, e)
                    _update_progress(run_id, message=f"{base}: overlay 失败: {e}")

            # Step 3: GIF
            gif_mb = 0.0
            if ip_diff_dir.is_dir():
                try:
                    gif_path = gif_dir / f"{ip_name}.gif"
                    pipe.render_ip_gif(ip_diff_dir, gif_path, scale, fps)
                    gif_mb = gif_path.stat().st_size / (1024 * 1024)
                except Exception as e:
                    logger.exception("{}: gif 失败: {}", base, e)

            units_done += 1
            _update_progress(
                run_id, units_done=units_done,
                message=f"{base}: diff {n_diff} 张 / overlay {coverage:.1f}% / gif {gif_mb:.1f} MB",
            )

        if stop_event.is_set():
            _update_progress(run_id, status="stopped", message="已停止")
            return

        # Step 4: 全局 overlay (跨所有 IP 的逐像素最大值)
        all_diff_files = list(diff_root.rglob("*.png"))
        if all_diff_files:
            first = np.asarray(Image.open(all_diff_files[0]), dtype=np.float64)
            h, w = first.shape[:2]
            is_rgb = first.ndim == 3 and first.shape[2] == 3
            overlay = np.zeros((h, w, 3) if is_rgb else (h, w), dtype=np.float64)
            for path in all_diff_files:
                img = np.asarray(Image.open(path), dtype=np.float64)
                if img.shape[:2] == (h, w):
                    overlay = np.maximum(overlay, img)
            global_overlay_path = output_dir / "global_overlay.png"
            Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(global_overlay_path)
            gray = np.mean(overlay, axis=2) if is_rgb else overlay
            coverage = float((gray > 0).sum()) / (h * w) * 100
            units_done += 1
            _update_progress(run_id, units_done=units_done, message=f"global overlay {coverage:.1f}%")
            logger.info("global overlay: {:.1f}%", coverage)
        else:
            units_done += 1
            _update_progress(run_id, units_done=units_done, message="global overlay: 无 diff 图像")

        # Step 5: combined GIF
        if gif_dir.is_dir() and list(gif_dir.glob("*.gif")):
            combined_path = output_dir / "combined.gif"
            try:
                pipe.combine_gifs(gif_dir, [input_dir], combined_path, fps)
                combined_mb = combined_path.stat().st_size / (1024 * 1024)
                units_done += 1
                _update_progress(run_id, units_done=units_done, message=f"combined.gif {combined_mb:.1f} MB")
            except Exception as e:
                logger.exception("combined gif 失败: {}", e)
                units_done += 1
                _update_progress(run_id, units_done=units_done, message=f"combined gif 失败: {e}")

        _update_progress(run_id, status="complete", message=f"流水线完成 → {output_dir}")
        logger.info("流水线完成 → {}", output_dir)
    except Exception as e:
        logger.exception("流水线失败: {}", e)
        _update_progress(run_id, status="error", message=f"流水线失败: {e}")


# =============================================================================
# 运行控制
# =============================================================================

def _is_running() -> bool:
    """当前是否有流水线在运行。"""
    run_id = st.session_state.mdimg_run_id
    if not run_id:
        return False
    return _get_progress(run_id).get("status") in ("starting", "running")


def _start_pipeline() -> None:
    """启动后台流水线线程 (参数快照 + stop_event, 线程内不读 session_state)。"""
    input_dir = _resolve_input_dir()
    if input_dir is None:
        set_feedback("输入目录无效, 请检查数据集选择或自定义路径", "error")
        return
    if _is_running():
        set_feedback("流水线正在运行中", "warning")
        return

    run_id = uuid.uuid4().hex
    stop_event = threading.Event()
    output_dir = _derive_output_dir(input_dir)
    snapshot = {
        "run_id": run_id,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "threshold": float(st.session_state.mdimg_threshold),
        "cmap": str(st.session_state.mdimg_cmap),
        "vmax": None if st.session_state.mdimg_vmax <= 0 else float(st.session_state.mdimg_vmax),
        "notch": bool(st.session_state.mdimg_notch),
        "scale": float(st.session_state.mdimg_scale),
        "fps": int(st.session_state.mdimg_fps),
        "skip_diff": bool(st.session_state.mdimg_skip_diff),
    }
    _update_progress(
        run_id, status="starting", message="启动中...",
        units_done=0, total_units=0, output_dir=str(output_dir),
    )
    st.session_state.mdimg_run_id = run_id
    st.session_state.mdimg_stop_event = stop_event
    threading.Thread(target=run_pipeline, args=(snapshot, stop_event), daemon=True).start()
    set_feedback(f"流水线已启动 → {output_dir}", "success")
    logger.info("流水线启动: {} → {}", input_dir, output_dir)


def _stop_pipeline() -> None:
    """请求停止后台流水线 (当前 IP 处理完后生效)。"""
    stop_event = st.session_state.mdimg_stop_event
    if stop_event is not None:
        stop_event.set()
    set_feedback("已请求停止, 将在当前 IP 处理完后生效", "info")


# =============================================================================
# Sidebar
# =============================================================================

def render_sidebar() -> None:
    """渲染侧边栏: 数据选择 + 流水线参数 + 运行控制。"""
    with st.sidebar:
        st.header("⚙️ 流水线设置")

        with st.container(border=True):
            st.markdown("##### 数据选择")
            options = _list_datasets()
            if not options:
                st.warning("未找到数据集 (data/md_test/md_img-*)")
            else:
                if st.session_state.mdimg_dataset not in options:
                    st.session_state.mdimg_dataset = options[0]
                st.selectbox("数据集", options=options, key="mdimg_dataset")
            st.text_input(
                "自定义输入目录 (覆盖数据集)",
                value=st.session_state.mdimg_custom_input,
                placeholder="data/md_test/md_img-80v",
                key="mdimg_custom_input",
            )
            input_dir = _resolve_input_dir()
            if input_dir is None:
                st.error("输入目录无效")
            else:
                st.caption(f"输入: {input_dir}")
                st.caption(f"输出: {_derive_output_dir(input_dir)}")

        with st.container(border=True):
            st.markdown("##### 流水线参数")
            st.number_input(
                "阈值 (threshold)", min_value=0.0, max_value=50.0,
                value=st.session_state.mdimg_threshold, step=1.0, format="%.1f",
                key="mdimg_threshold",
            )
            cmap_raw = st.segmented_control(
                "颜色映射", options=["jet", "gray"],
                default=st.session_state.mdimg_cmap, selection_mode="single",
                key="mdimg_cmap_sel",
            )
            if isinstance(cmap_raw, list):
                cmap_raw = cmap_raw[0] if cmap_raw else DEFAULT_CMAP
            st.session_state.mdimg_cmap = cmap_raw if cmap_raw else DEFAULT_CMAP
            st.number_input(
                "vmax (0 = 自动)", min_value=0.0, max_value=255.0,
                value=st.session_state.mdimg_vmax, step=1.0, format="%.1f",
                key="mdimg_vmax",
            )
            st.checkbox("FFT 去条纹 (notch)", value=st.session_state.mdimg_notch, key="mdimg_notch")
            st.number_input(
                "GIF 缩放 (scale)", min_value=0.1, max_value=1.0,
                value=st.session_state.mdimg_scale, step=0.05, format="%.2f",
                key="mdimg_scale",
            )
            st.number_input(
                "GIF 帧率 (fps)", min_value=1, max_value=30,
                value=st.session_state.mdimg_fps, step=1,
                key="mdimg_fps",
            )
            st.checkbox("跳过 diff (复用已有)", value=st.session_state.mdimg_skip_diff, key="mdimg_skip_diff")

        with st.container(border=True):
            st.markdown("##### 运行控制")
            col_run, col_stop = st.columns(2)
            with col_run:
                if st.button("▶ 运行流水线", type="primary", use_container_width=True, key="mdimg_run_btn"):
                    _start_pipeline()
                    st.rerun()
            with col_stop:
                if st.button("⏹ 停止", use_container_width=True, key="mdimg_stop_btn", disabled=not _is_running()):
                    _stop_pipeline()
                    st.rerun()


# =============================================================================
# Tab 1: 运行流水线 (实时进度)
# =============================================================================

def render_run_tab() -> None:
    """渲染运行页签: 实时进度 + 轮询刷新。"""
    show_and_clear_feedback()

    run_id = st.session_state.mdimg_run_id
    if not run_id:
        st.info("👈 在左侧选择数据集并点击「▶ 运行流水线」开始分析。")
        return

    prog = _get_progress(run_id)
    status = prog.get("status", "idle")
    message = prog.get("message", "")
    units_done = int(prog.get("units_done", 0))
    total_units = int(prog.get("total_units", 0))

    status_icon = {
        "starting": "🟡", "running": "🟢", "complete": "✅",
        "error": "❌", "stopped": "⏹",
    }.get(status, "⚪")
    st.markdown(f"##### 状态: {status_icon} {status}")

    if total_units > 0:
        percent = min(units_done / total_units, 1.0)
        st.progress(percent, text=message or "处理中...")
    else:
        st.progress(0.0, text=message or "启动中...")

    if status in ("complete", "error", "stopped"):
        if status == "complete":
            st.success(message)
        elif status == "error":
            st.error(message)
        else:
            st.warning(message)
        output_dir = prog.get("output_dir", "")
        if output_dir:
            st.caption(f"输出目录: {output_dir}")
    elif status in ("starting", "running"):
        time.sleep(POLL_INTERVAL)
        st.rerun()


# =============================================================================
# Tab 2: 浏览结果 (逐 IP)
# =============================================================================

def _centroid_for_channel(ip_diff_dir: Path, filename: str) -> str:
    """从 centroids.csv 查询指定文件的质心; 无信号返回 '无信号'。"""
    csv_path = ip_diff_dir / "centroids.csv"
    if csv_path.is_file():
        try:
            df = pd.read_csv(csv_path)
            row = df[df["filename"] == filename]
            if not row.empty:
                cx, cy = row.iloc[0]["cx"], row.iloc[0]["cy"]
                if pd.isna(cx) or pd.isna(cy):
                    return "无信号"
                return f"({cx:.1f}, {cy:.1f})"
        except Exception as e:
            logger.warning("读取质心失败: {}", e)
    return "无信号"


def render_browse_tab() -> None:
    """渲染浏览页签: 逐 IP 查看已有处理结果。"""
    show_and_clear_feedback()

    input_dir = _resolve_input_dir()
    if input_dir is None:
        st.info("请先在侧边栏选择数据集。")
        return
    output_dir = _derive_output_dir(input_dir)
    if not output_dir.is_dir():
        st.warning(f"尚无处理结果: {output_dir}\n\n请先在「运行流水线」页签运行分析。")
        return

    diff_root = output_dir / "diff"
    gif_dir = output_dir / "gif"
    overlay_dir = output_dir / "overlay"

    ip_dirs = sorted(p for p in diff_root.iterdir() if p.is_dir()) if diff_root.is_dir() else []
    if not ip_dirs:
        st.warning("diff 目录为空, 尚无处理结果。")
        return
    ip_names = [p.name for p in ip_dirs]

    col_sel, col_refresh = st.columns([4, 1])
    with col_sel:
        if st.session_state.mdimg_browse_ip not in ip_names:
            st.session_state.mdimg_browse_ip = ip_names[0]
        selected_ip = st.selectbox("控制器 IP", options=ip_names, key="mdimg_browse_ip")
    with col_refresh:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 刷新", use_container_width=True, key="mdimg_refresh_btn"):
            st.rerun()

    ip_diff_dir = diff_root / selected_ip
    png_files = sorted(ip_diff_dir.glob("*.png"))
    if not png_files:
        st.warning(f"{selected_ip} 无 diff 图像。")
        return

    n_channels = len(png_files)
    if st.session_state.mdimg_browse_ch >= n_channels:
        st.session_state.mdimg_browse_ch = 0
    ch = st.slider("通道", 0, n_channels - 1, st.session_state.mdimg_browse_ch, key="mdimg_browse_ch")

    # 按通道号定位文件 (文件名保留原始命名, 如 192.168.0.101-000.png)
    target = next((p for p in png_files if _channel_of_name(p.name) == ch), png_files[ch])

    col_img, col_gif = st.columns(2)
    with col_img:
        st.markdown(f"##### 通道 {ch} diff 图像")
        st.image(str(target), width="stretch")
        st.caption(f"质心: {_centroid_for_channel(ip_diff_dir, target.name)}")
    with col_gif:
        gif_path = gif_dir / f"{selected_ip}.gif"
        if gif_path.is_file():
            st.markdown(f"##### {selected_ip} 动画 GIF")
            st.image(str(gif_path), width="stretch")
        else:
            st.info("无 GIF")

    overlay_path = overlay_dir / f"{selected_ip}_overlay.png"
    if overlay_path.is_file():
        st.markdown(f"##### {selected_ip} 叠加图 (overlay)")
        st.image(str(overlay_path), width="stretch")

    csv_path = ip_diff_dir / "centroids.csv"
    if csv_path.is_file():
        st.markdown("##### 质心表 (centroids.csv)")
        try:
            df = pd.read_csv(csv_path)
            st.dataframe(df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.caption(f"读取 centroids.csv 失败: {e}")


# =============================================================================
# Tab 3: 汇总 (全局 overlay + combined GIF)
# =============================================================================

def _coverage_of(path: Path) -> float:
    """计算图像非零像素覆盖率 (%)。"""
    try:
        img = np.asarray(Image.open(path), dtype=np.float64)
        gray = np.mean(img, axis=2) if img.ndim == 3 else img
        return float((gray > 0).sum()) / gray.size * 100
    except Exception as e:
        logger.warning("计算覆盖率失败: {}", e)
        return 0.0


def render_merged_tab() -> None:
    """渲染汇总页签: 全局 overlay + combined GIF + 下载。"""
    show_and_clear_feedback()

    input_dir = _resolve_input_dir()
    if input_dir is None:
        st.info("请先在侧边栏选择数据集。")
        return
    output_dir = _derive_output_dir(input_dir)
    if not output_dir.is_dir():
        st.warning(f"尚无处理结果: {output_dir}\n\n请先在「运行流水线」页签运行分析。")
        return

    combined_path = output_dir / "combined.gif"
    global_path = output_dir / "global_overlay.png"

    col_global, col_combined = st.columns(2)
    with col_global:
        st.markdown("##### 全局叠加图 (global_overlay.png)")
        if global_path.is_file():
            st.image(str(global_path), width="stretch")
            st.caption(f"覆盖率: {_coverage_of(global_path):.1f}%")
        else:
            st.info("无 global_overlay.png")
    with col_combined:
        st.markdown("##### 合并 GIF (combined.gif)")
        if combined_path.is_file():
            st.image(str(combined_path), width="stretch")
            size_mb = combined_path.stat().st_size / (1024 * 1024)
            st.caption(f"大小: {size_mb:.1f} MB")
        else:
            st.info("无 combined.gif")

    if combined_path.is_file():
        st.divider()
        st.markdown("##### 下载")
        load_download = st.checkbox(
            "加载下载数据 (combined.gif 较大, 加载后每次刷新都会重新读取)",
            value=st.session_state.mdimg_load_download,
            key="mdimg_load_download",
        )
        if load_download:
            try:
                with open(combined_path, "rb") as f:
                    data = f.read()
                st.download_button(
                    "💾 下载 combined.gif",
                    data=data,
                    file_name=combined_path.name,
                    mime="image/gif",
                    key="mdimg_download_combined",
                )
            except Exception as e:
                st.caption(f"读取 combined.gif 失败: {e}")


# =============================================================================
# Main App
# =============================================================================

def main() -> None:
    """Streamlit 应用主入口。"""
    st.set_page_config(
        page_title="Micro-DM 差分分析",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🔬 Micro-DM 差分分析")
    st.caption("Micro-DM 逐通道响应分析 | diff → overlay → GIF → combined 流水线 · 结果浏览")

    _initialize_state()
    render_sidebar()

    tab_run, tab_browse, tab_merged = st.tabs(["运行流水线", "浏览结果", "汇总"])

    with tab_run:
        render_run_tab()

    with tab_browse:
        render_browse_tab()

    with tab_merged:
        render_merged_tab()


if __name__ == "__main__":
    main()