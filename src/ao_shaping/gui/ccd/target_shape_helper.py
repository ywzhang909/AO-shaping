"""
Target Shape Helper (Streamlit page)

Reads the current Daheng camera image and displays a target spot
(circle / square / rectangle) with matching intensity and center.

Features:
1. Daheng camera connection / disconnection from the sidebar
2. Live camera preview (auto-refresh fragment, 0.2 s cadence)
3. Simulation mode: when no camera is connected a random Gaussian-spot
   CCD frame is generated instead of a real capture
4. Target shape generation (circle / square / rectangle) at the beam center
5. Center calculation methods (checkboxes, multi-select): centroid,
   thresholded centroid, argmax, custom
6. Plotly overlay visualization (camera image + target contour)
"""

from __future__ import annotations

import os

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from loguru import logger
from plotly.subplots import make_subplots

from ao_shaping.utils.targets import create_target_shape, generate_target_mask
from ao_shaping.drivers.ccd.daheng import DahengCamManager
from ao_shaping.utils.spots_calc import center_of_brightness, centroid

# ── Constants ─────────────────────────────────────────────────────────────────
_DEFAULT_CAM_ID = int(os.environ.get("FAR_CAM_ID", "0"))
_REFRESH_INTERVAL = 0.2  # seconds – live preview fragment cadence

# Simulated CCD frame size (height, width) used when no camera is connected.
_SIM_FRAME_SHAPE = (1024, 1224)

_CENTER_METHODS = [
    ("centroid", "质心 (centroid)"),
    ("centroid_thresh", "亮度重心 (centroid_thresh)"),
    ("argmax", "峰值位置 (argmax)"),
    ("custom", "自定义"),
]

# Checkbox key prefix for center methods
_CM_KEY = "ts_cm_"


def _init_session_state() -> None:
    """Initialize the ``ts_*`` session state keys."""
    st.session_state.setdefault("ts_camera", None)
    st.session_state.setdefault("ts_camera_connected", False)
    st.session_state.setdefault("ts_cam_id", _DEFAULT_CAM_ID)
    st.session_state.setdefault("ts_exposure_ms", 50.0)
    st.session_state.setdefault("ts_window_size", 400)
    st.session_state.setdefault("ts_frame", None)
    st.session_state.setdefault("ts_centers", {})
    st.session_state.setdefault("ts_target", None)
    # Simulated-spot parameters (None → generate a fresh random spot)
    st.session_state.setdefault("ts_sim_params", None)
    # Center-method checkboxes (default: only centroid on)
    for key, _ in _CENTER_METHODS:
        st.session_state.setdefault(f"{_CM_KEY}{key}", key == "centroid")


def _generate_sim_frame() -> np.ndarray:
    """Generate a simulated CCD frame: a random Gaussian spot + noise.

    The spot geometry (center/sigma/peak/background) is stored in
    ``ts_sim_params`` and reused across fragment refreshes, so the live
    preview flickers (fresh noise every call) while the spot stays put.
    Set ``st.session_state.ts_sim_params = None`` to roll a new spot.

    Returns:
        uint16 array of shape :data:`_SIM_FRAME_SHAPE`.
    """
    h, w = _SIM_FRAME_SHAPE
    params = st.session_state.get("ts_sim_params")
    if params is None:
        rng = np.random.default_rng()
        params = {
            "cx": float(rng.uniform(0.3, 0.7) * w),
            "cy": float(rng.uniform(0.3, 0.7) * h),
            "sigma": float(rng.uniform(10.0, 40.0)),
            "peak": float(rng.uniform(120.0, 220.0)),
            "background": float(rng.uniform(2.0, 8.0)),
        }
        st.session_state.ts_sim_params = params

    cx = params["cx"]
    cy = params["cy"]
    sigma = params["sigma"]
    peak = params["peak"]
    background = params["background"]

    yy, xx = np.ogrid[:h, :w]
    gauss = peak * np.exp(
        -0.5 * ((xx - cx) ** 2 + (yy - cy) ** 2) / sigma**2
    )
    noise = np.random.default_rng().normal(0.0, 1.5, size=(h, w))
    frame = gauss + background + noise
    return np.clip(frame, 0, 4095).astype(np.uint16)


def _compute_center(
    frame: np.ndarray,
    method_key: str,
    custom_center: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Compute the beam center using the selected method."""
    if method_key == "centroid":
        return centroid(frame, moment=1, threshold=0.0, return_float=True)
    if method_key == "centroid_thresh":
        return centroid(frame, moment=1, threshold=0.1, return_float=True)
    if method_key == "argmax":
        return center_of_brightness(frame)
    if method_key == "custom":
        if custom_center is None:
            raise ValueError("自定义中心需要提供坐标")
        return custom_center
    raise ValueError(f"未知的中心计算方法: {method_key}")


# ── Plotly rendering ──────────────────────────────────────────────────────────
_METHOD_COLORS = {
    "centroid": "#ff4b4b",
    "centroid_thresh": "#ffa500",
    "argmax": "#00cc96",
    "custom": "#ab63fa",
}


def _display_overlay_plotly(
    frame: np.ndarray,
    target: np.ndarray,
    centers: dict[str, tuple[float, float]],
) -> None:
    """Render camera image, target shape and overlay with Plotly."""
    peak = float(np.max(frame)) if frame.size else 1.0
    target_scaled = target * peak

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Camera Image", "Target Shape", "Overlay"),
        horizontal_spacing=0.02,
    )

    # Panel 1 — camera
    fig.add_trace(
        go.Heatmap(
            z=frame,
            colorscale="gray",
            showscale=False,
            hovertemplate="x=%{x}<br>y=%{y}<br>I=%{z:.0f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Panel 2 — target
    fig.add_trace(
        go.Heatmap(
            z=target_scaled,
            colorscale="Hot",
            showscale=False,
            hovertemplate="x=%{x}<br>y=%{y}<br>I=%{z:.0f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # Panel 3 — overlay (camera + contour)
    fig.add_trace(
        go.Heatmap(
            z=frame,
            colorscale="gray",
            showscale=False,
            opacity=0.6,
            hovertemplate="x=%{x}<br>y=%{y}<br>I=%{z:.0f}<extra></extra>",
        ),
        row=1,
        col=3,
    )

    # Draw target contour on overlay panel
    _add_contour_trace(fig, target, row=1, col=3)

    # Draw center markers on overlay panel
    for method_key, (cx, cy) in centers.items():
        color = _METHOD_COLORS.get(method_key, "#ffffff")
        label = dict(_CENTER_METHODS).get(method_key, method_key)
        fig.add_trace(
            go.Scatter(
                x=[cx],
                y=[cy],
                mode="markers",
                marker=dict(size=8, color=color, symbol="x", line=dict(width=2, color="white")),
                name=label,
                showlegend=True,
                hovertemplate=f"{label}<br>x={cx:.1f} y={cy:.1f}<extra></extra>",
            ),
            row=1,
            col=3,
        )

    fig.update_layout(
        height=450,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.18,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
    )
    for i in range(1, 4):
        fig.update_xaxes(showticklabels=False, row=1, col=i)
        fig.update_yaxes(showticklabels=False, scaleanchor=f"x{i}" if i > 1 else None, row=1, col=i)

    st.plotly_chart(fig, use_container_width=True)


def _add_contour_trace(
    fig: go.Figure, target: np.ndarray, *, row: int, col: int
) -> None:
    """Add a contour line at the 0.5 level of a binary mask to *fig*."""
    # Use a thresholded version for contour extraction
    binary = (target > 0.5).astype(np.uint8)
    # Find contour points via boundary tracing (simple: row-by-row scan)
    ys, xs = np.where(np.diff(binary, axis=0) != 0)  # horizontal edges
    ys2, xs2 = np.where(np.diff(binary, axis=1) != 0)  # vertical edges
    # Combine into scatter points
    all_x = np.concatenate([xs, xs2 + 0.5]) if xs2.size else xs.astype(float)
    all_y = np.concatenate([ys + 0.5, ys2]) if ys2.size else ys.astype(float)
    if all_x.size == 0:
        return
    # Sort by angle from centroid for a closed path
    cx, cy = float(np.mean(all_x)), float(np.mean(all_y))
    angles = np.arctan2(all_y - cy, all_x - cx)
    order = np.argsort(angles)
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([all_x[order], [all_x[order[0]]]]),
            y=np.concatenate([all_y[order], [all_y[order[0]]]]),
            mode="lines",
            line=dict(color="#ff4b4b", width=2),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row,
        col=col,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
def _render_sidebar() -> None:
    """Render the sidebar with camera controls."""
    with st.sidebar:
        st.header("Target Shape Helper")
        st.caption("读取大恒相机图像并显示目标光斑")

        # ── 相机连接 ────────────────────────────────────────────────────────
        with st.container(border=True):
            st.subheader("📷 相机连接")

            st.session_state.ts_cam_id = st.number_input(
                "Camera ID",
                min_value=0,
                value=int(st.session_state.ts_cam_id),
                step=1,
            )
            st.session_state.ts_exposure_ms = st.number_input(
                "曝光时间 (ms)",
                min_value=0.02,
                max_value=1000.0,
                value=float(st.session_state.ts_exposure_ms),
                step=1.0,
            )
            st.session_state.ts_window_size = st.number_input(
                "窗口大小 (px)",
                min_value=50,
                max_value=2000,
                value=int(st.session_state.ts_window_size),
                step=50,
            )

            # Stale camera check
            cam = st.session_state.ts_camera
            if cam is not None:
                try:
                    if not cam.is_connected():
                        logger.warning("Camera connection lost, resetting state")
                        st.session_state.ts_camera = None
                        st.session_state.ts_camera_connected = False
                        st.session_state.ts_frame = None
                        st.session_state.ts_centers = {}
                        st.session_state.ts_target = None
                except Exception as exc:
                    logger.warning("Camera state check failed: {}", exc)
                    st.session_state.ts_camera = None
                    st.session_state.ts_camera_connected = False

            if not st.session_state.ts_camera_connected:
                if st.button("连接相机", type="primary"):
                    try:
                        cam = DahengCamManager(
                            cam_id=st.session_state.ts_cam_id,
                            exposure_time_ms=st.session_state.ts_exposure_ms,
                        )
                        cam.open()
                        st.session_state.ts_camera = cam
                        st.session_state.ts_camera_connected = True
                        st.success("相机已连接")
                        logger.info(
                            "Camera {} connected", st.session_state.ts_cam_id
                        )
                        st.rerun()
                    except Exception as exc:
                        logger.exception("Camera connection failed")
                        st.error(f"连接失败: {exc}")
            else:
                if st.button("断开相机"):
                    try:
                        cam = st.session_state.ts_camera
                        if cam is not None:
                            cam.close()
                        st.session_state.ts_camera = None
                        st.session_state.ts_camera_connected = False
                        st.session_state.ts_frame = None
                        st.session_state.ts_centers = {}
                        st.session_state.ts_target = None
                        st.info("相机已断开")
                        logger.info("Camera disconnected")
                        st.rerun()
                    except Exception as exc:
                        logger.exception("Camera disconnect failed")
                        st.error(f"断开失败: {exc}")

            # Connection status indicator
            if st.session_state.ts_camera_connected:
                st.success("● 已连接")
            else:
                st.warning("○ 未连接 — 使用仿真图像")

        st.divider()

        # ── 相机信息 ────────────────────────────────────────────────────────
        with st.container(border=True):
            st.subheader("⚙️ 相机信息")
            cam = st.session_state.ts_camera
            if st.session_state.ts_camera_connected and cam is not None:
                st.write(f"分辨率: {cam.cam_width} × {cam.cam_height}")
                st.write(f"曝光时间: {st.session_state.ts_exposure_ms} ms")
                st.write(f"窗口大小: {st.session_state.ts_window_size} px")
            else:
                sim_h, sim_w = _SIM_FRAME_SHAPE
                st.write(f"仿真分辨率: {sim_h} × {sim_w}")
                st.write("模式: 随机高斯光斑仿真")

        st.divider()

        # ── ROI 窗口 (可选) ─────────────────────────────────────────────────
        with st.container(border=True):
            st.subheader("🔲 ROI 窗口")
            use_roi = st.checkbox("启用 ROI 窗口", value=False)
            cam: DahengCamManager | None = st.session_state.ts_camera
            centers = st.session_state.ts_centers
            # Use the first available center for ROI placement
            roi_center: tuple[float, float] | None = None
            if centers:
                roi_center = next(iter(centers.values()))
            if (
                use_roi
                and st.session_state.ts_camera_connected
                and cam is not None
            ):
                if roi_center is None:
                    roi_center = (cam.cam_width / 2, cam.cam_height / 2)
                if st.button("应用 ROI 窗口"):
                    try:
                        cam.reset_window(
                            center=(int(roi_center[0]), int(roi_center[1])),
                            size=(
                                st.session_state.ts_window_size,
                                st.session_state.ts_window_size,
                            ),
                        )
                        st.success(
                            f"ROI 已应用: 中心 ({roi_center[0]:.0f}, {roi_center[1]:.0f}), "
                            f"大小 {st.session_state.ts_window_size}"
                        )
                        logger.info(
                            "ROI window applied: center={}, size={}",
                            roi_center,
                            st.session_state.ts_window_size,
                        )
                    except Exception as exc:
                        logger.exception("reset_window failed")
                        st.error(f"ROI 应用失败: {exc}")
            else:
                st.caption("默认使用全幅传感器")


# ── Shape controls (left column) ──────────────────────────────────────────────
def _render_shape_controls() -> tuple[
    str,
    dict,
    dict[str, tuple[float, float]],
    tuple[float, float] | None,
]:
    """Render the target shape parameter controls.

    Returns:
        (shape_type, shape_params, selected_centers, manual_center)
    """
    with st.container(border=True):
        st.subheader("🎯 目标形状")

        shape_type = st.selectbox(
            "形状类型", ["圆形", "方形", "长方形"], index=0
        )

        shape_params: dict = {}
        if shape_type == "圆形":
            shape_params["radius"] = st.number_input(
                "半径 (px)", min_value=1, max_value=1000, value=50, step=1
            )
        elif shape_type == "方形":
            shape_params["side"] = st.number_input(
                "边长 (px)", min_value=1, max_value=1000, value=100, step=1
            )
        else:  # 长方形
            c1, c2 = st.columns(2)
            with c1:
                shape_params["rect_w"] = st.number_input(
                    "宽度 (px)", min_value=1, max_value=2000, value=120, step=1
                )
            with c2:
                shape_params["rect_h"] = st.number_input(
                    "高度 (px)", min_value=1, max_value=2000, value=80, step=1
                )

    with st.container(border=True):
        st.subheader("📐 中心计算方法")
        st.caption("勾选多个方法可同时对比显示")

        # ── Center method checkboxes ────────────────────────────────────────
        selected_centers: dict[str, tuple[float, float]] = {}
        custom_center: tuple[float, float] | None = None

        for key, label in _CENTER_METHODS:
            checked = st.checkbox(
                label,
                value=st.session_state.get(f"{_CM_KEY}{key}", False),
                key=f"{_CM_KEY}{key}",
            )
            if checked and key == "custom":
                c1, c2 = st.columns(2)
                with c1:
                    cx = st.number_input(
                        "自定义 X",
                        min_value=0,
                        max_value=5000,
                        value=0,
                        step=1,
                        key="ts_custom_x",
                    )
                with c2:
                    cy = st.number_input(
                        "自定义 Y",
                        min_value=0,
                        max_value=5000,
                        value=0,
                        step=1,
                        key="ts_custom_y",
                    )
                custom_center = (float(cx), float(cy))

        # ── Manual center override ──────────────────────────────────────────
        manual_override = st.checkbox("手动指定中心 (覆盖上述方法)", value=False)
        manual_center: tuple[float, float] | None = None
        if manual_override:
            c1, c2 = st.columns(2)
            with c1:
                mx = st.number_input(
                    "中心 X",
                    min_value=0,
                    max_value=5000,
                    value=0,
                    step=1,
                    key="ts_manual_x",
                )
            with c2:
                my = st.number_input(
                    "中心 Y",
                    min_value=0,
                    max_value=5000,
                    value=0,
                    step=1,
                    key="ts_manual_y",
                )
            manual_center = (float(mx), float(my))

    return shape_type, shape_params, selected_centers, manual_center


# ── Overlay rendering ─────────────────────────────────────────────────────────
def _render_overlay(
    shape_type: str,
    shape_params: dict,
    selected_center_keys: list[str],
    custom_center: tuple[float, float] | None,
    manual_center: tuple[float, float] | None,
) -> None:
    """Compute centers + target from the latest frame and render the overlay."""
    frame = st.session_state.ts_frame
    if frame is None:
        st.caption("尚未采集图像 — 请点击「刷新并计算中心」")
        return

    height, width = frame.shape[:2]

    if frame.size == 0 or float(np.max(frame)) <= 0:
        st.warning("图像为空或全黑，无法计算中心")
        return

    # Compute centers for all checked methods
    centers: dict[str, tuple[float, float]] = {}

    if manual_center is not None:
        # Manual override: all methods map to the same point
        for key in selected_center_keys:
            label = dict(_CENTER_METHODS).get(key, key)
            centers[label] = manual_center
    else:
        for key in selected_center_keys:
            try:
                c = _compute_center(frame, key, custom_center)
                label = dict(_CENTER_METHODS).get(key, key)
                centers[label] = c
            except Exception as exc:
                logger.warning("Center computation failed for {}: {}", key, exc)

    st.session_state.ts_centers = centers

    if not centers:
        st.info("请在左侧勾选至少一种中心计算方法")
        return

    # Use the first selected center for target generation
    first_key = selected_center_keys[0] if selected_center_keys else None
    primary_center: tuple[float, float] | None = None
    if manual_center is not None:
        primary_center = manual_center
    elif first_key:
        label = dict(_CENTER_METHODS).get(first_key, first_key)
        primary_center = centers.get(label)

    # Generate target
    target = generate_target_mask(
        shape_type, shape_params, primary_center, height, width
    )
    st.session_state.ts_target = target

    # Show computed centers
    cols = st.columns(min(len(centers), 4))
    for i, (label, (cx, cy)) in enumerate(centers.items()):
        with cols[i % len(cols)]:
            st.metric(label=label, value=f"({cx:.1f}, {cy:.1f})")

    # Plotly overlay
    _display_overlay_plotly(frame, target, centers)


# ── Capture ───────────────────────────────────────────────────────────────────
def _capture_and_compute() -> None:
    """Capture a fresh averaged frame and store it for overlay computation.

    In simulation mode (no camera connected) a new random Gaussian spot
    is generated instead.
    """
    cam = st.session_state.ts_camera
    if cam is None or not st.session_state.ts_camera_connected:
        # Simulation mode: roll a new random spot
        st.session_state.ts_sim_params = None
        frame = _generate_sim_frame()
        st.session_state.ts_frame = frame
        st.success("已生成新的仿真光斑")
        logger.info(
            "Generated new simulated spot: shape={}, peak={:.0f}",
            frame.shape,
            float(np.max(frame)),
        )
        return
    try:
        frame = cam.get_numpy_image(n_sample=3, skip_first=True)
        st.session_state.ts_frame = frame
        st.success("图像已刷新")
        logger.info("Captured fresh frame: shape={}", frame.shape)
    except Exception as exc:
        logger.exception("Frame capture failed")
        st.error(f"采集失败: {exc}")


# ── Live preview fragment ─────────────────────────────────────────────────────
@st.fragment(run_every=_REFRESH_INTERVAL)
def live_display() -> None:
    """Auto-refreshing live camera preview (0.2 s cadence).

    Falls back to a simulated Gaussian-spot frame when the camera is not
    connected (spot geometry stable, per-frame noise only).
    """
    if not st.session_state.get("ts_camera_connected", False):
        frame = _generate_sim_frame()
        st.session_state.ts_frame = frame
        st.image(
            frame,
            caption="Simulated CCD Frame (随机高斯光斑)",
            clamp=True,
            use_container_width=True,
        )
        return
    cam = st.session_state.get("ts_camera")
    if cam is None:
        st.caption("相机对象不可用")
        return
    try:
        frame = cam.get_numpy_image(n_sample=1, skip_first=True)
        st.session_state.ts_frame = frame
        st.image(
            frame,
            caption="Current Camera Frame",
            clamp=True,
            use_container_width=True,
        )
    except Exception as exc:
        logger.exception("Frame capture failed")
        st.error(f"采集失败: {exc}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    """Target Shape Helper entry point."""
    st.set_page_config(layout="wide", page_title="Target Shape Helper")
    _init_session_state()
    _render_sidebar()

    st.title("🎯 Target Shape Helper")
    st.caption("读取大恒相机图像，显示目标光斑（圆形 / 方形 / 长方形）")

    left_col, right_col = st.columns([1, 1])

    with left_col:
        shape_type, shape_params, _, manual_center = _render_shape_controls()

    with right_col:
        st.subheader("📷 相机图像")
        live_display()

        st.subheader("🎯 目标叠加")
        if st.button("刷新并计算中心"):
            _capture_and_compute()

        # Collect checked center method keys for overlay
        checked_keys = [
            key for key, _ in _CENTER_METHODS
            if st.session_state.get(f"{_CM_KEY}{key}", False)
        ]
        custom_center = None
        if st.session_state.get(f"{_CM_KEY}custom", False):
            custom_center = (
                float(st.session_state.get("ts_custom_x", 0)),
                float(st.session_state.get("ts_custom_y", 0)),
            )
        _render_overlay(
            shape_type,
            shape_params,
            checked_keys,
            custom_center,
            manual_center,
        )


if __name__ == "__main__":
    main()