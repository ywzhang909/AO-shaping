"""
Target Shape Plotly views

Plotly figure construction for the Target Shape Helper page (camera image,
target shape and overlay panels). Extracted verbatim from
``ao_shaping.gui.ccd.target_shape_helper`` so both modules stay under the
800-line module-size gate; behaviour is unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ── Plotly rendering ──────────────────────────────────────────────────────────
_METHOD_COLORS = {
    "centroid": "#ff4b4b",
    "centroid_thresh": "#ffa500",
    "argmax": "#00cc96",
    "custom": "#ab63fa",
}


def display_overlay_plotly(
    frame: np.ndarray,
    target: np.ndarray,
    centers: dict[str, tuple[float, float]],
    method_labels: Iterable[tuple[str, str]] = (),
) -> None:
    """Render camera image, target shape and overlay with Plotly.

    *method_labels* holds the same ``(key, label)`` pairs the page uses for its
    center-method checkboxes, so the legend reads exactly as before.
    """
    labels = dict(method_labels)
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
        label = labels.get(method_key, method_key)
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
