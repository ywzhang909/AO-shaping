"""t-SNE 权重轨迹可视化 (纯离线)。

把 ``--debug-weights`` 在 fit 阶段产出的 ``weights_trajectory.npz`` 降维到 2D 并绘制，
用于观察在线调优过程中各参数（默认只取网络 Zernike 系数 ``residual_noll_*``）的
变化轨迹。

输入 / 输出
-----------
输入目录（``--input``）指 ``<out>/debug_weights``，其中 ``weights_trajectory.npz`` 的
每个键对应一条长度为 ``N`` 的数组（``N`` = 迭代轮数 + 1，含 iteration 0 的初始值），
另有一条 ``iterations`` 记录迭代序号。本脚本读取后：

* ``--mode zernike``（默认）只取 ``residual_noll_*`` 列；
* ``--mode all`` 取全部参数列。

堆叠成 ``(N, D)`` 矩阵（去掉 ``iterations`` 行），用 :class:`sklearn.manifold.TSNE`
降到 2D，绘制按迭代顺序连线的散点轨迹，颜色按迭代轮次渐变并标注迭代号。

``D < 2`` 时无法做 t-SNE，脚本报错退出；``N < 3`` 时给出警告但继续（点数太少，
t-SNE 的局部结构不可靠，仅作示意）。

示例
----
::

    .venv\\Scripts\\python.exe scripts\\slm\\visualize_weights_tsne.py \\
        --input data/debug/slm_transfer/<run>/fit/debug_weights \\
        --out data/debug/slm_transfer/<run>/fit/debug_weights/weights_tsne.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

# 无显示环境（CI / 后台批跑）下用 Agg 后端，避免绑定交互式窗口。
matplotlib.use("Agg")

import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRAJECTORY_FILENAME = "weights_trajectory.npz"
ITERATIONS_KEY = "iterations"
ZERNIKE_PREFIX = "residual_noll_"

#: 常见中文字体，按优先级探测；命中后用于渲染图内中文，避免出现方块（tofu）。
_CJK_FONT_CANDIDATES = (
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "PingFang SC",
    "Microsoft JhengHei",
    "STSong",
    "KaiTi",
    "FangSong",
    "SimSun",
)


def _resolve_cjk_font() -> str | None:
    """返回一个可用的中文字体名，找不到则返回 ``None``。"""
    available = {font.name for font in matplotlib.font_manager.fontManager.ttflist}
    for name in _CJK_FONT_CANDIDATES:
        if name in available:
            return name
    return None


_CJK_FONT = _resolve_cjk_font()
if _CJK_FONT is not None:
    matplotlib.rcParams["font.sans-serif"] = [_CJK_FONT, "DejaVu Sans"]
    # 负号在中文字体里常被渲染成方块。
    matplotlib.rcParams["axes.unicode_minus"] = False

#: 有中文字体时图内用中文，否则退回英文，保证任意环境都无方块。
_LABELS: dict[str, str] = (
    {
        "title": "t-SNE 权重轨迹（mode={mode}, D={n_params}, N={n}, perplexity={perplexity}）",
        "xlabel": "t-SNE 坐标 1",
        "ylabel": "t-SNE 坐标 2",
        "trajectory": "轨迹",
        "cbar": "迭代轮次",
    }
    if _CJK_FONT is not None
    else {
        "title": "t-SNE weight trajectory (mode={mode}, D={n_params}, N={n}, perplexity={perplexity})",
        "xlabel": "t-SNE coordinate 1",
        "ylabel": "t-SNE coordinate 2",
        "trajectory": "trajectory",
        "cbar": "iteration",
    }
)


def _load_trajectory(input_dir: Path, mode: str) -> tuple[np.ndarray, list[str], np.ndarray]:
    """读取 ``weights_trajectory.npz``，按 ``mode`` 过滤参数列。

    Returns
    -------
    tuple[np.ndarray, list[str], np.ndarray]
        ``(X, keys, iterations)``：``X`` 为 ``(N, D)`` 参数矩阵，``keys`` 为列名，
        ``iterations`` 为长度 ``N`` 的迭代序号（缺失时退化为 ``arange``）。

    Raises
    ------
    FileNotFoundError
        目录或 ``weights_trajectory.npz`` 不存在。
    ValueError
        ``mode`` 非法，或没有选中任何参数列。
    """
    trajectory_path = input_dir / TRAJECTORY_FILENAME
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"找不到权重轨迹文件：{trajectory_path}")

    data = np.load(trajectory_path)
    param_keys = [key for key in data.files if key != ITERATIONS_KEY]
    if mode == "zernike":
        selected = [key for key in param_keys if key.startswith(ZERNIKE_PREFIX)]
    elif mode == "all":
        selected = list(param_keys)
    else:
        raise ValueError(f"未知的 mode：{mode!r}（可选：all / zernike）")

    if not selected:
        raise ValueError(f"mode={mode!r} 未选中任何参数列（npz 键：{param_keys}）")

    selected = sorted(selected)
    X = np.column_stack([np.asarray(data[key], dtype=float) for key in selected])
    iterations = (
        np.asarray(data[ITERATIONS_KEY], dtype=float)
        if ITERATIONS_KEY in data.files
        else np.arange(X.shape[0], dtype=float)
    )
    logger.info(
        "读取 {}：{} 列，N={} 轮（mode={}）", trajectory_path, len(selected), X.shape[0], mode
    )
    return X, selected, iterations


def _embed_2d(X: np.ndarray, perplexity: float, random_state: int) -> np.ndarray:
    """用 t-SNE 把 ``X`` 降到 2D，返回 ``(N, 2)`` 坐标。

    sklearn 要求 ``perplexity < n_samples``；点数较少时把 perplexity 夹到
    ``N - 1`` 并记录，避免直接抛 ``ValueError``。
    """
    n_samples = X.shape[0]
    effective_perplexity = float(perplexity)
    if effective_perplexity >= n_samples:
        effective_perplexity = max(1.0, float(n_samples - 1))
        logger.warning("perplexity={} ≥ N={}，夹到 {} 以避免 sklearn 报错", perplexity, n_samples, effective_perplexity)

    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    Y = tsne.fit_transform(X)
    logger.info("t-SNE 完成：(N, D)={} -> (N, 2)", X.shape)
    return Y


def _render(
    Y: np.ndarray,
    iterations: np.ndarray,
    out: Path,
    mode: str,
    n_params: int,
    perplexity: float,
) -> None:
    """绘制并保存 2D 轨迹散点图。"""
    n = Y.shape[0]
    norm = plt.Normalize(vmin=0.0, vmax=max(1.0, float(n - 1)))
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    # 按迭代顺序连线，形成轨迹。
    ax.plot(
        Y[:, 0],
        Y[:, 1],
        linestyle="-",
        color="0.75",
        linewidth=1.2,
        zorder=1,
        label=_LABELS["trajectory"],
    )
    for index in range(n):
        color = cmap(norm(index))
        ax.scatter(Y[index, 0], Y[index, 1], color=color, s=42, zorder=2)
        ax.annotate(
            f"iter {int(iterations[index])}",
            (Y[index, 0], Y[index, 1]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
            color="0.25",
        )

    scatter = ax.scatter(
        Y[:, 0], Y[:, 1], c=np.arange(n), cmap=cmap, norm=norm, s=1, alpha=0.0, zorder=0
    )
    cbar = fig.colorbar(scatter, ax=ax, label=_LABELS["cbar"])
    cbar.set_ticks(np.arange(n))

    ax.set_title(
        _LABELS["title"].format(
            mode=mode, n_params=n_params, n=n, perplexity=perplexity
        )
    )
    ax.set_xlabel(_LABELS["xlabel"])
    ax.set_ylabel(_LABELS["ylabel"])
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("保存 t-SNE 轨迹图：{}", out)


def run(
    input_dir: Path,
    out: Path,
    mode: str = "zernike",
    perplexity: float = 5.0,
    random_state: int = 42,
) -> Path:
    """执行 t-SNE 可视化，返回输出 PNG 路径。

    Raises
    ------
    ValueError
        参数列数 ``D < 2``（t-SNE 无法在少于 2 个特征上工作）。
    """
    X, keys, iterations = _load_trajectory(Path(input_dir), mode)
    n, n_params = X.shape
    if n_params < 2:
        raise ValueError(
            f"mode={mode!r} 只得到 {n_params} 个参数列，t-SNE 需要至少 2 个特征"
        )
    if n < 3:
        logger.warning(
            "仅 N={} 轮，点数太少，t-SNE 的局部结构不可靠（结果仅作示意）", n
        )

    Y = _embed_2d(X, perplexity, random_state)
    _render(Y, iterations, Path(out), mode, n_params, perplexity)
    return Path(out)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="t-SNE 可视化 --debug-weights 产出的权重轨迹（默认取 residual_noll_*）。",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="debug_weights 目录（含 weights_trajectory.npz），如 data/debug/slm_transfer/<run>/fit/debug_weights",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="输出 PNG 路径",
    )
    parser.add_argument(
        "--mode",
        choices=("all", "zernike"),
        default="zernike",
        help="zernike=只取 residual_noll_* 列；all=取全部参数列（默认：zernike）",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=5.0,
        help="t-SNE perplexity 参数（默认：5；点数少于该值时自动夹到 N-1）",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="t-SNE 随机种子，保证可复现（默认：42）",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        run(
            args.input,
            args.out,
            mode=args.mode,
            perplexity=args.perplexity,
            random_state=args.random_state,
        )
    except (ValueError, FileNotFoundError) as exc:
        logger.error("t-SNE 可视化失败：{}", exc)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
