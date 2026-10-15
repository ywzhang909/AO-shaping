"""Generate table image for 2026-10-14 model-in-loop report."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IMG_DIR = ROOT / "report" / "shape_trials" / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

def render_model_in_loop_20261014_table() -> Path:
    """Create a table image for the 2026-10-14 model-in-loop runs."""
    # Data from the report
    data = [
        ["20261014_105041", "无扰动，方形", 0.8153, 0.7473, 0.8129, "0.9019 / 0.9209", 0.7629],
        ["20261014_105205", "扰动，方形", 0.8084, 0.7601, 0.8096, "0.8650 / 0.8741", 0.7390],
        ["20261014_105339", "扰动，长方形", 0.8112, 0.7664, 0.8219, "0.7142 / 0.7622", 0.6907],
        ["20261014_105438", "扰动，三角形", 0.8173, 0.7846, 0.8226, "0.6624 / 0.7173", 0.6710],
        ["20261014_105535", "扰动，五角星", 0.8112, 0.7722, 0.8430, "0.6529 / 0.6134", 0.6101],
        ["20261014_110203", "扰动，长方形，Step A 零步，种子 86", 0.8316, 0.8213, 0.8016, "0.6851 / 0.6944", 0.6540],
        ["20261014_110305", "扰动，长方形，Step A 10 步，种子 86", 0.8417, 0.7767, 0.8037, "0.8088 / 0.7471", 0.6694],
    ]
    col_labels = ["运行目录", "条件", "6 留出相关性均值", "自由相位相关性均值", "成形预测相关性", "实测 CV / EE", "分数"]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    ax.axis('tight')
    
    # Add title
    fig.suptitle("Model-in-loop 2026-10-14 Run Results Table", fontsize=14, y=0.95)
    
    # Create table
    table = ax.table(cellText=data, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    fig.tight_layout()
    
    out = IMG_DIR / "mil_model_in_loop_20261014_table.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out

if __name__ == "__main__":
    print("Generating model-in-loop 2026-10-14 table...")
    img = render_model_in_loop_20261014_table()
    print(f"  -> {img.relative_to(ROOT)}")
