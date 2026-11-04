"""Inventory shaping reports and preserve evidence excerpts for the 2026-11-03 review.

Read-only with respect to devices and source reports. Regenerate the JSON/Markdown
inventory after adding a shaping report, then revise the human-written synthesis.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts._common.provenance import insert_header

OUT = ROOT / "report/shape_trials/experience_review/artifacts"
INVENTORY_KEY = "report/shape_trials/experience_review/artifacts/source_inventory.md"
# Direct shaping evidence plus the calibration, model, loss, and device-health
# reports that shaped its acceptance criteria. Pure performance, centroid, and
# atmospheric-correction reports stay outside this evidence inventory.
GROUPS = (
    "benchmarks/device_less_full",
    "diff_beam",
    "fouriergsnet_pipeline",
    "fouriergsnet_sim",
    "hwdataset_corpus",
    "inverse_shaping_unet",
    "loss_algorithms_smoke",
    "loss_defects",
    "miicam_auto_exposure",
    "models_analysis",
    "pib_loss_terms",
    "shape_trials",
    "sim_tue_methods",
    "sim2f_bench",
    "sim2f_shaping",
    "simulation",
    "slm",
    "slm_pib_bench",
    "slm_pib_heuristic_hw",
    "slm_pib_online",
    "slm_pib_online_hw",
    "slm_pib_shape_50px",
    "slm_pib_sim",
    "slm_shaping_sim",
    "slm_zernike_shaping",
    "strehl_benchmark",
    "zernike_coeff2amp",
    "zernike_farfield_sim",
    "zernike_phase2amp",
    "zernike_r2_baseline",
)
ROOT_FILES = (
    "report/pib_optimizer_functional_report.md",
)
SIGNALS = re.compile(
    r"结论|总结|有效|无效|改善|下降|提高|失败|成功|拒绝|通过|不能|门禁|"
    r"accept|reject|conclusion|CV.*[0-9]|NCC.*[0-9]", re.IGNORECASE
)


def main() -> None:
    """Hash scoped sources and record exact line-numbered evidence excerpts."""
    OUT.mkdir(parents=True, exist_ok=True)
    paths = sorted(
        {p for group in GROUPS for p in (ROOT / "report" / group).rglob("*.md")}
        | {ROOT / root_file for root_file in ROOT_FILES}
    )
    records = []
    for path in paths:
        if "experience_review" in path.parts:
            continue
        raw = path.read_bytes()
        lines = raw.decode("utf-8-sig").splitlines()
        records.append({
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "title": next((line for line in lines if line.startswith("# ")), path.stem),
            "lines": len(lines),
            "excerpts": [{"line": i + 1, "text": line} for i, line in enumerate(lines) if line.startswith("#") or SIGNALS.search(line)],
        })
    payload = {"scope": list(GROUPS), "count": len(records), "sources": records}
    (OUT / "source_inventory.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = ["# 整形经验报告来源清单", "", "由脚本自动生成；哈希与逐行证据见 `source_inventory.json`。", "", "| 报告 | 行数 | SHA-256 前 12 位 |", "|---|---:|---|"]
    for record in records:
        relative = Path(record["path"]).relative_to("report")
        rows.append(f"| [{record['title'].removeprefix('# ')}](../../../{relative.as_posix()}) | {record['lines']} | `{record['sha256'][:12]}` |")
    text = insert_header("\n".join(rows) + "\n", INVENTORY_KEY)
    (OUT / "source_inventory.md").write_text(text, encoding="utf-8")
    logger.info("Inventoried {} shaping-related Markdown sources", len(records))


if __name__ == "__main__":
    main()
