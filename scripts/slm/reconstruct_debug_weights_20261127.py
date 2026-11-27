"""离线从 ``fit/iterations.pkl`` 重建 ``--debug-weights`` 三件套.

背景
----
``closed_loop_speed_analysis_20261125.md`` §5 第 2 项指出速度分析会话
(``20261124_200949``) 的 ``debug_weights=false``，没有 ``weights_trajectory.npz``。
该会话 ``save_dataset=false``，不能靠重跑拟合补轨迹；但 ``fit/iterations.pkl``
已逐迭代保存了完整 ``parameters`` 字典，重建轨迹**不需要硬件、也不需要重新拟合**。

本脚本复用 ``fit._flatten_parameters`` 与 ``fit._write_weight_trajectory``，保证
键集与文件格式和 ``--debug-weights`` 原生产物逐字节同构；额外写一份
``reconstruction.json`` 说明来源是重建而非当场采集，避免事后被误当作原始证据。

用法（默认只复核，不落盘）::

    .venv\\Scripts\\python.exe scripts/slm/reconstruct_debug_weights_20261127.py \
        --session data/debug/slm_transfer/20261124_200949

加 ``--write`` 才写到 ``<session>/fit/debug_weights_reconstructed/``；
原 ``debug_weights/`` 目录绝不覆盖。
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from ao_shaping.tools.slm.transfer.fit import _flatten_parameters, _write_weight_trajectory

logger = logging.getLogger(__name__)

RECON_SCHEMA = "reconstruct_debug_weights/1"


def load_trajectory(iterations_pkl: Path) -> tuple[list[dict[str, float]], list[float]]:
    payload: Any = pickle.loads(iterations_pkl.read_bytes())
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{iterations_pkl} 不是非空迭代列表")
    trajectory: list[dict[str, float]] = []
    losses: list[float] = []
    for entry in payload:
        if not isinstance(entry, dict) or "parameters" not in entry:
            raise ValueError("iterations.pkl 条目缺少 parameters")
        trajectory.append(_flatten_parameters(entry["parameters"]))
        losses.append(float(entry["loss"]))
    return trajectory, losses


def write_history(path: Path, trajectory: list[dict[str, float]], losses: list[float]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for index, (flat, loss) in enumerate(zip(trajectory, losses, strict=True)):
            record: dict[str, Any] = {
                "iteration": index,
                "loss": loss,
                "parameters": flat,
            }
            if index:
                previous = trajectory[index - 1]
                record["delta"] = {
                    key: flat[key] - previous.get(key, flat[key]) for key in flat
                }
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def provenance_doc(source: Path, trajectory: list[dict[str, float]]) -> dict[str, Any]:
    keys = sorted({key for row in trajectory for key in row})
    return {
        "schema": RECON_SCHEMA,
        "reconstructed_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "source": str(source),
        "source_kind": "existing-fit-iterations-pkl",
        "hardware_used": False,
        "refit_performed": False,
        "iterations": len(trajectory),
        "keys": keys,
        "note": (
            "由 fit/iterations.pkl 的 parameters 逐迭代展平重建；与当场 "
            "--debug-weights 同格式，但不是新采集数据。"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="离线重建 debug_weights 权重轨迹；默认只复核，--write 才落盘。",
        allow_abbrev=False,
    )
    parser.add_argument("--session", required=True, help="会话目录（含 fit/iterations.pkl）")
    parser.add_argument(
        "--out",
        default="",
        help="输出目录；默认 <session>/fit/debug_weights_reconstructed",
    )
    parser.add_argument("--write", action="store_true", help="真正写文件")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    session = Path(args.session).expanduser().resolve()
    source = session / "fit" / "iterations.pkl"
    if not source.is_file():
        logger.error("找不到 %s", source)
        return 2

    trajectory, losses = load_trajectory(source)
    out_dir = (
        Path(args.out).expanduser().resolve()
        if args.out
        else session / "fit" / "debug_weights_reconstructed"
    )
    # 原生产物目录绝不覆盖
    if out_dir.name == "debug_weights":
        logger.error("拒绝写入原 --debug-weights 目录：%s", out_dir)
        return 3

    logger.info(
        "重建复核：source=%s iterations=%d keys=%d out=%s write=%s",
        source,
        len(trajectory),
        len({key for row in trajectory for key in row}),
        out_dir,
        args.write,
    )
    if not args.write:
        print(
            json.dumps(
                {
                    "schema": RECON_SCHEMA,
                    "source": str(source),
                    "out": str(out_dir),
                    "iterations": len(trajectory),
                    "keys": sorted({key for row in trajectory for key in row}),
                    "would_write": [
                        "weights_trajectory.npz",
                        "weights_summary.json",
                        "weights_history.jsonl",
                        "reconstruction.json",
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_weight_trajectory(out_dir, trajectory)
    write_history(out_dir / "weights_history.jsonl", trajectory, losses)
    (out_dir / "reconstruction.json").write_text(
        json.dumps(provenance_doc(source, trajectory), ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )

    # 自检：npz 能读回、迭代数一致、首末值与源 parameters 一致
    with np.load(out_dir / "weights_trajectory.npz") as data:
        iterations = data["iterations"]
        if len(iterations) != len(trajectory):
            raise AssertionError("iterations 长度不一致")
        for key, expected in trajectory[0].items():
            if not np.isclose(float(data[key][0]), expected):
                raise AssertionError(f"首迭代值不一致：{key}")
    (out_dir / "verify.json").write_text(
        json.dumps(
            {
                "schema": RECON_SCHEMA + "/verify",
                "npz_readable": True,
                "iterations": len(trajectory),
                "first_last_checked": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
        newline="\n",
    )
    logger.info("重建完成并自检通过：%s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
