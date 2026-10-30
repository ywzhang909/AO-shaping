"""迁移来源候选层的纯数据用例（不碰 streamlit）。

候选行 = 磁盘上的历史产物 → 一行文案。判据同源：检查点走 ``checkpoint_registry``
+ ``optical_session``，``--seed-run`` 走 ``manifest.json`` + ``config.json`` +
``<形状>_final``（``validate.replay_seed`` 查的同一行）。控件接线（下拉、折叠的
手动区、session_state 指纹）在 ``test_shaping_controls.py``。
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from ao_shaping.gui.slm import transfer_candidates


INDEXED_ENTRY: dict[str, Any] = {
    "path": "ck/square_20261027",
    "file": "ck/square_20261027/fit/frozen_parameters.json",
    "session": "square_20261027",
    "kind": "checkpoint",
    "shape": "square",
    "created_at": "2026-10-27T18:10:38+00:00",
    "sha256_ok": True,
    "stale": False,
}


def _make_run(base: Path, rel: str, labels: list[str], *,
              config: bool = True) -> Path:
    run = base / rel
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(
        json.dumps([{"label": label} for label in labels]), encoding="utf-8")
    if config:
        (run / "config.json").write_text("{}", encoding="utf-8")
    return run


class TestTransferCheckpointCandidates:
    """候选行 = ``checkpoint_registry`` 的索引条目 + ``optical_session`` 的分数。

    本模块只排版：分数、拒绝理由一律转手，它自己不打分也不读指纹（那样就有了
    第二个口径）。
    """

    @staticmethod
    def _patch(monkeypatch, *, score: float = 0.87,
               dimensions: list[str] | None = None,
               entry: dict[str, Any] | None = None) -> None:
        from ao_shaping.tools.slm.transfer import checkpoint_registry, optical_session

        row = dict(entry or INDEXED_ENTRY)
        dims = ["geometry"] if dimensions is None else list(dimensions)
        monkeypatch.setattr(checkpoint_registry, "scan_checkpoint_index",
                            lambda root, **k: [row])
        monkeypatch.setattr(
            optical_session, "recommend_load",
            lambda current, candidates, **k: {
                "ranked": [{"candidate": candidates[0], "score": score,
                            "dimensions": dims, "rejected": False,
                            "reasons": []}],
                "recommended": None,
                "threshold": 0.6,
            },
        )

    def test_label_carries_shape_time_score_and_check_state(self,
                                                            monkeypatch) -> None:
        self._patch(monkeypatch)
        rows = transfer_candidates.transfer_checkpoint_candidates(root="data")
        assert rows is not None
        label = rows[0]["label"]
        # 形状 · 注册时间 · 兼容分（结论）· 是否已校验 —— 操作员不必再记路径。
        assert label == "正方形 · 2026-10-27 18:10 · 兼容 0.87（推荐） · 已校验"
        assert rows[0]["file"] == INDEXED_ENTRY["file"]
        assert rows[0]["score"] == pytest.approx(0.87)
        assert rows[0]["verified"] is True

    @pytest.mark.parametrize(("score", "tag"), [
        (0.85, "推荐"),
        (0.70, "谨慎"),
        (0.50, "不推荐"),
    ])
    def test_label_carries_the_verdict_of_the_shared_function(
            self, monkeypatch, score: float, tag: str) -> None:
        """三档分数在候选标题里给对应结论词 —— 结论不是本模块判的，它只转手。"""
        self._patch(monkeypatch, score=score)
        rows = transfer_candidates.transfer_checkpoint_candidates(root="data")
        assert rows is not None
        assert f"兼容 {score:.2f}（{tag}）" in rows[0]["label"]

    def test_verdict_tiers_split_exactly_at_the_shared_thresholds(self) -> None:
        """边界值对得上 ``optical_session`` 的两个阈值：本模块没有自己的一套。"""
        from ao_shaping.tools.slm.transfer.optical_session import (
            AUTO_APPLY_THRESHOLD,
            COMPAT_THRESHOLD,
        )

        def _tag(value: float) -> str:
            return transfer_candidates._checkpoint_label(
                INDEXED_ENTRY, score=value, verified=True)

        assert "（推荐）" in _tag(AUTO_APPLY_THRESHOLD)
        assert "（谨慎）" in _tag(AUTO_APPLY_THRESHOLD - 0.01)
        assert "（谨慎）" in _tag(COMPAT_THRESHOLD)
        assert "（不推荐）" in _tag(COMPAT_THRESHOLD - 0.01)

    def test_unscored_row_shows_dash_not_zero(self, monkeypatch) -> None:
        """``dimensions == []`` ⇔ 没能评分（缺当前指纹/候选指纹）。

        写 0.00 会让操作员以为「测过，不兼容」——那是两个完全不同的结论。
        而「不兼容」正是三档结论里最重的那一档，所以无分数时**一个结论词都不能出现**。
        """
        self._patch(monkeypatch, dimensions=[])
        rows = transfer_candidates.transfer_checkpoint_candidates(root="data")
        assert rows is not None
        assert rows[0]["score"] is None
        assert "兼容 —" in rows[0]["label"]
        for tag in ("推荐", "谨慎", "不推荐"):
            assert f"（{tag}）" not in rows[0]["label"]

    def test_shape_missing_from_local_table_uses_canonical_vocabulary(self) -> None:
        """本地表没有 ``circle``，canonical 词表有 —— 下拉里不该露英文原键。

        真实索引（2026-10-29）的第一条候选就是 ``circle``：这里不复制第二份形状
        词表，缺项时借 :mod:`ao_shaping.utils.image.target.shapes`。
        """
        assert transfer_candidates._shape_label("square") == "正方形"
        assert transfer_candidates._shape_label("circle") == "圆形"
        assert transfer_candidates._shape_label("") == "形状未记录"
        assert transfer_candidates._shape_label("pentagon") == "pentagon"

    def test_unscored_row_shows_dash_not_zero(self, monkeypatch) -> None:
        """``dimensions == []`` ⇔ 没能评分（缺当前指纹/候选指纹）。

        写 0.00 会让操作员以为「测过，不兼容」——那是两个完全不同的结论。
        """
        self._patch(monkeypatch, dimensions=[])
        rows = transfer_candidates.transfer_checkpoint_candidates(root="data")
        assert rows is not None
        assert rows[0]["score"] is None
        assert "兼容 —" in rows[0]["label"]

    def test_stale_candidate_is_marked_unverified(self, monkeypatch) -> None:
        self._patch(monkeypatch,
                    entry={**INDEXED_ENTRY, "stale": True})
        rows = transfer_candidates.transfer_checkpoint_candidates(root="data")
        assert rows is not None
        assert rows[0]["verified"] is False
        assert rows[0]["label"].endswith("未校验")

    def test_missing_transfer_package_returns_none(self, monkeypatch) -> None:
        """只装了 GUI 依赖的机器上 transfer 包 import 失败 ⇒ 调用方回落文本框。"""
        monkeypatch.setitem(
            sys.modules,
            "ao_shaping.tools.slm.transfer.checkpoint_registry",
            None,
        )
        assert transfer_candidates.transfer_checkpoint_candidates(root="data") is None

    def test_unreadable_index_returns_none(self, monkeypatch) -> None:
        from ao_shaping.tools.slm.transfer import checkpoint_registry

        def _boom(root, **k):
            raise OSError("索引读不动")

        monkeypatch.setattr(checkpoint_registry, "scan_checkpoint_index", _boom)
        assert transfer_candidates.transfer_checkpoint_candidates(root="data") is None

    def test_absent_candidate_root_returns_none(self, monkeypatch) -> None:
        monkeypatch.setattr(transfer_candidates, "_candidate_root",
                            lambda *a, **k: None)
        assert transfer_candidates.transfer_checkpoint_candidates() is None


def test_verdict_comes_from_optical_session_and_no_bare_threshold_lives_here() -> None:
    """阈值只有 ``optical_session`` 一份：本模块不许出现 ``0.6`` / ``0.8`` 这类裸字面量。

    源码级守卫（沿用 ``test_multishape_campaign_ui`` 的同类口径）：一旦有人在这里
    写 ``if score > 0.8``，候选标题的结论就跟 :func:`compatibility_verdict` 分叉了，
    而分叉的那一档（「推荐」= 可不经确认自动应用）恰恰是最危险的。用 AST 扫
    ``(0, 1)`` 区间的浮点常量 —— 本层只有「分数/权重」量级，任何裸阈值都落在这个区间，
    文案里的 ``0.87`` 之类字符串常量不会误报。
    """
    tree = ast.parse(Path(transfer_candidates.__file__).read_text(encoding="utf-8"))
    stray = [node.value for node in ast.walk(tree)
             if isinstance(node, ast.Constant) and isinstance(node.value, float)
             and 0.0 < node.value < 1.0]
    assert not stray, f"候选层出现了裸阈值字面量 {stray}：结论应问 compatibility_verdict"
    calls = {node.func.id for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "compatibility_verdict" in calls, \
        "候选标题的结论必须来自 optical_session.compatibility_verdict（不是本地判断）"


class TestTransferSeedRunCandidates:
    """``--seed-run`` 候选：判据与消费者自己的校验同源，不另立一条。"""

    def test_keeps_only_dirs_with_accepted_phase(self, tmp_path: Path) -> None:
        ok = _make_run(tmp_path, "live/anchored", ["square_final", "star_final"])
        # capture/ 也有 manifest.json + config.json，但只有原始帧记录：
        # 选中它要等 replay_seed 抛 StopIteration。
        _make_run(tmp_path, "live/capture", ["frame_0"])
        # 没 config.json 的目录 TransferSession 运行前就拒。
        _make_run(tmp_path, "live/noconfig", ["star_final"], config=False)
        # 会话内部的 shape/<形状>/<阶段>/ 不是整轮种子。
        _make_run(tmp_path, "a/b/c/d", ["square_final"])

        rows = transfer_candidates.transfer_seed_run_candidates(root=tmp_path)
        assert rows is not None
        assert [row["path"] for row in rows] == [str(ok)]
        assert rows[0]["label"].startswith(str(Path("live") / "anchored"))

    def test_empty_root_is_empty_not_none(self, tmp_path: Path) -> None:
        """空目录 = 「还没发现候选」（下拉只留冷启动项），不是「索引不可用」。"""
        assert transfer_candidates.transfer_seed_run_candidates(root=tmp_path) == []

    def test_missing_candidate_root_returns_none(self, monkeypatch) -> None:
        monkeypatch.setattr(transfer_candidates, "_candidate_root",
                            lambda *a, **k: None)
        assert transfer_candidates.transfer_seed_run_candidates() is None


class TestFingerprintFromState:
    """GUI 读完 ``st.session_state`` 后交给这里的纯映射转换。"""

    def test_mapping_becomes_dict(self) -> None:
        assert transfer_candidates.fingerprint_from_state({"geometry": {"a": 1}}) == {
            "geometry": {"a": 1}}

    def test_absent_or_mistyped_state_is_none(self) -> None:
        # 没放指纹 / 放了字符串 —— 都按「没放」处理，候选就不带兼容分。
        assert transfer_candidates.fingerprint_from_state(None) is None
        assert transfer_candidates.fingerprint_from_state("nav") is None
