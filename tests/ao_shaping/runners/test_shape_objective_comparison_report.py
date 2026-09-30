"""回归: 三目标离线对比脚本的两处静默正确性缺陷。

两个 bug 都是"能跑、但结论是错的"类型, 因此必须有测试锁住:

1. **tie-break 方向反了** (``_disagreement_order``)
   ``np.lexsort`` 以**最后一个** key 为主键。原文写作
   ``np.lexsort((pearson_loss, -gap))`` —— 主键变成 ``pearson_loss`` 升序,
   即"并列时最好的排前面", 与 docstring 声明的 *worst first* 相反, 会把最该
   被检视的帧藏到表格末尾。正确写法是 ``np.lexsort((-pearson_loss, -gap))``。

   注意 gap 是在 **min-max 归一化之后**计算的, 所以构造并列数据时必须先算
   归一化值, 不能拿原始值想当然 (原始值相等不代表归一化后相等)。

2. **rows / epochs / gates 错位** (``load_frame_sets``)
   ``frames`` 用 ``[r for r in rows[1:] if "_img" in r]`` 过滤, 而 ``epochs`` /
   ``gates`` 各自独立地从**未过滤**的 ``rows[1:]`` 推导。只要有任何一行缺
   ``_img``, 三个列表就会错位, 之后每个 gate 标签都会被安到错误的 epoch 上。
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from generate_shape_objective_comparison import (  # noqa: E402
    _disagreement_order,
    load_frame_sets,
)


class _FakeRecorder:
    """Module-level (hence picklable) stand-in for ``utils.io.file.Recorder``."""

    def __init__(self, history: list[dict]) -> None:
        self.history = history


def _write_run(root: Path, rows: list[dict], *, as_dict: bool = False,
              config: dict | None = None) -> Path:
    """Write one run dir.

    ``as_dict=True`` reproduces what ``save_recorder_debug_artifacts`` writes for
    the search runs: a plain ``{epoch: row}`` mapping rather than a pickled
    ``Recorder``. The loader used to require ``.history`` and silently skipped
    every such artifact, so a staged hardware run reported "no recorded frames".
    """
    import json as _json

    run = root / "20260101_000000_run"
    run.mkdir(parents=True)
    payload = dict(enumerate(rows)) if as_dict else _FakeRecorder(rows)
    (run / "recorder_tag.pkl").write_bytes(pickle.dumps(payload))
    if config is not None:
        (run / "summary_tag.json").write_text(_json.dumps(config), encoding="utf-8")
    return run


# comp/loss chosen so that, AFTER the min-max normalisation the script applies,
# indices 0 and 1 tie on the disagreement gap while index 1 has the worse loss.
_TIE_COMP = np.array([0.0, 1.0, 0.0, 1.0])
_TIE_LOSS = np.array([0.0, 0.9, 0.9, 0.0])


class TestDisagreementTieBreak:
    """Ties must resolve worst-Pearson-first, as the docstring promises."""

    def test_tie_breaks_toward_worst_loss(self) -> None:
        order = _disagreement_order(_TIE_COMP, _TIE_LOSS)

        # gap-1.0 group is {0, 1}; within it loss is 0.0 vs 0.9, so index 1
        # (worse) must come first.
        assert order[0] == 1
        assert order[1] == 0

    def test_regression_old_lexsort_was_wrong(self) -> None:
        """The pre-fix expression produced the opposite tie order.

        Replicates the original body exactly, including the normalisation, so the
        only difference from :func:`_disagreement_order` is the ``lexsort`` key
        order.
        """
        from generate_shape_objective_comparison import _higher_is_better

        comp_n = _higher_is_better(_TIE_COMP, lower_is_better=False)
        loss_n = _higher_is_better(_TIE_LOSS, lower_is_better=True)
        gap = np.abs(comp_n - loss_n)
        old = np.lexsort((_TIE_LOSS, -gap))

        new = _disagreement_order(_TIE_COMP, _TIE_LOSS)

        # gap = [1, 1, 0, 0]; within the tied {0, 1} group loss is 0.0 vs 0.9.
        assert list(old[:2]) == [0, 1], "old: tie broken best-loss-first"
        assert list(new[:2]) == [1, 0], "new: tie broken worst-loss-first"
        assert old[0] != new[0], "the fix must actually change the leading frame"

    def test_gap_is_the_primary_key(self) -> None:
        """A high-loss frame with a small gap must not outrank a large-gap one."""
        order = _disagreement_order(_TIE_COMP, _TIE_LOSS)

        assert set(order[:2]) == {0, 1}, "the two gap-1.0 frames must rank first"

    def test_returns_every_index_once(self) -> None:
        order = _disagreement_order(_TIE_COMP, _TIE_LOSS)

        assert sorted(order.tolist()) == list(range(len(_TIE_COMP)))


class TestFrameSetAlignment:
    """``frames`` / ``epochs`` / ``gates`` must describe the same rows."""

    def test_rows_missing_img_do_not_shift_labels(self, tmp_path: Path) -> None:
        # Row 2 deliberately lacks ``_img``; it must be dropped from ALL three
        # series so surviving gate labels stay attached to their own epoch.
        rows = [
            {"_img": np.zeros((8, 8), np.uint8), "_epoch": 0, "_gate": "init"},
            {"_img": np.zeros((8, 8), np.uint8), "_epoch": 1, "_gate": "applied"},
            {"_epoch": 2, "_gate": "no_image"},
            {"_img": np.zeros((8, 8), np.uint8), "_epoch": 3, "_gate": "stalled"},
        ]
        _write_run(tmp_path, rows)

        (fs,) = load_frame_sets(tmp_path)

        assert fs.epochs.size == len(fs.frames) == len(fs.gates) == 2
        assert list(fs.epochs) == [1.0, 3.0]
        assert fs.gates == ["applied", "stalled"]
        assert "no_image" not in fs.gates

    def test_all_lists_share_length(self, tmp_path: Path) -> None:
        rows = [
            {"_img": np.zeros((8, 8), np.uint8), "_epoch": 0},
            {"_img": np.zeros((8, 8), np.uint8), "_epoch": 1, "_gate": "applied"},
            {"_img": np.zeros((8, 8), np.uint8), "_epoch": 2, "_gate": "fold"},
        ]
        _write_run(tmp_path, rows)

        (fs,) = load_frame_sets(tmp_path)

        assert fs.epochs.size == len(fs.frames) == len(fs.gates)
        assert fs.gates == ["applied", "fold"]

    def test_run_with_no_scored_frames_is_skipped(self, tmp_path: Path) -> None:
        rows = [
            {"_img": np.zeros((8, 8), np.uint8), "_epoch": 0},
            {"_epoch": 1, "_gate": "no_image"},
        ]
        _write_run(tmp_path, rows)

        assert load_frame_sets(tmp_path) == []

    def test_anchor_is_row0_not_first_scored_frame(self, tmp_path: Path) -> None:
        """The anchor must be the pre-optimization row, never a scored frame."""
        anchor = np.full((8, 8), 7, np.uint8)
        scored = np.full((8, 8), 3, np.uint8)
        rows = [
            {"_img": anchor, "_epoch": 0},
            {"_img": scored, "_epoch": 1, "_gate": "applied"},
        ]
        _write_run(tmp_path, rows)

        (fs,) = load_frame_sets(tmp_path)

        assert np.array_equal(fs.anchor, anchor)
        assert np.array_equal(fs.frames[0], scored)
        assert not np.array_equal(fs.anchor, fs.frames[0])


class TestDictShapedPickles:
    """``save_recorder_debug_artifacts`` writes ``{epoch: row}``, not a Recorder."""

    @staticmethod
    def _rows() -> list[dict]:
        return [
            {"_img": np.full((8, 8), 1, np.uint8), "_epoch": 0},
            {"_img": np.full((8, 8), 2, np.uint8), "_epoch": 1, "_gate": "applied"},
            {"_img": np.full((8, 8), 3, np.uint8), "_epoch": 2, "_gate": "noise"},
        ]

    def test_dict_payload_is_loaded(self, tmp_path: Path) -> None:
        _write_run(tmp_path, self._rows(), as_dict=True)

        (fs,) = load_frame_sets(tmp_path)

        assert fs.epochs.size == 2
        assert fs.gates == ["applied", "noise"]

    def test_both_pickle_shapes_load_identically(self, tmp_path: Path) -> None:
        a, b = tmp_path / "recorder", tmp_path / "dict"
        a.mkdir()
        b.mkdir()
        _write_run(a, self._rows(), as_dict=False)
        _write_run(b, self._rows(), as_dict=True)

        (fs_rec,), (fs_dict,) = load_frame_sets(a), load_frame_sets(b)

        assert list(fs_rec.epochs) == list(fs_dict.epochs)
        assert fs_rec.gates == fs_dict.gates

    def test_sidecar_config_is_attached(self, tmp_path: Path) -> None:
        _write_run(
            tmp_path,
            self._rows(),
            as_dict=True,
            config={"objective": "pearson", "cam_type": "daheng", "delta": 0.1},
        )

        (fs,) = load_frame_sets(tmp_path)

        assert fs.config is not None
        assert fs.config["objective"] == "pearson"
        assert fs.config["cam_type"] == "daheng"
        assert fs.config["delta"] == 0.1

    def test_missing_sidecar_leaves_config_none(self, tmp_path: Path) -> None:
        _write_run(tmp_path, self._rows(), as_dict=True)

        (fs,) = load_frame_sets(tmp_path)

        assert fs.config is None

    def test_none_payload_is_still_skipped(self, tmp_path: Path) -> None:
        """The SNR sweeps dump ``None``; those must not raise."""
        run = tmp_path / "20260101_000000_snr"
        run.mkdir(parents=True)
        (run / "recorder_snr.pkl").write_bytes(pickle.dumps(None))

        assert load_frame_sets(tmp_path) == []
