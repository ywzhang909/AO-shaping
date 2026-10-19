"""`round_to_int` 与 `truncate_to_int` 的落盘语义契约 (#142)。

这两个 helper 只差一次取整，却**不可互换**：合并任意一个都会静默改变
`save_best` 写出的电压数值。本测试把 #142的对拍结论固定为回归门。
"""

from __future__ import annotations

import io

import numpy as np

from ao_shaping.runners.runner_common import round_to_int, truncate_to_int

#: 含半整数、负数与 99.5，取整口径在这些点上必然分化。
SAMPLE = [0.5, 1.5, 2.5, 2.7, 2.9, 3.2, 3.8, 4.0, -2.5, -2.7, 5.5, 99.5, -0.5]


def _savetxt(arr: np.ndarray) -> list[str]:
    buf = io.StringIO()
    np.savetxt(buf, arr, fmt="%d")
    return buf.getvalue().split()


class TestRoundToInt:
    def test_matches_both_original_spellings(self) -> None:
        """`round_to_int` 与两个被合并的原写法逐位相同 (wf 用 np.round，
        axis_beam / combined 用 np.around().astype(int))。"""
        arr = np.asarray(SAMPLE)
        assert np.array_equal(round_to_int(arr), np.round(arr))
        assert np.array_equal(round_to_int(arr), np.around(arr).astype(int))

    def test_saved_artifact_is_byte_identical_to_np_round(self) -> None:
        """落盘产物不变：dtype 由 float64 变 int64，但 `fmt="%d"` 下逐字节一致。"""
        arr = np.asarray(SAMPLE)
        assert _savetxt(round_to_int(arr)) == _savetxt(np.round(arr))


class TestTruncateToInt:
    def test_matches_pipeline_original_definition(self) -> None:
        """`truncate_to_int` 复刻 pipeline_runner 原先的本地 `arr.astype(int)`。"""
        arr = np.asarray(SAMPLE)
        assert np.array_equal(truncate_to_int(arr), arr.astype(int))

    def test_is_NOT_interchangeable_with_round_to_int(self) -> None:
        """两者必须在采样值上确有分歧 —— 否则本可合并，#142 的裁决就不成立。"""
        arr = np.asarray(SAMPLE)
        assert not np.array_equal(round_to_int(arr), truncate_to_int(arr))
        assert _savetxt(round_to_int(arr)) != _savetxt(truncate_to_int(arr))

    def test_divergence_is_on_halfway_values(self) -> None:
        """分歧点固定在半整数与 .7/.9 这类值上，防止未来「顺手统一」。"""
        arr = np.asarray(SAMPLE)
        assert list(round_to_int(arr)) != list(truncate_to_int(arr))
        assert round_to_int(np.asarray([2.7]))[0] == 3
        assert truncate_to_int(np.asarray([2.7]))[0] == 2
        assert round_to_int(np.asarray([99.5]))[0] == 100
        assert truncate_to_int(np.asarray([99.5]))[0] == 99
