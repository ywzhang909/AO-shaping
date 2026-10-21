"""``shaping_score`` (T 系列 composite) 的单一真源测试 (issue #154).

T1/T2 历史上各自内联 ``0.5*EE + 0.5/(1+CV)``; 现在统一走
``report/sim_tue_methods/_common.py:shaping_score``。本测试断言:

1. 公式值正确: ``shaping_score(0.8, 0.5) == 0.5*0.8 + 0.5/1.5``;
2. 数值守恒: 与 T1/T2 旧内联表达式在同一输入下**逐位**相等 (refactor 不改口径);
3. 语义不变: EE 越大、CV 越小, 分数越大 (单调性)。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# _common.py 与 t1/t2 同目录; 该目录不是 package, 测试需把目录加进 sys.path。
_TUE_DIR = Path(__file__).resolve().parents[3] / "report" / "sim_tue_methods"


@pytest.fixture()
def shaping_score():
    # 不用 ``importlib.import_module("_common")``：仓库里另有 ``scripts/_common/``
    # 包，若先前任何测试把 scripts/ 加进 sys.path 并 import 过它，
    # ``sys.modules["_common"]`` 会被那个包污染，本 fixture 会拿到包
    # (无 shaping_score) 而 AttributeError。这里按绝对路径
    # spec_from_file_location 加载本目录的 _common.py，并清缓存，
    # 彻底免疫命名冲突与模块缓存残留。
    import importlib.util

    path = _TUE_DIR / "_common.py"
    spec = importlib.util.spec_from_file_location("tue_sim_methods_common", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    yield mod.shaping_score


def test_shaping_score_matches_published_formula(shaping_score) -> None:
    """与 T1/T2 旧内联表达式逐位相等 (数值守恒断言)。"""
    for ee, cv in [(0.8, 0.5), (0.99, 0.01), (0.1, 3.0), (1.0, 0.0), (0.0, 1.0)]:
        legacy = 0.5 * ee + 0.5 / (1.0 + cv)  # T1/T2 历史内联
        assert shaping_score(ee, cv) == pytest.approx(legacy, rel=0.0, abs=0.0)


def test_shaping_score_reference_value(shaping_score) -> None:
    """BRIEF 指定参考值: shaping_score(0.8, 0.5) == 0.5*0.8 + 0.5/1.5。"""
    assert shaping_score(0.8, 0.5) == pytest.approx(0.5 * 0.8 + 0.5 / 1.5)


def test_shaping_score_monotonic(shaping_score) -> None:
    """EE 越大分数越大; CV 越大分数越小。"""
    assert shaping_score(0.9, 0.3) > shaping_score(0.5, 0.3)
    assert shaping_score(0.8, 0.1) > shaping_score(0.8, 0.9)
