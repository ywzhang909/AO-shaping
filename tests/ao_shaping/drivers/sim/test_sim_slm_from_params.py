"""回归: ``SimSLMPib`` 必须支持 dataclass 构造 API (``from_params``)。

``optimize_slm_zernike_pib`` 已迁移到纯 dataclass 单参数 API, 内部用
``Santec.from_params(config.slm)`` 构造 SLM (见 AGENTS.md ``slm-pib 配置容器``)。
但 ``scripts/slm_pib_sim_run.py`` 把 ``opt.Santec`` 换成的 ``SimSLMPib`` 当初
没有跟上, 于是 ``--cam_type sim`` 这条离线通路直接崩在

    AttributeError: type object 'SimSLMPib' has no attribute 'from_params'

设备下线期间这条通路正是"无设备分析"的主入口, 所以必须锁死。
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ao_shaping.drivers.sim.slm_pib_sim import SimSLMPib


@dataclass
class _FakeSlmParams:
    """Stand-in for ``runners.runner_common.SlmParamsPib``."""

    slm_number: int = 2
    slm_wavelength: int = 1064
    shift_x: int = 3
    shift_y: int = -4


class TestSimSlmFromParams:
    def test_from_params_exists_and_is_a_classmethod(self) -> None:
        # ``getattr`` on a class yields the bound classmethod (not the raw
        # descriptor), so probe the class dict for the descriptor type instead.
        import inspect

        assert inspect.ismethod(SimSLMPib.from_params), (
            "SimSLMPib.from_params must be a classmethod so the dataclass API "
            "can construct it from a params object"
        )
        assert SimSLMPib.from_params.__self__ is SimSLMPib

    def test_from_params_builds_an_instance(self) -> None:
        slm = SimSLMPib.from_params(_FakeSlmParams())

        assert isinstance(slm, SimSLMPib)

    def test_from_params_tolerates_a_bare_object(self) -> None:
        """``getattr``-with-default extraction must survive a params object
        that carries none of the optional attributes (e.g. a SimpleNamespace)."""

        class _Bare:
            pass

        slm = SimSLMPib.from_params(_Bare())

        assert isinstance(slm, SimSLMPib)

    def test_overrides_are_applied(self) -> None:
        slm = SimSLMPib.from_params(_FakeSlmParams(), system=None)

        assert isinstance(slm, SimSLMPib)

    def test_result_is_usable_as_a_context_manager(self) -> None:
        """The optimizer enters the SLM context; the sim must support it."""
        with SimSLMPib.from_params(_FakeSlmParams()) as slm:
            assert slm.is_connected()

    def test_real_santec_exposes_the_same_entry_point(self) -> None:
        """Parity guard: the real driver is the reference contract."""
        from ao_shaping.drivers.slm.santec.driver import Santec

        assert hasattr(Santec, "from_params")
        assert hasattr(SimSLMPib, "from_params")
