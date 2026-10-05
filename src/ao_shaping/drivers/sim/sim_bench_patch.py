"""把某个 SLM 优化器模块的 ``Santec`` 绑定改道到模拟 SLM。

每个 SLM-Zernike 整形优化器都通过在调用时从一个模块级全局 ``Santec`` 名字解析出的
``Santec.from_params(config.slm)`` 构造设备。正是这层间接让整个环路能跑在 2f 傅里叶
数字孪生上而非硬件上, 而有三处各自在手动利用它:

* ``runner_common.patch_sim_square_shaping`` -> ``slm_square_shaping``
* ``scripts/slm_pib_sim_run.py::_patch_santec`` -> ``slm_zernike_pib``
* ``tests/.../test_slm_zernike_objectives_sim.py::_patch_slm`` -> 两者

这种重复并非无害, 因为**每处打的模块集合都不一样**。于是端到端离线测试测的是
``slm_zernike_shaping``, 而生产调用方实际使用的引擎 ``slm_zernike_pib`` 完全没有
离线覆盖。改成一个接受"要打哪些模块"这一显式、可检查参数的函数, 覆盖面就成了
参数本身。

注册 ``"sim"`` 相机类型与重置共享光学状态这两件事刻意**不在**这里做: 各调用方的意图
不同 (``slm_pib_sim_run`` 那套测试脚手架需要一个感知干扰的 ``reset_system`` 包装),
所以它们留在调用方。

分层 (Layering): 本模块位于 ``drivers/sim``, 且不从 ``optimizer`` import 任何东西 ——
目标模块
是以参数传入的。生产优化器代码从不 import 本模块, 因此硬件路径保持其惰性、无仿真的
导入图 (见 ``sim/AGENTS.md`` 以及"硬件包不得 import 仿真包"这条仓库规则)。
"""

from __future__ import annotations

from types import ModuleType

from ao_shaping.drivers.sim.slm_pib_sim import SimSLMPib


def install_sim_slm(*optimizer_modules: ModuleType) -> None:
    """把各模块的 ``Santec`` 名字指向 :class:`SimSLMPib`。

    Args:
        *optimizer_modules: 需要替换其 ``Santec`` 全局的模块, 例如
            ``slm_zernike_pib``。它们每一个都必须已经 import 了 ``Santec``;
            没有 import 的模块属于调用方 bug。

    Raises:
        AttributeError: 给定的模块没有 ``Santec`` 全局。若不做这项检查, ``setattr``
            会欣然*创建*该属性, 补丁看上去生效了, 而被测模块仍会去打开真实硬件 ——
            这正是这层间接所要避免的失败。报错信息会指出是哪个模块。
    """
    if not optimizer_modules:
        raise ValueError(
            "install_sim_slm() needs at least one optimizer module to patch; "
            "an empty call would silently leave the real Santec in place"
        )
    for module in optimizer_modules:
        if not hasattr(module, "Santec"):
            raise AttributeError(
                f"{module.__name__!r} has no 'Santec' global to patch. The "
                "optimizer resolves its SLM from that module-level import; "
                "pass the module that performs that import (e.g. "
                "ao_shaping.optimizer.wfless.slm_zernike_pib), not its "
                "transitive dependency."
            )
    for module in optimizer_modules:
        module.Santec = SimSLMPib  # type: ignore[attr-defined]


__all__ = ["SimSLMPib", "install_sim_slm"]
