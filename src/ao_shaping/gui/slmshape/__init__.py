"""闭环整形页面的**共享**构件（被两个以上页面消费才会搬进来）。

为什么是独立包而**不是**放进 ``gui/slm/shaping_controls.py``
------------------------------------------------------------
``diff_opt_controller`` 与 ``multi_slm_controller`` 之间存在 60+ 符号的循环
import（前者要设备枚举与侧栏，后者要 ``render_closed_loop_shaping``），而
``shaping_controls`` 已经 import ``multi_slm_controller``。共享件必须是无上游
页面的**叶子**，否则「谁 import 谁」会立刻又成一团。

模块（组织方式同 :mod:`ao_shaping.tools.slm.panel`)
--------------------------------------------------
- ``layout`` —— 页面渲染目标 :class:`Layout` 与侧栏「页面布局」选择器：把主体
  画在主区，或画进 ``st.status`` 折叠面板。Streamlit 的渲染位置由调用点的容器
  决定，所以它是**显式参数**而不是全局状态。
- ``connect`` —— 首次进入时枚举设备（每轮 rerun 重扫等于每次点按钮白等十几秒）、
  已连接 SLM 编号、无设备时的可执行提示。
- ``method_panel`` —— 「参数 → 校验 → 调试开关 + 运行按钮」这一段。顺序是契约：
  校验必须早于按钮，按钮才能 ``disabled``。
- ``trace_plot`` —— matplotlib 的 Agg 后端 + CJK 字体 + 「画完即关」的
  :func:`~ao_shaping.gui.slmshape.trace_plot.show`，是本包唯一的绘图入口。
- ``trace_tabs`` —— 一张轨迹图一块：CCD 迭代过程 / 拟合轨迹 / 模型预测 /
  逐块验收。每个模块的输入都是 :func:`ao_shaping.gui.slm.transfer_progress.
  collect_traces` 视图里的一小块，**只画、不读盘、不重算指标**。

本包有意**不做** eager re-export：``trace_plot`` 在模块作用域把 matplotlib 切到
Agg 后端，包级导入会把这个副作用带给只想要 ``layout`` 的调用方（以及那些在
``plt`` 上做替身的测试）。
"""

from __future__ import annotations

__all__: list[str] = []
