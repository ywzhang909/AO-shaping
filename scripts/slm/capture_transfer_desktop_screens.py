"""离屏截取「迁移学习闭环整形」页的三张手册配图。

为什么要有它
------------
用户手册要展示这一页**真实跑起来之后**长什么样: 表单填的是哪几个值、设备自检
写了什么、⑤ 进度表里的阶段与耗时是多少。手工截屏做不到 —— 它要求现场有一台开着
的相机和 SLM, 而写进手册的图又不该随某次现场状态漂移。所以这里用离屏后端把页面
渲出来, 数据全部从一次**已经跑完**的实验会话目录里注入。

注入的都是「页面自己的路径」, 不是手工 ``setText``:

* 表单走公开的 :meth:`~ao_shaping.gui.slm.transfer_desktop.page.TransferPage.set_camera`
  / ``set_shapes`` / ``set_period``;
* 零级读数走页面自己的结果槽 ``_on_zero_done`` (因此 ② 的说明、目标边长上限、
  状态行都是页面的真实渲染结果);
* 自检结论走 ``_on_check_done``;
* 进度走 ``_refresh_progress()``, 它读 ``journal.json`` 与各阶段产物
  (:func:`~ao_shaping.gui.slm.transfer_progress.collect_progress`)。

只伪造一个环境事实: torch 已安装
---------------------------------
本机 ``.venv`` 里没有 torch, 而 :func:`~ao_shaping.gui.slm.transfer_desktop.params.validate_form`
的第一道闸就是 ``import torch`` —— 不处理的话 ③ 校验区只会显示一行「需要 torch
(未安装)」的红字, 图就废了。所以这里把
:func:`~ao_shaping.gui.slm.transfer_desktop.params._torch_missing` 换成一个「没缺」
的返回值, 让**真正的** ``validate_form`` 跑后面的几何 / 形状校验
(``preflight.check_parameters`` 是纯函数, 只需 numpy)。校验结论本身是真算的:
图上那条「参数自检通过」是 900 px 边长、128 px 载波周期真过了 preflight 的结果,
不是写死的字。

两种模式: 仿真 (默认) 与实验系统 (``--live``)
----------------------------------------
页面里只有两个入口会真正读写设备: 「读取零级」的 ``zero_order.read_zero_order``
与「设备自检」的 ``hardware_check.check_hardware``。

* 默认 (仿真): 在实例化页面**之前**把这两个函数换成直接抛
  :class:`AssertionError` 的桩 —— 于是「这次渲染没有碰过硬件」不是一句声明,
  而是一条被断言保护的事实: 任何一条代码路径想开相机或探 SLM, 脚本当场就炸。
  零级读数与自检结论用已有实验会话里的值走页面的结果槽上屏。
* ``--live`` (实验系统): 不装桩, 直接点页面自己的「读取零级」与「设备自检」,
  等后台任务把**真实读数 / 真实自检结论**送回页面再抓图。这会短暂占用
  SLM + CCD (读零级开关一次相机、自检枚举相机并探一次 SLM), 全程只读。

输出
----
固定三张, 落在 ``docs/GUI/slm/transfer_desktop/figures/`` (手册直接引用)::

    gui_console.png    整窗 (含左侧导航栏) 在页 1 的状态
    gui_form.png       左侧表单列 ① 设备 / ② 目标与载波 / ③ 校验 / ④ 运行
    gui_progress.png   右侧 ⑤ 进度 (当前阶段 + 进度条 + 计数 + 阶段表)

``--replay`` 模式不渲染页面, 只离屏打开归档回放弹窗 (递归扫描整个 debug 根
``data/debug/slm_transfer/`` 下的所有历史 ``*.pkl``, 不碰设备),
输出到 ``docs/GUI/slm/transfer_desktop/``。可用 ``--debug-root`` 改扫描根::

    gui_archive_replay.png          预选会话第 1 块 capture (CCD + 剖面 + 相位预览, 左侧为归档列表)
    gui_archive_replay_middle.png   中间某块 (证明切块真的换了画面)

脚本自己对每张图做自检 (尺寸、字节数、灰度标准差) 并把结论打出来, 任何一张像
空白图 (std 过低) 或尺寸过小都以非零退出码结束 —— 手册图不能悄悄变成一片白。

用法::

    python scripts/slm/capture_transfer_desktop_screens.py
    python scripts/slm/capture_transfer_desktop_screens.py --replay
    python scripts/slm/capture_transfer_desktop_screens.py --sim-live

``--sim-live`` 额外抓一张 ⑥ 实时画面 (``gui_live_closed_loop_sim.png``)。帧不是画
出来的: 直接读实验会话归档 ``anchored/capture_*.npz`` 的 ``raw_frames`` (与回放弹窗
同源), 走 ``block_frame_payload`` → ``LiveView.apply_payload`` 这条**页面真正的消费
路径** 上屏。``on_frame`` 是每块一次、由后台会话回调的, 而这里不跑会话, 所以直接喂
负载 —— 被验证的是显示链路, 不是采集。
"""

from __future__ import annotations

import os

# 必须在任何 PySide6 import 之前: 无显示环境 (CI / 无桌面会话) 一律走离屏后端。
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import argparse
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from loguru import logger
from PIL import Image
from PySide6.QtGui import QFont, QFontDatabase, QFontInfo
from PySide6.QtWidgets import (
    QApplication,
    QGroupBox,
    QLabel,
    QScrollArea,
)

from ao_shaping.gui.ccd.beam_quality.shell import DesktopShellApp
from ao_shaping.gui.slm.transfer_desktop import params

__all__ = ["main"]

#: 实验那次运行 (``--execute`` 跑完 11 个阶段) 的会话目录 —— 表单与进度的唯一口径。
SESSION = ROOT / "data" / "debug" / "slm_transfer" / "physical_20261102c"

#: 图落在这里; 手册 ``docs/GUI/slm/transfer_desktop/`` 按文件名引用。
FIGURES = ROOT / "docs" / "GUI" / "slm" / "transfer_desktop" / "figures"

#: 归档回放截图落在手册目录本身 (不是 ``figures/``), 文件名见 ``--replay``。
REPLAY_DIR = FIGURES.parent

#: 外壳窗口尺寸 —— 与 :class:`~ao_shaping.gui.ccd.beam_quality.shell.DesktopShellApp`
#: 的默认尺寸一致 (1560x940), 图与操作员看到的比例相同。
WINDOW = (1560, 940)

#: 归档回放弹窗的离屏尺寸 (与弹窗默认一致)。
REPLAY_WINDOW = (1360, 860)

#: 自检项的最小尺寸与灰度标准差 (防全空白)。帧图 (gui_live_*) 的宽小于 300
#: (它是弹窗/右栏的一部分), 所以只卡灰度标准差 —— 一片全黑同样会露馅。
_MIN_SIDE = 300
_MIN_STD = 8.0

#: 中文字符集, 用来挑出页面里的中文标签做「不是方框」的旁证。
_CJK = re.compile(r"[\u4e00-\u9fff]")

#: 界面字体的候选家族 (按优先级)。前三个是 Windows 自带, 后面是常见 Linux 打包。
_CJK_FAMILIES: tuple[str, ...] = (
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "Noto Sans CJK SC",
    "WenQuanYi Zen Hei",
)

#: 与上面一一对应的字体文件。离屏后端的字体库是空的, 只能按路径装。
_CJK_FONT_FILES: tuple[str, ...] = (
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
)

#: 界面字号 (pt)。与 Windows 上原生控件默认的 9 pt 一致。
_CJK_POINT_SIZE = 9


def _quiet_logs() -> None:
    """把 loguru 压到 WARNING: 驱动的模块级 DEBUG (相机注册……) 与本脚本的结论无关。"""
    logger.remove()
    logger.add(sys.stderr, level="WARNING")


def _forbid_hardware() -> None:
    """把两个只读硬件入口换成断言 —— 这次渲染不许碰任何设备。"""

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("截图脚本不得触碰硬件")

    from ao_shaping.gui.slm.transfer_desktop import hardware_check, zero_order

    # 页面里的调用点都是**惰性** import 再取属性 (``check_task`` / ``_ZeroOrderTask``
    # 各自在 ``run`` 里 import), 所以换模块属性就足以挡住它们。
    zero_order.read_zero_order = _boom
    hardware_check.check_hardware = _boom


def _pretend_torch_installed() -> None:
    """只伪造「torch 已安装」这一个环境事实, 让 ``validate_form`` 跑到真校验。

    校验链: torch → 相机类型 → 形状 → ``preflight.check_parameters``。跳过第一
    道闸, 后面三道照常跑, 所以 ③ 区的结论是真算的。
    """
    params._torch_missing = lambda: None


def _apply_font(app: QApplication) -> str:
    """让界面用上带中文字形的字体, 返回实际解析到的家族。

    ``QT_QPA_PLATFORM=offscreen`` 的字体库在 Windows 上是**空的**
    (``QFontDatabase.families() == []``), 于是每个汉字都渲成方框 —— 只有显式
    ``addApplicationFont`` 装一个字体文件, 离屏渲染才有字形可查。所以这里先看
    字体库有没有中文家族, 没有就从系统字体目录里装一个, 再把它设成界面字体。
    """
    families = set(QFontDatabase.families())
    if not families & set(_CJK_FAMILIES):
        for candidate in _CJK_FONT_FILES:
            path = Path(candidate)
            if path.exists():
                QFontDatabase.addApplicationFont(str(path))
        families = set(QFontDatabase.families())
    for family in _CJK_FAMILIES:
        if family in families:
            app.setFont(QFont(family, _CJK_POINT_SIZE))
            break
    return QFontInfo(app.font()).family()


def _inject_form(page) -> None:
    """表单填成实验那次运行的值 (``session.json`` 的 ``settings``)。"""
    page.set_camera("daheng")
    page._cam_id.setValue(0)
    page._slm_number.setValue(1)
    page.set_shapes(("square", "rectangle", "triangle"))
    page.set_period(128.0)


def _inject_zero_order(page) -> None:
    """零级读数走页面自己的结果槽 —— 上限、② 说明、状态行都由页面渲染。

    顺序要紧: 目标边长控件的上限初始是保守值 400 px, 直接把值写成 900 会被
    clamp 成 400; 先送读数把上限抬到实测的 1858 px, 再写 900 才是真实通路。
    """
    from ao_shaping.gui.slm.transfer_desktop.zero_order import ZeroOrderReading

    page._on_zero_done(
        ZeroOrderReading(
            center=(1075, 939),
            max_side_px=1858,
            exposure_ms=0.5,
            peak=1653.0,
            frame_shape=(2048, 2048),
        )
    )
    page._side.setValue(900.0)


def _inject_hardware_report(page) -> None:
    """自检结论走页面自己的结果槽 (行文案是真实语义, 不含序列号与 IP)。"""
    from ao_shaping.gui.slm.transfer_desktop.hardware_check import (
        CheckItem,
        HardwareReport,
    )

    # 与 ``_run_check`` 一致: 结论是「对着这组设备参数」跑出来的, 不该标成过期。
    page._check_key = page._device_key()
    page._on_check_done(
        HardwareReport(
            items=(
                CheckItem("大恒相机", "ok", "已枚举到 1 台；打开 → 抓帧 → 释放完成"),
                CheckItem("SLM", "ok", "SLM #1 探针通过（已释放）"),
                CheckItem(
                    "MiiCam",
                    "warn",
                    "未枚举到相机（迁移学习闭环只用大恒）",
                    required=False,
                ),
                CheckItem(
                    "曝光上限",
                    "warn",
                    "AO_DAHENG_MAX_EXPOSURE_MS 未配置（自检只读，不改曝光上限）",
                    required=False,
                ),
            )
        )
    )


def _wait_until(app, predicate, *, timeout: float, what: str) -> None:
    """泵 Qt 事件直到 ``predicate()`` 为真 (后台任务的结果靠排队信号送回 UI 线程)。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            app.processEvents()
            return
        time.sleep(0.05)
    raise SystemExit(f"等待「{what}」超时（{timeout:.0f} s）")


def _read_hardware_live(app, page) -> None:
    """实验系统: 点页面自己的两个只读入口, 等真实读数 / 真实自检结论回到页面。

    这里不注入任何伪造数据 —— ``page._on_zero_done`` / ``page._on_check_done``
    收到的是后台任务从 SLM + CCD 读回来的结果。
    """
    page._read_zero_order()
    _wait_until(app, lambda: not page._zero_reading, timeout=240, what="读取零级")
    if page._zero_state is None or page._zero_state[0] != "ok":
        raise SystemExit(f"读取零级失败：{page._zero_state}")
    # 上限已按实测放宽; 再写目标边长 (超过上限会被页面 clamp, 那是物理事实)。
    page._side.setValue(900.0)

    page._run_check()
    _wait_until(app, lambda: not page._checking, timeout=240, what="设备自检")
    heading = page._check_heading or ("", "")
    if heading[0] != "ok":
        raise SystemExit(f"设备自检未通过：{heading[1]}")


def _inject_progress(page) -> None:
    """进度面板读真实会话目录, 状态行写成那次运行结束后的样子。

    ``_settle`` 跑完时写的就是 ``会话完成：<会话目录>``; 这里多一句「本次会话
    已完成」, 手册要说明的是「这一页在会话结束后的样子」。
    """
    page._session_dir = SESSION
    # 与 ``_start_run`` 同一条文案: 运行目录确定后就写进 ④ 运行区。
    page._session_label.setText(f"会话目录: {SESSION}")
    page._refresh_progress()
    page._revalidate()
    page._set_run_state("complete", f"本次会话已完成：{SESSION}")


def _grab(widget, name: str, folder: Path = FIGURES) -> str:
    """抓一张 PNG 并做空白 / 尺寸自检, 返回一行结论。"""
    path = folder / name
    if not widget.grab().save(str(path), "PNG"):
        raise SystemExit(f"无法写出 {path}")
    with Image.open(path) as image:
        width, height = image.size
        grayscale = np.asarray(image.convert("L"), dtype=np.float64)
    size = path.stat().st_size
    std = float(grayscale.std())
    line = f"{name}: {width}x{height} px / {size} B / std={std:.1f}"
    if min(width, height) <= _MIN_SIDE:
        raise SystemExit(f"{line} —— 尺寸不足 {_MIN_SIDE}x{_MIN_SIDE}, 图是空的")
    if std <= _MIN_STD:
        raise SystemExit(f"{line} —— 灰度标准差过低, 图几乎是一片纯色")
    return line


def _chinese_labels(page) -> tuple[int, int]:
    """页面里的中文标签: (个数, 最小 ``sizeHint().width()``)。

    OCR 在离屏环境里不可行, 所以「中文没渲成方框」只能旁证: 只要中文标签有
    正的尺寸提示宽度, 说明字体给了这些字形一套实际度量。
    """
    widths = [
        label.sizeHint().width()
        for label in page.findChildren(QLabel)
        if _CJK.search(label.text() or "")
    ]
    return len(widths), (min(widths) if widths else 0)


def _form_host(page):
    """左侧表单列的宿主控件 (①-④ 都在它里面, 抓它才不丢区块)。"""
    scroll = page.findChildren(QScrollArea)[0]
    host = scroll.widget()
    # 一屏放不下时 ``widgetResizable`` 会把 host 撑到内容高度, ``grab`` 不受
    # viewport 裁剪, 所以抓 host 拿到的是完整一列。
    return host if host is not None else scroll


def _progress_box(page) -> QGroupBox:
    """右侧 ⑤ 进度分组框 (按标题找, 不依赖控件层级)。"""
    for box in page.findChildren(QGroupBox):
        if box.title().startswith("⑤"):
            return box
    raise SystemExit("页面上找不到 ⑤ 进度分组框")


def _live_box(page) -> QGroupBox:
    """右侧 ⑥ 实时画面分组框 (按标题找, 不依赖控件层级)。"""
    for box in page.findChildren(QGroupBox):
        if box.title().startswith("⑥"):
            return box
    raise SystemExit("页面上找不到 ⑥ 实时画面分组框")


def _inject_live_frame(page) -> None:
    """``--sim-live``: 用实验会话归档的真实 raw 帧喂出 ⑥ 实时画面。

    后端 ``validate.block_frame_payload`` 就是会话每块发给 GUI 的负载构造器, 这里
    拿同一构造器 + 同一批数据走 ``apply_payload``, 保证图上就是真实链路渲染的样子。
    """
    from ao_shaping.tools.slm.transfer.validate import block_frame_payload

    archive = SESSION / "anchored"
    files = sorted(archive.glob("capture_*.npz"))
    if not files:
        raise SystemExit(f"没有可用的采集归档: {archive}")
    # 取中间一块: 它确有前后文, 进度条与倒计时都不会停在 0。
    path = files[len(files) // 2]
    with np.load(path) as data:
        raw = data["raw_frames"]
        xy = data["camera_xy"] if "camera_xy" in data.files else None
    total = len(files)
    payload = block_frame_payload(
        frame=raw[-1],
        xy=xy,
        shape="triangle",
        block=len(files) // 2,
        block_index=len(files) // 2 + 1,
        blocks_total=total,
        iteration=12,
        iterations_total=12,
        remaining=12,
        label="验收采集（最后一帧）",
    )
    page._live_view.apply_payload(payload)


def _render_replay(app: QApplication, root: Path) -> int:
    """``--replay``: 只开归档回放弹窗, 用整个 debug 根的历史 pkl 离屏上屏。

    不建页面、不做硬件桩: 弹窗本身递归扫描 ``root`` 下所有 ``*.pkl`` (只读, 与
    相机 / SLM 驱动无 import 关系), 左侧列表即真实历史全量。``preselect=SESSION``
    把预选落到这次实验会话的 ``capture_records.pkl``, 由它带出 CCD 三视图。
    抓两块 (第 1 块 + 中间一块), 后者证明切块确实换了画面。
    """
    from ao_shaping.gui.slm.transfer_desktop.archive_replay import (
        ArchiveReplayDialog,
    )

    dialog = ArchiveReplayDialog(root, preselect=SESSION)
    dialog.resize(*REPLAY_WINDOW)
    dialog.show()
    for _ in range(3):
        app.processEvents()

    count = dialog.capture_count()
    if count == 0:
        raise SystemExit(f"归档列表为空或预选归档没有记录: {root}")
    targets = [(0, "gui_archive_replay.png")]
    if count > 1:
        targets.append((count // 2, "gui_archive_replay_middle.png"))

    lines = []
    for index, name in targets:
        dialog.set_index(index)
        for _ in range(2):
            app.processEvents()
        lines.append(_grab(dialog, name, folder=REPLAY_DIR))
    print("\n".join(lines))
    print(
        f"归档共 {dialog.archive_count()} 个 pkl; 当前归档 {count} 条记录, 已截下标 {[index for index, _ in targets]}"
    )
    print(f"提示行: {dialog.message_text()!r}")
    dialog.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    """渲染 → (仿真注入 | 实验系统读取) → 抓图 → 自检; 返回进程退出码。"""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--live",
        action="store_true",
        help="实验系统模式: 点页面的「读取零级」「设备自检」用真实 SLM + CCD (只读), 不注入伪造读数",
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="归档回放模式: 只离屏渲染归档回放弹窗 (真实历史 pkl, 不碰设备)",
    )
    parser.add_argument(
        "--debug-root",
        default=str(SESSION.parent),
        help="归档回放扫描根 (默认 data/debug/slm_transfer, 递归所有历史 pkl)",
    )
    parser.add_argument(
        "--sim-live",
        action="store_true",
        help="额外抓一张 ⑥ 实时画面 (gui_live_closed_loop_sim.png, 用实验归档 raw 帧)",
    )
    args = parser.parse_args(argv)

    _quiet_logs()
    FIGURES.mkdir(parents=True, exist_ok=True)
    SESSION_JSON = SESSION / "session.json"
    if not SESSION_JSON.exists():
        raise SystemExit(f"会话目录里没有 session.json: {SESSION_JSON}")

    app = QApplication.instance() or QApplication([])
    app.setApplicationName("AO-Shaping 截图")
    family = _apply_font(app)
    app.setStyle("Fusion")

    if args.replay:
        # 回放模式不建页面, 也不需要硬件桩 / torch 事实 —— 那些是页面路径的事。
        REPLAY_DIR.mkdir(parents=True, exist_ok=True)
        print(f"字体: {family}")
        return _render_replay(app, Path(args.debug_root))

    # 顺序: 仿真模式先说好「不许碰硬件」, 再伪造 torch 事实, 最后才让页面存在。
    # 实验系统模式不装桩 —— 硬件调用留给页面自己的按钮路径。
    if not args.live:
        _forbid_hardware()
    _pretend_torch_installed()

    shell = DesktopShellApp()
    shell.resize(*WINDOW)
    shell.show()
    for _ in range(3):
        app.processEvents()

    # 页 1 惰性创建: 切过去之后要泵一遍事件, 否则布局还没算尺寸。
    shell.select_page(1)
    for _ in range(3):
        app.processEvents()
    page = shell._transfer_page
    if page is None:
        raise SystemExit("页 1 没有建出来")

    _inject_form(page)
    if args.live:
        _read_hardware_live(app, page)
    else:
        _inject_zero_order(page)
        _inject_hardware_report(page)
    _inject_progress(page)
    for _ in range(3):
        app.processEvents()

    lines = [
        _grab(shell, "gui_console.png"),
        _grab(_form_host(page), "gui_form.png"),
        _grab(_progress_box(page), "gui_progress.png"),
    ]
    if args.sim_live:
        REPLAY_DIR.mkdir(parents=True, exist_ok=True)
        _inject_live_frame(page)
        for _ in range(3):
            app.processEvents()
        lines.append(
            _grab(_live_box(page), "gui_live_closed_loop_sim.png", folder=REPLAY_DIR)
        )

    errors, warnings = page.current_problems()
    label_count, min_label_width = _chinese_labels(page)
    print("\n".join(lines))
    print(f"字体: app.font()={app.font().family()!r} 解析到 {family!r}")
    print(
        f"中文标签: {label_count} 个, 最小 sizeHint 宽度 {min_label_width} px"
        " (0 就意味着字形没有度量)"
    )
    print(f"表单值: {page.form_values()}")
    print(f"current_problems: {len(errors)} error / {len(warnings)} warning {errors}")
    print(f"check_report: {page.check_report()[:80]!r}")
    print(
        "进度条: value={} range={}..{}".format(
            page._bar.value(), page._bar.minimum(), page._bar.maximum()
        )
    )
    print(f"状态行: {page._status_message}")
    print(f"开始运行可点: {page.run_button().isEnabled()}")

    page.close()
    shell.close()
    app.processEvents()
    if errors or min_label_width <= 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
