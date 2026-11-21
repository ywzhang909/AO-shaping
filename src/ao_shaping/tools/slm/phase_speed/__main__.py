"""``python -m ao_shaping.tools.slm.phase_speed`` 命令行入口.

参数类 :class:`SpeedConfig` 是本工具 CLI 的**唯一**选项真源; 字段顺序即
``--help`` 顺序. 默认值只写在 dataclass 字段上, ``with_params`` 负责注入
click 选项 (在 ``option(...)`` 里再写 ``default=`` 会 ``TypeError``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import click

from ao_shaping.drivers.ccd.common import CAMERA_TYPE_CHOICES
from ao_shaping.utils.cli.params import option, with_params

from .measure import run_speed_measurement


@dataclass
class SpeedConfig:
    """相位变化速度测量 CLI 参数.

    关键默认值说明:
        * ``gray-a=0`` / ``gray-b=993``: 1064nm 台的 2π 典型灰度读数 (设备
          动态查询为准), 覆盖一个完整 2π 相位周期, 平场强度几乎不变 ⇒
          用 ``--pattern blaze`` 才能看到 0 级调制度变化.
        * ``n-frames=60``: 逐帧记录的默认长度; ``fps`` 由实测帧间隔推算.
    """

    out_dir: Annotated[
        Path,
        option("--out-dir", type=click.Path(path_type=Path), help="产物目录 (默认 data/phase_speed)"),
    ] = Path("data/phase_speed")
    slm_number: Annotated[int, option("--slm-number", help="SLM 设备编号 (默认 1)")] = 1
    wavelength_nm: Annotated[
        int, option("--wavelength-nm", help="SLM 工作波长 nm (默认 1064)")
    ] = 1064
    cam_type: Annotated[
        str,
        option(
            "--cam-type",
            type=click.Choice(CAMERA_TYPE_CHOICES),
            help="相机后端类型 (默认 daheng)",
        ),
    ] = "daheng"
    cam_id: Annotated[int, option("--cam-id", help="相机编号 (默认 0)")] = 0
    exposure_ms: Annotated[
        float, option("--exposure-ms", help="曝光时间 ms (默认 5.0)")
    ] = 5.0
    gray_a: Annotated[int, option("--gray-a", help="阶跃前平场灰度 (默认 0)")] = 0
    gray_b: Annotated[int, option("--gray-b", help="阶跃后平场灰度 (默认 993)")] = 993
    n_frames: Annotated[
        int, option("--n-frames", help="阶跃后采集帧数 (默认 60)")
    ] = 60
    settle_s: Annotated[
        float,
        option("--settle-s", help="写入相位后的额外稳定等待秒数 (默认 0.0, 即不额外等)"),
    ] = 0.0
    pattern: Annotated[
        str,
        option(
            "--pattern",
            type=click.Choice(["flat", "blaze"]),
            help="相位图案: flat=平场两灰度 (0级基本不变); blaze=闪耀光栅深度阶跃 (0级亮度变化)",
        ),
    ] = "flat"
    period_px: Annotated[
        float, option("--period-px", help="blaze 图案的光栅周期 px (默认 16.0)")
    ] = 16.0
    save_frames: Annotated[
        bool,
        option("--save-frames", is_flag=True, help="保存原始帧为 npz (默认关闭)"),
    ] = False
    tag: Annotated[
        str,
        option("--tag", help="产物文件前缀标签 (默认 phase_speed)"),
    ] = "phase_speed"
    baseline_window: Annotated[
        int, option("--baseline-window", help="基准电平用起始帧数 (默认 3)")
    ] = 3
    settle_window: Annotated[
        int, option("--settle-window", help="终值电平用末尾帧数 (默认 3)")
    ] = 3
    settle_fraction: Annotated[
        float, option("--settle-fraction", help="稳定带占阶跃幅度比例 (默认 0.1)")
    ] = 0.1
    response_fraction: Annotated[
        float, option("--response-fraction", help="响应判据占阶跃幅度比例 (默认 0.9)")
    ] = 0.9


@click.command()
@with_params(SpeedConfig, kw_name="config")
def main(config: SpeedConfig) -> None:
    """测量 SLM 相位阶跃后远场 0 级强度随时间的变化, 提取响应/稳定时间.

    ⚠️ 本工具测的是 SLM+CCD **组合系统**的响应: 时间分辨率受相机帧率限制
    (结果报告实测 fps), 不是孤立的 SLM 像素上升时间; 快于单帧间隔的响应无法
    分辨. t=0 定义为写入新相位后的第一帧, 相机曝光/读出延迟已包含在内.

    实用提示: 平场 piston 阶跃 (如 0→993) 在 0 级光强上几乎不可见; 若指标无
    可分辨的跃迁, 结果里响应/稳定时间返回 None. 此时请用 ``--pattern blaze``
    改用闪耀光栅深度阶跃, 让 0 级亮度随相位变化.
    """
    run_speed_measurement(config)


if __name__ == "__main__":
    main()
