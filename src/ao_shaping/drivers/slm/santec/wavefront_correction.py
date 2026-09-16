"""波前误差矫正工具类

提供矫正CSV文件加载、灰度矫正图计算（含基于前后数组的异常点检测与剔除、
余弦拟合与相位提取、矫正映射图计算），以及矫正应用功能，供SLM驱动类调用。
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Callable

import numpy as np
from loguru import logger

from ao_shaping.drivers.slm.santec.slm200_constants import (
    PANEL_RES,
    get_max_grayscale,
)


class WavefrontCorrection:
    """波前误差矫正工具类

    加载矫正CSV文件，通过 calc() 函数（默认实现进行数据验证、
    异常点检测与剔除、统计信息记录）计算灰度矫正映射图，
    并提供 map_error() 供SLM驱动调用。

    Attributes:
        csv_path: 矫正CSV文件路径
        correction_map: 计算后的矫正映射图 (None 表示尚未计算)
        panel_resolution: SLM面板分辨率 (width, height)
    """

    def __init__(
        self,
        csv_path: str | Path | None = None,
        calc_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        """初始化波前误差矫正器

        文件路径的有效性（None、空字符串、文件不存在）在此处统一判断。
        无效路径将设置 csv_path=None，调用方可通过 is_valid 属性查询。

        Args:
            csv_path: 矫正CSV文件路径，None/空/不存在都属于无效
            calc_fn: 自定义 calc 函数，接收原始数据 (np.ndarray, float64)
                     返回矫正映射图。默认实现进行异常点检测、剔除与统计。
        """
        self.csv_path: Path | None = None
        self._calc_fn = calc_fn
        self._raw_data: np.ndarray | None = None
        self._correction_map: np.ndarray | None = None
        self._panel_resolution: tuple[int, int] = (1920, 1200)

        if csv_path is not None and csv_path != "":
            p = Path(csv_path)
            if p.exists():
                self.csv_path = p
            else:
                logger.warning("矫正文件不存在，跳过: {}", p)

    # ── 属性 ──────────────────────────────────────────

    @property
    def is_valid(self) -> bool:
        """文件路径是否有效（csv_path 不为 None 且文件存在）"""
        return self.csv_path is not None

    @property
    def correction_map(self) -> np.ndarray | None:
        """矫正映射图 (None 表示尚未调用 calc())"""
        return self._correction_map

    @property
    def raw_data(self) -> np.ndarray | None:
        """原始CSV加载数据 (None 表示尚未调用 load_csv())"""
        return self._raw_data

    # ── 公开方法 ──────────────────────────────────────

    def load_csv(self, skiprows: int = 1, delimiter: str = ",") -> None:
        """从CSV文件加载原始矫正数据

        Santec 官方 Wavefront_correction_Data CSV 格式:
        第一行为列标题 (Y/X,0,1,2,...)，第二列起为行索引，
        数据区域为灰度偏移值。

        Args:
            skiprows: 跳过的行数（标题行），默认 1
            delimiter: 分隔符，默认逗号
        """
        if self.csv_path is None:
            raise RuntimeError("未设置有效的矫正文件路径")

        raw = np.loadtxt(self.csv_path, delimiter=delimiter, skiprows=skiprows)
        # 跳过第一列（行索引列），CSV 存储的是 uint16 灰度偏移值
        if raw.ndim == 2:
            raw = raw[:, 1:]

        self._raw_data = raw.astype(np.uint16)

        logger.info(
            f"已加载矫正数据: {self.csv_path.name}, "
            f"形状: {self._raw_data.shape}, "
            f"范围: [{np.nanmin(self._raw_data):.1f}, {np.nanmax(self._raw_data):.1f}]"
        )

    def load(
        self,
        panel_resolution: tuple[int, int] = (1920, 1200),
        outlier_threshold: float = 3.0,
        median_filter_size: int = 5,
        max_grayscale: int = 1023,
        measurement_gray: int | None = None,
    ) -> None:
        """加载CSV并计算矫正映射（load_csv + calc 一步完成）

        Args:
            panel_resolution: 面板分辨率 (width, height)
            outlier_threshold: 前后邻域比较的异常阈值 (默认 3.0)
            median_filter_size: 保留参数（向后兼容）
            max_grayscale: 最大灰度值（2π 对应的灰度级）
            measurement_gray: 测量时施加的灰度级
        """
        self.load_csv()
        self.calc(
            panel_resolution=panel_resolution,
            outlier_threshold=outlier_threshold,
            median_filter_size=median_filter_size,
            max_grayscale=max_grayscale,
            measurement_gray=measurement_gray,
        )

    def calc(
        self,
        panel_resolution: tuple[int, int] = (1920, 1200),
        outlier_threshold: float = 3.0,
        median_filter_size: int = 5,
        max_grayscale: int = 1023,
        measurement_gray: int | None = None,
    ) -> None:
        """计算矫正映射图

        默认实现流程:
          1. 复制原始数据并转为 float64
          2. 基于前后邻域比较的异常点检测与剔除
          3. 余弦拟合提取每像素相位偏移 φ
          4. 构建矫正映射图: correction = -φ·MAX_GRAY/(2π)
          5. 调整尺寸到面板分辨率

        Args:
            panel_resolution: 目标面板分辨率 (width, height)
            outlier_threshold: 前后邻域比较的异常阈值 (默认 3.0)
            median_filter_size: 保留参数，未在邻域法中使用（向后兼容）
            max_grayscale: 最大灰度值（2π 对应的灰度级）
            measurement_gray: 测量时施加的灰度级，为 None 时取 max_grayscale
        """
        if self._raw_data is None:
            raise RuntimeError("请先调用 load_csv() 加载原始数据")

        self._panel_resolution = panel_resolution

        if self._calc_fn is not None:
            result = self._calc_fn(self._raw_data)
        else:
            result = self._default_calc(
                self._raw_data,
                outlier_threshold=outlier_threshold,
                median_filter_size=median_filter_size,
                max_grayscale=max_grayscale,
                measurement_gray=measurement_gray,
            )

        self._correction_map = self.resize_to_panel(result, panel_resolution)

        logger.info(
            f"矫正映射图已计算: "
            f"形状={self._correction_map.shape}, "
            f"范围=[{np.nanmin(self._correction_map):.1f}, "
            f"{np.nanmax(self._correction_map):.1f}]"
        )

    def map_error(self, grayscale: np.ndarray, max_grayscale: int = 1023) -> np.ndarray:
        """应用误差矫正到灰度相位图

        将计算好的矫正映射图叠加到输入灰度图上（模 max_grayscale+1 环绕）。
        输入输出尺寸必须匹配。

        Args:
            grayscale: 输入灰度相位图 (uint16 或 float)
            max_grayscale: 最大灰度值（2π 对应的灰度值），默认 1023

        Returns:
            矫正后的灰度相位图 (float64)，调用者自行 .astype(np.uint16)
        """
        # 无效配置或尚未计算矫正图 → 原样返回
        if not self.is_valid or self._correction_map is None:
            return grayscale

        if grayscale.shape != self._correction_map.shape:
            logger.warning(
                f"输入灰度图尺寸 {grayscale.shape} 不匹配 "
                f"矫正图尺寸 {self._correction_map.shape}，跳过矫正"
            )
            return grayscale

        return np.mod(
            grayscale.astype(np.float64) + self._correction_map,
            max_grayscale + 1,
        )

    # ── 工厂方法 ──────────────────────────────────────

    @classmethod
    def resolve(
        cls,
        explicit_path: str | Path | None = None,
        config: dict | None = None,
        panel_resolution: tuple[int, int] = (1920, 1200),
        default_path: str | Path | None = None,
        **calc_kwargs: int,
    ) -> "WavefrontCorrection | None":
        """从优先级链解析并加载矫正数据

        优先级顺序: explicit → config['correction_csv_path'] → default_path
        跳过不存在的路径，从第一个有效路径加载并返回完整实例。
        均无效时返回 None。

        Args:
            explicit_path: __init__ 显式指定的路径（最高优先级）
            config: 配置字典（含 correction_csv_path 键）
            panel_resolution: 面板分辨率 (width, height)
            default_path: 默认备选路径（最低优先级）
            **calc_kwargs: 传递给 calc() 的额外参数

        Returns:
            已加载完成的 WavefrontCorrection 实例，或 None
        """
        candidates: list[tuple[str, Path]] = []

        if explicit_path is not None and explicit_path != "":
            candidates.append(("显式指定", Path(explicit_path)))
        if config is not None and "correction_csv_path" in config:
            candidates.append(("配置文件", Path(config["correction_csv_path"])))
        if default_path is not None:
            candidates.append(("默认路径", Path(default_path)))

        for source, path in candidates:
            if not path.exists():
                logger.debug("矫正文件不存在，跳过[{}]: {}", source, path)
                continue
            try:
                instance = cls(path)
                instance.load(panel_resolution=panel_resolution, **calc_kwargs)
                logger.info("从{}加载矫正数据: {}", source, path.name)
                return instance
            except Exception as e:
                logger.warning("从{}加载矫正数据失败 ({}): {}", source, path.name, e)

        return None

    @classmethod
    def from_file(
        cls,
        csv_path: str | Path,
        panel_resolution: tuple[int, int] = (1920, 1200),
        calc_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        outlier_threshold: float = 3.0,
        median_filter_size: int = 5,
        max_grayscale: int = 1023,
        measurement_gray: int | None = None,
    ) -> "WavefrontCorrection":
        """便捷方法：从CSV文件创建并计算矫正映射（load_csv + calc 一步完成）

        Args:
            csv_path: CSV文件路径
            panel_resolution: 面板分辨率 (width, height)
            calc_fn: 自定义 calc 函数
            outlier_threshold: 前后邻域比较的异常阈值 (默认 3.0)
            median_filter_size: 保留参数（向后兼容）
            max_grayscale: 最大灰度值（2π 对应的灰度级）
            measurement_gray: 测量时施加的灰度级

        Returns:
            已初始化的 WavefrontCorrection 实例
        """
        instance = cls(csv_path, calc_fn=calc_fn)
        instance.load_csv()
        instance.calc(
            panel_resolution=panel_resolution,
            outlier_threshold=outlier_threshold,
            median_filter_size=median_filter_size,
            max_grayscale=max_grayscale,
            measurement_gray=measurement_gray,
        )
        return instance

    # ── 静态工具方法 ──────────────────────────────────

    @staticmethod
    def resize_to_panel(
        data: np.ndarray, panel_resolution: tuple[int, int]
    ) -> np.ndarray:
        """将数据调整到面板分辨率（裁切或补零）

        若输入尺寸超过面板分辨率，从中心裁切；
        若不足，则居中补零。

        Args:
            data: 输入数组，shape (height, width)
            panel_resolution: 面板分辨率 (width, height)

        Returns:
            调整后的数组，shape (panel_resolution[1], panel_resolution[0]),
            dtype float64
        """
        target_w, target_h = panel_resolution
        h, w = data.shape

        if (h, w) == (target_h, target_w):
            return np.ascontiguousarray(data, dtype=np.float64)

        if h > target_h or w > target_w:
            # 从中心裁切
            start_y = (h - target_h) // 2
            start_x = (w - target_w) // 2
            result = data[start_y : start_y + target_h, start_x : start_x + target_w]
        else:
            # 居中补零
            result = np.zeros((target_h, target_w), dtype=np.float64)
            start_y = (target_h - h) // 2
            start_x = (target_w - w) // 2
            result[start_y : start_y + h, start_x : start_x + w] = data

        return np.ascontiguousarray(result, dtype=np.float64)

    # ── CSV I/O 静态工具 (SLM 相位/灰度 CSV 统一格式) ────

    @staticmethod
    def load_gray_from_csv(
        filepath: str | Path,
        skiprows: int = 1,
        delimiter: str = ",",
        panel_resolution: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """从CSV文件加载灰度数据 (uint16)

        CSV 格式:
        - 第1行第1列: "Y/X" (标题), 第1行第2~1921列: 0~1919 (列索引)
        - 第2~1201行第1列: 0~1199 (行索引)
        - 数据区域: 0~2^GRAY_SCALE_BITS-1 之间的整数 (最大1023)
        - 示例文件: C:\\santec\\SLM-200\\Files\\All 1023.csv

        Args:
            filepath: CSV文件路径
            skiprows: 跳过的行数，默认为1（跳过标题行）
            delimiter: 分隔符，默认为逗号
            panel_resolution: 面板分辨率 (宽, 高)，默认 (1920, 1200)

        Returns:
            灰度数据数组，shape=(1200, 1920)，dtype=uint16

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: CSV 格式校验失败（标题、尺寸、值范围）
        """
        if panel_resolution is None:
            panel_resolution = (1920, 1200)
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"灰度文件不存在: {filepath}")

        # ── 格式校验 ──────────────────────────────────────────
        # 1. 标题行首列必须为 "Y/X"
        with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
            header_line = fh.readline().strip()
        header_fields = header_line.split(delimiter)
        if not header_fields or header_fields[0].strip().upper() != "Y/X":
            raise ValueError(
                f"CSV 格式错误: 首行首列应为 'Y/X' 标题，实际为 "
                f"'{header_fields[0] if header_fields else ''}'"
            )

        # 2. 读取数据区域（跳过标题行与行索引列）
        raw = np.loadtxt(filepath, delimiter=delimiter, skiprows=skiprows)
        if raw.ndim != 2:
            raise ValueError(f"CSV 格式错误: 数据区域应为 2D 矩阵，实际维度 {raw.ndim}")
        data = raw[:, 1:]

        # 3. 尺寸校验：必须与 SLM 面板分辨率一致
        target_h, target_w = panel_resolution[1], panel_resolution[0]
        if data.shape != (target_h, target_w):
            raise ValueError(
                f"CSV 尺寸错误: 数据区域 {data.shape}，"
                f"应与 SLM 面板分辨率一致 ({target_h}, {target_w})"
            )

        # 4. 值范围校验：灰度必须在 0..1023 范围内
        gray = data.astype(np.uint16)
        if gray.max() > 1023 or gray.min() < 0:
            raise ValueError(
                f"CSV 灰度值越界: 范围 [{gray.min()}, {gray.max()}]，允许范围 [0, 1023]"
            )

        logger.info(f"已从 {filepath} 加载灰度数据，形状: {gray.shape}")
        return gray

    @staticmethod
    def csv_to_phase(
        filepath: str | Path,
        skiprows: int = 1,
        delimiter: str = ",",
    ) -> np.ndarray:
        """将CSV中的灰度矩阵（0~2^GRAY_SCALE_BITS-1）还原为弧度制相位数组。

        将CSV文件中的原始灰度值（0~1023）按设备常量 :func:`get_max_grayscale`
        转换为弧度制相位（0~2π），然后可用于 :meth:`create_phase_from_array`
        进行最终的相位→灰度转换。

        Args:
            filepath: CSV文件路径
            skiprows: 跳过的行数，默认为1（跳过标题行）
            delimiter: 分隔符，默认为逗号

        Returns:
            弧度制相位数组，dtype=float64，形状与CSV数据一致
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"相位文件不存在: {filepath}")

        phase_gray = np.loadtxt(filepath, delimiter=delimiter, skiprows=skiprows)[
            :, 1:
        ].astype(np.float64)

        phase_rad = phase_gray / 1023 * 2 * np.pi
        logger.info(
            f"CSV灰度数据已转换为弧度相位: {filepath.name}, "
            f"形状: {phase_rad.shape}, "
            f"范围: [0, 2π]"
        )
        return phase_rad

    @staticmethod
    def save_phase_to_csv(
        phase_rad: np.ndarray,
        filepath: str | Path | io.BufferedIOBase | io.TextIOBase,
        delimiter: str = ",",
        panel_resolution: tuple[int, int] | None = None,
    ) -> None:
        """将弧度制相位矩阵导出为带行列索引的 CSV。

        CSV格式与 :meth:`load_gray_from_csv` 保持一致：首行首列为 ``Y/X``，
        首列是行索引，首行其余列是列索引；数据区保存弧度制相位值。

        Args:
            phase_rad: 弧度制相位数组，shape 必须等于 ``PANEL_RES`` 的 ``(高, 宽)``
            filepath: 输出文件路径，或可写入字节/文本的流
            delimiter: CSV 分隔符，默认为逗号
            panel_resolution: 面板分辨率 (宽, 高)，默认 (1920, 1200)

        Raises:
            ValueError: 相位数组维度、尺寸或数值无效
        """
        if panel_resolution is None:
            panel_resolution = (1920, 1200)
        phase = np.asarray(phase_rad, dtype=np.float64)
        if phase.ndim != 2:
            raise ValueError(f"相位数据必须是2D数组，当前维度: {phase.ndim}")

        target_h, target_w = panel_resolution[1], panel_resolution[0]
        if phase.shape != (target_h, target_w):
            raise ValueError(
                f"相位尺寸错误: {phase.shape}，应与 SLM 面板分辨率一致 "
                f"({target_h}, {target_w})"
            )
        if not np.isfinite(phase).all():
            raise ValueError("相位数据包含 NaN 或无穷值")

        output = io.StringIO()
        output.write(
            delimiter.join(["Y/X", *(str(index) for index in range(target_w))])
        )
        output.write("\n")

        indexed_phase = np.column_stack((np.arange(target_h, dtype=np.int64), phase))
        np.savetxt(
            output,
            indexed_phase,
            delimiter=delimiter,
            fmt=["%d", *["%.12g"] * target_w],
        )
        content = output.getvalue()

        if isinstance(filepath, (str, Path)):
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        else:
            if isinstance(filepath, io.TextIOBase):
                filepath.write(content)
            else:
                filepath.write(content.encode("utf-8"))

        logger.info(
            f"弧度相位已导出: 形状={phase.shape}, "
            f"范围=[{phase.min():.6f}, {phase.max():.6f}] rad"
        )

    @staticmethod
    def correction_gray_offsets(
        phase_rad: np.ndarray,
        max_grayscale: int | None = None,
    ) -> np.ndarray:
        """弧度矫正相位 → 灰度偏移 (10 位, 加法语义, mod 环绕, 不含 shift).

        官方软件约定: **2π = max_grayscale**, 默认取设备常量
        ``slm200_constants.get_max_grayscale()`` (= 1023) —— 与
        :meth:`csv_to_phase` (``gray/1023·2π``) 与 :meth:`_default_calc`
        (``correction = -φ·MAX/(2π)``) 的 1023 满量程约定一致。
        ⚠️ **不能**用波长相关 ``two_pi_gray`` (如 532nm→998) 做满量程:
        官方软件按 1023 换算会引入 ``1023/two_pi_gray ≈ 1.025×`` 缩放误差。

        ⚠️ **不烘焙 shift**: 输入必须是面板坐标中**未平移**的矫正相位
        (Zernike 图案居中于面板几何中心); shift 是显示时参数, 由官方软件
        shift 设置或驱动 ``Santec.apply_shift`` 单独应用, 本函数不涉及任何平移。

        Args:
            phase_rad: 弧度矫正相位 (未 shift, 面板坐标), shape (H, W)
            max_grayscale: 满量程灰度 (2π 对应), None → 设备常量 1023

        Returns:
            uint16 灰度偏移, shape 同 phase_rad, 范围 0..max_grayscale
        """
        if max_grayscale is None:
            max_grayscale = get_max_grayscale()
        # 加法语义下偏移运行在循环域 (mod max_grayscale+1): 先 mod 再 rint 时
        # (max, max+1) 区间的值会被舍入到 max+1 → 溢出。故 rint 之后再 mod
        # (max+1) 把 max+1 环绕回 0, 保证输出严格落在 0..max_grayscale。
        scaled = phase_rad / (2.0 * np.pi) * max_grayscale
        return np.mod(
            np.rint(np.mod(scaled, max_grayscale + 1)),
            max_grayscale + 1,
        ).astype(np.uint16)

    @staticmethod
    def save_gray_correction_csv(
        gray_offsets: np.ndarray,
        filepath: str | Path | io.BufferedIOBase | io.TextIOBase,
        delimiter: str = ",",
        panel_resolution: tuple[int, int] | None = None,
        max_grayscale: int | None = None,
    ) -> None:
        """导出灰度偏移矫正 CSV (官方 Wavefront_correction_Data 格式, 不含 shift).

        与 :meth:`load_gray_from_csv` 完全互逆: 首行 ``Y/X,0..W-1``, 数据行
        ``行索引, g_{i,0}..g_{i,W-1}``, 数据区为 0..max_grayscale 的 uint16
        灰度偏移 (加法语义: ``displayed = mod(grayscale + corr, max_gray + 1)``)。

        ⚠️ **不烘焙 shift**: 本函数只写出逐像素偏移图 (面板坐标), **不含任何
        平移**; shift 由官方软件 / 驱动 ``Santec.apply_shift`` 在显示时单独应用。
        导出的 CSV 可直接被官方软件作为 ``Wavefront_correction_Data`` 加载
        (满量程 2π = max_grayscale, 默认设备常量 1023)。

        Args:
            gray_offsets: (H, W) 灰度偏移 (未 shift), 数值须在 0..max_grayscale
            filepath: 输出路径, 或可写入字节/文本的流
            delimiter: CSV 分隔符, 默认为逗号
            panel_resolution: 面板分辨率 (宽, 高), 默认 PANEL_RES (1920, 1200)
            max_grayscale: 满量程灰度, None → 设备常量 1023

        Raises:
            ValueError: 维度、尺寸或数值范围不合法
        """
        if panel_resolution is None:
            panel_resolution = PANEL_RES
        if max_grayscale is None:
            max_grayscale = get_max_grayscale()
        target_h, target_w = panel_resolution[1], panel_resolution[0]

        f = np.rint(np.asarray(gray_offsets, dtype=np.float64))
        if f.ndim != 2:
            raise ValueError(f"矫正灰度必须是2D数组，当前维度: {f.ndim}")
        if f.shape != (target_h, target_w):
            raise ValueError(
                f"矫正灰度尺寸必须为 ({target_h}, {target_w})，得到 {f.shape}"
            )
        if f.min() < 0 or f.max() > max_grayscale:
            raise ValueError(
                f"矫正灰度数值必须在 0..{max_grayscale}，"
                f"得到范围 [{f.min():.1f}, {f.max():.1f}]"
            )
        arr = f.astype(np.uint16)

        output = io.StringIO()
        output.write(
            delimiter.join(["Y/X", *(str(index) for index in range(target_w))])
        )
        output.write("\n")
        body = np.column_stack((np.arange(target_h, dtype=np.uint16), arr))
        np.savetxt(output, body, delimiter=delimiter, fmt="%d")
        content = output.getvalue()

        if isinstance(filepath, (str, Path)):
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        else:
            if isinstance(filepath, io.TextIOBase):
                filepath.write(content)
            else:
                filepath.write(content.encode("utf-8"))

        logger.info(
            f"灰度矫正 CSV 已导出: shape={arr.shape}, max_grayscale={max_grayscale}, "
            f"范围=[{arr.min()}, {arr.max()}], 不含 shift"
        )

    # ── 内部方法 ──────────────────────────────────────

    @staticmethod
    def _default_calc(
        raw_data: np.ndarray,
        outlier_threshold: float = 3.0,
        median_filter_size: int = 5,
        max_grayscale: int = 1023,
        measurement_gray: int | None = None,
    ) -> np.ndarray:
        """默认 calc 实现：异常点检测、余弦拟合与矫正映射图计算

        基于物理模型:
          B(g) = A·cos(2π·g/MAX_GRAY + φ_pixel) + C
        其中 B(g) 是灰度 g 下的亮度测量值，φ_pixel 是每像素的固有相位偏移。

        处理流程:
          1. 基于前后邻域比较的异常点检测与剔除（"前后数组"法）
             逐行比较每个像素与左右邻居的线性插值，残差超过阈值则标记替换
          2. 全局估计 A、C，逐像素解算相位偏移 φ
              amplitude = (max - min)/2, offset = mean
              cos(2π·g_ref/MAX_GRAY + φ) = (B - C) / A
              → φ = arccos((B - C) / A) - 2π·g_ref/MAX_GRAY
          3. 构建矫正映射图: correction = -φ·MAX_GRAY/(2π)

        Args:
            raw_data: 原始数据（uint16 或 float64），形状 (height, width) 的空间图
            outlier_threshold: 前后邻域比较的残差 / 行标准差阈值
            median_filter_size: （保留，不再用于邻域法，仅向后兼容）
            max_grayscale: 最大灰度值（2π 对应的灰度级）
            measurement_gray: 测量时施加的灰度级。为 None 时取 max_grayscale

        Returns:
            矫正映射图 (float64)，形状与 raw_data 相同
        """
        data = np.nan_to_num(raw_data).copy().astype(np.float64)
        h, w = data.shape

        # ============================================================
        # Step 1: 基于前后数组的异常检测与剔除
        #         对每个元素，比较其与左右邻居的线性插值
        # ============================================================
        orig_mean = float(np.mean(data))
        orig_std = float(np.std(data))
        orig_min = float(np.min(data))
        orig_max = float(np.max(data))

        # 构建左右邻居（首尾边界处复制自身）
        left_neighbor = np.pad(data[:, :-1], ((0, 0), (1, 0)), mode="edge")
        right_neighbor = np.pad(data[:, 1:], ((0, 0), (0, 1)), mode="edge")

        # 线性插值参考 = (left + right) / 2
        expected = (left_neighbor + right_neighbor) / 2.0

        # 残差 = |实测值 - 插值|
        residual = np.abs(data - expected)

        # 行方向局部标准差
        local_std = np.std(data, axis=1, keepdims=True) + 1e-10

        # 异常点标记（排除首尾列）
        outlier_mask = residual / local_std > outlier_threshold
        outlier_mask[:, 0] = False
        outlier_mask[:, -1] = False

        n_outliers = int(np.sum(outlier_mask))
        n_total = data.size

        if n_outliers > 0:
            data[outlier_mask] = expected[outlier_mask]
            logger.debug(
                f"矫正数据异常点（前后数组法）: {n_outliers}/{n_total} "
                f"({100.0 * n_outliers / n_total:.2f}%) "
                f"已替换为线性插值"
            )

        # ============================================================
        # Step 2: 余弦拟合 — 提取每像素相位偏移 φ
        # ============================================================
        g_ref = max_grayscale if measurement_gray is None else measurement_gray

        # 估计数据动态范围 (非异常点)
        clean_data = data.copy()
        clean_data[outlier_mask] = np.nan
        data_min = float(np.nanmin(clean_data))
        data_max = float(np.nanmax(clean_data))
        data_range = data_max - data_min

        if data_range < 1e-10:
            logger.warning(
                "数据动态范围过小 (range={:.4f})，跳过余弦拟合",
                data_range,
            )
            logger.info(
                f"原始数据统计: mean={orig_mean:.2f}, "
                f"std={orig_std:.2f}, range=[{orig_min:.1f}, {orig_max:.1f}]"
            )
            return np.zeros_like(data)

        # ============================================================
        # Step 2a: 数据归一化 → cos(θ) 估计
        # 物理模型: B = C + A·cos(θ), θ = 2π·g_ref/MAX + φ
        # cos_target = (B - B_min)/(B_max - B_min) ∈ [0, 1]
        #
        # 数据在余弦上的位置由 θ_ref = 2π·g_ref/MAX 决定:
        #   cos(θ_ref) ≥ 0 → 数据在余弦上半支 [0, 1]，可用 cos_target = cos(θ)
        #   cos(θ_ref) < 0 → 数据在余弦下半支 [-1, 0]，cos_target = cos(θ) + 1
        #
        # 本假设要求像素 φ 分布覆盖半支余弦（cos 从极值到过零点）。
        # 若 φ 范围不足，结果为单调近似（矫正图案空间分布正确，幅度近似）。
        # ============================================================
        cos_target = (data - data_min) / data_range
        cos_target = np.clip(cos_target, 0.0, 1.0)

        theta_ref = 2.0 * np.pi * g_ref / max_grayscale
        cos_ref = np.cos(theta_ref)
        sin_ref = np.sin(theta_ref)

        if cos_ref >= 0:
            # 余弦上半支: cos(θ) ∈ [0, 1]
            cos_theta = cos_target
        else:
            # 余弦下半支: cos(θ) ∈ [-1, 0]
            cos_theta = cos_target - 1.0

        # θ_principal = arccos(cos(θ)) ∈ [0, π]
        theta_principal = np.arccos(np.clip(cos_theta, -1.0, 1.0))

        # ============================================================
        # Step 2b: 象限确定 — 从主值恢复真值 θ
        #
        # cos(θ) = cos(θ_principal)，但真值 θ 可能在 [π, 2π]。
        # 导数符号决定:
        #   sin(θ_ref) ≥ 0 → cos递减 → θ = θ_principal + 2πk
        #   sin(θ_ref) < 0 → cos递增 → θ = 2π - θ_principal + 2πk
        #   sin=0 时根据 cos 符号判断极值类型:
        #     cos_ref > 0 (cos极大值) → 递减分支
        #     cos_ref < 0 (cos极小值) → 递增分支
        # ============================================================
        # sin(θ_ref) 在 θ_ref = kπ 处因浮点误差不为精确 0 (如 sin(2π) ≈ -2.4e-16),
        # 用容差判断"极值点"分支, 否则 θ_ref = 2π 时会被误判为递增分支导致 φ 符号反转。
        if sin_ref > 1e-12 or (abs(sin_ref) <= 1e-12 and cos_ref > 0):
            theta_base = theta_principal  # θ ∈ [0, π]，cos递减
        else:
            theta_base = 2.0 * np.pi - theta_principal  # θ ∈ [π, 2π]，cos递增

        # 周期对齐: 使 θ_true 接近 θ_ref
        k_period = int(np.floor(theta_ref / (2.0 * np.pi)))
        theta_true = theta_base + k_period * 2.0 * np.pi

        # ============================================================
        # Step 2c: 提取 φ 并计算矫正映射图
        # ============================================================
        phi = theta_true - theta_ref
        # φ 归一化到 [-π, π)：矫正模 MAX 等效为 0
        phi = (phi + np.pi) % (2.0 * np.pi) - np.pi

        # 矫正: correction = -φ · MAX / (2π)
        correction_map = -phi * max_grayscale / (2.0 * np.pi)

        # ============================================================
        # Step 4: 统计与日志
        # ============================================================
        correction_mean = float(np.mean(correction_map))
        correction_std = float(np.std(correction_map))
        correction_min = float(np.min(correction_map))
        correction_max = float(np.max(correction_map))

        # 拟合优度：非异常点残差 / 原始标准差
        clean_residual = residual.copy()
        clean_residual[outlier_mask] = 0.0  # replaced outliers have 0 residual
        if orig_std > 1e-10:
            fit_quality = float(np.std(clean_residual)) / orig_std
            fit_grade = (
                "优"
                if fit_quality < 0.1
                else "良"
                if fit_quality < 0.25
                else "中"
                if fit_quality < 0.5
                else "差"
            )
        else:
            fit_quality = 0.0
            fit_grade = "均匀数据"

        logger.info(f"矫正数据角度拟合评估: {fit_grade} (残差/原始={fit_quality:.3f})")
        logger.info(
            f"测量条件: measurement_gray={g_ref}, "
            f"max_grayscale={max_grayscale}, "
            f"data_range={data_range:.2f}"
        )
        logger.info(
            f"原始数据统计: mean={orig_mean:.2f}, std={orig_std:.2f}, "
            f"range=[{orig_min:.1f}, {orig_max:.1f}], "
            f"异常点={n_outliers}/{n_total} ({100.0 * n_outliers / n_total:.2f}%)"
        )
        logger.info(
            f"矫正映射图统计: mean={correction_mean:.2f}, "
            f"std={correction_std:.2f}, "
            f"range=[{correction_min:.1f}, {correction_max:.1f}]"
        )

        return correction_map
