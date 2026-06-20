"""波前误差矫正工具类

提供矫正CSV文件加载、灰度矫正图计算（含异常点检测与剔除）、
以及矫正应用功能，供SLM驱动类调用。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from loguru import logger
from scipy import ndimage


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

        raw = np.loadtxt(
            self.csv_path, delimiter=delimiter, skiprows=skiprows
        )
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
    ) -> None:
        """加载CSV并计算矫正映射（load_csv + calc 一步完成）

        Args:
            panel_resolution: 面板分辨率 (width, height)
            outlier_threshold: 异常点 Z-score 阈值 (默认 3.0)
            median_filter_size: 中值滤波窗口大小 (默认 5)
        """
        self.load_csv()
        self.calc(
            panel_resolution=panel_resolution,
            outlier_threshold=outlier_threshold,
            median_filter_size=median_filter_size,
        )

    def calc(
        self,
        panel_resolution: tuple[int, int] = (1920, 1200),
        outlier_threshold: float = 3.0,
        median_filter_size: int = 5,
    ) -> None:
        """计算矫正映射图

        默认实现流程:
          1. 复制原始数据并转为 float64
          2. 计算局部中值参考面（median filter）
          3. 计算残差 Z-score，标记并剔除异常点（替换为局部中值）
          4. 记录统计指标（均值、标准差、范围、异常点比例）
          5. 调整尺寸到面板分辨率

        Args:
            panel_resolution: 目标面板分辨率 (width, height)
            outlier_threshold: 异常点 Z-score 阈值 (默认 3.0)
            median_filter_size: 中值滤波窗口大小 (默认 5)
        """
        if self._raw_data is None:
            raise RuntimeError("请先调用 load_csv() 加载原始数据")

        self._panel_resolution = panel_resolution

        if self._calc_fn is not None:
            # 用户自定义 calc 函数
            result = self._calc_fn(self._raw_data)
        else:
            result = self._default_calc(
                self._raw_data,
                outlier_threshold=outlier_threshold,
                median_filter_size=median_filter_size,
            )

        # 调整尺寸到面板分辨率
        self._correction_map = self.resize_to_panel(result, panel_resolution)

        logger.info(
            f"矫正映射图已计算: "
            f"形状={self._correction_map.shape}, "
            f"范围=[{np.nanmin(self._correction_map):.1f}, "
            f"{np.nanmax(self._correction_map):.1f}]"
        )

    def map_error(
        self, grayscale: np.ndarray, max_grayscale: int = 1023
    ) -> np.ndarray:
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
    ) -> "WavefrontCorrection":
        """便捷方法：从CSV文件创建并计算矫正映射（load_csv + calc 一步完成）

        Args:
            csv_path: CSV文件路径
            panel_resolution: 面板分辨率 (width, height)
            calc_fn: 自定义 calc 函数
            outlier_threshold: 异常点 Z-score 阈值
            median_filter_size: 中值滤波窗口大小

        Returns:
            已初始化的 WavefrontCorrection 实例
        """
        instance = cls(csv_path, calc_fn=calc_fn)
        instance.load_csv()
        instance.calc(
            panel_resolution=panel_resolution,
            outlier_threshold=outlier_threshold,
            median_filter_size=median_filter_size,
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
            result = data[
                start_y : start_y + target_h, start_x : start_x + target_w
            ]
        else:
            # 居中补零
            result = np.zeros((target_h, target_w), dtype=np.float64)
            start_y = (target_h - h) // 2
            start_x = (target_w - w) // 2
            result[start_y : start_y + h, start_x : start_x + w] = data

        return np.ascontiguousarray(result, dtype=np.float64)

    # ── 内部方法 ──────────────────────────────────────

    @staticmethod
    def _default_calc(
        raw_data: np.ndarray,
        outlier_threshold: float = 3.0,
        median_filter_size: int = 5,
    ) -> np.ndarray:
        """默认 calc 实现：异常点检测、剔除与统计信息输出

        处理流程:
          1. 复制原始数据
          2. 用局部中值滤波构建参考面
          3. 计算残差 (data - local_median) 及其标准差
          4. Z-score > threshold → 标记为异常点，替换为局部中值
          5. 记录详细统计: 均值、标准差、范围、异常点比例、拟合度

        Args:
            raw_data: 原始数据（uint16 或 float64）
            outlier_threshold: Z-score 异常阈值
            median_filter_size: 中值滤波窗口

        Returns:
            清理后的数据 (float64)
        """
        data = np.nan_to_num(raw_data).copy()

        # ── 基础统计 ──
        orig_mean = float(np.mean(data))
        orig_std = float(np.std(data))
        orig_min = float(np.min(data))
        orig_max = float(np.max(data))

        # ── 局部中值参考面 ──
        local_median = ndimage.median_filter(data, size=median_filter_size)
        residual = data - local_median
        residual_std = float(np.std(residual))

        # ── 异常点检测与剔除 ──
        n_outliers = 0
        if residual_std > 1e-10:
            z_scores = np.abs(residual) / residual_std
            outlier_mask = z_scores > outlier_threshold
            n_outliers = int(np.sum(outlier_mask))

            if n_outliers > 0:
                data[outlier_mask] = local_median[outlier_mask]
                logger.debug(
                    f"矫正数据异常点: {n_outliers}/{data.size} "
                    f"({100.0 * n_outliers / data.size:.2f}%) "
                    f"已替换为局部中值"
                )

        # ── 清理后统计 ──
        clean_mean = float(np.mean(data))
        clean_std = float(np.std(data))

        # ── 拟合度评估 ──
        #   使用残差与原始波动的比值来量化拟合优度
        if orig_std > 1e-10:
            # rms 拟合度: 残差标准差 / 原始标准差（越小说明参考面拟合越好）
            fit_quality = residual_std / orig_std
            fit_grade = (
                "优" if fit_quality < 0.1
                else "良" if fit_quality < 0.25
                else "中" if fit_quality < 0.5
                else "差"
            )
        else:
            fit_quality = 0.0
            fit_grade = "均匀数据"

        # ── 日志输出 ──
        logger.info(
            f"矫正数据拟合评估: {fit_grade} "
            f"(残差/原始={fit_quality:.3f})"
        )
        logger.info(
            f"矫正数据统计: "
            f"mean={orig_mean:.2f}→{clean_mean:.2f}, "
            f"std={orig_std:.2f}→{clean_std:.2f}, "
            f"range=[{orig_min:.1f}, {orig_max:.1f}], "
            f"异常点={n_outliers}/{data.size} "
            f"({100.0 * n_outliers / data.size:.2f}%)"
        )

        return data
