"""Standalone Zernike polynomial utilities.

Provides coefficient parsing, validation, and phase generation without
requiring the full PatternHelper or SLM context. 从
`gui/slm/multi_slm_controller.py` 中工具化提取, 供 SLM 相位优化 runner
(如 `optimizer/wfless/slm_square_shaping.py`) 与外部脚本复用。

==================== 使用示例 ====================

1) 解析系数 (支持 3 种输入格式 → (n, m) dict):

    >>> from ao_shaping.utils.wavefront.zernike_utils import (
    ...     parse_zernike_coefficients, generate_zernike_phase, list_zernike_modes,
    ... )
    >>> # 格式 A: Noll 索引 dict (str/int 键, 1-based)
    >>> coeffs = parse_zernike_coefficients({"5": 1.0, "13": 0.5})
    >>> # 格式 B: (n, m) 元组 dict
    >>> coeffs = parse_zernike_coefficients({(2, 0): 1.0, (4, 0): 0.5})
    >>> # 格式 C: Noll 序扁平数组 (索引 0 = Noll 1 = piston; 索引 3 = Noll 4 = defocus)
    >>> coeffs = parse_zernike_coefficients([0.0, 0.0, 0.0, 1.0])

2) 生成相位图 (SLM 灰度相位, 直接用于 SLM 写入):

    >>> phase = generate_zernike_phase(coeffs, n_max=4, resolution=(1920, 1200))
    >>> phase.shape  # (1200, 1920)

3) 枚举模式:

    >>> modes = list_zernike_modes(4)
    >>> modes[0]  # (1, 0, 0, 'Piston') → (noll_index, n, m, name)

==================== Noll 索引约定 (重要!) ====================

本模块使用 aotools `zernike` 库的标准 Noll 1976 约定 (与
`zernike_calc.ZernikeGenerator` 一致)。前 15 阶映射:

    Noll 1  → (0, 0)   Piston           Noll 9  → (3, -3)  Trefoil
    Noll 2  → (1, 1)   Tip              Noll 10 → (3, 3)   Trefoil
    Noll 3  → (1, -1)  Tilt             Noll 11 → (4, 0)   Spherical
    Noll 4  → (2, 0)   Defocus          Noll 12 → (4, 2)
    Noll 5  → (2, -2)  Astig.           Noll 13 → (4, -2)
    Noll 6  → (2, 2)   Astig.           Noll 14 → (4, 4)
    Noll 7  → (3, -1)  Coma             Noll 15 → (4, -4)
    Noll 8  → (3, 1)   Coma

注意: 本模块的 `noll_to_nm()` 直接导入自 `zernike_calc` (aotools Noll 1976
约定, 支持任意 Noll 索引), 是全项目唯一的索引→(n, m) 约定。`optimizer/wf/`
三个优化器 (ga_zernike / greedy_zernike / rms_by_zernike) 的模式枚举与索引
映射均使用 canonical 约定; 历史遗留的 legacy 硬编码查表 (`noll_to_nm_legacy`,
Noll 5 = (2, 0)) 已删除。新代码一律用 canonical 的 `noll_to_nm()` /
`list_zernike_modes()`, 不要混用两套索引。

==================== 输出格式说明 ====================

`generate_zernike_phase()` 返回 **float64** 相位图 (非 uint16):
- 孔径内: 浮点相位值 (单位: 弧度, 未包裹 raw radians — 对应 raw-only 契约)
- 孔径外 (圆形孔径之外): NaN
- 空系数 / None: 全零 **uint16** 数组

若需要 SLM 灰度图, 请对孔径内区域自行做 2π 取模并量化到目标位深,
或直接用 `gui/slm/multi_slm_controller.py` 的 PatternHelper (含灰度转换)。

==================== 与 ZernikeGenerator 的关系 ====================

`generate_zernike_phase()` 内部创建 `ZernikeGenerator` (带网格缓存)。
反复生成相同分辨率的相位时, 建议直接持有 ZernikeGenerator 实例复用:

    >>> from ao_shaping.utils.wavefront.zernike_calc import ZernikeGenerator
    >>> gen = ZernikeGenerator((1920, 1200), n_orders=4)
    >>> gen.set_bits(10)
    >>> img = gen.generate_polynomial(coeffs)  # 与 generate_zernike_phase 等价
"""

from __future__ import annotations

import numpy as np

from ao_shaping.utils.wavefront.zernike_calc import (
    ZernikeGenerator,
    calc_n_zernike_terms,
    get_zernike_name,
    noll_to_nm,
)

# 工作波长 (µm): WFS get_zernike() 返回 µm, 参与矫正前必须换算为 λ。
# ⚠️ 单位一致性 (2026-09-16 实测定位的矫正失效根因): 响应矩阵必须与矫正时的 `w`
#   用**同一单位**。曾因矩阵用原始 µm 构建、而矫正用 `w = z/0.532` (λ), 反解系数
#   被放大 1/0.532 = 1.88×, 矫正过驱动 88% → 实测闭环仅 19.7% (离线最优可达 75%)。
LAMBDA_UM = 0.532
UM_TO_WAVES = 1.0 / LAMBDA_UM


def um_to_waves(z: np.ndarray) -> np.ndarray:
    """WFS Zernike 系数 µm → λ (工作波长 532nm)。矩阵与矫正**必须**用同一单位。"""
    return np.asarray(z, dtype=float) * UM_TO_WAVES


def list_zernike_modes(n_max: int) -> list[tuple[int, int, str]]:
    """List all Zernike modes up to a given radial order.

    Args:
        n_max: Maximum Zernike radial order.

    Returns:
        List of (noll_index, n, m, name) tuples, sorted by Noll index.
        Noll index is 1-based.
    """
    n_terms = calc_n_zernike_terms(n_max)
    modes = []
    for j in range(1, n_terms + 1):
        n, m = noll_to_nm(j)
        if n <= n_max:
            name = get_zernike_name(n, m)
            modes.append((j, n, m, name))
    return modes


def parse_zernike_coefficients(
    raw: dict[str | int | tuple[int, int], float | int] | list[float] | np.ndarray | None,
    n_max: int | None = None,
) -> dict[tuple[int, int], float]:
    """Parse Zernike coefficients from various input formats.

    Accepts:
    - dict with Noll index keys (str/int): ``{"5": 1.0, "13": 0.5}``
    - dict with (n, m) tuple keys: ``{(2, 0): 1.0, (4, 0): 0.5}``
    - list/array of floats (Noll order, 1-based index): ``[0, 0, 0, 0, 1.0, ...]``
    - None → empty dict

    Args:
        raw: Raw coefficient input.
        n_max: Maximum Zernike radial order (used for validation).
            If None, no validation is performed.

    Returns:
        Dictionary mapping (n, m) to amplitude.
    """
    if raw is None:
        return {}

    # Already (n, m) dict
    if isinstance(raw, dict):
        first_key = next(iter(raw), None)
        if first_key is None:
            return {}
        if isinstance(first_key, tuple):
            # Already (n, m) format — cast for pyright variance
            if n_max is not None:
                return {tuple(k): float(v) for k, v in raw.items() if k[0] <= n_max}  # type: ignore[misc]
            return {tuple(k): float(v) for k, v in raw.items()}  # type: ignore[misc]
        # Noll index format (str or int keys)
        result: dict[tuple[int, int], float] = {}
        for k, v in raw.items():
            j = int(k)  # type: ignore[arg-type]
            if j < 1:
                continue
            n, m = noll_to_nm(j)
            if n_max is not None and n > n_max:
                continue
            result[(n, m)] = float(v)
        return result

    # List/array format (Noll order)
    if isinstance(raw, (list, np.ndarray)):
        arr = np.asarray(raw, dtype=float)
        result = {}
        for j_idx in range(len(arr)):
            if abs(arr[j_idx]) < 1e-15:
                continue
            j = j_idx + 1  # 1-based Noll index
            n, m = noll_to_nm(j)
            if n_max is not None and n > n_max:
                continue
            result[(n, m)] = float(arr[j_idx])
        return result

    raise TypeError(f"Unsupported coefficient type: {type(raw)}")


def coefficients_to_array(
    coeffs: dict[tuple[int, int], float],
    n_max: int,
) -> np.ndarray:
    """Convert (n, m) coefficient dict to flat array in Noll order.

    Args:
        coeffs: Dictionary mapping (n, m) to amplitude.
        n_max: Maximum Zernike radial order.

    Returns:
        1D array of length calc_n_zernike_terms(n_max) in Noll order.
    """
    nk = calc_n_zernike_terms(n_max)
    arr = np.zeros(nk, dtype=np.float64)
    for (n, m), amp in coeffs.items():
        from zernike import RZern

        cart = RZern(max(1, n))
        j = cart.nm2noll(n, m) - 1  # 0-based
        if 0 <= j < nk:
            arr[j] = amp
    return arr


def generate_zernike_phase(
    coefficients: dict[str | int | tuple[int, int], float | int] | list[float] | np.ndarray | None,
    resolution: tuple[int, int],
    n_max: int = 6,
    radius: float | None = None,
    bits: int = 10,
) -> np.ndarray:
    """Generate Zernike phase pattern from coefficients.

    Standalone function that creates a ZernikeGenerator internally.
    For repeated calls with the same resolution, use ZernikeGenerator
    directly for better performance.

    Args:
        coefficients: Zernike coefficients in any format accepted by
            parse_zernike_coefficients.
        resolution: Output resolution as (width, height).
        n_max: Maximum Zernike radial order.
        radius: Aperture radius in pixels. Defaults to min(w, h) / 2.
        bits: Output bit depth (default 10 → 0-1023).

    Returns:
        Phase pattern array with shape (height, width). Pixels inside the
        aperture contain float phase values; pixels outside are NaN.
        When coefficients is empty/None, returns all-zeros uint16 array.
    """
    coeffs_dict = parse_zernike_coefficients(coefficients, n_max=n_max)

    gen = ZernikeGenerator(
        resolution=resolution,
        radius=radius,
        n_orders=n_max,
    )
    gen.set_bits(bits)

    if not coeffs_dict:
        return np.zeros((resolution[1], resolution[0]), dtype=np.uint16)

    return gen.generate_polynomial(coeffs_dict)
