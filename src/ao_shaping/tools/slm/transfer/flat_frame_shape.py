"""平场参照帧的形状判据 (flat-frame shape criteria)。

光路复用 (``optical_probe.py``) 的两条形状判据都住在这里, 加上那一帧证据本身的
落盘/读回:

1. **逐像素 (NCC)**: 两帧平场斑各自取亚像素零点, 以该零点为心重采样成同尺寸方形
   窗, 平滑去均值后取归一化互相关 (:func:`compare_flat_frames`)。对平移与整体亮度
   免疫 (这正是要的), 对「光路真的变了」太宽容。
2. **二阶矩**: 同一对窗口再算一次强度加权二阶矩 (:func:`flat_shape` →
   :func:`compare_flat_shape`), 管 NCC 看不见的那半边 —— 斑多大 (``sigma_g``)、多扁
   (``aspect``)、朝哪边 (``theta_deg``)。

形状判据的证据就是那一帧平场: 采集段写下它 (:func:`save_flat_frame`), 拟合段复制进
``latest``, 探针在当前会话目录写自己那一帧 (:func:`load_flat_frame` 读回)。

本模块顶层**不 import 任何设备/SDK**, 全部是纯函数; ``scipy.ndimage`` 在函数内
惰性 import, 与 :mod:`ao_shaping.tools.slm.transfer.optical_probe` 同一口径。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger


#: latest 目录里记录探针条件/格式的文件名。
FLAT_FROZEN_SIDECAR = "_optical_probe.json"
#: 平场参照帧 (平面相位下的 0 级斑) 的文件名。与 :data:`FLAT_FROZEN_SIDECAR`
#: 同目录: 采集段写进采集目录, 拟合段复制进 ``latest``; 探针在当前会话目录写
#: 自己那一帧。缺它就是「没有形状证据」, 退到
#: :data:`ao_shaping.tools.slm.transfer.optical_probe.DEFAULT_TOL_ZERO_PX` 的零点判比。
FLAT_FRAME_FILE = "_optical_probe_flat.npz"
#: 形状比对的方形窗半宽 (px)。实测平场斑 FWHM 约 22 px; 65x65 窗里噪声占比
#: 太高 (同一条光路只剩 0.81 相似度), 129x129 窗把斑连同第一圈衍射环框住。
DEFAULT_FLAT_HALF_WINDOW = 64


#: 形状相似度下限 (亚像素对心后的归一化互相关)。
#:
#: 2026-11-11 离线标定 (``data/debug/slm_transfer/20261105_*/response/flat_probe.npz``):
#: 同一条光路的重复取帧 0.94-0.99 (含 16 px 的零点漂移), 换了光路 (2026-11-02 的
#: 相机 ROI + 斑点形状) 只有 0.70-0.83。0.90 落在两簇中间。
DEFAULT_MIN_FLAT_SIMILARITY = 0.90


#: 求亚像素零点时峰值窗口的半宽 (px), 与采集段 ``locate`` 的局部矩窗口同量级。
DEFAULT_CENTRE_HALF_WINDOW = 60


# --- 形状 (二阶矩) 判据 -------------------------------------------------------
# 为什么要第二条判据: 归一化互相关 (NCC) 对**平移**和**整体亮度**确实是对的
# (2026-11-11 用当天实测平场帧 ``data/debug/slm_transfer/20261111_204230_optical_
# reuse_calib/_optical_probe_flat.npz`` 做的合成敏感性扫描: 平移 40 px → 1.0000,
# 增益 ×0.25~×3 → 1.0000), 但它对「光路真的变了」的那一类改动**几乎完全瞎**:
#   各向异性拉伸 6%  → NCC 0.9964 / 20% → 0.9874
#   旋转 12°        → NCC 0.9963 / 25°  → 0.9860
#   各向同性放宽 30% (离焦代理) → NCC 0.976
# 这些全都远高于 0.90 的相似度下限, 于是「换了光路」会被判成「可复用」。同一批
# 合成扰动下**二阶矩**分得很干净: sigma 变化 6-17%, 椭圆度变化 0.03-0.24, 主轴
# 角度跟着转。所以形状判据不是替代 NCC, 而是补上 NCC 看不见的那半边: NCC 管
# 「逐像素长什么样」, 二阶矩管「斑多大、多扁、朝哪边」。
#: 二阶矩的圆形加权窗半径 (px)。取 40 ≈ 实测平场斑 (FWHM 22 px, sigma 11.4/11.7)
#: 的 3.5 倍 sigma: 再小会切掉衍射环 (尺寸比被压扁), 再大就把背景噪声和邻近杂散
#: 光纳入矩, 椭圆度开始受噪声支配。必须是 :data:`DEFAULT_FLAT_HALF_WINDOW` 以内。
DEFAULT_SHAPE_RADIUS_PX = 40
#: 回旋半径 (sigma_g) 之比的上限; 下限取其倒数 (1/1.12 ≈ 0.89), 收紧与放宽同等
#: 对待 —— 斑变小同样是光路变了 (口径/焦距), 不是「只有曝光变了」。合成扫描里
#: 实测 6% 拉伸给 sigma_g ±3%, 30% 放宽给 1.31 (受 40 px 窗截断影响后仍 > 1.12),
#: 而纯平移/纯增益是 1.00。1.12 留了约 4 倍于噪声的余量, **暂定值, 待实验标定**。
DEFAULT_MAX_SIZE_RATIO = 1.12
#: 椭圆度 (sqrt(长轴特征值/短轴特征值), >=1) 的失配上限。实测平场斑接近圆
#: (aspect 1.028), 各向异性 6% 拉伸给失配 ~1.06, 20% 给 ~1.24。1.12 **暂定**。
DEFAULT_MAX_ASPECT_MISMATCH = 1.12
#: 长短轴朝向 (主轴夹角) 的上限 (度)。旋转 12° 时 NCC 还有 0.9963, 只有矩能看见;
#: 但实测 rot 的角不确定度很大 (近圆斑的主轴方向由噪声决定), 所以只在**两个斑都
#: 明显椭圆**时才拿它当判据, 见 :data:`DEFAULT_MIN_ASPECT_FOR_ANGLE`。15° **暂定**。
DEFAULT_MAX_AXIS_TILT_DEG = 15.0
#: 主轴夹角**参与判比**的门槛: 两边椭圆度都至少这么大才认为朝向是可测的。
#: 近圆斑 (实测 aspect 1.03) 的 theta 是噪声, 用它否决会把好光路判成坏光路。
DEFAULT_MIN_ASPECT_FOR_ANGLE = 1.15


@dataclass(frozen=True)
class ShapeMetrics:
    """一帧平场斑的**二阶矩**描述子 (对心后, 圆形窗内)。

    NCC 看不见的那些改动 (离焦/像差、各向异性拉伸、旋转) 都会写进这四个数里:
    ``sigma_g`` 管斑多大, ``aspect`` 管多扁, ``theta_deg`` 管朝哪边。见
    :data:`DEFAULT_SHAPE_RADIUS_PX` 上方的扫描数据。
    """

    #: 回旋半径 (RMS 半径, px) = sqrt(cxx + cyy) = sqrt(两特征值之和)。
    sigma_g: float
    #: 对重心的 x 方向标准差 (px)。
    sigma_x: float
    #: 对重心的 y 方向标准差 (px)。
    sigma_y: float
    #: 椭圆度, 恒 >= 1: sqrt(lambda_major / lambda_minor)。完美圆 = 1.0。
    aspect: float
    #: 长轴朝向 (度), 取值范围 (-90, 90]; 图像坐标 (x 向右, y 向下, 角从 +x 轴
    #: 转向 +y 轴)。近圆斑时它是噪声, 由 :func:`compare_flat_shape` 负责不用它。
    theta_deg: float

    def to_dict(self) -> dict[str, Any]:
        # 原样给数 (退化时是 NaN): 分析脚本要拿这些数算中位数, None 会直接把
        # np.median 打挂。落盘用的 ``FlatComparison.to_dict`` 才做 NaN→None。
        return asdict(self)


@dataclass(frozen=True)
class ShapeComparison:
    """两帧二阶矩的比对结论 (三条轴 + 一个总开关)。"""

    #: ``cur.sigma_g / ref.sigma_g`` (1.0 = 同尺寸); 不可算时为 NaN。
    size_ratio: float
    #: 椭圆度失配 ``max(a/b, b/a)`` (对称, >= 1.0); 不可算时为 NaN。
    aspect_mismatch: float
    #: 主轴夹角, 折到 ``[0, 90]`` (度); 不可算时为 NaN。
    axis_tilt_deg: float
    #: 夹角是否**参与判比**: 只有两边都明显椭圆 (``min(aspect) >=
    #: DEFAULT_MIN_ASPECT_FOR_ANGLE``) 才为 True —— 近圆斑的角度是噪声, 不能拿它否决。
    angle_significant: bool
    #: 三条轴是否全部落在容差内 (**失败关闭**: 任何不可算/退化都是 False)。
    within_tolerance: bool
    #: 三个数是否都算得出来; False 表示至少一边退化 (无有效形状证据)。
    known: bool = True

    def to_dict(self) -> dict[str, Any]:
        # 同 :class:`ShapeMetrics.to_dict`: 保留 NaN, 让分析脚本能继续做数值统计。
        return asdict(self)


@dataclass(frozen=True)
class FlatComparison:
    """两帧平场斑在**亚像素对心之后**的形状比对结果。

    只描述「形状像不像」: 平移已经被对心消掉, 所以 ``drift_px`` 是信息量, 不是
    判据的一部分 (:func:`assess_reuse` 另外拿它卡硬上限)。
    """

    #: 归一化互相关 (-1..1); 1.0 = 逐像素一致。
    similarity: float
    #: 当前帧相对参照帧的零点位移 ``(dx, dy)`` (px, 亚像素)。
    drift_px: tuple[float, float]
    #: 比对窗口边长 (px)。
    window_px: int
    #: 当前帧亚像素零点 ``(x, y)``。
    current_xy: tuple[float, float]
    #: 参照帧亚像素零点 ``(x, y)``。
    reference_xy: tuple[float, float]
    #: 当前帧的二阶矩 (与 NCC 用**同一片**平滑去均值窗); 手工构造的旧实例为 None。
    shape: ShapeMetrics | None = None
    #: 参照帧的二阶矩。
    reference_shape: ShapeMetrics | None = None
    #: 两者的比对结论 (:func:`compare_flat_shape`)。
    shape_comparison: ShapeComparison | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_number(asdict(self))


def _json_number(value: Any) -> Any:
    """递归把 NaN/Inf 换成 ``None`` —— ``optical_probe.json`` 必须能严格解析。

    退化的平场帧 (全背景) 会让二阶矩算不出数, 而这些数要落盘; Python 的 ``json``
    默认写 ``NaN`` (非标准 JSON), 别的读者会用 ``allow_nan=False`` 或直接拒绝。
    """
    if isinstance(value, dict):
        return {k: _json_number(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_json_number(v) for v in value)
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _nan_shape() -> ShapeMetrics:
    """退化输入的统一返回值 (不抛异常, 但绝不能被读成「通过」)。"""
    nan = float("nan")
    return ShapeMetrics(
        sigma_g=nan, sigma_x=nan, sigma_y=nan, aspect=nan, theta_deg=nan
    )


def _is_num(value: float) -> bool:
    """``value`` 是否为可用的有限数 (NaN/Inf 都不行)。"""
    return bool(np.isfinite(value))


def _fold_theta(theta_deg: float) -> float:
    """把长轴朝向折进 (-90, 90] (``atan2`` 取到 ``-pi`` 时是 -90, 与 +90 同一方向)。"""
    out = float(theta_deg) % 180.0
    return out - 180.0 if out > 90.0 else out


def _fold_angle(deg: float) -> float:
    """把角度差折进 [0, 90]: 长轴是无向轴, 170° 差 = 10° 差。"""
    out = abs(float(deg)) % 180.0
    return 180.0 - out if out > 90.0 else out


def flat_shape(
    window: np.ndarray,
    *,
    radius: int = DEFAULT_SHAPE_RADIUS_PX,
) -> ShapeMetrics:
    """纯函数: 以窗心为界, 圆形加权窗内的强度加权二阶矩。

    为什么需要它
    ------------
    NCC 把「斑的形状」当一张图逐像素比, 于是它对**平移**和**整体亮度**天然免疫
    (这正是我们要的), 但对二阶矩层面的形变几乎无感: 2026-11-11 的合成扫描里,
    20% 各向异性拉伸只把 NCC 拉到 0.9874, 25° 旋转 0.9860, 30% 各向同性放宽
    (离焦代理) 0.976 —— 全部远高于 0.90 下限。二阶矩是这些改动的直接观测量:
    同样三档扰动给出 sigma 变化 6-17%、椭圆度变化 0.03-0.24、主轴角度平移。

    做法
    ----
    权重 = ``clip(window, 0, None)`` × 半径 ``radius`` 的圆窗 (窗心即窗口几何中心,
    调用方已经对过心)。取负号截掉背景 (减过均值的窗里背景是负数, 留着会把矩往
    外拉) ; 圆窗而非方窗, 免得方形边界对各向异性给出假的偏好方向。质心用强度
    加权一次, 二阶矩对**质心**取 (不是对窗心), 于是残余的小偏移不会伪装成展宽。
    2x2 协方差做特征分解 (闭式, 免 ``numpy.linalg`` 的额外开销与非收敛风险):
    特征比开方 = 椭圆度, 长轴特征向量角度 = 主轴朝向。

    Args:
        window: 2D 窗 (通常是 :func:`compare_flat_frames` 里那片平滑去均值窗)。
        radius: 圆形加权窗半径 (px), 默认 :data:`DEFAULT_SHAPE_RADIUS_PX`。

    Returns:
        :class:`ShapeMetrics`。总权重 <= 0 (全背景/空图/非 2D) 时五个字段都是
        NaN, **不抛异常** —— 判定端把 NaN 当作「没有形状证据」而否决 (fail-closed)。
    """
    data = np.asarray(window, dtype=np.float64)
    if data.ndim != 2 or data.size == 0:
        return _nan_shape()
    n_rows, n_cols = data.shape
    centre_y = (n_rows - 1) / 2.0
    centre_x = (n_cols - 1) / 2.0
    yy, xx = np.mgrid[0:n_rows, 0:n_cols]
    rr = float(radius)
    if not np.isfinite(rr) or rr <= 0.0:
        rr = max(centre_x, centre_y)
    inside = (yy - centre_y) ** 2 + (xx - centre_x) ** 2 <= rr * rr
    weights = np.clip(data, 0.0, None) * inside
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        return _nan_shape()

    mean_x = float((xx * weights).sum()) / total
    mean_y = float((yy * weights).sum()) / total
    dx = xx - mean_x
    dy = yy - mean_y
    cxx = float((weights * dx * dx).sum()) / total
    cyy = float((weights * dy * dy).sum()) / total
    cxy = float((weights * dx * dy).sum()) / total
    if not all(np.isfinite(v) for v in (cxx, cyy, cxy)):
        return _nan_shape()

    trace = cxx + cyy
    if trace <= 0.0:
        # 权重全压在一个像素上 (斑被截断到不存在): 矩退化为 0, 不是「很圆的窄斑」。
        return _nan_shape()
    half_diff = (cxx - cyy) / 2.0
    radius_off = float(np.hypot(half_diff, cxy))
    lam_major = trace / 2.0 + radius_off
    lam_minor = trace / 2.0 - radius_off
    aspect = float("inf") if lam_minor <= 0.0 else float(np.sqrt(lam_major / lam_minor))
    theta = _fold_theta(float(np.degrees(0.5 * np.arctan2(2.0 * cxy, cxx - cyy))))
    return ShapeMetrics(
        sigma_g=float(np.sqrt(trace)),
        sigma_x=float(np.sqrt(max(cxx, 0.0))),
        sigma_y=float(np.sqrt(max(cyy, 0.0))),
        aspect=aspect,
        theta_deg=theta,
    )


def compare_flat_shape(cur: ShapeMetrics, ref: ShapeMetrics) -> ShapeComparison:
    """纯函数: 两帧二阶矩 → 三条轴的失配 + 总开关 (逐条对应三个上限常量)。

    三条轴各管一种物理变化: ``size_ratio`` (离焦/口径/传播距离: 斑整体变大变小),
    ``aspect_mismatch`` (像散/柱面差: 变扁), ``axis_tilt_deg`` (主轴转动: 光路里
    转了东西)。**失败关闭**: 任何一边退化 (NaN) → ``known=False`` →
    ``within_tolerance=False``, 绝不把「算不出」读成「通过」。

    角度这一条带一个门槛: 只有**两边都明显椭圆**时主轴朝向才是可测的量。实测平场
    斑 aspect 1.028, 这种近圆斑的 theta 完全由噪声决定 (1° 的角度不确定度对应
    椭圆度要到 1.05 以上才谈得上方向), 拿它否决会把好光路判成坏光路 —— 于是
    ``angle_significant=False`` 时这条直接放行。

    Args:
        cur: 当前帧二阶矩。
        ref: 参照帧二阶矩。

    Returns:
        :class:`ShapeComparison`。
    """
    sigma_cur = float(cur.sigma_g)
    sigma_ref = float(ref.sigma_g)
    size_ratio = (
        sigma_cur / sigma_ref
        if _is_num(sigma_cur) and _is_num(sigma_ref) and sigma_ref > 0.0
        else float("nan")
    )

    a_cur = float(cur.aspect)
    a_ref = float(ref.aspect)
    if _is_num(a_cur) and _is_num(a_ref) and a_cur > 0.0 and a_ref > 0.0:
        aspect_mismatch = max(a_cur / a_ref, a_ref / a_cur)
    else:
        aspect_mismatch = float("nan")

    t_cur = float(cur.theta_deg)
    t_ref = float(ref.theta_deg)
    axis_tilt_deg = (
        _fold_angle(t_cur - t_ref)
        if _is_num(t_cur) and _is_num(t_ref)
        else float("nan")
    )

    known = bool(
        _is_num(size_ratio)
        and _is_num(aspect_mismatch)
        and _is_num(axis_tilt_deg)
        and size_ratio > 0.0
    )
    angle_significant = bool(
        known
        and min(a_cur, a_ref) >= DEFAULT_MIN_ASPECT_FOR_ANGLE
        and _is_num(a_cur)
        and _is_num(a_ref)
    )
    size_ok = bool(
        known and 1.0 / DEFAULT_MAX_SIZE_RATIO <= size_ratio <= DEFAULT_MAX_SIZE_RATIO
    )
    aspect_ok = bool(known and aspect_mismatch <= DEFAULT_MAX_ASPECT_MISMATCH)
    tilt_ok = bool(
        (not angle_significant) or axis_tilt_deg <= DEFAULT_MAX_AXIS_TILT_DEG
    )
    return ShapeComparison(
        size_ratio=float(size_ratio),
        aspect_mismatch=float(aspect_mismatch),
        axis_tilt_deg=float(axis_tilt_deg),
        angle_significant=angle_significant,
        within_tolerance=bool(size_ok and aspect_ok and tilt_ok),
        known=known,
    )


def _shape_gate_text(flat: FlatComparison | None) -> tuple[bool, str]:
    """形状判据的中文一行话 (给 GUI 看是哪条轴动了) + 是否放行。"""
    sc = None if flat is None else flat.shape_comparison
    if sc is None:
        return False, "形状判据未过: 没有二阶矩度量"
    if not sc.known:
        return False, "形状判据未过: 斑点退化, 二阶矩算不出 (无形状证据)"
    if not (1.0 / DEFAULT_MAX_SIZE_RATIO <= sc.size_ratio <= DEFAULT_MAX_SIZE_RATIO):
        if sc.size_ratio > DEFAULT_MAX_SIZE_RATIO:
            return False, (
                f"形状判据未过: 尺寸比 {sc.size_ratio:.2f} > {DEFAULT_MAX_SIZE_RATIO:.2f}"
            )
        return False, (
            f"形状判据未过: 尺寸比 {sc.size_ratio:.2f} < "
            f"{1.0 / DEFAULT_MAX_SIZE_RATIO:.2f}"
        )
    if sc.aspect_mismatch > DEFAULT_MAX_ASPECT_MISMATCH:
        return False, (
            f"形状判据未过: 椭圆度失配 {sc.aspect_mismatch:.2f} > "
            f"{DEFAULT_MAX_ASPECT_MISMATCH:.2f}"
        )
    if sc.angle_significant and sc.axis_tilt_deg > DEFAULT_MAX_AXIS_TILT_DEG:
        return False, (
            f"形状判据未过: 主轴夹角 {sc.axis_tilt_deg:.1f}° > "
            f"{DEFAULT_MAX_AXIS_TILT_DEG:.0f}°"
        )
    tail = "主轴夹角不计(近圆斑)" if not sc.angle_significant else ""
    ok = (
        f"尺寸比 {sc.size_ratio:.2f} / 椭圆度失配 {sc.aspect_mismatch:.2f} / "
        f"主轴夹角 {sc.axis_tilt_deg:.1f}°"
    )
    return True, f"{ok}{' ' + tail if tail else ''} (通过)"


def _shape_tail(flat: FlatComparison | None) -> str:
    """判决文案里的形状摘要片段 (`` · 尺寸比 …``); 没有二阶矩时返回**空串**。

    为什么必须能返回空串: :func:`optical_probe.format_probe_verdict` 的老调用点
    (GUI 的跳过文案、没有平场帧的老 latest 目录) 传的是 ``flat=None`` 或不含二阶矩的旧式
    :class:`FlatComparison`, 那些句子里多一个字符都会动既有文案。
    """
    sc = None if flat is None else flat.shape_comparison
    if sc is None:
        return ""
    if not sc.known:
        return ", 形状 二阶矩不可判 (退化斑)"
    angle = (
        f"{sc.axis_tilt_deg:.1f}°"
        if sc.angle_significant
        else f"{sc.axis_tilt_deg:.1f}°(近圆不计)"
    )
    return (
        f", 形状 尺寸比 {sc.size_ratio:.2f}/椭圆度失配 {sc.aspect_mismatch:.2f}"
        f"/主轴夹角 {angle}"
    )


def _prepare_flat(frame: Any) -> np.ndarray:
    """平场帧 → 去背景的 float64 2D 图 (背景取中位数, 抗热点)。"""
    data = np.asarray(frame, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"平场帧必须是 2D, 实际 shape={data.shape}")
    if data.size == 0:
        raise ValueError("平场帧是空的")
    return data - float(np.median(data))


def _flat_centre(frame: np.ndarray, half: int) -> tuple[float, float]:
    """亚像素零点: 峰值窗口内的强度加权重心 (与采集段同一原语)。"""
    from ao_shaping.utils.image.centroid import windowed_centroid

    return windowed_centroid(frame, half)


def _flat_window(frame: np.ndarray, xy: tuple[float, float], half: int) -> np.ndarray:
    """以 ``xy`` 为心取 ``(2*half+1)`` 方形窗, 三次样条重采样到整数网格。

    直接 ``frame[y0:y1, x0:x1]`` 只能取整像素位置, 亚像素零点就被丢掉了 —— 而
    剩下的半个像素正好落在斑点最陡的斜坡上。这里用 ``map_coordinates`` 在**小数
    坐标**上取样, 于是两帧的斑点都落在窗口正中心, 误差只剩插值阶数。
    """
    from scipy.ndimage import map_coordinates

    offsets = np.arange(-half, half + 1, dtype=np.float64)
    rows, cols = np.meshgrid(xy[1] + offsets, xy[0] + offsets, indexing="ij")
    return map_coordinates(frame, [rows, cols], order=3, mode="nearest")


def compare_flat_frames(
    current: Any,
    reference: Any,
    *,
    half_window: int = DEFAULT_FLAT_HALF_WINDOW,
) -> FlatComparison:
    """纯函数: 亚像素对心后比较两帧平场斑的形状相似度。

    为什么不是零点漂移
    ------------------
    零点位置对**光路是否同一条**是个坏观测量: 它把「整条光轴平移」和「光路真
    变了」混在一个数里, 于是判据只能在「容忍 5 px (漏判)」和「容忍 100 px (误
    判)」之间二选一。实测: 2026-11-05 同一条光路的两次采集零点就差 16 px, 而
    斑形几乎逐像素一致; 反过来 2026-11-02 的零点差异不大, 斑却从 FWHM 12 px
    变成 22 px。先对心再比形状, 才把「平移」和「形变」分成两件事。

    判据 (一)
    ---------
    两帧各自取亚像素零点 (加权重心), 以该零点为心重采样出同尺寸方形窗, 轻度高斯
    平滑 (压单像素噪声), 减均值后取归一化互相关。对整体亮度标定不敏感 (曝光不同
    不影响), 只对**形状**敏感。

    判据 (二)
    ---------
    同一对窗口再算一次**二阶矩** (:func:`flat_shape` → :func:`compare_flat_shape`),
    结果放进新增的 ``shape`` / ``reference_shape`` / ``shape_comparison`` 三个字段。
    原因是 NCC 太宽容: 实测平场帧的合成扰动扫描里, 20% 各向异性拉伸 NCC 0.9874、
    25° 旋转 0.9860、30% 各向同性放宽 (离焦代理) 0.976 —— 都会穿过 0.90 的相似度
    下限被判成「同一条光路」。二阶矩对这三类扰动的响应是 6-17% / 0.03-0.24 / 主轴
    转角, 分得开; 而对平移与增益它仍然给 1.00/1.00, 与 NCC 的强项不冲突。两条判据
    在 :func:`optical_probe.assess_reuse` 里**同时**要求通过。

    Args:
        current: 当前平场帧 (2D)。
        reference: 参照平场帧 (2D); 两帧尺寸必须一致 (同一相机同一 ROI)。
        half_window: 方形窗半宽 (px)。

    Returns:
        :class:`FlatComparison`。

    Raises:
        ValueError: 帧不是 2D / 空 / 两帧尺寸不同。
    """
    from scipy.ndimage import gaussian_filter

    a = _prepare_flat(current)
    b = _prepare_flat(reference)
    if a.shape != b.shape:
        raise ValueError(f"平场帧尺寸不一致: {a.shape} vs {b.shape}")
    centre_a = _flat_centre(a, DEFAULT_CENTRE_HALF_WINDOW)
    centre_b = _flat_centre(b, DEFAULT_CENTRE_HALF_WINDOW)
    window_a = gaussian_filter(_flat_window(a, centre_a, half_window), 1.5)
    window_b = gaussian_filter(_flat_window(b, centre_b, half_window), 1.5)
    window_a = window_a - float(window_a.mean())
    window_b = window_b - float(window_b.mean())
    norm = float(np.linalg.norm(window_a) * np.linalg.norm(window_b))
    similarity = float((window_a * window_b).sum() / norm) if norm > 0.0 else 0.0
    # 二阶矩复用上面这两片窗 (平滑 + 去均值 + 对心重采样都做过了): 再重采样一次
    # 只会让 NCC 与形状判据看的不是同一个东西, 出分歧时无法归因。
    shape_radius = int(min(DEFAULT_SHAPE_RADIUS_PX, half_window))
    shape_a = flat_shape(window_a, radius=shape_radius)
    shape_b = flat_shape(window_b, radius=shape_radius)
    return FlatComparison(
        similarity=similarity,
        drift_px=(centre_a[0] - centre_b[0], centre_a[1] - centre_b[1]),
        window_px=int(2 * half_window + 1),
        current_xy=(float(centre_a[0]), float(centre_a[1])),
        reference_xy=(float(centre_b[0]), float(centre_b[1])),
        shape=shape_a,
        reference_shape=shape_b,
        shape_comparison=compare_flat_shape(shape_a, shape_b),
    )


def save_flat_frame(
    directory: Path | str,
    frame: Any,
    *,
    spot_xy: tuple[float, float] | None = None,
    exposure_ms: float | None = None,
    slm_serial: str | None = None,
) -> str | None:
    """把一帧平场存成 :data:`FLAT_FRAME_FILE` (float32, 压缩)。

    写不下只记一条 warning 并返回 None —— 少一帧形状证据不该让会话失败
    (下次复用自动退回零点判比)。

    Args:
        directory: 写到哪个目录 (采集目录 / 会话目录)。
        frame: 2D 平场帧。
        spot_xy: 该帧的零点 ``(x, y)``, 可缺。
        exposure_ms: 该帧的曝光 (ms), 可缺。
        slm_serial: SLM 序列号, 可缺。

    Returns:
        写下的文件名 (:data:`FLAT_FRAME_FILE`), 失败为 None。
    """
    target = Path(directory) / FLAT_FRAME_FILE
    payload: dict[str, Any] = {
        "frame": np.asarray(frame, dtype=np.float32),
        "spot_xy": np.asarray(
            spot_xy if spot_xy is not None else (np.nan, np.nan), dtype=np.float64
        ),
        "exposure_ms": np.asarray(
            np.nan if exposure_ms is None else float(exposure_ms), dtype=np.float64
        ),
        "slm_serial": np.asarray("" if slm_serial is None else str(slm_serial)),
    }
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **payload)
    except OSError as exc:
        logger.warning(f"平场参照帧写不下 ({target}): {exc!r}")
        return None
    return FLAT_FRAME_FILE


def load_flat_frame(
    directory: Path | str, name: str | None = None
) -> np.ndarray | None:
    """读回 :func:`save_flat_frame` 写下的平场帧。

    Args:
        directory: 该帧所在目录。
        name: 文件名, 通常取自侧车的 ``flat_frame`` 键。**只认真给出的名字**:
            ``None`` 与空串都返回 None, 不会去猜 :data:`FLAT_FRAME_FILE` ——
            侧车缺这一键就表示「这次没有形状证据」, 调用方退回零点判比;
            若这里替它补默认名, 目录里恰好躺着一帧旧文件就会被当成证据。

    Returns:
        2D float64 帧; 文件缺失或解不开返回 None (**不抛**)。
    """
    if not name:
        return None
    path = Path(directory) / str(name)
    try:
        with np.load(path, allow_pickle=False) as payload:
            return np.asarray(payload["frame"], dtype=np.float64)
    except (OSError, KeyError, ValueError) as exc:
        logger.warning(f"平场参照帧读不出 ({path}): {exc!r}")
        return None


__all__ = [
    "DEFAULT_CENTRE_HALF_WINDOW",
    "DEFAULT_FLAT_HALF_WINDOW",
    "DEFAULT_MAX_ASPECT_MISMATCH",
    "DEFAULT_MAX_AXIS_TILT_DEG",
    "DEFAULT_MAX_SIZE_RATIO",
    "DEFAULT_MIN_ASPECT_FOR_ANGLE",
    "DEFAULT_MIN_FLAT_SIMILARITY",
    "DEFAULT_SHAPE_RADIUS_PX",
    "FLAT_FRAME_FILE",
    "FLAT_FROZEN_SIDECAR",
    "FlatComparison",
    "ShapeComparison",
    "ShapeMetrics",
    "compare_flat_frames",
    "compare_flat_shape",
    "flat_shape",
    "load_flat_frame",
    "save_flat_frame",
]
