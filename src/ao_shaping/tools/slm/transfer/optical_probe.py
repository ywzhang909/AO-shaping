"""光路复用探针 (optical-path reuse probe)。

这是一个**便宜探针**, 不是新的优化器: 在 ``reuse_latest`` 开启时、重建采集与
拟合之前, 打开相机 + SLM, 量三个「当前光路条件」:

1. 零点位置 ``zero_xy`` (0 级 = 帧全局最大, 用 :func:`capture.locate`);
2. 相机的可用曝光区间 ``exposure_range_ms``;
3. 相机的取图格式 (``camera_format``, 如 ``"MONO8"``, 及位深 ``bit_depth``)。

然后与保存的 ``latest`` 模型 (``data/slm_models/<serial>/latest``) 里记录的
同一批次条件做比对 (:func:`same_path`)。一致 → 光路可复用 → workflow 跳过
采集+拟合, 直接复用 latest 冻结参数跑 CCD 锚定块; 不一致 → 走正常全流程。

判定是**纯函数**, 任何一步读不到/解不开都只返回不一致 (或 ``None``), 绝不抛
异常。本模块顶层**不 import 任何设备/SDK**; 设备在 :func:`measure_probe` 内部
惰性打开。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger


#: 零点漂移容忍 (像素)。**只用于没有平场参照帧时的退路**: 探针复用是「便宜
#: 比对」, 几像素内的零点平移认为光路没变。有参照帧时改由
#: :func:`compare_flat_frames` 的形状相似度判比 —— 见 :func:`same_path`。
DEFAULT_TOL_ZERO_PX = 5.0
#: 曝光区间上界放宽倍数 (下界通常触及传感器最小档, 不稳定, 不参与判比)。
#: 曝光是试出来的自适应结果, 允许 1.5 倍以内的浮动。
DEFAULT_TOL_EXPOSURE_FACTOR = 1.5
#: latest 目录里记录探针条件/格式的文件名。
FLAT_FROZEN_SIDECAR = "_optical_probe.json"
#: 平场参照帧 (平面相位下的 0 级斑) 的文件名。与 :data:`FLAT_FROZEN_SIDECAR`
#: 同目录: 采集段写进采集目录, 拟合段复制进 ``latest``; 探针在当前会话目录写
#: 自己那一帧。缺它就是「没有形状证据」, 退到 ``DEFAULT_TOL_ZERO_PX`` 的零点判比。
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
#: 零点漂移的**硬上限** (px)。形状判据已经把纯平移对心掉, 但 latest 模型里
#: ``origin``/``amplitude_center``/``amplitude_width`` 都是按旧零点写下的相机像元量;
#: 平移超过这个量级就不再是「同一条光路」的复用, 宁可重采。40 px 约当平场斑
#: 第一圈衍射环直径的 2 倍, 也是光照包络宽度 (约 600 px) 的 7%。
DEFAULT_MAX_ZERO_DRIFT_PX = 40.0
#: 零点漂移**黄警**阈值 (px)，#243 前置二。派生自硬上限之半：40 px ≈ 平场斑
#: 第一圈衍射环直径的 2 倍（见 :data:`DEFAULT_MAX_ZERO_DRIFT_PX` 注释），
#: 20 px 即一个环径 —— 斑整体挪动一个环径时复用仍成立（不否决），但 #242 的
#: G2b 门应显黄提示「光轴在走」。三档关系：``5 px``（无平场帧退路容忍，
#: :data:`DEFAULT_TOL_ZERO_PX`）< ``20 px``（黄警）< ``40 px``（硬上限否决）。
DEFAULT_WARN_ZERO_DRIFT_PX = DEFAULT_MAX_ZERO_DRIFT_PX / 2.0
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
class OpticalProbe:
    """一次光路探针的量测结果。"""

    #: 0 级光斑的 (x, y) 像素坐标 (帧内全局最大背后的局部矩, 同 capture.locate)。
    zero_xy: tuple[float, float]
    #: 相机曝光区间 ``(min_ms, max_ms)``。
    exposure_range_ms: tuple[float, float]
    #: 相机取图格式名 (如 ``"MONO8"``); 读不到为 None。
    camera_format: str | None = None
    #: 像素位深 (8/12/16); 读不到为 None。
    bit_depth: int | None = None
    #: 本次探针用的 SLM 序列号 (与 capture.py 同一口径: 打开后读 ``serial_number``)。
    #: 读不到为 None —— 调用方据此认为「没有可比的 latest 模型」。
    slm_serial: str | None = None
    #: 这一帧平场的文件名 (:data:`FLAT_FRAME_FILE`), 与该探针记录同目录;
    #: 没有落盘的平场帧时为 None (于是形状判比不可用)。
    flat_frame: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


@dataclass(frozen=True)
class ReuseDecision:
    """光路复用判比的完整结论 (布尔 + 逐条判据 + 形状细节)。"""

    #: 全部判据通过 = 同一条光路。
    same: bool
    #: 判据 → 人类可读结论 (键: ``flat`` / ``shape`` / ``zero`` / ``exposure`` /
    #: ``camera_format`` / ``bit_depth`` / ``reference``)。写进
    #: ``optical_probe.json`` 供 GUI 逐条解释, 也便于事后追哪一项变了。
    criteria: dict[str, str]
    #: 形状比对细节; 没有平场帧时为 None。
    flat: FlatComparison | None = None

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

    为什么必须能返回空串: :func:`format_probe_verdict` 的老调用点 (GUI 的跳过文案、
    没有平场帧的老 latest 目录) 传的是 ``flat=None`` 或不含二阶矩的旧式
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
    在 :func:`assess_reuse` 里**同时**要求通过。

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


def measure_probe(
    *,
    camera: str = "daheng",
    cam_id: int = 0,
    slm_number: int = 1,
    save_frame_dir: Path | str | None = None,
) -> OpticalProbe:
    """打开相机 + SLM, 取一帧平场, 量一次光路探针。

    Args:
        camera: 相机后端名 (传给 ``create_camera``)。
        cam_id: 相机 id。
        slm_number: SLM 编号。
        save_frame_dir: 给出时把这一帧平场写成
            :data:`FLAT_FRAME_FILE` 存进该目录 (形状判比要用它), 并把文件名记进
            ``flat_frame``。写不下只是少一条判据, 不影响探针本身。

    Returns:
        :class:`OpticalProbe`。

    Raises:
        DeviceOpenError / ImportError: 设备打开失败或 SDK 缺失时抛出 (硬件层语义)。
    """
    from .hardware import opened_device, prepare_camera_sdk
    from ao_shaping.drivers.ccd.common import (
        create_camera,
        get_camera_exposure_range,
    )
    from ao_shaping.drivers.slm.santec import Santec
    from ao_shaping.utils.image.exposure_control import configured_daheng_exposure_limit

    prepare_camera_sdk(camera)
    cam = create_camera(camera, cam_id, exposure_time_ms=0.02)

    # 全函数最长的设备占用: 相机 + SLM, 退出即释放。探针只读一帧、不下发图案,
    # 但为与采集段一致 (零点装在「平场面」上) 还是保持零相位覆盖层。
    with opened_device(cam) as cam_handle:
        with opened_device(
            Santec(slm_number=slm_number, wavelength=1064, video_mode=0)
        ) as slm:
            # 序列号: 打开后才有 (与 capture.py 同一口径); 读不到保持 None,
            # 不报错 (仿真面板没有该属性)。
            slm_serial = None
            try:
                slm_serial = getattr(slm, "serial_number", None)
            except Exception as exc:  # pragma: no cover - 设备属性读取失败
                logger.warning(f"光路探针: 无法读取 SLM 序列号: {exc!r}")
            slm.set_phase_overlay(None)
            slm.display_data(slm.create_phase_from_array(np.zeros((1200, 1920))))
            floor, ceiling = get_camera_exposure_range(cam_handle)
            limit = configured_daheng_exposure_limit() if camera == "daheng" else 0
            if limit > 0:
                ceiling = min(ceiling, limit)
            raw = cam_handle.get_numpy_image(n_sample=1, skip_first=False)
            frame = np.asarray(raw, dtype=np.float64)
            zero = locate_zero(frame)
            exposure_range = (floor, ceiling)
            camera_format = read_camera_format(cam_handle)
            bit_depth = read_bit_depth(cam_handle)
            # 复位平场, 与采集段 finally 一致
            slm.display_data(slm.create_phase_from_array(np.zeros((1200, 1920))))
    probe = OpticalProbe(
        zero_xy=zero,
        exposure_range_ms=exposure_range,
        camera_format=camera_format,
        bit_depth=bit_depth,
        slm_serial=None if slm_serial is None else str(slm_serial),
        flat_frame=(
            None
            if save_frame_dir is None
            else save_flat_frame(
                save_frame_dir,
                frame,
                spot_xy=zero,
                exposure_ms=0.02,
                slm_serial=None if slm_serial is None else str(slm_serial),
            )
        ),
    )
    logger.info(
        "光路探针: zero={} exposure={} format={} bit={}",
        probe.zero_xy,
        probe.exposure_range_ms,
        probe.camera_format,
        probe.bit_depth,
    )
    return probe


def locate_zero(frame: Any) -> tuple[float, float]:
    """定位 0 级光斑: 帧全局最大 + 局部矩 (同 capture.locate 口径)。

    纯函数, 不打开设备; 供 :func:`measure_probe` 与纯测试直接使用。
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter

    smooth = gaussian_filter(np.asarray(frame, dtype=float), 3)
    row, col = np.unravel_index(smooth.argmax(), smooth.shape)
    y0, y1 = max(0, row - 30), min(frame.shape[0], row + 31)
    x0, x1 = max(0, col - 30), min(frame.shape[1], col + 31)
    weights = np.maximum(
        np.asarray(frame, dtype=float)[y0:y1, x0:x1]
        - np.median(np.asarray(frame, dtype=float)),
        0,
    )
    yy, xx = np.mgrid[y0:y1, x0:x1]
    if float(weights.sum()) <= 0:
        return 0.0, 0.0
    return float((xx * weights).sum() / weights.sum()), float(
        (yy * weights).sum() / weights.sum()
    )


def read_camera_format(cam: Any) -> str | None:
    """读相机当前取图格式名 (如 ``"MONO8"``); 读不到返回 None。"""
    try:
        pf = cam.cam.PixelFormat
        return str(pf.get())
    except Exception as exc:  # 格式读不到不是光路判定错误
        logger.warning(f"无法读相机格式: {exc!r}")
        return None


def read_bit_depth(cam: Any) -> int | None:
    """读相机位深 (优先驱动缓存 ``bit_depth``, 再尝试 gxipy PixelSize)。"""
    bd = getattr(cam, "bit_depth", None)
    if isinstance(bd, (int, float)) and bd > 0:
        return int(bd)
    try:
        pixel_size = cam.cam.PixelSize.get()
        if pixel_size is not None:
            # GenICam PixelSize 即位深 (8/12/16); 驱动缓存缺失时才回退到这里。
            depth = int(pixel_size)
            if depth > 0:
                return depth
    except Exception as exc:  # pragma: no cover - 设备回退路径
        logger.warning(f"无法读相机位深: {exc!r}")
    return None


def _frozen_camera_format(frozen: dict[str, Any]) -> str | None:
    """从冻结参数里取相机格式 (latest 目录 / 冻结模型可能存 ``camera_format``)。"""
    value = frozen.get("camera_format")
    if value is None and frozen.get("probe", {}):
        value = frozen["probe"].get("camera_format")
    return str(value) if value else None


def probe_from_latest(latest_dir: Path | str) -> OpticalProbe | None:
    """从 ``latest`` 目录重建上一次会话记录的探针条件。

    读 ``frozen_parameters.json`` 的 ``origin`` (零点), 并从 ``_optical_probe.json``
    侧车读曝光区间 / 取图格式 / 位深 (侧车缺失时退回 ``provenance.json``)。任何一步
    读不到/解不开 → 返回 None (表示没有可比的参照, 调用方按「不一致」回退)。

    Args:
        latest_dir: latest 目录 (含 ``frozen_parameters.json``)。

    Returns:
        该会话的 :class:`OpticalProbe`, 或 None。
    """
    latest_dir = Path(latest_dir)
    frozen_path = latest_dir / "frozen_parameters.json"
    if not frozen_path.is_file():
        return None
    try:
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning(f"latest 探针对照读不出 frozen_parameters.json: {exc!r}")
        return None
    origin = frozen.get("origin")
    if not isinstance(origin, (list, tuple)) or len(origin) != 2:
        logger.warning("latest frozen_parameters.json 缺有效 origin, 无零点头")
        return None
    try:
        zero = (float(origin[0]), float(origin[1]))
    except (TypeError, ValueError):
        return None

    # 侧车 (``_optical_probe.json``) 是完整的参照: 曝光区间 + 取图格式 + 位深。
    # 侧车缺失时退回冻结参数里可能带的字段, 再缺就留 None —— 缺项在
    # :func:`same_path` 里视作「不可比」, 于是走正常全流程。
    sidecar = _read_sidecar(latest_dir)
    camera_format = sidecar.get("camera_format") or _frozen_camera_format(frozen)
    bit_depth = _as_int(sidecar.get("bit_depth")) or _as_int(frozen.get("bit_depth"))
    exposure_range = _as_range(sidecar.get("exposure_range_ms"))
    if exposure_range is None:
        exposure_range = _read_exposure_range(latest_dir)
    return OpticalProbe(
        zero_xy=zero,
        exposure_range_ms=exposure_range,
        camera_format=camera_format,
        bit_depth=bit_depth,
        flat_frame=_latest_flat_frame(latest_dir, sidecar),
    )


def _latest_flat_frame(latest_dir: Path, sidecar: dict[str, Any]) -> str | None:
    """latest 目录里的平场参照帧文件名; 侧车没登记或文件不在 → None。

    侧车登记的名字优先 (允许将来换文件名), 但必须真的存在 —— 只信登记不看文件
    会让 :func:`load_flat_frame` 每场都多打一条 warning。
    """
    name = sidecar.get("flat_frame")
    if not isinstance(name, str) or not name:
        return None
    return name if (Path(latest_dir) / name).is_file() else None


def _read_sidecar(latest_dir: Path) -> dict[str, Any]:
    """读探针参照侧车; 读不到或坏了就返回空字典。"""
    sidecar = Path(latest_dir) / FLAT_FROZEN_SIDECAR
    if not sidecar.is_file():
        return {}
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning(f"latest 探针侧车读不出: {exc!r}")
        return {}
    return data if isinstance(data, dict) else {}


def _as_int(value: Any) -> int | None:
    """把可转的数转成 int; 不可转返回 None。"""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_range(value: Any) -> tuple[float, float] | None:
    """把 ``[lo, hi]`` 形状的值转成 ``(float, float)``; 不可用返回 None。"""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


def _read_exposure_range(latest_dir: Path) -> tuple[float, float]:
    """从 latest 目录取曝光区间; 无可读来源时回落到默认传感器区间。"""
    for candidate in ("provenance.json", "result.json"):
        path = Path(latest_dir) / candidate
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        rng = _as_range(data.get("exposure_range_ms"))
        if rng is None:
            rng = _as_range((data.get("capture") or {}).get("exposure_range"))
        if rng is not None:
            return rng
    # 无可读曝光区间 → 用默认传感器区间 (与 get_camera_exposure_range 兜底一致)。
    return (0.011, 10_000.0)


def assess_reuse(
    current: OpticalProbe,
    reference: OpticalProbe | None,
    *,
    current_frame: Any = None,
    reference_frame: Any = None,
    tol_zero_px: float = DEFAULT_TOL_ZERO_PX,
    tol_exposure_factor: float = DEFAULT_TOL_EXPOSURE_FACTOR,
    min_similarity: float = DEFAULT_MIN_FLAT_SIMILARITY,
    max_zero_drift_px: float = DEFAULT_MAX_ZERO_DRIFT_PX,
) -> ReuseDecision:
    """纯函数判比: 当前探针 vs latest 参照是否同一条光路, 逐条给结论。

    判据 (全部通过才算同一条光路):

    - **形状 · 逐像素** (有平场帧时): :func:`compare_flat_frames` 的相似度
      ``>= min_similarity``; 零点漂移只作硬上限 ``<= max_zero_drift_px``
      —— 斑形一致但整体平移几十像素是「同一条光路, 光轴挪了一下」, 而斑形变了
      是「光路真的变了」;
    - **形状 · 二阶矩** (有平场帧时): ``flat.shape_comparison.within_tolerance``
      —— NCC 对「斑变大/变扁/转了」太宽容 (实测扫描: 20% 拉伸 0.9874, 25° 旋转
      0.9860, 30% 放宽 0.976, 全部能穿过 0.90), 所以三条轴 (尺寸比 / 椭圆度失配 /
      主轴夹角) 要各自在容差内, 见 :func:`compare_flat_shape`。逐条结论写进
      ``criteria["shape"]``; 算不出 (退化帧) 一律按**不过**处理;
    - **形状** (缺平场帧时, 老 latest 目录): 退回 ``zero_xy`` 各轴
      ``<= tol_zero_px`` —— 没有形状证据就只能用位置, 宁可严格;
    - 曝光上界在当前区间 ``[参考/因子, 参考*因子]`` 内 (上界稳; 下界不稳不判);
    - ``camera_format`` 完全相等 (None 之一视作不可比 → 不同);
    - ``bit_depth`` 完全相等 (None 之一视作不可比 → 不同)。

    任何一步配不上 (含 ``reference=None``) → ``same=False``, **绝不抛异常**;
    形状比对本身抛 ValueError (帧尺寸不一致等) 也按「不同」处理。

    Args:
        current: 本次实测探针。
        reference: latest 记录的参照探针; None 表示没有参照。
        current_frame: 本次平场帧 (2D); None 表示没有形状证据。
        reference_frame: 参照平场帧 (2D); None 表示没有形状证据。
        tol_zero_px: 缺形状证据时的零点容忍 (px)。
        tol_exposure_factor: 曝光上界放宽倍数。
        min_similarity: 形状相似度下限。
        max_zero_drift_px: 零点漂移硬上限 (px)。

    Returns:
        :class:`ReuseDecision`。
    """
    criteria: dict[str, str] = {}
    flat: FlatComparison | None = None
    if reference is None:
        criteria["reference"] = "没有 latest 参照"
        return ReuseDecision(False, criteria, None)

    # 形状 (主判据) / 零点 (退路)
    if current_frame is not None and reference_frame is not None:
        try:
            flat = compare_flat_frames(current_frame, reference_frame)
        except ValueError as exc:
            criteria["flat"] = f"平场帧不可比 ({exc})"
            criteria["shape"] = "平场帧不可比, 无形状度量"
            return ReuseDecision(False, criteria, None)
        drift = max(abs(flat.drift_px[0]), abs(flat.drift_px[1]))
        criteria["flat"] = (
            f"平场斑相似度 {flat.similarity:.3f} (下限 {min_similarity:.2f}), "
            f"漂移 {flat.drift_px[0]:+.1f}/{flat.drift_px[1]:+.1f} px "
            f"(上限 {max_zero_drift_px:.0f} px)"
        )
        flat_ok = flat.similarity >= min_similarity and drift <= max_zero_drift_px
        # 第二条 (互补) 判据: NCC 放行不代表形状没变, 二阶矩再卡一遍三条轴。
        moment_ok, criteria["shape"] = _shape_gate_text(flat)
    else:
        try:
            dz = (
                abs(current.zero_xy[0] - reference.zero_xy[0]),
                abs(current.zero_xy[1] - reference.zero_xy[1]),
            )
        except (TypeError, IndexError):
            criteria["zero"] = "零点不可比"
            criteria["shape"] = "无平场帧, 形状判据不可用"
            return ReuseDecision(False, criteria, None)
        criteria["zero"] = (
            f"零点漂移 {dz[0]:.1f}/{dz[1]:.1f} px (无平场帧, 容忍 {tol_zero_px:.0f} px)"
        )
        # 没有平场帧 = 没有形状证据: 二阶矩判据不参与, 退回零点判比 (既有语义)。
        criteria["shape"] = "无平场帧, 形状判据不可用 (退到零点判比)"
        flat_ok = dz[0] <= tol_zero_px and dz[1] <= tol_zero_px
        moment_ok = True

    # 曝光上界 (下界通常触及传感器最小档, 不稳定, 不参与判比)
    ref_hi = reference.exposure_range_ms[1] if reference.exposure_range_ms else 0.0
    cur_hi = current.exposure_range_ms[1] if current.exposure_range_ms else 0.0
    exposure_ok = True
    if ref_hi > 0 and not (
        ref_hi / tol_exposure_factor <= cur_hi <= ref_hi * tol_exposure_factor
    ):
        exposure_ok = False
    criteria["exposure"] = f"曝光上界 {cur_hi:g} ms vs 参照 {ref_hi:g} ms"

    # 格式与位深: 都必须非空且完全一致
    format_ok = (
        current.camera_format is not None
        and reference.camera_format is not None
        and current.camera_format == reference.camera_format
    )
    criteria["camera_format"] = (
        f"取图格式 {current.camera_format!r} vs 参照 {reference.camera_format!r}"
    )
    bit_ok = (
        current.bit_depth is not None
        and reference.bit_depth is not None
        and current.bit_depth == reference.bit_depth
    )
    criteria["bit_depth"] = (
        f"像素位深 {current.bit_depth!r} vs 参照 {reference.bit_depth!r}"
    )
    return ReuseDecision(
        bool(flat_ok and moment_ok and exposure_ok and format_ok and bit_ok),
        criteria,
        flat,
    )


def same_path(
    current: OpticalProbe,
    reference: OpticalProbe | None,
    *,
    tol_zero_px: float = DEFAULT_TOL_ZERO_PX,
    tol_exposure_factor: float = DEFAULT_TOL_EXPOSURE_FACTOR,
    current_frame: Any = None,
    reference_frame: Any = None,
    min_similarity: float = DEFAULT_MIN_FLAT_SIMILARITY,
    max_zero_drift_px: float = DEFAULT_MAX_ZERO_DRIFT_PX,
) -> bool:
    """:func:`assess_reuse` 的布尔简写 (判据与兼容语义见前者)。

    没有传 ``current_frame``/``reference_frame`` 时行为与历史版本一致: 零点各轴
    ``<= tol_zero_px`` 才算同一条光路。
    """
    return assess_reuse(
        current,
        reference,
        current_frame=current_frame,
        reference_frame=reference_frame,
        tol_zero_px=tol_zero_px,
        tol_exposure_factor=tol_exposure_factor,
        min_similarity=min_similarity,
        max_zero_drift_px=max_zero_drift_px,
    ).same


def zero_drift_verdict(
    current_zero_xy: tuple[float, float],
    reference_zero_xy: tuple[float, float] | None,
    *,
    warn_px: float = DEFAULT_WARN_ZERO_DRIFT_PX,
    hard_px: float = DEFAULT_MAX_ZERO_DRIFT_PX,
) -> tuple[str, str]:
    """零级漂移三档结论 (``"ok"`` / ``"warn"`` / ``"hard"``)，#243 前置二。

    逐轴绝对差取较大者，与 :func:`assess_reuse` 的 ``drift`` 同一算法。
    ``reference_zero_xy`` 为 None 或坐标不可比时返回 ``("ok", 说明)`` ——
    没有参照就没有漂移证据，判定端不得凭空报警（与 ``assess_reuse``
    的 fail-open on missing reference 一致）。
    """
    if reference_zero_xy is None:
        return "ok", "没有参照零点, 漂移判据不参与"
    try:
        dx = abs(float(current_zero_xy[0]) - float(reference_zero_xy[0]))
        dy = abs(float(current_zero_xy[1]) - float(reference_zero_xy[1]))
    except (TypeError, ValueError, IndexError):
        return "ok", "零点不可比, 漂移判据不参与"
    drift = max(dx, dy)
    if drift > hard_px:
        return "hard", f"零级漂移 {dx:.1f}/{dy:.1f} px 超硬上限 {hard_px:.0f} px"
    if drift > warn_px:
        return "warn", f"零级漂移 {dx:.1f}/{dy:.1f} px 超黄警 {warn_px:.0f} px (硬上限 {hard_px:.0f} px 内, 复用仍成立)"
    return "ok", f"零级漂移 {dx:.1f}/{dy:.1f} px (黄警 {warn_px:.0f} px)"


def format_probe_verdict(
    verdict: bool,
    *,
    current: OpticalProbe | None = None,
    reference: OpticalProbe | None = None,
    flat: FlatComparison | None = None,
) -> str:
    """把光路复用判决格式化成一行人类可读的文案。纯函数, 不读文件。

    ``flat`` 给出时按形状判据描述 (相似度 + 漂移), 否则沿用零点/曝光口径。
    ``flat`` 里带二阶矩 (:func:`compare_flat_frames` 总是带) 时再补一段形状摘要,
    让 GUI 看得出是**哪条轴**动了; 没有平场帧时文案一字不变。
    """
    shape_tail = _shape_tail(flat)
    if verdict:
        if flat is not None:
            return (
                f"光路复用: 与 latest 模型同一条光路 (平场斑相似度 "
                f"{flat.similarity:.3f}, 零点漂移 {flat.drift_px[0]:+.1f}/"
                f"{flat.drift_px[1]:+.1f} px{shape_tail}), 跳过采集+拟合直接复用"
            )
        return "光路复用: 与 latest 模型同一条光路, 跳过采集+拟合直接复用"
    if reference is None:
        return "光路复用: 无 latest 参照, 走正常全流程"
    if current is None:
        return "光路复用: 当前探针缺失, 走正常全流程"
    if flat is not None:
        return (
            "光路复用: 光路已变 (平场斑相似度 {:.3f} < {:.2f}, 漂移 {:+.1f}/{:+.1f} px{}), "
            "走正常全流程"
        ).format(
            flat.similarity,
            DEFAULT_MIN_FLAT_SIMILARITY,
            flat.drift_px[0],
            flat.drift_px[1],
            shape_tail,
        )
    return (
        "光路复用: 光路已变 (zero=({:.1f},{:.1f}) vs {:.1f},{:.1f}, "
        "exposure_up={} vs {}), 走正常全流程"
    ).format(
        current.zero_xy[0],
        current.zero_xy[1],
        reference.zero_xy[0],
        reference.zero_xy[1],
        current.exposure_range_ms[1] if current.exposure_range_ms else float("nan"),
        reference.exposure_range_ms[1] if reference.exposure_range_ms else float("nan"),
    )


#: 光路复用被跳过 (而非「未通过」) 的原因。写进 ``optical_probe.json`` 的
#: ``reason`` 键, 人类可读; ``REUSE_SKIP_PROBE_FAILED`` 一条由调用方拼上具体
#: 异常类型与消息, 不在此列。这些原因供「运行处」说明「为什么没走复用」。
REUSE_SKIP_DISABLED = "光路复用未开启"
REUSE_SKIP_DATASET = "数据集会话不参与光路复用"
REUSE_SKIP_OFFLINE = "离线模式不参与光路复用"
REUSE_SKIP_NO_SERIAL = "读不到 SLM 序列号"
REUSE_SKIP_NO_LATEST = "该 SLM 没有 latest 模型"
REUSE_SKIP_PROBE_FAILED = "光路探针失败"
#: latest 模型的采集网格小于本次目标需要的网格（复用会把够不着的目标交回来）。
REUSE_SKIP_GRID = "latest 模型网格不足以覆盖本次目标"


def format_probe_skip_line(reason: str) -> str:
    """光路复用被跳过 → 一行「光路复用：跳过（原因） → 走正常全流程」。

    纯函数, 不读文件。``reason`` 为空/空白时退到无原因的兜底文案, 绝不抛。
    """
    reason = (reason or "").strip()
    if not reason:
        return "光路复用：跳过 → 走正常全流程"
    return f"光路复用：跳过（{reason}） → 走正常全流程"


__all__ = [
    "DEFAULT_TOL_ZERO_PX",
    "DEFAULT_WARN_ZERO_DRIFT_PX",
    "DEFAULT_TOL_EXPOSURE_FACTOR",
    "FLAT_FROZEN_SIDECAR",
    "OpticalProbe",
    "measure_probe",
    "locate_zero",
    "read_camera_format",
    "read_bit_depth",
    "probe_from_latest",
    "same_path",
    "zero_drift_verdict",
    "format_probe_verdict",
    "format_probe_skip_line",
    "REUSE_SKIP_DISABLED",
    "REUSE_SKIP_DATASET",
    "REUSE_SKIP_OFFLINE",
    "REUSE_SKIP_NO_SERIAL",
    "REUSE_SKIP_NO_LATEST",
    "REUSE_SKIP_GRID",
    "REUSE_SKIP_PROBE_FAILED",
]
