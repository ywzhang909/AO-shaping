# -*- coding: utf-8 -*-
"""
fouriergsnet_optimize.py — SLM+CCD 束整形一体化工具 (标定 + FourierGSNet 闭环优化)
====================================================================================

单文件包含四部分内容:
  [A] 几何标定类 SLMCCDCalibrator   装配辅助/光束位置/K-尺度/旋转/验证/workzone
  [B] LUT 标定类 SLMLUTCalibrator   灰度-相位曲线/标定显示通道/残差验证
  [C] FourierGSNet-lite            GS物理展开(FFT双向投影) + CNN残差回归
  [D] ShapingSystem                测量/显示/初值/扰动采样/微调/闭环(含在线再学习)

三大数据驱动升级(相对早期版本):
  1) 域随机化采样   —— 扰动幅度混合 {0.3,0.5,0.8}rad(低阶)/{0.1,0.2,0.3}rad(高阶),
                       目标尺寸 HALF 在 0.30~0.40 随机, 提升网络对工况变化的鲁棒性
  2) 残差头扩维     —— c_head 从 8 维扩到 24 维(Z4~Z27: 低阶8 + 高阶16),
                       可吸收几何标定未建模的静态像差, 同时保持参数化安全
  3) 在线 replay   —— 闭环中监测均匀度退化与 |c_hat| 漂移, 超阈值自动采50组
                       扰动数据做3个epoch快调, 把失配校正变成持续学习

CLI:
  python fouriergsnet_optimize.py calibrate                  # 全流程标定(含装配辅助)
  python fouriergsnet_optimize.py calibrate --skip-align --skip-beam
  python fouriergsnet_optimize.py verify --calib calib.npz [--lut lut.npz]
  python fouriergsnet_optimize.py run --calib calib.npz --lut lut.npz [--steps 50]
  python fouriergsnet_optimize.py run ... --retrain          # 重跑初值+微调
  python fouriergsnet_optimize.py run ... --no-replay        # 关闭在线再学习

=========================== 调试指南(遇到问题先查这里) ===========================
| 现象                                   | 可能原因                     | 对策 |
|----------------------------------------|------------------------------|------|
| workzone 无信号/能量极低               | 曝光过短、衰减过重、裁剪框错位 | 调曝光; 重跑 calibrate 的 align/beam |
| init GS 的"信号区RMSE"纹丝不动         | workzone 套错位置            | 先 verify --calib 复核几何标定 |
| init GS RMSE 降但均匀度仍差           | 目标尺寸不匹配               | 看启动日志"物理边长", 调 HALF |
| finetune 的 |c|err 不下降               | 扰动信号弱/标签错            | 增曝光; 确认 place_on_panel 生效 |
| 闭环中均匀度持续走低                   | 热漂移超网络跟踪能力         | 在线 replay 会自动触发; 或降步长延长 settle |
| LUT verify 残差 std > 0.05 rad         | 半屏功率不对称/0级污染       | 重跑 beam 定位; 焦面加针孔挡0级 |
| 拟合残差 > 5%K (几何标定警告)          | 0级污染/曝光不足             | 提高闪耀周期避开0级; 重标 |
| 检测到 mirror=True                     | 光路奇次反射                 | 翻转SLM坐标或检查反射次数 |
====================================================================================
"""
from __future__ import annotations

import math
import time
from collections import deque
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

# =====================================================================
# 0. 全局常量 (所有可调参数集中于此, 调试优先改这里)
# =====================================================================
# ---- 模型 ----
N = 64                     # 模型工作区 (FFT网格, 与SLM面/焦面成归一化傅里叶对)
K_UNROLL = 5               # GS 物理展开层数 M (论文 M=5/10/15, 越大越准越慢)
N_ZERN_LOW = 8             # 低阶 Zernike 数 (Z4~Z11, 物理像差主导项)
N_ZERN = 24                # 总回归维度 (Z4~Z27: 低阶8 + 高阶16, 见 zernike_basis)
CH = 32                    # CNN 通道数
HALF = 0.35                # 闭环目标方形归一化半宽 (物理边长≈2*HALF*K/2 px, 启动日志会打印)
# ---- 域随机化采样 (升级1) ----
AMP_LOW_POOL = (0.3, 0.5, 0.8)     # 低阶扰动幅度池(rad), 1/3概率各取
AMP_HIGH_POOL = (0.1, 0.2, 0.3)    # 高阶扰动幅度池(rad), 高阶图案SLM实现损耗大, 取小
HALF_RANGE = (0.30, 0.40)          # 目标尺寸随机范围 (训练时目标尺寸也随机, 提升泛化)
# ---- 微调 ----
N_PERTURB = 240            # 扰动样本数 (低/中/高幅度各约1/3)
FT_EPOCHS = 12
FT_BATCH = 16
FT_LR = 2e-4
# ---- 在线 replay (升级3) ----
REPLAY_WINDOW = 10         # 滑动窗口步数
REPLAY_DROP = 0.6          # 均匀度 < 窗口中位数×此值 → 触发
REPLAY_C_MAX = 0.8         # |c_hat|均值 > 此值(rad) → 触发 (网络拼命补偿=系统变了)
REPLAY_EVERY = 0           # 周期性再微调间隔步数 (0=关闭; 如 200 = 每200步小调一次)
REPLAY_N = 50              # 触发后再采样数量
REPLAY_EPOCHS = 3          # 触发后快调轮数
REPLAY_LR = 1e-4           # 快调学习率(更小, 防灾难性遗忘)
REPLAY_COOLDOWN = 30       # 两次触发的最小间隔步数
# ---- 硬件 ----
SETTLE_S = 0.2             # SLM 显示后稳定等待(s); 若条纹/光斑有拖影调大
CAPTURE_TIMEOUT_S = 30.0
N_SAMPLE = 3               # 相机平均帧数
GS_ITERS_INIT = 15         # 设备自适应GS迭代数(初值)
EXPOSURE_MS = 1.2
CKPT = Path("data/shaping_test/fouriergsnet_lite.pth")
OUT_DIR = Path("data/shaping_test")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# =====================================================================
# 1. 物理基础 (归一化傅里叶对: prop 即"模型SLM面->模型焦面"的酉变换)
# =====================================================================
def prop(U):
    """简化前向: 傅里叶平面关系(对应论文 Fourier forward path).
    真实系统的光学链只出现在训练数据采集端, 网络训练/推理均不调用它."""
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(U, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def prop_inv(G):
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(G, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def wrap_pi(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


def zernike_basis(n, n_terms, device):
    """Z4~Z27 共24项在单位圆盘上的取值 (n_terms≤24).
    顺序: 离焦,像散×2,彗差×2,三叶×2,球差, 二级像散×2,四叶×2,
          二级彗差×2,二级三叶×2,五叶×2, 二级球差,三级像散×2,六叶×2, ...
    注: 用于回归基即可, 与Noll编号严格对应关系不影响使用."""
    t = torch.linspace(-1, 1, n, device=device)
    y, x = torch.meshgrid(t, t, indexing="ij")
    r = torch.sqrt(x * x + y * y)
    th = torch.arctan2(y, x)
    rc = torch.clamp(r, max=1.0)
    r2, r3, r4, r5, r6 = rc**2, rc**3, rc**4, rc**5, rc**6
    def pair(m, R):   # sin/cos 一对
        return [R * torch.sin(m * th), R * torch.cos(m * th)]
    Z = [
        2 * r2 - 1,                          # Z4  defocus
        *pair(2, r2),                        # Z5,6  astig
        *pair(1, 3 * r3 - 2 * rc),           # Z7,8  coma
        *pair(3, r3),                        # Z9,10 trefoil
        6 * r4 - 6 * r2 + 1,                 # Z11  spherical
        *pair(2, 4 * r4 - 3 * r2),           # Z12,13 2nd astig
        *pair(4, r4),                        # Z14,15 tetrafoil
        *pair(1, 10 * r5 - 12 * r3 + 3 * rc),  # Z16,17 2nd coma
        *pair(3, 5 * r5 - 4 * r3),           # Z18,19 2nd trefoil
        *pair(5, r5),                        # Z20,21 pentafoil
        20 * r6 - 30 * r4 + 12 * r2 - 1,     # Z22  2nd spherical
        *pair(2, 15 * r6 - 20 * r4 + 6 * r2),  # Z23,24 3rd astig
        *pair(4, 6 * r6 - 5 * r4),           # Z25,26 (6ρ⁶−5ρ⁴)
        *pair(6, r6),                        # Z27,28 hexafoil (取第24项止)
    ][:n_terms]
    Z = torch.stack(Z, 0)
    return Z * (r <= 1.0).float()


def aberration(Z, c):
    """系数->相位面. c:(n,)或(B,n) -> (N,N)或(B,N,N). 与einsum维度匹配见下方."""
    squeeze = c.dim() == 1
    if squeeze:
        c = c.unsqueeze(0)
    ab = torch.einsum("bn,nm->bm", c, Z.flatten(1)).unflatten(1, Z.shape[-2:])
    return ab.squeeze(0) if squeeze else ab


def make_square_target(n, half, device):
    """方形 top-hat 目标强度(单位能量). half: 归一化半宽."""
    t = torch.linspace(-1, 1, n, device=device)
    y, x = torch.meshgrid(t, t, indexing="ij")
    I = ((x.abs() <= half) & (y.abs() <= half)).float()
    return I / I.sum()


def gs_unroll(A_src, A_tgt, phi0, K, src_mask):
    """简化前向(无像差FFT)下K层GS双向投影 -> (相位, K通道物理特征).
    对应论文的 GS physics pipeline: 目标面替换幅值 -> 反投影 -> 源面保留相位."""
    phi = phi0
    feats = []
    for _ in range(K):
        G = prop(A_src * torch.exp(1j * phi))
        G = A_tgt * torch.exp(1j * torch.angle(G))     # 目标面幅值约束
        g = prop_inv(G)
        phi = wrap_pi(torch.angle(g)) * src_mask       # 源面幅值约束由A_src保证
        feats.append(phi)
    return phi, torch.stack(feats, 1)


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.c2(self.act(self.c1(x))))


class FourierGSNetLite(nn.Module):
    """GS物理展开(FFT双向投影, K通道特征) + CNN注入.
    c_head 输出 N_ZERN 维像差系数; 控制律: phi = phi_GS_unroll - c_hat·Z.
    物理结构 psi=phi+aberr 保证: 只要 c_hat 接近真实失配, 残差即被解析抵消."""

    def __init__(self, K, n_zern, ch, src_mask):
        super().__init__()
        self.K = K
        self.enc1 = ConvBlock(K + 2, ch)
        self.enc2 = ConvBlock(ch, ch * 2)
        self.dec = ConvBlock(ch * 3, ch)
        self.c_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                    nn.Linear(ch, n_zern))
        self.register_buffer("src_mask", src_mask)

    def forward(self, I_meas, I_tgt, A_src, A_tgt, phi_prev, Z):
        phi_g, feats = gs_unroll(A_src, A_tgt, phi_prev, self.K, self.src_mask)
        x = torch.cat([feats, I_meas.unsqueeze(1), I_tgt.unsqueeze(1)], 1)
        h1 = self.enc1(x)
        h2 = self.enc2(F.avg_pool2d(h1, 2))
        u = F.interpolate(h2, scale_factor=2, mode="bilinear", align_corners=False)
        c_hat = self.c_head(self.dec(torch.cat([u, h1], 1)))
        return wrap_pi(phi_g - aberration(Z, c_hat)), c_hat


# =====================================================================
# 2. [A] 几何标定类
# =====================================================================
def _display_radians(slm, phase_rad: np.ndarray, settle_s: float) -> int:
    """显示弧度相位: 显式内存槽轮换(2..125) + 内存模式(AGENTS.md 铁律).
    弧度由驱动 display_phase 内部 create_phase_from_array 单点 wrap."""
    from ao_shaping.drivers.slm.santec import MEMORY_MODE_INTERNAL
    from ao_shaping.utils.slm.phase_display import pick_slm_slot
    slot = pick_slm_slot(slm)
    slm.display_phase(np.asarray(phase_rad, np.float32), wait_time_s=0.0,
                      memory_number=slot, memory_mode=MEMORY_MODE_INTERNAL)
    time.sleep(settle_s)
    return slot


def _display_grayscale(slm, gray_uint16: np.ndarray, settle_s: float) -> int:
    """显示 uint16 原始灰度图 (自定义 LUT 通道 / 平场, 不走弧度转换)."""
    from ao_shaping.drivers.slm.santec import MEMORY_MODE_INTERNAL
    from ao_shaping.utils.slm.phase_display import pick_slm_slot
    slot = pick_slm_slot(slm)
    slm.display_data(np.asarray(gray_uint16, np.uint16), wait_time_s=0.0,
                     memory_number=slot, memory_mode=MEMORY_MODE_INTERNAL)
    time.sleep(settle_s)
    return slot


class SLMCCDCalibrator:
    """几何标定: 装配辅助 -> 光束位置 -> K/旋转 -> 验证 -> 运行时workzone/place_on_panel.

    标定量:
      center        0级光斑质心 (CCD px)
      Kx, Ky        尺度常数 K=λf/(d_slm·d_ccd): 每(1/SLM像素)频率对应的CCD像素数
                    -> ±1/(2d_slm) 带宽 = K/2 px; 目标半宽h(归一化) = h·K/2 px
      rotation_deg  CCD 相对 SLM 轴的旋转角;  crop_side  建议工作区边长 = 1.15·max(K)
      beam_center   光束在SLM面板的中心(刀口扫描), 所有图案以此为原点
    """

    def __init__(self, slm, ccd, acquire, panel_res, settle_s: float = SETTLE_S):
        self.slm = slm
        self.ccd = ccd
        self.acquire = acquire
        self.panel_res = tuple(panel_res)
        self.settle_s = settle_s
        self.calib: dict | None = None

    # ---------------- 底层工具 ----------------
    @staticmethod
    def _moments(img: np.ndarray, exclude=None, thresh_frac: float = 0.15):
        """阈值质心. exclude=((cy,cx),r): 挖掉半径r的圆(屏蔽0级污染)."""
        img = np.asarray(img, np.float64)
        if exclude is not None:
            (cy, cx), r = exclude
            yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
            img = img.copy()
            img[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = 0.0
        thr = img.max() * thresh_frac
        m = img >= thr
        if int(m.sum()) < 5:
            raise RuntimeError("未找到光斑(阈值内像素<5): 检查曝光/衰减/0级遮挡")
        yy, xx = np.nonzero(m)
        wv = img[m]
        return np.array([(yy * wv).sum() / wv.sum(), (xx * wv).sum() / wv.sum()])

    @staticmethod
    def _fwhm1d(profile: np.ndarray) -> float:
        peak = profile.max()
        if peak <= 0:
            return 0.0
        above = np.nonzero(profile >= peak / 2)[0]
        return float(above[-1] - above[0] + 1) if len(above) >= 2 else 0.0

    def spot_fwhm(self, img: np.ndarray, c) -> float:
        cy, cx = int(round(c[0])), int(round(c[1]))
        h, w = img.shape
        col = img[:, min(max(cx, 0), w - 1)]
        row = img[min(max(cy, 0), h - 1), :]
        return 0.5 * (self._fwhm1d(col) + self._fwhm1d(row))

    def _show(self, phase: np.ndarray):
        """显示弧度相位图 (本类图案均为弧度: blaze/棋盘/0位)."""
        _display_radians(self.slm, phase, self.settle_s)

    def _show_flat(self):
        """平坦相位: 直写 uint16 全0灰度, 不走弧度转换 (AGENTS.md 平场规则)."""
        _display_grayscale(self.slm, np.zeros(self.panel_res, np.uint16), self.settle_s)

    @staticmethod
    def _blaze(period_px: float, axis: str, panel_res) -> np.ndarray:
        """满屏线性闪耀光栅(弧度). 衍射位移=K/P, 用于测K与旋转."""
        h, w = panel_res
        if axis == "x":
            ramp = 2 * np.pi * np.arange(w) / period_px
            return np.tile(ramp, (h, 1)).astype(np.float32)
        ramp = 2 * np.pi * np.arange(h) / period_px
        return np.tile(ramp[:, None], (1, w)).astype(np.float32)

    # ---------------- 阶段1: 装配辅助 ----------------
    def align(self, max_rounds: int = 10, center_tol_px: float = 30.0,
              blaze_period: float = 24.0) -> dict:
        """装配辅助: 检查0级与±1级是否入视场并大致居中, 给出调节引导."""
        report = {"rounds": []}
        for rnd in range(max_rounds):
            self._show_flat()
            F0 = np.asarray(self.acquire(self.ccd), np.float64)
            c0 = self._moments(F0)
            f0 = max(self.spot_fwhm(F0, c0), 3.0)
            H, W = F0.shape
            spots = {"0级": c0}
            for axis, tag in (("x", "+1级x"), ("y", "+1级y")):
                self._show(self._blaze(blaze_period, axis, self.panel_res))
                F = np.asarray(self.acquire(self.ccd), np.float64)
                spots[tag] = self._moments(F, exclude=(c0, 3.0 * f0))
            off = {k: v - np.array([H, W]) / 2.0 for k, v in spots.items()}
            in_fov = {k: (0 <= v[0] < H and 0 <= v[1] < W) for k, v in spots.items()}
            centered = all(abs(o).max() < center_tol_px for o in off.values())
            report["rounds"].append({"spots": spots, "offsets": off, "in_fov": in_fov})
            logger.info("装配检查[{}/{}]:", rnd + 1, max_rounds)
            for k, o in off.items():
                logger.info("  {:5s} 位置=({:7.1f},{:7.1f}) 距视场中心=({:+6.1f},{:+6.1f}) {}",
                            k, spots[k][0], spots[k][1], o[0], o[1],
                            "在视场内" if in_fov[k] else "出视场!")
            if all(in_fov.values()) and centered:
                logger.info("装配满足要求 ✓")
                report["ok"] = True
                return report
            dy, dx = off["0级"]
            sug = []
            if abs(dx) > center_tol_px:
                sug.append(f"{'←' if dx > 0 else '→'}平移使0级向视场中心(移动{abs(dx):.0f}px)")
            if abs(dy) > center_tol_px:
                sug.append(f"{'↑' if dy > 0 else '↓'}俯仰调节使0级向视场中心(移动{abs(dy):.0f}px)")
            for k, ok_ in in_fov.items():
                if not ok_:
                    sug.append(f"{k}出视场: 减小闪耀周期(当前{blaze_period}px)或增大视场/减小f")
            logger.info("建议: {}", " ; ".join(sug) if sug else "微调后复测")
            report["ok"] = False
        logger.warning("装配辅助达到最大轮数仍未满足, 请人工检查光路")
        return report

    # ---------------- 阶段1.5: 光束在SLM上的位置 ----------------
    def find_beam_on_slm(self, n_scan: int = 25, checker_period: int = 8,
                         window: int | None = None) -> dict:
        """刀口扫描测光束在SLM面板的中心/宽度.
        图案: 分割位置s一侧0相位(全通), 另一侧0/π棋盘(散射走0级);
        0级功率P(s)=光束截面累积分布 -> 中心=50% crossing, sigma=(84%-16%)/2."""
        self._show_flat()
        F0 = np.asarray(self.acquire(self.ccd), np.float64)
        c0 = self._moments(F0)
        h, w = self.panel_res
        win = int(window or 6 * max(self.spot_fwhm(F0, c0), 5))

        def power(frame: np.ndarray) -> float:
            cy, cx = int(round(c0[0])), int(round(c0[1]))
            hh, ww = frame.shape
            y0, x0 = cy - win // 2, cx - win // 2
            pt, pb = max(0, -y0), max(0, y0 + win - hh)
            pl, pr = max(0, -x0), max(0, x0 + win - ww)
            p = frame[max(0, y0):min(hh, y0 + win), max(0, x0):min(ww, x0 + win)]
            if pt or pb or pl or pr:
                p = np.pad(p, ((pt, pb), (pl, pr)), mode="edge")
            return float(p.sum())

        def scan_pattern(s: int, axis: str) -> np.ndarray:
            yy, xx = np.mgrid[0:h, 0:w]
            clear = (xx < s) if axis == "x" else (yy < s)
            checker = ((xx // checker_period + yy // checker_period) % 2) * np.pi
            return np.where(clear, 0.0, checker).astype(np.float32)

        def scan(axis: str):
            H = w if axis == "x" else h
            ss = np.linspace(0, H, n_scan)
            P = []
            for s in ss:
                self._show(scan_pattern(int(round(s)), axis))
                P.append(power(np.asarray(self.acquire(self.ccd), np.float64)))
            P = np.asarray(P)
            Pn = (P - P.min()) / (np.ptp(P) + 1e-12)
            c50 = float(np.interp(0.50, Pn, ss))
            sigma = float(np.interp(0.84, Pn, ss) - np.interp(0.16, Pn, ss)) / 2.0
            return c50, max(sigma, 1.0)

        cx, sx = scan("x")
        cy, sy = scan("y")
        if self.calib is None:
            self.calib = {}
        self.calib["beam_center"] = np.array([cy, cx])
        self.calib["beam_sigma"] = np.array([sy, sx])
        off_y, off_x = cy - h / 2, cx - w / 2
        logger.info("SLM面板上的光束: 中心=({:.0f},{:.0f}) 偏移面板中心=({:+.0f},{:+.0f})px "
                    "sigma=({:.0f},{:.0f})px", cy, cx, off_y, off_x, sy, sx)
        if abs(off_y) > 0.1 * h or abs(off_x) > 0.1 * w:
            logger.warning("光束偏离面板中心>10%, 建议装配调节或后续图案均以beam_center为原点")
        return dict(beam_center=self.calib["beam_center"],
                    beam_sigma=self.calib["beam_sigma"])

    def place_on_panel(self, pattern_nx: np.ndarray, n: int = N) -> np.ndarray:
        """N×N弧度相位 -> 面板, 以beam_center为原点(仿射平移).
        模型图案坐标(0,0) ≡ 光束轴线 ≡ 0级质心 —— 这是'相机-系统对应关系'的核心."""
        if self.calib is None or "beam_center" not in self.calib:
            raise RuntimeError("先find_beam_on_slm()")
        h, w = self.panel_res
        full = F.interpolate(torch.from_numpy(np.asarray(pattern_nx, np.float32))[None, None],
                             size=(h, w), mode="bilinear", align_corners=False)
        cy, cx = self.calib["beam_center"]
        tx, ty = 2 * (cx - w / 2) / w, 2 * (cy - h / 2) / h
        theta = torch.tensor([[1.0, 0.0, tx], [0.0, 1.0, ty]],
                             dtype=torch.float32)[None]
        grid = F.affine_grid(theta, list(full.shape), align_corners=False)
        return F.grid_sample(full, grid, align_corners=False,
                             padding_mode="zeros")[0, 0].numpy().astype(np.float32)

    # ---------------- 阶段2: 几何标定 ----------------
    def calibrate(self, periods=(16, 24, 32), exclude_radius_factor: float = 3.0) -> dict:
        """闪耀光栅位移=K/P -> 拟合Kx/Ky/旋转. 保留已有beam_center."""
        self._show_flat()
        F0 = np.asarray(self.acquire(self.ccd), np.float64)
        c0 = self._moments(F0)
        f0 = max(self.spot_fwhm(F0, c0), 3.0)
        logger.info("0级质心=({:.1f}, {:.1f})  FWHM≈{:.1f}px", c0[0], c0[1], f0)

        K, u, resid = {}, {}, {}
        for axis in ("x", "y"):
            disps, invs = [], []
            for P in periods:
                self._show(self._blaze(P, axis, self.panel_res))
                F = np.asarray(self.acquire(self.ccd), np.float64)
                d = self._moments(F, exclude=(c0, exclude_radius_factor * f0)) - c0
                disps.append(d)
                invs.append(1.0 / P)
                logger.info("  {}光栅 P={:2d}px -> 位移=({:+7.1f}, {:+7.1f})px",
                            axis, P, d[0], d[1])
            disps = np.asarray(disps)
            invs = np.asarray(invs)
            u0 = disps.sum(axis=0)
            nrm = np.linalg.norm(u0)
            if nrm < 1e-6:
                raise RuntimeError(f"标定失败: {axis}方向光栅无可测位移")
            u0 /= nrm
            K[axis] = float((disps @ u0 @ invs) / (invs @ invs))
            u[axis] = u0
            resid[axis] = float(np.abs(disps - np.outer(invs * K[axis], u0)).max())
            logger.info("  {}轴: K={:.1f}px·SLMpx  线性拟合残差={:.1f}px", axis, K[axis], resid[axis])

        rotation_deg = math.degrees(math.acos(float(np.clip(u["x"] @ u["y"], -1, 1)))) - 90.0
        mirror = bool(u["x"][1] < 0)
        prev = self.calib or {}
        self.calib = dict(
            center=c0, Kx=K["x"], Ky=K["y"],
            rotation_deg=float(rotation_deg), fwhm0=float(f0),
            mirror=mirror, resid_x=resid["x"], resid_y=resid["y"],
            crop_side=int(round(max(K["x"], K["y"]) * 1.15)),
        )
        for k_src in ("beam_center", "beam_sigma"):
            if k_src in prev:
                self.calib[k_src] = prev[k_src]
        logger.info("标定完成: Kx={:.1f} Ky={:.1f} rot={:+.2f}° crop={}px mirror={}",
                    self.calib["Kx"], self.calib["Ky"], self.calib["rotation_deg"],
                    self.calib["crop_side"], mirror)
        if mirror:
            logger.warning("检测到x方向镜像, 检查光路奇次反射或翻转SLM坐标")
        if max(resid.values()) > 0.05 * min(K.values()):
            logger.warning("拟合残差偏大(>5%K): 0级污染? 曝光不足? 建议verify()复核")
        return self.calib

    def save(self, path) -> None:
        if self.calib is None:
            raise RuntimeError("无标定结果, 先calibrate()或load()")
        np.savez(path, **self.calib)
        logger.info("标定已保存: {}", path)

    def load(self, path) -> dict:
        self.calib = dict(np.load(path, allow_pickle=True))
        logger.info("加载标定: Kx={:.1f} Ky={:.1f} rot={:+.2f}°",
                    self.calib["Kx"], self.calib["Ky"], self.calib["rotation_deg"])
        return self.calib

    # ---------------- 阶段3: 几何验证 ----------------
    def verify(self, test_period: float = 20.0,
               drift_tol_px: float = 3.0, pred_tol_px: float = 5.0) -> dict:
        """①0级质心重复漂移 ②标定期外周期(test_period)的+1级位置预测残差 ③能量集中度."""
        if self.calib is None:
            raise RuntimeError("先calibrate()或load()")
        c0_ref = np.asarray(self.calib["center"], np.float64)
        Kx, Ky = self.calib["Kx"], self.calib["Ky"]
        report: dict = {"items": {}}

        self._show_flat()
        c0 = self._moments(np.asarray(self.acquire(self.ccd), np.float64))
        drift = float(np.linalg.norm(c0 - c0_ref))
        report["items"]["center_drift_px"] = drift

        errors = {}
        for axis, K in (("x", Kx), ("y", Ky)):
            self._show(self._blaze(test_period, axis, self.panel_res))
            F = np.asarray(self.acquire(self.ccd), np.float64)
            c_meas = self._moments(F, exclude=(c0, 3.0 * max(self.calib["fwhm0"], 3.0)))
            direction = np.array([0.0, 1.0]) if axis == "x" else np.array([1.0, 0.0])
            if abs(self.calib["rotation_deg"]) > 0.3:
                th = math.radians(self.calib["rotation_deg"])
                R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
                direction = R @ direction
            c_pred = c0 + direction * (K / test_period)
            errors[axis] = float(np.linalg.norm(c_meas - c_pred))
        report["items"]["pred_err_px"] = errors

        patch = self.workzone(np.asarray(self.acquire(self.ccd), np.float64))
        In = patch / max(patch.sum(), 1e-12)
        yy, xx = np.mgrid[0:In.shape[0], 0:In.shape[1]]
        for frac, tag in ((0.25, "中心1/2区域"), (0.0625, "中心1/4区域")):
            r = math.sqrt(frac) * In.shape[0] / 2.0
            m = (yy - In.shape[0] / 2) ** 2 + (xx - In.shape[1] / 2) ** 2 <= r * r
            report["items"][f"encircled_{tag}"] = float(In[m].sum())

        ok = (drift < drift_tol_px and max(errors.values()) < pred_tol_px)
        report["pass"] = bool(ok)
        logger.info("几何验证: 漂移={:.2f}px(限{:.1f}) 预测残差 x={:.2f}/y={:.2f}px(限{:.1f})  {}",
                    drift, drift_tol_px, errors["x"], errors["y"], pred_tol_px,
                    "通过 ✓" if ok else "未通过 ✗")
        if not ok:
            logger.warning("验证未通过: 检查装配松动/温度漂移, 或重新calibrate()")
        return report

    # ---------------- 运行时接口 ----------------
    def workzone(self, img: np.ndarray, n: int = N) -> np.ndarray:
        """标定取窗: 定心(center) -> 纠旋(rotation) -> 重采样n×n.
        所有闭环采图必须走这里, 否则与模型网格错位."""
        if self.calib is None:
            raise RuntimeError("先calibrate()或load()")
        cy, cx = self.calib["center"]
        side = int(self.calib["crop_side"])
        h, w = img.shape
        y0, x0 = int(round(cy)) - side // 2, int(round(cx)) - side // 2
        pt, pb = max(0, -y0), max(0, y0 + side - h)
        pl, pr = max(0, -x0), max(0, x0 + side - w)
        p = img[max(0, y0):min(h, y0 + side), max(0, x0):min(w, x0 + side)]
        if pt or pb or pl or pr:
            p = np.pad(p, ((pt, pb), (pl, pr)), mode="edge")
        t = torch.from_numpy(p.astype(np.float32))[None, None]
        if abs(self.calib["rotation_deg"]) > 0.3:
            th = math.radians(-self.calib["rotation_deg"])   # 方向反了就去掉负号
            rot = torch.tensor([[math.cos(th), -math.sin(th), 0.0],
                                [math.sin(th), math.cos(th), 0.0]],
                               dtype=torch.float32)[None]
            grid = F.affine_grid(rot, list(t.shape), align_corners=False)
            t = F.grid_sample(t, grid, align_corners=False, padding_mode="border")
        return F.interpolate(t, size=(n, n), mode="bilinear",
                             align_corners=False)[0, 0].numpy().astype(np.float64)

    def gauss_amp_from_farfield(self, flat_frame: np.ndarray, n: int = N,
                                w0_override: float | None = None) -> torch.Tensor:
        """flat相位远场高斯拟合 -> 源面高斯幅值(近似). 有近场实测时传w0_override."""
        if w0_override is not None:
            w0 = w0_override
        else:
            fine = self.workzone(flat_frame, 256)
            In = fine / max(fine.sum(), 1e-12)
            fwhm_n = self.spot_fwhm(In, np.array(In.shape) / 2.0) / 256.0
            w0 = float(np.clip(2 * math.sqrt(2 * math.log(2)) / (math.pi * max(fwhm_n, 1e-3)),
                               0.2, 1.5))
            logger.info("远场FWHM={:.3f}(归一化) -> 源面高斯束腰 w0≈{:.3f}(可用--w0覆盖)", fwhm_n, w0)
        t = torch.linspace(-1, 1, n)
        y, x = torch.meshgrid(t, t, indexing="ij")
        A = torch.exp(-(x * x + y * y) / w0 ** 2)
        return A / A.amax()

    def target_half_to_ccd_px(self, half_norm: float) -> float:
        return float(half_norm) * max(self.calib["Kx"], self.calib["Ky"])


# =====================================================================
# 3. [B] LUT 标定类
# =====================================================================
class SLMLUTCalibrator:
    """灰度-相位 LUT 标定 (自参考干涉法, 无需额外干涉仪).

    原理: 左半屏测试灰度g, 右半屏参考灰度g_ref(不加闪耀, 利用0级直反光),
    远场叠加出杨氏条纹 I(x)=包络·cos(2πx/Λ+ψ(g)), ψ(g)=φ(g)−φ(g_ref);
    行平均+Hann窗+FFT主峰相位即得ψ(g), 全灰度扫描解卷绕得LUT曲线.

    输出 lut.npz: gray, phase_of_gray, gray_of_phase(256点), phase_range, g_ref
    """

    def __init__(self, slm, ccd, acquire, panel_res, calib: dict | None = None,
                 settle_s: float = SETTLE_S, factory_2pi: float = 255.0,
                 window: int = 384):
        self.slm = slm
        self.ccd = ccd
        self.acquire = acquire
        self.panel_res = tuple(panel_res)
        self.calib = calib
        self.settle_s = settle_s
        self.factory_2pi = factory_2pi
        self.window = window
        self.result: dict | None = None

    def _split_pattern(self, g_left: float, g_right: float) -> np.ndarray:
        """左半g_left/右半g_right, 分割位置取beam_center(若已测)保证两半功率对称."""
        h, w = self.panel_res
        bc = None
        if self.calib is not None:
            bc = self.calib.get("beam_center", None)
        cx = int(round(bc[1])) if bc is not None else w // 2
        pat = np.empty((h, w), np.float32)
        pat[:, :cx] = g_left * (2 * np.pi / self.factory_2pi)
        pat[:, cx:] = g_right * (2 * np.pi / self.factory_2pi)
        return pat

    def _center(self, img: np.ndarray):
        if self.calib is not None:
            return np.asarray(self.calib["center"], np.float64)
        thr = img.max() * 0.15
        m = img >= thr
        yy, xx = np.nonzero(m)
        return np.array([yy.mean(), xx.mean()])

    def _fringe_phase(self, frame: np.ndarray):
        """中心窗口 -> 行平均 -> Hann+FFT主峰相位. 条纹频率自动检测."""
        img = np.asarray(frame, np.float64)
        cy, cx = self._center(img)
        h, w = img.shape
        half = self.window // 2
        y0, x0 = int(round(cy)) - half, int(round(cx)) - half
        pt, pb = max(0, -y0), max(0, y0 + self.window - h)
        pl, pr = max(0, -x0), max(0, x0 + self.window - w)
        p = img[max(0, y0):min(h, y0 + self.window), max(0, x0):min(w, x0 + self.window)]
        if pt or pb or pl or pr:
            p = np.pad(p, ((pt, pb), (pl, pr)), mode="edge")
        prof = p.mean(axis=0)
        prof = prof - prof.mean()
        spec = np.fft.rfft(prof * np.hanning(len(prof)))
        mag = np.abs(spec)
        mag[:3] = 0
        k = int(np.argmax(mag))
        period = len(prof) / k if k > 0 else np.inf
        if not (3.0 <= period <= len(prof) / 8):
            logger.warning("条纹周期={:.1f}px 异常(期望3~{:.0f}px), 检查窗口/半屏分割",
                           period, len(prof) / 8)
        return float(np.angle(spec[k])), k, float(period)

    def calibrate(self, grays=None, g_ref: float = 128.0, drift_correct: bool = True):
        """全灰度扫描, 缓存 dict(gray, phase_of_gray, gray_of_phase, phase_range)."""
        if grays is None:
            grays = np.linspace(0, 255, 64)
        grays = np.asarray(grays, np.float64)

        def psi(g):
            _display_radians(self.slm, self._split_pattern(g, g_ref), self.settle_s)
            ph, k, period = self._fringe_phase(self.acquire(self.ccd))
            return ph

        logger.info("LUT标定: g_ref={:.0f}, {}个灰度点 ...", g_ref, len(grays))
        psi_ref0 = psi(g_ref)
        phases = np.array([psi(g) for g in grays])
        psi_ref1 = psi(g_ref) if drift_correct else psi_ref0
        frac = np.linspace(0, 1, len(grays))
        phases = phases - (psi_ref0 + frac * (psi_ref1 - psi_ref0))   # 热漂移线性修正
        curve = np.unwrap(phases - phases[np.argmin(np.abs(grays - g_ref))])
        curve = np.maximum.accumulate(curve)          # 单调化(相位随灰度物理递增)
        k = 5
        ker = np.ones(2 * k + 1) / (2 * k + 1)
        smooth = np.convolve(curve, ker, mode="same")
        smooth[:k] = curve[:k]
        smooth[-k:] = curve[-k:]

        phase_grid = np.linspace(smooth.min(), smooth.max(), 256)
        gray_of_phase = np.interp(phase_grid, smooth, grays)
        self.result = dict(
            gray=grays, phase_of_gray=smooth,
            gray_of_phase=gray_of_phase,
            phase_range=float(smooth.max() - smooth.min()),
            g_ref=g_ref,
        )
        logger.info("LUT标定完成: 调制深度={:.2f} rad ({:.2f}×2π)",
                    self.result["phase_range"], self.result["phase_range"] / (2 * np.pi))
        if self.result["phase_range"] < 1.8 * np.pi:
            logger.warning("调制深度不足2π({:.2f}rad), 相位会量化失真, 考虑换波长/型号",
                           self.result["phase_range"])
        return self.result

    def save(self, path) -> None:
        if self.result is None:
            raise RuntimeError("先calibrate()或load()")
        np.savez(path, **self.result)
        logger.info("LUT已保存: {}", path)

    def load(self, path) -> dict:
        self.result = dict(np.load(path, allow_pickle=True))
        logger.info("加载LUT: 调制深度={:.2f} rad", float(self.result["phase_range"]))
        return self.result

    def phase2gray(self, phase: np.ndarray) -> np.ndarray:
        """弧度相位 -> 灰度图(查LUT, 按实测深度wrap). 输入任意尺寸."""
        if self.result is None:
            raise RuntimeError("先calibrate()或load()")
        ph = np.asarray(phase, np.float64)
        depth = float(self.result["phase_range"])
        ph01 = np.mod(ph, depth) / depth
        gray = np.interp(ph01, np.linspace(0, 1, 256),
                         np.asarray(self.result["gray_of_phase"], np.float64))
        return gray.astype(np.float32)

    def verify(self, n_test: int = 9) -> dict:
        """LUT显示n_test个目标相位, 测残余条纹相位std, <0.05rad PASS."""
        if self.result is None:
            raise RuntimeError("先calibrate()或load()")
        depth = float(self.result["phase_range"])
        targets = np.linspace(0, 0.9 * depth, n_test)
        res = []
        for t in targets:
            gray_test = float(np.interp(t / depth, np.linspace(0, 1, 256),
                                        np.asarray(self.result["gray_of_phase"], np.float64)))
            _display_radians(self.slm, self._split_pattern(gray_test, self.result["g_ref"]), self.settle_s)
            ph, _, _ = self._fringe_phase(self.acquire(self.ccd))
            res.append((t, ph))
        res = np.asarray(res)
        err = np.unwrap(res[:, 1] - res[0, 1]) - (res[:, 0] - res[0, 0])
        std = float(err.std())
        report = dict(err_rad=err.tolist(), residual_std_rad=std,
                      pass_=bool(std < 0.05))
        logger.info("LUT验证: 残差std={:.3f} rad ({:.1f}°)  {}",
                    std, np.degrees(std), "通过 ✓" if report["pass_"] else "未通过 ✗")
        return report


# =====================================================================
# 4. [D] 硬件封装: 标定存档 -> 测量/显示/初值/采样/微调/闭环
# =====================================================================
class ShapingSystem:
    """把标定存档变成可用的测量/显示通道, 并实现 初值->微调->闭环(在线replay) 全流程.

    坐标链(每一环都经过标定):
      真实CCD帧 --workzone()--> 模型网格 (定心+纠旋+重采样)
      模型相位 --place_on_panel()+LUT--> SLM灰度 (光束中心原点+标定灰度)
    """

    def __init__(self, slm, ccd, acquire, calib: dict, lut_result: dict,
                 half: float = HALF):
        self.slm = slm
        self.ccd = ccd
        self.acquire = acquire
        from ao_shaping.drivers.slm import Santec
        panel = Santec.Panel_Res
        self.geo = SLMCCDCalibrator(slm, ccd, acquire, panel, settle_s=SETTLE_S)
        self.geo.calib = calib
        self.lut = SLMLUTCalibrator(slm, ccd, acquire, panel, calib=calib,
                                    settle_s=SETTLE_S)
        self.lut.result = lut_result
        self.half = half
        self.Z = zernike_basis(N, N_ZERN, DEV)
        self.src_mask = (self.Z[0] != 0).float()
        self.I_tgt = make_square_target(N, half, DEV)
        self.A_tgt = self.I_tgt.sqrt()
        self.roi = (self.I_tgt > 0).float()
        logger.info("目标方形物理边长 ≈ {:.1f} px (CCD); 工作区 {}×{}",
                    self.geo.target_half_to_ccd_px(half), N, N)

    # ---------------- 测量与显示 ----------------
    def measure(self) -> torch.Tensor:
        """采图 -> workzone -> 单位能量归一化. 调试: 打印峰值位置可判断是否套准."""
        frame = np.asarray(self.acquire(self.ccd), np.float64)
        p = self.geo.workzone(frame, N)
        s = p.sum()
        if s <= 1e-12:
            raise RuntimeError("工作区无信号: 检查曝光/光路/标定文件")
        t = torch.from_numpy((p / s).astype(np.float32)).to(DEV)
        return t

    def display(self, phi_nx: torch.Tensor):
        """显示: beam_center原点 + LUT标定通道(绕过驱动出厂LUT)."""
        panel = self.geo.place_on_panel(phi_nx.detach().cpu().numpy(), N)
        gray = self.lut.phase2gray(panel)
        _display_grayscale(self.slm, np.clip(gray, 0.0, 255.0).astype(np.uint16), SETTLE_S)

    # ---------------- 初值: 设备自适应GS (修正版: 回退保留仿真相位) ----------------
    def adaptive_gs_init(self, iters: int = GS_ITERS_INIT,
                         w0_override: float | None = None) -> torch.Tensor:
        """真实相机=前向模型: 显示->实测幅值(保留仿真相位)->反投影.
        返回phi0并估计A_src. 调试: 盯'信号区RMSE'应稳定下降."""
        self.display(torch.zeros(N, N, device=DEV))
        I_flat = self.measure()
        self.A_src = self.geo.gauss_amp_from_farfield(
            (I_flat.cpu().numpy() * I_flat.numel()).astype(np.float64), N,
            w0_override=w0_override).to(DEV)
        phi = torch.zeros(N, N, device=DEV)
        for k in range(iters):
            self.display(phi)
            I = self.measure()
            U = prop(self.A_src * torch.exp(1j * phi))            # 仿真前向(取相位)
            U = I.sqrt() * torch.exp(1j * torch.angle(U))         # 实测幅值替换
            g = prop_inv(U)
            phi = wrap_pi(torch.angle(g)) * self.src_mask
            logger.info("  init GS {}/{}  信号区RMSE={:.5f}", k + 1, iters,
                        (I - self.I_tgt).pow(2)[self.roi > 0].mean().item())
        return phi

    # ---------------- 升级1: 域随机化扰动采样 ----------------
    def sample_c(self) -> torch.Tensor:
        """随机像差系数(24维): 低阶幅度池AMP_LOW_POOL, 高阶AMP_HIGH_POOL(小).
        幅度混合让网络见过轻/中/重失配, 而不是只会补偿一种."""
        c = torch.zeros(N_ZERN)
        for i in range(N_ZERN_LOW):
            c[i] = AMP_LOW_POOL[torch.randint(0, len(AMP_LOW_POOL), (1,)).item()] \
                * (2 * torch.rand(1) - 1)
        for i in range(N_ZERN_LOW, N_ZERN):
            c[i] = AMP_HIGH_POOL[torch.randint(0, len(AMP_HIGH_POOL), (1,)).item()] \
                * (2 * torch.rand(1) - 1)
        return c

    def collect(self, phi0: torch.Tensor, n: int):
        """在phi0上加已知扰动采样: (I_meas, c, half).
        half(目标尺寸)也随机 —— 网络输入含I_tgt, 学会'按目标尺寸整形', 泛化更好."""
        data = []
        for i in range(n):
            c = self.sample_c()
            half = float(np.random.uniform(*HALF_RANGE))
            self.display(wrap_pi(phi0 + aberration(self.Z, c.to(DEV))))
            data.append((self.measure(), c, half))
            if (i + 1) % 60 == 0:
                logger.info("  扰动采样 {}/{}", i + 1, n)
        return data

    # ---------------- 微调 (升级2: 24维残差头在此训练) ----------------
    def finetune(self, net: FourierGSNetLite, data, phi0: torch.Tensor,
                 epochs: int = FT_EPOCHS, lr: float = FT_LR, tag: str = "ft"):
        """只训网络(物理展开无参数). 损失: |c_hat−c|_1 + 0.5×强度MSE(按每样本目标)."""
        opt = torch.optim.AdamW(net.parameters(), lr=lr)
        for ep in range(epochs):
            perm = torch.randperm(len(data))
            tot, nb = 0.0, 0
            for i in range(0, len(perm) - FT_BATCH + 1, FT_BATCH):
                idx = perm[i:i + FT_BATCH]
                I_m = torch.stack([data[j][0] for j in idx])
                c_gt = torch.stack([data[j][1] for j in idx]).to(DEV)
                halves = torch.tensor([data[j][2] for j in idx], device=DEV)
                B = I_m.shape[0]
                # 按每样本随机half重建目标 (域随机化的关键: 目标和扰动一起变化)
                I_t = torch.stack([make_square_target(N, float(h), DEV) for h in halves])
                A_t = I_t.sqrt()
                phi_new, c_hat = net(I_m, I_t,
                                     self.A_src.expand(B, -1, -1), A_t,
                                     phi0.expand(B, -1, -1), self.Z)
                loss_c = (c_hat - c_gt).abs().mean()
                I_sim = prop(self.A_src.expand(B, -1, -1)
                             * torch.exp(1j * phi_new)).abs() ** 2
                loss_i = F.mse_loss(I_sim / I_sim.sum(dim=(-2, -1), keepdim=True), I_t)
                loss = loss_c + 0.5 * loss_i
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss_c.item(); nb += 1
            logger.info("  [{}] epoch {}/{}  |c|err={:.4f} rad",
                        tag, ep + 1, epochs, tot / max(nb, 1))
        return net

    # ---------------- 闭环 (升级3: 在线replay) ----------------
    @staticmethod
    def _metrics(I: torch.Tensor, roi: torch.Tensor):
        m = roi > 0
        imin = torch.where(m, I, torch.full_like(I, float("inf"))).min()
        uni = (imin / ((I * roi).sum() / m.sum())).item()
        ee = (I * roi).sum().item()
        return uni, ee

    def closed_loop(self, net: FourierGSNetLite, phi: torch.Tensor,
                    steps: int, recorder: Recorder, replay: bool = True):
        """闭环: 采图->推理->显示. 触发条件(replay=True时):
        ① 均匀度 < 滑动窗口中位数×REPLAY_DROP  (束形退化)
        ② |c_hat|均值 > REPLAY_C_MAX          (网络拼命补偿=系统已变)
        ③ 每REPLAY_EVERY步(>0时)              (周期性保养)
        触发后: 采REPLAY_N组 + REPLAY_EPOCHS轮快调(小学习率防遗忘), 冷却REPLAY_COOLDOWN步."""
        logger.info("进入闭环: {}步, replay={}", steps, replay)
        uni_hist: deque = deque(maxlen=REPLAY_WINDOW)
        c_hist: deque = deque(maxlen=REPLAY_WINDOW)
        since_trigger = REPLAY_COOLDOWN   # 开局即允许触发一次
        for step in range(steps):
            I = self.measure()
            t0 = time.time()
            phi, c_hat = net(I.unsqueeze(0), self.I_tgt.unsqueeze(0),
                             self.A_src.unsqueeze(0), self.A_tgt.unsqueeze(0),
                             phi.unsqueeze(0), self.Z)
            phi = phi.squeeze(0)
            dt = (time.time() - t0) * 1000
            self.display(phi)
            uni, ee = self._metrics(I, self.roi)
            c_abs = float(c_hat.abs().max())
            logger.info("step {:3d}  推理{:5.1f}ms  均匀度={:.3f}  封闭能量={:.3f}  |c_hat|max={:.2f}",
                        step + 1, dt, uni, ee, c_abs)
            recorder.append({"step": step, "uniformity": uni, "encircled": ee,
                             "inference_ms": dt, "ccd": I.cpu().numpy(),
                             "phase": phi.cpu().numpy()})
            # ---- replay 监测 ----
            if not replay:
                continue
            uni_hist.append(uni)
            c_hist.append(c_abs)
            since_trigger += 1
            if len(uni_hist) < REPLAY_WINDOW or since_trigger < REPLAY_COOLDOWN:
                continue
            med = float(np.median(uni_hist))
            triggered = (uni < REPLAY_DROP * med) or (float(np.mean(c_hist)) > REPLAY_C_MAX)
            if REPLAY_EVERY and (step + 1) % REPLAY_EVERY == 0:
                triggered = True
            if triggered:
                reason = ("均匀度退化({:.3f}<{:.3f}×{:.2f})".format(uni, med, REPLAY_DROP)
                          if uni < REPLAY_DROP * med else
                          "|c_hat|偏高({:.2f}>{:.2f})或周期保养".format(float(np.mean(c_hist)), REPLAY_C_MAX))
                logger.info(">> 在线replay触发: {} — 采{}组快调{}轮", reason, REPLAY_N, REPLAY_EPOCHS)
                data = self.collect(phi, REPLAY_N)
                net = self.finetune(net, data, phi, epochs=REPLAY_EPOCHS,
                                    lr=REPLAY_LR, tag="replay")
                torch.save(net.state_dict(), CKPT)
                uni_hist.clear(); c_hist.clear(); since_trigger = 0
        return net, phi


# =====================================================================
# 5. CLI (click 子命令)
# =====================================================================
def _acquire_factory(exposure_ms: float):
    """采图回调工厂: 平均帧 + 超时看门狗(防原生SDK挂起)"""
    from ao_shaping.drivers.ccd import DahengCamera  # noqa: F401 (类型提示用)
    from ao_shaping.utils.hardware_utils import call_with_timeout

    def acquire(ccd, n_sample=N_SAMPLE):
        return call_with_timeout(lambda: ccd.get_numpy_image(n_sample=n_sample),
                                 CAPTURE_TIMEOUT_S, "CCD capture")
    return acquire


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def cli():
    """SLM+CCD 束整形一体化工具: calibrate(标定) / verify(验证) / run(闭环优化)."""


@cli.command()
@click.option("--out-calib", "out_calib", default="calib.npz", show_default=True)
@click.option("--out-lut", "out_lut", default="lut.npz", show_default=True)
@click.option("--skip-align", is_flag=True, help="跳过装配辅助")
@click.option("--skip-beam", is_flag=True, help="跳过光束位置测量")
@click.option("--geo-only", is_flag=True, help="只做几何标定")
@click.option("--lut-only", is_flag=True, help="只做LUT标定")
@click.option("--exposure-ms", default=EXPOSURE_MS, show_default=True)
@click.option("--settle-s", default=SETTLE_S, show_default=True)
@click.option("--cam-type", default="daheng", show_default=True,
              type=click.Choice(["daheng", "miicam"]))
@click.option("--cam-id", default=0, show_default=True, type=int)
@click.option("--slm-number", default=1, show_default=True, type=int)
@click.option("--slm-wavelength", default=1064.0, show_default=True, type=float)
@click.option("--shift-x", default=0, show_default=True, type=int)
@click.option("--shift-y", default=0, show_default=True, type=int)
def calibrate(out_calib, out_lut, skip_align, skip_beam, geo_only, lut_only,
              exposure_ms, settle_s, cam_type, cam_id, slm_number,
              slm_wavelength, shift_x, shift_y):
    """标定全流程: 装配辅助 -> 光束位置 -> 几何标定 -> LUT标定 -> 双验证."""
    from ao_shaping.drivers.ccd.common import create_camera
    from ao_shaping.drivers.slm import Santec

    acquire = _acquire_factory(exposure_ms)
    with Santec(slm_number=slm_number, wavelength=slm_wavelength,
                shift_x=shift_x, shift_y=shift_y) as slm, \
         create_camera(cam_type, cam_id=cam_id, exposure_time_ms=exposure_ms) as ccd:
        geo = SLMCCDCalibrator(slm, ccd, acquire, Santec.Panel_Res,
                               settle_s=settle_s)
        if not lut_only:
            if not skip_align:
                geo.align()
            if not skip_beam:
                geo.find_beam_on_slm()
            geo.calibrate()
            geo.save(out_calib)
            geo.verify()
            if geo_only:
                return
        calib = geo.calib if geo.calib is not None else dict(np.load(out_calib, allow_pickle=True))
        lut = SLMLUTCalibrator(slm, ccd, acquire, Santec.Panel_Res,
                               calib=calib, settle_s=settle_s)
        lut.calibrate()
        lut.save(out_lut)
        lut.verify()


@cli.command()
@click.option("--calib", "calib_path", required=True, type=click.Path(exists=True))
@click.option("--lut", "lut_path", default=None, type=click.Path(exists=True))
@click.option("--exposure-ms", default=EXPOSURE_MS, show_default=True)
@click.option("--cam-type", default="daheng", show_default=True,
              type=click.Choice(["daheng", "miicam"]))
@click.option("--cam-id", default=0, show_default=True, type=int)
@click.option("--slm-number", default=1, show_default=True, type=int)
@click.option("--slm-wavelength", default=1064.0, show_default=True, type=float)
@click.option("--shift-x", default=0, show_default=True, type=int)
@click.option("--shift-y", default=0, show_default=True, type=int)
def verify(calib_path, lut_path, exposure_ms, cam_type, cam_id, slm_number,
           slm_wavelength, shift_x, shift_y):
    """验证已有标定(开机自检): 几何漂移/预测残差/LUT残差, 全PASS再进闭环."""
    from ao_shaping.drivers.ccd.common import create_camera
    from ao_shaping.drivers.slm import Santec

    acquire = _acquire_factory(exposure_ms)
    with Santec(slm_number=slm_number, wavelength=slm_wavelength,
                shift_x=shift_x, shift_y=shift_y) as slm, \
         create_camera(cam_type, cam_id=cam_id, exposure_time_ms=exposure_ms) as ccd:
        geo = SLMCCDCalibrator(slm, ccd, acquire, Santec.Panel_Res)
        geo.load(calib_path)
        rep = geo.verify()
        if lut_path:
            lut = SLMLUTCalibrator(slm, ccd, acquire, Santec.Panel_Res,
                                   calib=geo.calib)
            lut.load(lut_path)
            lut.verify()
        if not rep.get("pass"):
            raise SystemExit("几何验证未通过, 建议重新 calibrate")


@cli.command()
@click.option("--calib", "calib_path", required=True, type=click.Path(exists=True))
@click.option("--lut", "lut_path", required=True, type=click.Path(exists=True))
@click.option("--steps", default=50, show_default=True, help="闭环步数")
@click.option("--exposure-ms", default=EXPOSURE_MS, show_default=True)
@click.option("--retrain", is_flag=True, help="忽略ckpt, 重跑初值+微调")
@click.option("--no-replay", is_flag=True, help="关闭在线再学习")
@click.option("--w0", "w0_override", default=None, type=float,
              help="源面高斯束腰(归一化), 覆盖自动拟合值")
@click.option("--init-iters", default=GS_ITERS_INIT, show_default=True,
              help="设备自适应GS迭代数")
@click.option("--cam-type", default="daheng", show_default=True,
              type=click.Choice(["daheng", "miicam"]))
@click.option("--cam-id", default=0, show_default=True, type=int)
@click.option("--slm-number", default=1, show_default=True, type=int)
@click.option("--slm-wavelength", default=1064.0, show_default=True, type=float)
@click.option("--shift-x", default=0, show_default=True, type=int)
@click.option("--shift-y", default=0, show_default=True, type=int)
def run(calib_path, lut_path, steps, exposure_ms, retrain, no_replay,
        w0_override, init_iters, cam_type, cam_id, slm_number,
        slm_wavelength, shift_x, shift_y):
    """闭环束整形: 无ckpt->初值+微调; 有ckpt->直接闭环. 含在线replay."""
    from ao_shaping.drivers.ccd.common import create_camera
    from ao_shaping.drivers.slm import Santec
    from ao_shaping.utils.image.beam_metrics import compute_metrics
    from ao_shaping.utils.io.file import Recorder

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    recorder = Recorder("uniformity", "max")
    calib = dict(np.load(calib_path, allow_pickle=True))
    lut_result = dict(np.load(lut_path, allow_pickle=True))
    torch.manual_seed(22)
    np.random.seed(22)

    acquire = _acquire_factory(exposure_ms)
    with Santec(slm_number=slm_number, wavelength=slm_wavelength,
                shift_x=shift_x, shift_y=shift_y) as slm, \
         create_camera(cam_type, cam_id=cam_id, exposure_time_ms=exposure_ms) as ccd:
        sys_ = ShapingSystem(slm, ccd, acquire, calib, lut_result)
        net = FourierGSNetLite(K_UNROLL, N_ZERN, CH, sys_.src_mask).to(DEV)

        if CKPT.exists() and not retrain:
            logger.info("加载模型 {}", CKPT)
            net.load_state_dict(torch.load(CKPT, map_location=DEV))
            # 平坦相位起步(网络主导): iters=0 仍会测量并构建 A_src (闭环必需), 但返回零相位.
            phi = sys_.adaptive_gs_init(iters=0, w0_override=w0_override)
        else:
            logger.info("初值: 设备自适应GS ({}轮) ...", init_iters)
            phi = sys_.adaptive_gs_init(iters=init_iters, w0_override=w0_override)
            logger.info("域随机化扰动采样 {}组 ...", N_PERTURB)
            data = sys_.collect(phi, N_PERTURB)
            net = sys_.finetune(net, data, phi)
            torch.save(net.state_dict(), CKPT)
            logger.info("模型已保存 {}", CKPT)

        net, phi = sys_.closed_loop(net, phi, steps, recorder, replay=not no_replay)

        final = sys_.geo.workzone(np.asarray(acquire(ccd), np.float64), N)
        metrics = compute_metrics(final, sys_.I_tgt.cpu().numpy())
        logger.info("整帧: mse={:.5f} corr={:.4f} eff={:.4f}",
                    metrics["mse"], metrics["correlation"], metrics["efficiency"])
        recorder.append({"ccd": final, **metrics})
        recorder.save_dataframe(OUT_DIR / "history.csv", sidecar_dir=OUT_DIR)
        logger.info("已保存: {}", OUT_DIR)


if __name__ == "__main__":
    cli()
