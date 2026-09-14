"""
traditional_gs_hardware.py
传统 Gerchberg-Saxton 方形 top-hat 束整形 + 真实硬件
(Santec SLM-200 / 大恒 CCD, ao_shaping 驱动)

流程:
  1) flat(+闪耀光栅) 采初始帧 -> 0级位置/质心标定
  2) 构建方形目标 (单位能量)
  3) 离线GS(可选MRAF) 或 混合GS(每N轮用CCD实测幅值替换, 自适应真实系统)
  4) 显示相位 -> 采图评估 -> 保存

光路要求:
  激光(线偏振, 对准SLM慢轴) -> SLM(加闪耀光栅移至+1级) -> 傅里叶透镜L1
  -> L1后焦面 = 整形面, CCD取焦斑(方案i)或4f中继到PV(方案ii)
  注: mod 2pi 与灰度标定由 slm.display_data 内部完成, 本脚本只发弧度相位矩阵.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from ao_shaping.drivers.ccd import DahengCamera
from ao_shaping.drivers.slm import SantecSLM200
from ao_shaping.utils.beam_metrics import compute_metrics
from ao_shaping.utils.file import Recorder

# ---------------- 可调参数 ----------------
EXPOSURE_MS = 0.7
SIDE_PX = 30.0            # 目标方形边长 (CCD px)
N = 64                    # 工作区(FFT网格, 与SLM面/焦面成归一化傅里叶对)
GS_ITERS = 60             # GS 迭代数
BLAZE_PERIOD_PX = 20.0    # 闪耀光栅周期(SLM px), 0=不加
USE_MRAF = True           # 混合区域幅值自由: 只约束方形信号区, 区外幅值自由
FEEDBACK_EVERY = 0        # >0: 混合自适应GS(每N轮用CCD实测幅值回注), 0=纯离线
SETTLE_S = 0.2
CAPTURE_TIMEOUT_S = 30.0
N_SAMPLE = 3
OUT_DIR = Path("data/shaping_test_gs")
SRC_AMP = "gauss"         # 源面幅值: 'gauss'/'uniform' (有近场实测图可改为加载)
SRC_W0 = 0.55             # 高斯束腰(归一化半径, ≤1)


# ---------------- 硬件/图像工具 ----------------
def acquire(ccd, n_sample=N_SAMPLE):
    return ccd.get_numpy_image(n_sample=n_sample)


def crop(img, cy, cx, n=N):
    h, w = img.shape
    y0, x0 = int(round(cy)) - n // 2, int(round(cx)) - n // 2
    pt, pb = max(0, -y0), max(0, y0 + n - h)
    pl, pr = max(0, -x0), max(0, x0 + n - w)
    p = img[max(0, y0):min(h, y0 + n), max(0, x0):min(w, x0 + n)]
    if pt or pb or pl or pr:
        p = np.pad(p, ((pt, pb), (pl, pr)), mode="edge")
    return p.astype(np.float64)


def norm_unit_sum(a):
    s = a.sum()
    return a / s if s > 1e-12 else a


def add_blaze(phi, n=N, period=BLAZE_PERIOD_PX):
    """线性闪耀相位斜坡(弧度), 周期按SLM像素计(映射到N×N网格).
    把整形图样衍射到+1级, 避开LCOS强镜面0级; period=0 不加."""
    if period <= 0:
        return phi
    ramp = 2 * np.pi * (np.arange(n) / period)
    return phi + ramp[None, :].astype(np.float32)


def display(slm:SantecSLM200, phi_n):
    """phi_n: N×N 弧度相位矩阵 -> 上采样到SLM面板分辨率后直接发送.
    mod 2π 与灰度标定由 slm.display_data 内部完成, 这里不再处理."""
    full = F.interpolate(torch.from_numpy(phi_n.astype(np.float32))[None, None],
                         size=SantecSLM200.Panel_Res, mode="bilinear",
                         align_corners=False)[0, 0].numpy().transpose()
    slm.display_data(full)


# ---------------- GS 核心 ----------------
def prop(U):
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(U, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def prop_inv(G):
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(G, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def make_source_amp(n, kind=SRC_AMP, w0=SRC_W0):
    t = torch.linspace(-1, 1, n)
    y, x = torch.meshgrid(t, t, indexing="ij")
    r2 = x * x + y * y
    if kind == "gauss":
        A = torch.exp(-r2 / w0**2) * (r2 <= 1.0)
    else:
        A = (r2 <= 1.0).float()
    return A / A.amax()


def make_square_target(n, half):
    t = torch.linspace(-1, 1, n)
    y, x = torch.meshgrid(t, t, indexing="ij")
    I = ((x.abs() <= half) & (y.abs() <= half)).float()
    return I / I.sum()


def gs_solve(A_src, I_tgt, signal_mask, iters=GS_ITERS, mraf=USE_MRAF,
             A_measured=None):
    """经典GS: 目标面幅值替换 -> 反投影 -> 源面幅值替换(保留相位).
    mraf=True: 信号区用目标幅值, 区外保留当前幅值(自由), 提升均匀度/效率.
    A_measured: 若给定(实测幅值), 信号区内用它替换理想目标幅值(自适应)."""
    A_tgt = I_tgt.sqrt()
    A_use = torch.where(signal_mask > 0, A_measured, A_tgt) if A_measured is not None else A_tgt
    phi = torch.zeros_like(A_src)
    for k in range(iters):
        U = prop(A_src * torch.exp(1j * phi))
        A_new = torch.where(signal_mask > 0, A_use, U.abs()) if mraf else A_use
        U = A_new * torch.exp(1j * torch.angle(U))
        g = prop_inv(U)
        phi = torch.atan2(g.imag, g.real)      # 源面幅值约束由A_src保证
        if (k + 1) % 20 == 0:
            err = (U.abs() - A_tgt).pow(2)[signal_mask > 0].mean()
            logger.info("  GS iter {}/{}  信号区幅值RMSE={:.5f}", k + 1, iters, err)
    return phi


# ---------------- 主流程 ----------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    recorder = Recorder("mse", "min")
    A_src = make_source_amp(N)

    with SantecSLM200() as slm, DahengCamera(exposure_time_ms=EXPOSURE_MS) as ccd:
        logger.info("SLM {}x{} / CCD {}ms", *SantecSLM200.Panel_Res, EXPOSURE_MS)

        # 1) flat(+blaze) 采初始帧, 定位工作区
        display(slm, np.zeros((N, N), np.float32))
        init_img = acquire(ccd)
        cy, cx = np.unravel_index(np.argmax(init_img), init_img.shape)
        logger.info("初始帧 peak={:.0f} 质心=({}, {})", float(np.max(init_img)), cy, cx)

        # 2) 目标方形 top-hat (单位能量)
        I0 = norm_unit_sum(crop(init_img, cy, cx, N))
        half = (SIDE_PX / 2.0) / (N / 2.0)
        I_tgt = make_square_target(N, half)
        signal_mask = (I_tgt > 0).float()

        # 3) GS: 纯离线 或 混合自适应(实测幅值回注)
        phi = None
        if FEEDBACK_EVERY <= 0:
            phi = gs_solve(A_src, I_tgt, signal_mask).numpy()
        else:
            A_meas = None
            for rnd in range(GS_ITERS // FEEDBACK_EVERY):
                phi = gs_solve(A_src, I_tgt, signal_mask,
                               iters=FEEDBACK_EVERY, A_measured=A_meas).numpy()
                display(slm, phi)
                I_meas = norm_unit_sum(crop(acquire(ccd), cy, cx, N))
                A_meas = torch.from_numpy(np.sqrt(I_meas)).float()
                logger.info("  混合GS round {}/{} (实测幅值已回注)", rnd + 1,
                            GS_ITERS // FEEDBACK_EVERY)

        # 4) 显示整形相位并评估
        display(slm, phi)
        result = acquire(ccd)

        # 裁剪区指标 (与仿真版一致)
        I_r = norm_unit_sum(crop(result, cy, cx, N))
        It = I_tgt.numpy()
        m = It > 0
        uni = I_r[m].min() / I_r[m].mean()
        ee = I_r[m].sum()
        logger.info("裁剪区: 均匀度={:.3f} 封闭能量={:.3f}", uni, ee)

        # 整帧指标 (沿用你的utils)
        metrics = compute_metrics(result, It / It.sum())
        logger.info("整帧: mse={:.5f} corr={:.4f} eff={:.4f}",
                    metrics["mse"], metrics["correlation"], metrics["efficiency"])

        recorder.append({"ccd": result, "phase": phi, **metrics})
        recorder.save_dataframe(OUT_DIR / "history.csv", sidecar_dir=OUT_DIR)
        logger.info("已保存: {}", OUT_DIR)


if __name__ == "__main__":
    main()