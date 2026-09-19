"""
fouriergsnet_hardware.py
FourierGSNet-lite 闭环束整形 + 真实硬件 (Santec SLM-200 / 大恒 CCD)

流程 (模式B):
  1) 加载预生成的全分辨率 GS 相位 (phi_full)  --init-phase 指定路径, 默认 data/shaping_test/gs_phase.npy
  2) flat 相位采初始帧 -> 以 0 级光斑质心为中心裁剪 N×N 工作区
  3) 在 phi_full (全面板) 上加已知 Zernike 扰动采数据 -> 微调 CNN 的 c_head
  4) 闭环运行: 采图 -> 网络推理出上游像差系数 -> phi = phi_full - c_hat*Z_full
     -> 全面板显示 -> 循环, 每步记录 metrics

依赖: 与主项目相同的 ao_shaping 包 (drivers/optimizer/utils/display)
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from ao_shaping.drivers.ccd import DahengCamera
from ao_shaping.drivers.slm import Santec
from ao_shaping.utils.beam_metrics import (
    compute_metrics,
    compute_quality_score,
    compute_square_metrics,
)
from ao_shaping.utils.file import Recorder

# ---------------------------------------------------------------------------
# 可调参数
# ---------------------------------------------------------------------------
EXPOSURE_MS = 1.2  # CCD 曝光 (ms)
SIDE_PX = 30.0  # 目标方形在 CCD 上的边长 (px)
N = 64  # 模型工作区 (CCD 裁剪 N×N)
K_UNROLL = 5  # GS 展开层数
N_ZERN = 8  # Zernike 项数 (Z4~Z11)
CH = 32  # CNN 通道数
EPOCHS_FT = 15  # c_head 微调轮数
BATCH_FT = 16
LR_FT = 5e-4
N_PERTURB = 200  # 微调用扰动样本数 (±0.6 rad)
SETTLE_S = 0.0  # SLM 显示后等待 (s)
CAPTURE_TIMEOUT_S = 30.0
N_SAMPLE = 3  # 相机平均帧数
GS_ENERGY = 0.90  # 光斑环围能量比例 (尺寸标定/指标)
LOOP_STEPS = 50  # 闭环步数
CKPT = Path("data/shaping_test/fouriergsnet_lite.pth")
OUT_DIR = Path("data/shaping_test")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =====================================================================
# 1. 硬件/图像工具
# =====================================================================
def acquire(ccd: DahengCamera, n_sample: int = N_SAMPLE) -> np.ndarray:
    return ccd.get_numpy_image(n_sample=n_sample)


def crop_to_workzone(img: np.ndarray, cy: float, cx: float, n: int = N) -> np.ndarray:
    """以 (cy,cx) 为中心裁剪 n×n, 越界用边缘补."""
    h, w = img.shape
    y0, x0 = int(round(cy)) - n // 2, int(round(cx)) - n // 2
    pad_top, pad_bot = max(0, -y0), max(0, y0 + n - h)
    pad_lft, pad_rgt = max(0, -x0), max(0, x0 + n - w)
    patch = img[max(0, y0) : min(h, y0 + n), max(0, x0) : min(w, x0 + n)]
    if pad_top or pad_bot or pad_lft or pad_rgt:
        patch = np.pad(patch, ((pad_top, pad_bot), (pad_lft, pad_rgt)), mode="edge")
    return patch.astype(np.float64)


def norm_unit_sum(a: np.ndarray) -> np.ndarray:
    s = a.sum()
    return a / s if s > 1e-12 else a


def slm_panel_phase(phi_n: np.ndarray) -> np.ndarray:
    """N×N 弧度相位 -> SLM 全面板 float32, 直接发送.

    用 **nearest-exact** 复制 (每个模型单元->整块宏像素) 而非双线性插值:
    - 双线性对已 wrap 的相位做插值会在 2π 折痕处产生虚假梯度, 且平滑后
      高频分量比模型假设的少, 导致物理远场更窄/更像高斯 (实测 iter1→iter30
      只变宽不展平)
    - 单元恒定复制使 SLM 上渲染的相位与 64×64 模型假设一致, GS 才能在
      真实远场形成方形
    mod 2π 与灰度标定由 slm.display_data 内部完成, 这里只上采样."""
    return F.interpolate(
        torch.from_numpy(phi_n.astype(np.float32))[None, None],
        size=Santec.Panel_Res[::-1],
        mode="nearest-exact",
    )[0, 0].numpy()


def _display_with_retry(slm: Santec, gray: np.ndarray, retries: int = 2):
    """显示灰度相位, 对瞬时 USB/SDK 错误整路径重试.

    驱动内建 _write_phase_with_retry 只覆盖 write_phase; 这里再包一层,
    把 display_data (写入+display_memory) 作为整体重试, 抵御实测到的
    SLM 内存槽写入瞬时错误 (如 -10032)."""
    from ao_shaping.drivers.slm.santec import SantecError

    for attempt in range(retries + 1):
        try:
            slm.display_data(gray)
            return
        except SantecError as e:
            if attempt >= retries:
                raise
            logger.warning("SLM 显示失败 ({}), 重试 {}/{}", e, attempt + 1, retries)
            time.sleep(0.5)


def display_phase(slm: Santec, phi_n: np.ndarray):
    """显示弧度相位: 全面板尺寸直接下发, 其他尺寸 nearest-exact 上采样."""
    grid_w, grid_h = Santec.Panel_Res
    if phi_n.shape != (grid_h, grid_w):
        phi_n = slm_panel_phase(phi_n)
    gray = slm.create_phase_from_array(phi_n.astype(np.float32))
    _display_with_retry(slm, gray)


def _phase_thumbs(phi_full: torch.Tensor, n: int = N) -> np.ndarray:
    """全分辨率弧度相位 -> n×n 缩略图 (bilinear 插值 cos/sin 再 atan2).

    对已 wrap 的相位直接插值会在 2π 折痕处产生虚假值, 因此插值复数场."""
    with torch.no_grad():
        c = F.interpolate(
            torch.cos(phi_full)[None, None],
            size=(n, n),
            mode="bilinear",
            align_corners=False,
        )
        s = F.interpolate(
            torch.sin(phi_full)[None, None],
            size=(n, n),
            mode="bilinear",
            align_corners=False,
        )
        return torch.atan2(s, c)[0, 0].cpu().numpy()


# =====================================================================
# 2. 物理模型 (归一化傅里叶对)
# =====================================================================
def prop(U):
    return torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(U, dim=(-2, -1)), norm="ortho"), dim=(-2, -1)
    )


def prop_inv(G):
    return torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(G, dim=(-2, -1)), norm="ortho"),
        dim=(-2, -1),
    )


def wrap_pi(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


def zernike_basis(n, n_terms, device):
    t = torch.linspace(-1, 1, n, device=device)
    y, x = torch.meshgrid(t, t, indexing="ij")
    r = torch.sqrt(x * x + y * y)
    th = torch.arctan2(y, x)
    rc = torch.clamp(r, max=1.0)
    Z = [
        2 * rc**2 - 1,
        rc**2 * torch.sin(2 * th),
        rc**2 * torch.cos(2 * th),
        (3 * rc**3 - 2 * rc) * torch.sin(th),
        (3 * rc**3 - 2 * rc) * torch.cos(th),
        rc**3 * torch.sin(3 * th),
        rc**3 * torch.cos(3 * th),
        6 * rc**4 - 6 * rc**2 + 1,
    ][:n_terms]
    Z = torch.stack(Z, 0)
    return Z * (r <= 1.0).float()


def zernike_basis_panel(h: int, w: int, n_terms: int, device: str) -> torch.Tensor:
    """(n_terms, h, w) Zernike 基, 在 SLM 面板网格上求值.

    网格按较长边取 n 使 x,y∈[-1,1] 等比例, 垂直居中裁剪到 h 行 —
    与 zernike_basis(n) 保持相同的圆形孔径约定; Zernike 系数分辨率无关,
    c_hat 可直接用于面板分辨率相位合成."""
    m = max(h, w)
    t = torch.linspace(-1, 1, m, device=device)
    y, x = torch.meshgrid(t, t, indexing="ij")
    r = torch.sqrt(x * x + y * y)
    th = torch.arctan2(y, x)
    rc = torch.clamp(r, max=1.0)
    Z = [
        2 * rc**2 - 1,
        rc**2 * torch.sin(2 * th),
        rc**2 * torch.cos(2 * th),
        (3 * rc**3 - 2 * rc) * torch.sin(th),
        (3 * rc**3 - 2 * rc) * torch.cos(th),
        rc**3 * torch.sin(3 * th),
        rc**3 * torch.cos(3 * th),
        6 * rc**4 - 6 * rc**2 + 1,
    ][:n_terms]
    Z = torch.stack(Z, 0)
    Z = Z * (r <= 1.0).float()
    y0 = (m - h) // 2
    return Z[:, y0 : y0 + h, :]


def aberration(Z, c):
    squeeze = c.dim() == 1
    if squeeze:
        c = c.unsqueeze(0)
    ab = torch.einsum("bn,nm->bm", c, Z.flatten(1)).unflatten(1, Z.shape[-2:])
    return ab.squeeze(0) if squeeze else ab


def make_square_target(n, half, device):
    t = torch.linspace(-1, 1, n, device=device)
    y, x = torch.meshgrid(t, t, indexing="ij")
    I = ((x.abs() <= half) & (y.abs() <= half)).float()
    return I / I.sum()


def gs_unroll(A_src, A_tgt, phi0, K, src_mask):
    """简化前向(无像差FFT)下K层GS双向投影 -> (相位, K通道特征)"""
    phi = phi0
    feats = []
    for k in range(K):
        G = prop(A_src * torch.exp(1j * phi))
        G = A_tgt * torch.exp(1j * torch.angle(G))
        g = prop_inv(G)
        phi = wrap_pi(torch.angle(g)) * src_mask
        feats.append(phi)
    return phi, torch.stack(feats, 1)


# =====================================================================
# 3. FourierGSNet-lite (与仿真版相同结构)
# =====================================================================
class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.c2(self.act(self.c1(x))))


class FourierGSNetLite(nn.Module):
    def __init__(self, K, n_zern, ch, src_mask):
        super().__init__()
        self.K = K
        self.enc1 = ConvBlock(K + 2, ch)
        self.enc2 = ConvBlock(ch, ch * 2)
        self.dec = ConvBlock(ch * 3, ch)
        self.c_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(ch, n_zern)
        )
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
# 4. 模式B: 扰动采样 + c_head 微调 (sim2real 轻量版)
# =====================================================================
def collect_finetune_data(
    slm, ccd, phi0_full, Z_full, roi_cy, roi_cx, n_samples=N_PERTURB
) -> tuple[list, list]:
    """在 phi0_full (全面板) 上加已知 Zernike 扰动, 记录 (I_meas, c).

    I_meas 裁剪为 N×N 工作区. 显示全分辨率相位, 与闭环显示口径一致.
    返回 (data, samples): data 供微调, samples 用于落盘留档."""
    data, samples = [], []
    for i in range(n_samples):
        c = torch.rand(Z_full.shape[0]) * 1.2 - 0.6  # ±0.6 rad
        phi = wrap_pi(phi0_full + aberration(Z_full, c.to(phi0_full.device)))
        display_phase(slm, phi.cpu().numpy())
        raw = norm_unit_sum(acquire(ccd))
        I = crop_to_workzone(raw, roi_cy, roi_cx, N)
        data.append((torch.from_numpy(I).float(), c))
        samples.append({"idx": i, "ccd": I.astype(np.float32), "c": c.numpy()})
        if (i + 1) % 50 == 0:
            logger.info("  扰动采样 {}/{}", i + 1, n_samples)
    return data, samples


def finetune_chead(net, data, A_tgt, A_src, phi0, Z, I_tgt):
    opt = torch.optim.AdamW(net.parameters(), lr=LR_FT)
    for ep in range(EPOCHS_FT):
        perm = torch.randperm(len(data))
        tot, nb = 0.0, 0
        for i in range(0, len(perm) - BATCH_FT + 1, BATCH_FT):
            batch = [data[j] for j in perm[i : i + BATCH_FT]]
            I_m = torch.stack([b[0] for b in batch]).to(DEVICE)
            c_gt = torch.stack([b[1] for b in batch]).to(DEVICE)
            B = I_m.shape[0]
            _, c_hat = net(
                I_m,
                I_tgt.expand(B, -1, -1),
                A_src.expand(B, -1, -1),
                A_tgt.expand(B, -1, -1),
                phi0.expand(B, -1, -1),
                Z,
            )
            loss = (c_hat - c_gt).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        logger.info(
            "  finetune epoch {}/{}  |c|err={:.4f} rad",
            ep + 1,
            EPOCHS_FT,
            tot / max(nb, 1),
        )
    return net


# =====================================================================
# 5. 闭环主循环
# =====================================================================
def closed_loop(
    slm,
    ccd,
    net,
    phi_full,
    Z_full,
    I_tgt,
    A_tgt,
    A_src,
    phi_lo,
    Z,
    roi,
    recorder,
    roi_cy,
    roi_cx,
    steps=LOOP_STEPS,
):
    logger.info("进入闭环: {} 步, 每步=采图+一次前馈推理", steps)
    for step in range(steps):
        raw = norm_unit_sum(acquire(ccd))
        I = crop_to_workzone(raw, roi_cy, roi_cx, N)
        I_t = torch.from_numpy(I).float().to(DEVICE)
        t0 = time.time()
        phi_lo, c_hat = net(
            I_t.unsqueeze(0),
            I_tgt.unsqueeze(0),
            A_src.unsqueeze(0),
            A_tgt.unsqueeze(0),
            phi_lo.unsqueeze(0),
            Z,
        )
        # 推理输出脱离计算图: 只用于显示/记录/下一轮输入, 避免跨步图累积
        phi_lo = phi_lo.squeeze(0).detach()
        c_hat = c_hat.detach()
        dt = (time.time() - t0) * 1000
        # 全面板合成: 显示相位 = phi_full - c_hat·Z_full (Zernike 系数分辨率无关)
        phi_disp = wrap_pi(phi_full - aberration(Z_full, c_hat.squeeze(0)))
        display_phase(slm, phi_disp.cpu().numpy())

        # 曝光无关评估 (裁剪区内, 与仿真指标一致)
        I_n = I_t / I_t.sum()
        imin = torch.where(roi > 0, I_n, torch.full_like(I_n, float("inf"))).min()
        uni = (imin / (I_n * roi).sum() * (roi > 0).sum()).item()
        ee = (I_n * roi).sum().item()
        roi_vals = I_n[roi > 0]
        cv = (roi_vals.std() / roi_vals.mean().clamp_min(1e-12)).item()
        logger.info(
            "step {:3d}  推理 {:5.1f}ms  均匀度={:.3f}  封闭能量={:.3f}  CV={:.3f}  |c_hat|max={:.2f}",
            step + 1,
            dt,
            uni,
            ee,
            cv,
            float(c_hat.abs().max()),
        )
        recorder.append(
            {
                "step": step,
                "uniformity": uni,
                "encircled": ee,
                "cv": cv,
                "inference_ms": dt,
                "ccd": I,
                "phase": _phase_thumbs(phi_disp, N),
            }
        )
    return phi_disp


# =====================================================================
# 6. main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="FourierGSNet-lite Mode B (perturbation fine-tune + closed loop)"
    )
    parser.add_argument(
        "--init-phase",
        type=str,
        default=str(OUT_DIR / "gs_phase.npy"),
        help="Path to the pre-generated full-resolution GS phase .npy file "
        "(radians, grid_h x grid_w).",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = OUT_DIR / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    recorder = Recorder("uniformity", "max")
    grid_w, grid_h = Santec.Panel_Res

    phi_path = Path(args.init_phase)
    if not phi_path.exists():
        raise FileNotFoundError(
            f"--init-phase 未找到: {phi_path}. "
            "请先在仿真/其他工具中生成全分辨率 GS 相位后再运行模式B."
        )
    phi_full = torch.from_numpy(np.load(phi_path)).float().to(DEVICE)
    if phi_full.shape != (grid_h, grid_w):
        raise ValueError(
            f"--init-phase 形状 {tuple(phi_full.shape)} 与 SLM 面板 {grid_h}x{grid_w} 不符."
        )
    logger.info("已加载初始全分辨率 GS 相位: {}", phi_path)

    with Santec() as slm, DahengCamera(exposure_time_ms=EXPOSURE_MS) as ccd:
        logger.info("设备已连接: SLM {}x{} / CCD {}ms", grid_h, grid_w, EXPOSURE_MS)

        # flat 相位 -> 初始帧 (用于工作区裁剪中心 + 源幅值)
        phase0 = np.zeros((grid_h, grid_w), dtype=np.uint16)
        slm.display_data(phase0)
        init_img = acquire(ccd)
        cy, cx = (
            float(v) for v in np.unravel_index(np.argmax(init_img), init_img.shape)
        )
        logger.info(
            "初始帧 shape={} peak={:.0f} 0级质心=({}, {})",
            init_img.shape,
            float(np.max(init_img)),
            cy,
            cx,
        )

        # 2) 工作区: 裁剪 N×N; 目标方形 top-hat (单位能量); 源幅值=sqrt(实测)
        I0 = norm_unit_sum(crop_to_workzone(init_img, cy, cx, N))
        A_src = torch.from_numpy(np.sqrt(I0)).float().to(DEVICE)
        half = (SIDE_PX / 2.0) / (N / 2.0)  # CCD px -> 归一化坐标
        I_tgt = make_square_target(N, half, DEVICE)
        A_tgt = I_tgt.sqrt()
        roi = (I_tgt > 0).float()
        Z = zernike_basis(N, N_ZERN, DEVICE)
        Z_full = zernike_basis_panel(grid_h, grid_w, N_ZERN, DEVICE)
        src_mask = (Z[0] != 0).float()

        # 3) 加载/微调模型, 闭环/微调的 64×64 低分辨率相位输入 (全面板缩略图)
        phi_lo = torch.from_numpy(_phase_thumbs(phi_full, N)).float().to(DEVICE)

        net = FourierGSNetLite(K_UNROLL, N_ZERN, CH, src_mask).to(DEVICE)
        if CKPT.exists():
            net.load_state_dict(torch.load(CKPT, map_location=DEVICE))
            logger.info("加载模型 {}", CKPT)
        else:
            logger.info("扰动采样+微调 c_head ...")
            data, samples = collect_finetune_data(slm, ccd, phi_full, Z_full, cy, cx)
            with open(run_dir / "finetune_samples.pkl", "wb") as fh:
                pickle.dump(samples, fh)
            net = finetune_chead(net, data, A_tgt, A_src, phi_lo, Z, I_tgt)
            torch.save(net.state_dict(), CKPT)
            logger.info("模型已保存 {}", CKPT)

        # 4) 闭环
        phi_disp = closed_loop(
            slm,
            ccd,
            net,
            phi_full,
            Z_full,
            I_tgt,
            A_tgt,
            A_src,
            phi_lo,
            Z,
            roi,
            recorder,
            cy,
            cx,
        )

        # 5) 整帧评估 + 保存 (与闭环步同口径: roi 内均匀度 + 封闭能量 + 全帧方形指标)
        final_img = acquire(ccd)
        ncy, ncx = (
            float(v) for v in np.unravel_index(np.argmax(final_img), final_img.shape)
        )
        mets = compute_square_metrics(
            final_img, int(round(SIDE_PX)), (ncx, ncy), energy=GS_ENERGY
        )
        qscore = compute_quality_score(mets)
        patch = crop_to_workzone(final_img, ncy, ncx, N)
        metrics = compute_metrics(patch, I_tgt.cpu().numpy() / I_tgt.sum().cpu())
        I_f = torch.from_numpy(patch).float().to(DEVICE)
        I_fn = I_f / I_f.sum()
        imin = torch.where(roi > 0, I_fn, torch.full_like(I_fn, float("inf"))).min()
        uni = (imin / (I_fn * roi).sum() * (roi > 0).sum()).item()
        ee = (I_fn * roi).sum().item()
        roi_vals = I_fn[roi > 0]
        cv = (roi_vals.std() / roi_vals.mean().clamp_min(1e-12)).item()
        logger.info(
            "整帧 metrics: mse={:.5f} corr={:.4f} eff={:.4f} 均匀度={:.3f} 封闭能量={:.3f} "
            "CV={:.3f} 方形评分={:.3f}",
            metrics["mse"],
            metrics["correlation"],
            metrics["efficiency"],
            uni,
            ee,
            cv,
            qscore,
        )
        recorder.append(
            {
                "ccd": patch.astype(np.float32),
                "phase": _phase_thumbs(phi_disp, N),
                "uniformity": uni,
                "encircled": ee,
                "cv": cv,
                "quality_score": qscore,
                "squareness": mets["squareness"],
                "uniformity_cv_full": mets["uniformity_cv"],
                "encircled_energy_full": mets["encircled_energy"],
                **metrics,
            }
        )

        # 6) 运行归档: 参数快照 + 全量记录
        params = {
            k: v
            for k, v in globals().items()
            if k.isupper() and isinstance(v, (int, float, str, bool, Path))
        }
        with open(run_dir / "params.json", "w", encoding="utf-8") as fh:
            json.dump({k: str(v) for k, v in params.items()}, fh, indent=2)
        archive = {
            "run_dir": str(run_dir),
            "init_frame": init_img.astype(np.float32),
            "centroid": (cy, cx),
            "loop": recorder.history,
            "final_frame": patch.astype(np.float32),
            "phi_full": phi_disp.cpu().numpy().astype(np.float32),
            "target": I_tgt.cpu().numpy(),
            "params": params,
        }
        with open(run_dir / "run.pkl", "wb") as fh:
            pickle.dump(archive, fh)
        recorder.save_dataframe(run_dir / "history.csv", sidecar_dir=run_dir)
        logger.info("结果已保存: {} ({} 条闭环记录)", run_dir, len(recorder.history))


if __name__ == "__main__":
    main()
