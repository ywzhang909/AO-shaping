"""
fouriergsnet_hardware.py
FourierGSNet-lite 闭环束整形 + 真实硬件 (Santec SLM-200 / 大恒 CCD)

流程:
  1) flat 相位采初始帧 -> 以 0 级光斑质心为中心裁剪 N×N 工作区
  2) 在工作区内构建方形 top-hat 目标 (单位能量)
  3) 模式A(无需训练): 设备上的自适应 GS —— 每轮"显示相位->CCD 测幅值->替换幅值->反投影",
     真实相机即真实前向模型, 物理保证收敛
  4) 模式B(推荐): 在 phi0 上加已知 Zernike 扰动采数据 -> 微调 CNN 的 c_head
     -> 之后每步闭环只需一次前馈推理 (毫秒级)
  5) 闭环运行: 采图 -> 网络推理出上游像差系数 -> phi = phi_GS_unroll - c_hat*Z
     -> 上采样到 SLM 面板显示 -> 循环, 每步记录 metrics

依赖: 与主项目相同的 ao_shaping 包 (drivers/optimizer/utils/display)
"""
from __future__ import annotations

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
from ao_shaping.drivers.slm import SantecSLM200
from ao_shaping.utils.beam_metrics import compute_metrics
from ao_shaping.utils.file import Recorder
from ao_shaping.utils.hardware_utils import call_with_timeout
from ao_shaping.utils.targets import build_target_from_frame

# ---------------------------------------------------------------------------
# 可调参数
# ---------------------------------------------------------------------------
EXPOSURE_MS = 1.2          # CCD 曝光 (ms)
SIDE_PX = 30.0             # 目标方形在 CCD 上的边长 (px)
N = 64                     # 模型工作区 (CCD 裁剪 N×N)
K_UNROLL = 5               # GS 展开层数
N_ZERN = 8                 # Zernike 项数 (Z4~Z11)
CH = 32                    # CNN 通道数
EPOCHS_FT = 15             # c_head 微调轮数
BATCH_FT = 16
LR_FT = 5e-4
N_PERTURB = 200            # 微调用扰动样本数 (±0.6 rad)
SETTLE_S = 0.0             # SLM 显示后等待 (s)
CAPTURE_TIMEOUT_S = 30.0
N_SAMPLE = 3               # 相机平均帧数
GS_ITERS_DEVICE = 30       # 模式A 设备上 GS 迭代数
LOOP_STEPS = 50            # 闭环步数
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
    patch = img[max(0, y0):min(h, y0 + n), max(0, x0):min(w, x0 + n)]
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
        size=SantecSLM200.Panel_Res[::-1], mode="nearest-exact",
    )[0, 0].numpy()


def display_phase(slm: SantecSLM200, phi_n: np.ndarray):
    gray = slm.create_phase_from_array(slm_panel_phase(phi_n))
    slm.display_data(gray)


# =====================================================================
# 2. 物理模型 (归一化傅里叶对)
# =====================================================================
def prop(U):
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(U, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def prop_inv(G):
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(G, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


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
        rc**2 * torch.sin(2 * th), rc**2 * torch.cos(2 * th),
        (3 * rc**3 - 2 * rc) * torch.sin(th), (3 * rc**3 - 2 * rc) * torch.cos(th),
        rc**3 * torch.sin(3 * th), rc**3 * torch.cos(3 * th),
        6 * rc**4 - 6 * rc**2 + 1,
    ][:n_terms]
    Z = torch.stack(Z, 0)
    return Z * (r <= 1.0).float()


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
# 4. 模式A: 设备上的自适应 GS (无需训练, 相机=真实前向模型)
# =====================================================================
def adaptive_gs_on_device(slm, ccd, A_src: torch.Tensor, A_tgt: torch.Tensor,
                          phi0: torch.Tensor, roi_cy: float, roi_cx: float,
                          iters: int = GS_ITERS_DEVICE, gamma: float = 0.6):
    """设备上的自适应 GS: 模型 GS 半步 + 实测幅值反馈, 相机即真实前向.

    每轮:
      1) 显示当前相位 φ_k, CCD 实测焦斑幅值 A_meas (真实前向)
      2) 模型前向 G = prop(A_src·exp(iφ_k)), 取模型相位 ∠G
      3) 目标幅值约束: A_n = A_tgt·(实测能量/目标能量), 并与实测混合:
         A_n = (1-gamma)·A_meas + gamma·A_tgt·(ΣA_meas/ΣA_tgt)
      4) 反投影 g = prop_inv(A_n·exp(i∠G)), φ_{k+1} = ∠g, 源平面全孔径自由

    相位更新由**目标幅值**驱动 (而非把实测幅值反投影回去), 这才在物理上
    等价于 GS; 实测幅值以 (1-gamma) 的权重混合进目标, 即 IFTA 阻尼:
    与仓库已验证的 adaptive_gerchberg_saxton (feedback_weight=0.3) 同一机制.
    纯模型目标 (gamma=1.0) 在硬件上有模型-实物失配时会把能量推到模型认为的
    方形位置, 实际 CCD 上仍是中心峰 — 必须让实测幅值参与更新.

    Parameters
    ----------
    A_src : 源平面幅值 (实测平场焦斑的 sqrt), 模型前向的入射场幅值
    A_tgt : 远场目标幅值 (N×N, sqrt of unit-sum square)
    gamma : 目标混合权重, A_fb = (1-gamma)·A_meas + gamma·A_tgt·scale
            gamma=1.0 -> 纯目标约束; gamma=0.6 -> 40% 实测 + 60% 目标 (推荐)
    roi_cy, roi_cx : CCD 全帧 0 级质心, 用于裁剪 N×N 工作区
    """
    n = A_tgt.shape[0]
    # 计算全程留在 GPU (与 gs_unroll / Z 一致), 显示侧维护一个持久 CPU 缓冲
    phi = phi0.clone().to(A_tgt.device)
    use_cuda = A_tgt.is_cuda
    phi_cpu = torch.empty_like(phi0, device="cpu", pin_memory=use_cuda)
    phi_cpu.copy_(phi0.detach())
    roi = (A_tgt > 0).float()                         # 远场方形掩码 (评价口径)
    src_mask = torch.ones(n, n, device=A_tgt.device)  # 源平面全孔径自由
    tgt_sum = A_tgt.sum().clamp_min(1e-12)
    history: list[dict] = []
    # 源幅值初始用平场实测 (sqrt(I0)); 之后每轮用**当前实测幅值**重新锚定,
    # 与仓库已验证的 gs_square_runner 外层机制一致 — 源幅值追踪真实光束,
    # 否则模型会用冻结源解出一个只在其内部成立的方形, 物理上不展平
    src = A_src.clone()
    for k in range(iters):
        display_phase(slm, phi_cpu.numpy())
        raw = acquire(ccd)
        # 裁剪到 N×N 工作区 (与 main 中 I0 一致: 先裁剪后归一)
        patch = crop_to_workzone(raw, roi_cy, roi_cx, n)
        I = torch.from_numpy(norm_unit_sum(patch)).float().to(A_tgt.device)
        A_meas = I.sqrt()                                     # 实测幅值 (真实前向输出)
        src = A_meas.detach().clone()                          # 源幅值 = 当前实测光束
        # 模型 GS 半步: 前向(源幅值×当前相位) -> 目标幅值(能量对齐实测) -> 反投影
        G = prop(src * torch.exp(1j * phi))
        A_n = A_tgt * (A_meas.sum() / tgt_sum)
        if gamma < 1.0:
            A_n = (1.0 - gamma) * A_meas + gamma * A_n
        G = A_n * torch.exp(1j * torch.angle(G))
        g = prop_inv(G)
        phi = wrap_pi(torch.angle(g)) * src_mask              # 源面相位约束 (全孔径)
        # 异步 H2D: 只写持久缓冲, 不产生新分配; 下次 display 前的 .numpy() 会同步
        phi_cpu.copy_(phi.detach(), non_blocking=use_cuda)
        # 与闭环同口径指标 (roi 内均匀度 + 封闭能量 + CV)
        I_n = I / I.sum()
        imin = torch.where(roi > 0, I_n, torch.full_like(I_n, float("inf"))).min()
        uni = (imin / (I_n * roi).sum() * roi.sum()).item()
        ee = (I_n * roi).sum().item()
        roi_vals = I_n[roi > 0]
        cv = (roi_vals.std() / roi_vals.mean().clamp_min(1e-12)).item()
        history.append({"iter": k, "uniformity": uni, "encircled": ee,
                        "cv": cv, "ccd": patch.astype(np.float32),
                        "phase": phi_cpu.numpy().copy()})
        logger.info("  设备GS iter {}/{}  均匀度={:.3f}  封闭能量={:.3f}  CV={:.3f}",
                    k + 1, iters, uni, ee, cv)
    return phi, history


# =====================================================================
# 5. 模式B: 扰动采样 + c_head 微调 (sim2real 轻量版)
# =====================================================================
def collect_finetune_data(slm, ccd, phi0, Z, roi_cy, roi_cx,
                          n_samples=N_PERTURB) -> tuple[list, list]:
    """在 phi0 上加已知 Zernike 扰动, 记录 (I_meas, c); I_meas 裁剪为 N×N 工作区.
    返回 (data, samples): data 供微调, samples 用于落盘留档."""
    data, samples = [], []
    for i in range(n_samples):
        c = (torch.rand(Z.shape[0]) * 1.2 - 0.6)              # ±0.6 rad
        phi = wrap_pi(phi0 + aberration(Z, c.to(phi0.device)))
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
            batch = [data[j] for j in perm[i:i + BATCH_FT]]
            I_m = torch.stack([b[0] for b in batch]).to(DEVICE)
            c_gt = torch.stack([b[1] for b in batch]).to(DEVICE)
            B = I_m.shape[0]
            _, c_hat = net(I_m, I_tgt.expand(B, -1, -1), A_src.expand(B, -1, -1),
                           A_tgt.expand(B, -1, -1), phi0.expand(B, -1, -1), Z)
            loss = (c_hat - c_gt).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        logger.info("  finetune epoch {}/{}  |c|err={:.4f} rad", ep + 1, EPOCHS_FT, tot / max(nb, 1))
    return net


# =====================================================================
# 6. 闭环主循环
# =====================================================================
def closed_loop(slm, ccd, net, A_tgt, A_src, phi, Z, I_tgt, roi, recorder,
                roi_cy: float, roi_cx: float, steps: int = LOOP_STEPS):
    logger.info("进入闭环: {} 步, 每步=采图+一次前馈推理", steps)
    for step in range(steps):
        raw = norm_unit_sum(acquire(ccd))
        I = crop_to_workzone(raw, roi_cy, roi_cx, N)
        I_t = torch.from_numpy(I).float().to(DEVICE)
        t0 = time.time()
        phi, c_hat = net(I_t.unsqueeze(0), I_tgt.unsqueeze(0), A_src.unsqueeze(0),
                         A_tgt.unsqueeze(0), phi.unsqueeze(0), Z)
        # 推理输出脱离计算图: 只用于显示/记录/下一轮输入, 且避免跨步图累积
        phi = phi.squeeze(0).detach()
        c_hat = c_hat.detach()
        dt = (time.time() - t0) * 1000
        display_phase(slm, phi.cpu().numpy())

        # 曝光无关评估 (裁剪区内, 与仿真指标一致)
        I_n = I_t / I_t.sum()
        imin = torch.where(roi > 0, I_n, torch.full_like(I_n, float("inf"))).min()
        uni = (imin / (I_n * roi).sum() * (roi > 0).sum()).item()
        ee = (I_n * roi).sum().item()
        logger.info("step {:3d}  推理 {:5.1f}ms  均匀度={:.3f}  封闭能量={:.3f}  |c_hat|max={:.2f}",
                    step + 1, dt, uni, ee, float(c_hat.abs().max()))
        recorder.append({"step": step, "uniformity": uni, "encircled": ee,
                         "inference_ms": dt, "ccd": I, "phase": phi.cpu().numpy()})
    return phi


# =====================================================================
# 7. main
# =====================================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = OUT_DIR / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    recorder = Recorder("uniformity", "max")
    grid_w, grid_h = SantecSLM200.Panel_Res

    with SantecSLM200() as slm, DahengCamera(exposure_time_ms=EXPOSURE_MS) as ccd:
        logger.info("设备已连接: SLM {}x{} / CCD {}ms", grid_h, grid_w, EXPOSURE_MS)

        # 1) flat 相位 -> 初始帧
        phase0 = np.zeros((grid_h, grid_w), dtype=np.uint16)
        slm.display_data(phase0)
        init_img = acquire(ccd)
        cy, cx = (float(v) for v in np.unravel_index(np.argmax(init_img), init_img.shape))
        logger.info("初始帧 shape={} peak={:.0f} 0级质心=({}, {})",
                    init_img.shape, float(np.max(init_img)), cy, cx)

        # 2) 工作区: 裁剪 N×N; 目标方形 top-hat (单位能量); 源幅值=sqrt(实测)
        I0 = norm_unit_sum(crop_to_workzone(init_img, cy, cx, N))
        A_src = torch.from_numpy(np.sqrt(I0)).float().to(DEVICE)
        half = (SIDE_PX / 2.0) / (N / 2.0)                      # CCD px -> 归一化坐标
        I_tgt = make_square_target(N, half, DEVICE)
        A_tgt = I_tgt.sqrt()
        roi = (I_tgt > 0).float()
        Z = zernike_basis(N, N_ZERN, DEVICE)
        src_mask = (Z[0] != 0).float()

        # 3) 初值: 优先加载已有模型+相位; 否则设备GS + 扰动微调 + 保存
        gs_history = []
        phi_path = OUT_DIR / "gs_phase.npy"
        if CKPT.exists():
            logger.info("加载模型 {}", CKPT)
            net = FourierGSNetLite(K_UNROLL, N_ZERN, CH, src_mask).to(DEVICE)
            net.load_state_dict(torch.load(CKPT, map_location=DEVICE))
            if phi_path.exists():
                phi = torch.from_numpy(np.load(phi_path)).float().to(DEVICE)
                logger.info("已加载上次 GS 相位 {}", phi_path)
            else:
                phi = torch.zeros(N, N, device=DEVICE)
        else:
            logger.info("无模型: 先运行设备自适应GS求初值 ...")
            phi, gs_history = adaptive_gs_on_device(slm, ccd, A_src, A_tgt,
                                                    torch.zeros(N, N, device=DEVICE),
                                                    roi_cy=cy, roi_cx=cx)
            np.save(phi_path, phi.cpu().numpy())
            logger.info("GS 相位已保存 {}", phi_path)
            with open(run_dir / "gs_device.pkl", "wb") as fh:
                pickle.dump(gs_history, fh)
            net = FourierGSNetLite(K_UNROLL, N_ZERN, CH, src_mask).to(DEVICE)
            logger.info("扰动采样+微调 c_head ...")
            data, samples = collect_finetune_data(slm, ccd, phi, Z, cy, cx)
            with open(run_dir / "finetune_samples.pkl", "wb") as fh:
                pickle.dump(samples, fh)
            net = finetune_chead(net, data, A_tgt, A_src, phi, Z, I_tgt)
            torch.save(net.state_dict(), CKPT)
            logger.info("模型已保存 {}", CKPT)

        # 4) 闭环
        phi = closed_loop(slm, ccd, net, A_tgt, A_src, phi, Z, I_tgt, roi,
                          recorder, cy, cx)

        # 5) 整帧评估 + 保存 (与闭环步同口径: roi 内均匀度 + 封闭能量)
        final_img = crop_to_workzone(acquire(ccd), cy, cx, N)
        metrics = compute_metrics(final_img, I_tgt.cpu().numpy() / I_tgt.sum().cpu())
        I_f = torch.from_numpy(final_img).float().to(DEVICE)
        I_fn = I_f / I_f.sum()
        imin = torch.where(roi > 0, I_fn, torch.full_like(I_fn, float("inf"))).min()
        uni = (imin / (I_fn * roi).sum() * (roi > 0).sum()).item()
        ee = (I_fn * roi).sum().item()
        logger.info("整帧 metrics: mse={:.5f} corr={:.4f} eff={:.4f} 均匀度={:.3f} 封闭能量={:.3f}",
                    metrics["mse"], metrics["correlation"], metrics["efficiency"], uni, ee)
        recorder.append({"ccd": final_img, "phase": phi.cpu().numpy(),
                         "uniformity": uni, "encircled": ee, **metrics})

        # 6) 运行归档: 参数快照 + 全量记录
        params = {k: v for k, v in globals().items()
                  if k.isupper() and isinstance(v, (int, float, str, bool, Path))}
        with open(run_dir / "params.json", "w", encoding="utf-8") as fh:
            json.dump({k: str(v) for k, v in params.items()}, fh, indent=2)
        archive = {"run_dir": str(run_dir), "init_frame": init_img.astype(np.float32),
                   "centroid": (cy, cx), "gs_device": gs_history,
                   "loop": recorder.history,
                   "final_frame": final_img.astype(np.float32),
                   "target": I_tgt.cpu().numpy(), "params": params}
        with open(run_dir / "run.pkl", "wb") as fh:
            pickle.dump(archive, fh)
        recorder.save_dataframe(run_dir / "history.csv", sidecar_dir=run_dir)
        logger.info("结果已保存: {} ({} 条闭环记录)", run_dir, len(recorder.history))


if __name__ == "__main__":
    main()