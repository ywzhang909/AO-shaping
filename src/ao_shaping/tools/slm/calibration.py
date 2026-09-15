# -*- coding: utf-8 -*-
"""
slm_calib_toolkit.py
SLM+CCD 一体化标定工具包 (单文件)

标定光路

激光器(单纵模, 线偏振)
   │  偏振方向对准 SLM 液晶取向
   ▼
扩束准直镜组 ──► 光斑均匀照亮 SLM 有效区(左右半屏照度均匀是 LUT 前提)
   │
   ▼
┌─────────────────────────────────────────────┐
│ 相位 SLM (Santec SLM-200, 反射式 LCOS)        │
│                                              │
│  ┌─────────────┬─────────────┐               │
│  │  左半屏       │   右半屏      │  ← LUT标定:   │
│  │  (灰度 g)    │  (灰度 g_ref)│   两半不同灰度  │
│  ├─────────────┴─────────────┤               │
│  │   几何标定: 满屏闪耀光栅      │   ← x/y 方向   │
│  │   (周期 16/20/24/32 px)     │    周期变化    │
│  └───────────────────────────┘               │
└─────────────────────────────────────────────┘
   │ 反射光
   ▼
傅里叶透镜 L1 (焦距 f)
   │
   ▼
L1 后焦面 (频谱面) ◄── CCD 放在这里
   ┌──────────────────────────┐
   │  · 0级光斑(直反光)        │ ← 位置=质心标定;
   │    └─ 内部含干涉条纹(Λ≈K/960px)│   LUT标定发生在这里
   │  · +1级光斑(K/P 位移)  · -1级 │ ← 几何标定测这两个
   └──────────────────────────┘

注意一个细节：LUT 标定时不要加挡 0 级的针孔（整形运行时才需要），几何标定反而要避开 0 级（代码里用 3×FWHM 排除盘 + 闪耀移到 +1 级）。

包含两个类:
  SLMCCDCalibrator   装配辅助 + 光束位置 + 几何标定 + 效果验证
  SLMLUTCalibrator   灰度-相位 LUT 标定(自参考干涉法) + 标定显示通道

完整流程(全部可独立调用):
  align()              装配辅助: 0级/±1级入视场检查与定心引导
  find_beam_on_slm()   刀口扫描测光束在SLM面板的中心/尺寸
  calibrate()          几何标定: center/Kx/Ky/rotation/crop_side
  verify()             几何验证: 质心漂移/预测残差/能量集中度 (PASS/FAIL)
  (LUT) calibrate()    半屏干涉法测灰度-相位曲线
  (LUT) verify()       标定残差检验 (PASS/FAIL)
  (LUT) display_phase() 标定后的弧度相位显示通道(绕过驱动自带LUT)

CLI:
  python slm_calib_toolkit.py                          # 全流程
  python slm_calib_toolkit.py --skip-align --skip-beam # 只重新标定
  python slm_calib_toolkit.py --geo-only               # 只做几何
  python slm_calib_toolkit.py --lut-only --calib calib.npz
  python slm_calib_toolkit.py --verify-only calib.npz --lut lut.npz
"""
from __future__ import annotations

import math
import time

import click
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from ao_shaping.drivers.ccd import DahengCamera
from ao_shaping.drivers.slm import Santec

SETTLE_S = 0.2
CAMERA_SAMPLES = 10


# =====================================================================
# 几何标定: 装配 -> 光束位置 -> K/旋转 -> 验证
# =====================================================================
class SLMCCDCalibrator:
    """SLM+CCD 几何标定工具. 硬件无关: 传入 slm/ccd 与 acquire 回调.

    Parameters
    ----------
    slm          具有 display_data(panel_float32) 的SLM驱动 (如 Santec)
    ccd          相机驱动
    acquire      callable(ccd) -> ndarray  采图回调 (建议含平均帧与超时保护)
    panel_res    (h, w) SLM面板分辨率, 如 Santec.Panel_Res
    settle_s     SLM显示后稳定等待时间
    """

    def __init__(self, slm:Santec, ccd:DahengCamera, settle_s: float = SETTLE_S):
        self.slm = slm
        self.ccd = ccd
        self.panel_res = slm.Panel_Res[::-1]
        self.settle_s = settle_s
        self.calib: dict | None = None

    # ------------------------------------------------------------ 底层工具
    @staticmethod
    def _moments(img: np.ndarray, exclude=None, thresh_frac: float = 0.15):
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
        self.slm.display_data(phase)
        time.sleep(self.settle_s)

    @staticmethod
    def _blaze(period_px: float, axis: str, panel_res) -> np.ndarray:
        h, w = panel_res
        if axis == "x":
            ramp = 2 * np.pi * np.arange(w) / period_px
            return np.tile(ramp, (h, 1)).astype(np.float32)
        ramp = 2 * np.pi * np.arange(h) / period_px
        return np.tile(ramp[:, None], (1, w)).astype(np.float32)

    # ------------------------------------------------------------ 阶段1: 装配辅助
    def align(self, max_rounds: int = 10, center_tol_px: float = 30.0,
              blaze_period: float = 24.0) -> dict:
        """装配辅助: 检查0级与±1级是否都进入CCD视场, 并给出定心引导.
        返回装配报告 dict(ok, rounds)."""
        report = {"rounds": [], "ok": False}
        for rnd in range(max_rounds):
            self._show(np.zeros(self.panel_res, np.float32))
            F0 = np.asarray(self.ccd.get_numpy_image(CAMERA_SAMPLES), np.float64)
            c0 = self._moments(F0)
            f0 = max(self.spot_fwhm(F0, c0), 3.0)
            H, W = F0.shape
            spots = {"0级": c0}
            for axis, tag in (("x", "+1级x"), ("y", "+1级y")):
                self._show(self._blaze(blaze_period, axis, self.panel_res))
                F = np.asarray(self.ccd.get_numpy_image(CAMERA_SAMPLES), np.float64)
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
                logger.info("装配满足要求: 所有级次在视场内且大致居中")
                report["ok"] = True
                return report
            dy, dx = off["0级"]
            sug = []
            if abs(dx) > center_tol_px:
                sug.append(f"{'←' if dx > 0 else '→'}平移CCD/光路使0级向视场中心(需移动{abs(dx):.0f}px)")
            if abs(dy) > center_tol_px:
                sug.append(f"{'↑' if dy > 0 else '↓'}俯仰调节使0级向视场中心(需移动{abs(dy):.0f}px)")
            for k, ok_ in in_fov.items():
                if not ok_:
                    sug.append(f"{k}出视场: 减小闪耀周期(当前{blaze_period}px)或增大CCD视场/减小焦距")
            logger.info("建议: {}", " ; ".join(sug) if sug else "微调后复测")

        logger.warning("装配辅助达到最大轮数仍未满足, 请人工检查光路")
        return report

    # ------------------------------------------------------------ 阶段1.5: 光束在SLM上的位置
    def find_beam_on_slm(self, n_scan: int = 25, checker_period: int = 8,
                         window: int | None = None) -> dict:
        """刀口扫描: 测光束在SLM面板上的中心(beam_center)与宽度(beam_sigma), 单位panel px.
        图案: 分割位置s一侧为0相位(全通), 另一侧为0/pi棋盘(散射走0级);
        0级功率 P(s) = 光束截面累积分布 -> 中心=50% crossing, sigma=(84%-16%)/2.
        结果写入 self.calib['beam_center']=(cy,cx), ['beam_sigma']=(sy,sx)."""
        self._show(np.zeros(self.panel_res, np.float32))
        F0 = np.asarray(self.ccd.get_numpy_image(CAMERA_SAMPLES), np.float64)
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
                P.append(power(np.asarray(self.ccd.get_numpy_image(CAMERA_SAMPLES), np.float64)))
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
                    "sigma=({:.0f},{:.0f})px 直径(1/e²)≈{:.0f}px",
                    cy, cx, off_y, off_x, sy, sx, math.sqrt(2) * (sy + sx))
        if abs(off_y) > 0.1 * h or abs(off_x) > 0.1 * w:
            logger.warning("光束偏离面板中心>10%, 建议装配调节(见align()引导)或后续图案均以beam_center为原点")
        return dict(beam_center=self.calib["beam_center"],
                    beam_sigma=self.calib["beam_sigma"])

    def place_on_panel(self, pattern_nx: np.ndarray, N: int = 64) -> np.ndarray:
        """把N×N弧度相位放到面板, 以beam_center(而非面板几何中心)为原点.
        模型里的图案坐标(0,0) ≡ 光束轴线 ≡ 0级质心."""
        if self.calib is None or "beam_center" not in self.calib:
            raise RuntimeError("先find_beam_on_slm()")
        h, w = self.panel_res
        full = F.interpolate(torch.from_numpy(np.asarray(pattern_nx, np.float32))[None, None],
                             size=(h, w), mode="bilinear", align_corners=False)
        cy, cx = self.calib["beam_center"]
        tx, ty = 2 * (cx - w / 2) / w, 2 * (cy - h / 2) / h
        theta = torch.tensor([[1.0, 0.0, tx], [0.0, 1.0, ty]],
                             dtype=torch.float32)[None]
        grid = F.affine_grid(theta, full.shape, align_corners=False)
        return F.grid_sample(full, grid, align_corners=False,
                             padding_mode="zeros")[0, 0].numpy().astype(np.float32)

    # ------------------------------------------------------------ 阶段2: 几何标定
    def calibrate(self, periods=(16, 24, 32), exclude_radius_factor: float = 3.0) -> dict:
        """自动标定 center/Kx/Ky/rotation, 结果存 self.calib 并返回.
        保留已有 beam_center/beam_sigma(阶段1.5)不丢失."""
        self._show(np.zeros(self.panel_res, np.float32))
        F0 = np.asarray(self.ccd.get_numpy_image(CAMERA_SAMPLES), np.float64)
        c0 = self._moments(F0)
        f0 = max(self.spot_fwhm(F0, c0), 3.0)
        logger.info("0级质心=({:.1f}, {:.1f})  FWHM≈{:.1f}px", c0[0], c0[1], f0)

        K, u, resid = {}, {}, {}
        for axis in ("x", "y"):
            disps, invs = [], []
            for P in periods:
                self._show(self._blaze(P, axis, self.panel_res))
                F = np.asarray(self.ccd.get_numpy_image(CAMERA_SAMPLES), np.float64)
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
        for k_src in ("beam_center", "beam_sigma"):   # 保留光束位置结果
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

    # ------------------------------------------------------------ 阶段3: 几何验证
    def verify(self, test_period: float = 20.0,
               drift_tol_px: float = 3.0, pred_tol_px: float = 5.0) -> dict:
        """定量验证标定效果. 返回报告 dict(pass, items).
        1) 0级质心漂移(重复性) 2) ±1级位置预测残差 3) 工作区能量集中度."""
        if self.calib is None:
            raise RuntimeError("先calibrate()或load()")
        c0_ref = np.asarray(self.calib["center"], np.float64)
        Kx, Ky = self.calib["Kx"], self.calib["Ky"]
        report: dict = {"items": {}}

        self._show(np.zeros(self.panel_res, np.float32))
        c0 = self._moments(np.asarray(self.ccd.get_numpy_image(CAMERA_SAMPLES), np.float64))
        drift = float(np.linalg.norm(c0 - c0_ref))
        report["items"]["center_drift_px"] = drift

        errors = {}
        for axis, K in (("x", Kx), ("y", Ky)):
            self._show(self._blaze(test_period, axis, self.panel_res))
            F = np.asarray(self.ccd.get_numpy_image(CAMERA_SAMPLES), np.float64)
            c_meas = self._moments(F, exclude=(c0, 3.0 * max(self.calib["fwhm0"], 3.0)))
            direction = np.array([0.0, 1.0]) if axis == "x" else np.array([1.0, 0.0])
            if abs(self.calib["rotation_deg"]) > 0.3:
                th = math.radians(self.calib["rotation_deg"])
                R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
                direction = R @ direction
            c_pred = c0 + direction * (K / test_period)
            errors[axis] = float(np.linalg.norm(c_meas - c_pred))
        report["items"]["pred_err_px"] = errors

        patch = self.workzone(np.asarray(self.ccd.get_numpy_image(CAMERA_SAMPLES), np.float64))
        In = patch / max(patch.sum(), 1e-12)
        yy, xx = np.mgrid[0:In.shape[0], 0:In.shape[1]]
        for frac, tag in ((0.25, "中心1/2区域"), (0.0625, "中心1/4区域")):
            r = math.sqrt(frac) * In.shape[0] / 2.0
            m = (yy - In.shape[0] / 2) ** 2 + (xx - In.shape[1] / 2) ** 2 <= r * r
            report["items"][f"encircled_{tag}"] = float(In[m].sum())

        ok = (drift < drift_tol_px and max(errors.values()) < pred_tol_px)
        report["pass"] = bool(ok)
        logger.info("几何验证: 质心漂移={:.2f}px(限{:.1f})  预测残差 x={:.2f}/y={:.2f}px(限{:.1f})  {}",
                    drift, drift_tol_px, errors["x"], errors["y"], pred_tol_px,
                    "通过 ✓" if ok else "未通过 ✗")
        for k, v in report["items"].items():
            if k.startswith("encircled"):
                logger.info("  {}: {:.3f}", k, v)
        if not ok:
            logger.warning("验证未通过: 检查装配松动/温度漂移, 或重新calibrate()")
        return report

    # ------------------------------------------------------------ 运行时接口
    def workzone(self, img: np.ndarray, N: int = 64) -> np.ndarray:
        """按标定参数取工作区: 定心(calibrated center) -> 纠旋 -> 重采样N×N.
        所有后续帧(闭环采图同理)都必须走这个函数, 保证与模型网格严格对齐."""
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
            th = math.radians(-self.calib["rotation_deg"])   # 若旋转方向反了, 去掉负号
            rot = torch.tensor([[math.cos(th), -math.sin(th), 0.0],
                                [math.sin(th), math.cos(th), 0.0]],
                               dtype=torch.float32)[None]
            grid = F.affine_grid(rot, t.shape, align_corners=False)
            t = F.grid_sample(t, grid, align_corners=False, padding_mode="border")
        return F.interpolate(t, size=(N, N), mode="bilinear",
                             align_corners=False)[0, 0].numpy().astype(np.float64)

    def gauss_amp_from_farfield(self, flat_frame: np.ndarray, N: int = 64,
                                w0_override: float | None = None) -> torch.Tensor:
        """由flat相位远场高斯拟合估计源面高斯幅值(近似).
        有更准的近场测量时直接传 w0_override 或替换此函数."""
        if w0_override is not None:
            w0 = w0_override
        else:
            fine = self.workzone(flat_frame, 256)
            In = fine / max(fine.sum(), 1e-12)
            fwhm_n = self.spot_fwhm(In, np.array(In.shape) / 2.0) / 256.0
            w0 = float(np.clip(2 * math.sqrt(2 * math.log(2)) / (math.pi * max(fwhm_n, 1e-3)),
                               0.2, 1.5))
            logger.info("远场FWHM={:.3f}(归一化) -> 源面高斯束腰 w0≈{:.3f}(可调)", fwhm_n, w0)
        t = torch.linspace(-1, 1, N)
        y, x = torch.meshgrid(t, t, indexing="ij")
        A = torch.exp(-(x * x + y * y) / w0 ** 2)
        return A / A.amax()

    def target_half_to_ccd_px(self, half_norm: float) -> float:
        """归一化半宽 -> CCD像素边长(核对目标尺寸): side_px = 2*half*K/2."""
        return float(half_norm) * max(self.calib["Kx"], self.calib["Ky"])


# =====================================================================
# LUT 标定: 自参考干涉法测灰度-相位曲线
# =====================================================================
class SLMLUTCalibrator:
    """SLM 灰度-相位 LUT 标定. 与几何标定相互独立, 可任意顺序执行.

    Parameters
    ----------
    slm         SLM驱动 (display_data 接收弧度矩阵, 内部做线性灰度映射)
    ccd, acquire 相机与采图回调
    panel_res   (h, w)
    calib       几何标定 dict(可选): 提供 0级质心与 beam_center,
                条纹窗口更准且半屏分割对称; 无则用全帧质心/面板中心
    factory_2pi 驱动内部 "弧度->灰度" 的满量程灰度 (默认255). display_phase 利用
                逆映射 gray*(2π/factory_2pi) 绕过驱动自带LUT, 强制写入标定灰度
    """

    def __init__(self, slm:Santec, ccd:DahengCamera, calib: dict | None = None,
                 settle_s: float = SETTLE_S, factory_2pi: float = 255.0,
                 window: int = 384):
        self.slm = slm
        self.ccd = ccd
        self.panel_res = slm.Panel_Res
        self.calib = calib
        self.settle_s = settle_s
        self.factory_2pi = factory_2pi
        self.window = window
        self.result: dict | None = None

    # ------------------------------------------------------------ 图案与条纹分析
    def _split_pattern(self, g_left: float, g_right: float) -> np.ndarray:
        """左半 g_left, 右半 g_right(参考), 不加闪耀(利用0级直反光干涉).
        分割位置取beam_center(若已测), 保证两半功率对称/条纹可见度."""
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
        """取中心窗口, 行平均成一维, Hann窗+FFT主峰相位. 自动找条纹频率."""
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

    # ------------------------------------------------------------ 标定主流程
    def calibrate(self, grays=None, g_ref: float = 128.0, drift_correct: bool = True):
        """全灰度扫描, 返回并缓存 dict(gray, phase_of_gray, gray_of_phase, phase_range)."""
        if grays is None:
            grays = np.linspace(0, 255, 64)
        grays = np.asarray(grays, np.float64)

        def psi(g):
            self.slm.display_data(self._split_pattern(g, g_ref))
            time.sleep(self.settle_s)
            ph, k, period = self._fringe_phase(self.ccd.get_numpy_image(CAMERA_SAMPLES))
            return ph

        logger.info("LUT标定: 参考灰度 g_ref={:.0f}, {}个灰度点 ...", g_ref, len(grays))
        psi_ref0 = psi(g_ref)
        phases = np.array([psi(g) for g in grays])
        psi_ref1 = psi(g_ref) if drift_correct else psi_ref0
        frac = np.linspace(0, 1, len(grays))
        phases = phases - (psi_ref0 + frac * (psi_ref1 - psi_ref0))
        curve = np.unwrap(phases - phases[np.argmin(np.abs(grays - g_ref))])
        curve = np.maximum.accumulate(curve)
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

    # ------------------------------------------------------------ 标定显示通道
    def phase2gray(self, phase: np.ndarray) -> np.ndarray:
        """弧度相位矩阵 -> 灰度图 (查标定LUT, 自动按调制深度缩放)"""
        if self.result is None:
            raise RuntimeError("先calibrate()或load()")
        ph = np.asarray(phase, np.float64)
        depth = float(self.result["phase_range"])
        ph01 = np.mod(ph, depth) / depth
        gray = np.interp(ph01, np.linspace(0, 1, 256),
                         np.asarray(self.result["gray_of_phase"], np.float64))
        return gray.astype(np.float32)

    def display_phase(self, phase_nx: np.ndarray):
        """标定后的相位显示: N×N弧度相位 -> 上采样 -> 查LUT -> 经驱动逆映射写入灰度.
        替代原有的 slm.display_data(interpolate(phase)) 调用链."""
        full = F.interpolate(torch.from_numpy(np.asarray(phase_nx, np.float32))[None, None],
                             size=self.panel_res, mode="bilinear",
                             align_corners=False)[0, 0].numpy()
        gray = self.phase2gray(full)
        self.slm.display_data(gray * (2 * np.pi / self.factory_2pi))
        time.sleep(self.settle_s)

    # ------------------------------------------------------------ LUT验证
    def verify(self, n_test: int = 9) -> dict:
        """用标定LUT显示 n_test 个目标相位(0~深度), 测残余条纹相位, 报告std."""
        if self.result is None:
            raise RuntimeError("先calibrate()或load()")
        depth = float(self.result["phase_range"])
        targets = np.linspace(0, 0.9 * depth, n_test)
        res = []
        for t in targets:
            gray_test = float(np.interp(t / depth, np.linspace(0, 1, 256),
                                        np.asarray(self.result["gray_of_phase"], np.float64)))
            self.slm.display_data(self._split_pattern(gray_test, self.result["g_ref"]))
            time.sleep(self.settle_s)
            ph, _, _ = self._fringe_phase(self.ccd.get_numpy_image(CAMERA_SAMPLES))
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
# CLI: 全流程 装配 -> 光束位置 -> 几何标定 -> 几何验证 -> LUT标定 -> LUT验证
# =====================================================================
@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--out-calib", "out_calib", default="calib.npz", show_default=True,
              help="几何标定输出路径")
@click.option("--out-lut", "out_lut", default="lut.npz", show_default=True,
              help="LUT标定输出路径")
@click.option("--calib", "calib_path", default=None,
              help="已有几何标定文件(lut-only/verify-only时作为输入)")
@click.option("--lut", "lut_path", default=None,
              help="已有LUT文件(verify-only时作为输入)")
@click.option("--skip-align", is_flag=True, help="跳过装配辅助")
@click.option("--skip-beam", is_flag=True, help="跳过光束位置测量(沿用已有beam_center)")
@click.option("--geo-only", is_flag=True, help="只做几何标定")
@click.option("--lut-only", is_flag=True, help="只做LUT标定")
@click.option("--verify-only", is_flag=True, help="只验证已有标定(需 --calib 和/或 --lut)")
@click.option("--exposure-ms", default=1.2, show_default=True, help="CCD曝光(ms)")
@click.option("--settle-s", default=0.2, show_default=True, help="SLM显示稳定等待(s)")
def main(out_calib, out_lut, calib_path, lut_path, skip_align, skip_beam,
         geo_only, lut_only, verify_only, exposure_ms, settle_s):
    """SLM+CCD 一体化标定工具包: 装配 -> 光束位置 -> 几何标定 -> LUT标定 -> 验证."""
    with Santec() as slm, DahengCamera(exposure_time_ms=exposure_ms) as ccd:
        geo = SLMCCDCalibrator(slm, ccd, settle_s=settle_s)

        # ---------- 仅验证 ----------
        if verify_only:
            if not calib_path and not lut_path:
                raise click.UsageError("--verify-only 需要 --calib 和/或 --lut")
            if calib_path:
                geo.load(calib_path)
                geo.verify()
            if lut_path:
                lut = SLMLUTCalibrator(slm, ccd,
                                       calib=geo.calib, settle_s=settle_s)
                lut.load(lut_path)
                lut.verify()
            return

        # ---------- 几何部分 ----------
        if not lut_only:
            if calib_path and skip_align and skip_beam:
                geo.load(calib_path)          # 以已有标定为底, 增量复标
            if not skip_align:
                geo.align()
            if not skip_beam:
                geo.find_beam_on_slm()
            geo.calibrate()
            geo.save(out_calib)
            geo.verify()
            if geo_only:
                return

        # ---------- LUT部分 ----------
        calib = geo.calib if geo.calib is not None else (
            dict(np.load(calib_path, allow_pickle=True)) if calib_path else None)
        lut = SLMLUTCalibrator(slm, ccd, calib=calib, settle_s=settle_s)
        if lut_path and lut_only:
            lut.load(lut_path)
        else:
            lut.calibrate()
            lut.save(out_lut)
        lut.verify()


if __name__ == "__main__":
    main()