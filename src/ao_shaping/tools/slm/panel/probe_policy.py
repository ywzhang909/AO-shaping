"""Policy shared by the SLM panel / optical-path probes.

Two things live here, both of them "what the probe family decided once, and
must not decide twice":

1. **Fixed-budget acquisition** — ``beam_extent`` and ``panel_locate`` each
   measured the spot displacement / extent as an energy ratio between a flat
   reference and a tilted panel state, and each carried its own copy of the
   same fixed-frame budget (``_PROBE_DISCARD`` / ``_PROBE_FRAMES`` /
   ``_probe_policy``). The budget is one bench measurement, so it lives here
   once.

   ``PROBE_FRAMES`` was measured on the 2026-10-24 run: 7 kept frames give
   the per-pixel median enough samples to reject one corrupted frame (peak
   sigma 1.4%). ``--frames`` may raise the budget, never lower it.

2. **Four-state optical-path verdict** — :func:`decide`, the judgement that
   ``scripts/slm/probe_optical_path_validity.py`` prints and that the GUI
   「光路快速探测」button shows. It started inside that script, where it was
   reachable only by loading the script file; the GUI needs the very same
   judgement, and a second copy would let the CLI and the button disagree
   about the same readings. Thresholds and their measured basis therefore
   moved here unchanged (see :data:`WIDTH_RESOLUTION_MAX` for the 2026-10-22
   low-SNR evidence and :data:`STATE_WEAK_PATH` for the 2026-10-29 weak-path
   evidence), and the script imports them back so its behaviour is identical.

No GUI dependency: this module is importable from ``gui/`` without pulling in
Streamlit, a camera, or an SLM.
"""

from __future__ import annotations

from typing import Any

from ao_shaping.tools.kernels.bench_kernels import ProbePolicy
from ao_shaping.utils.image.frame_analysis import DEFAULT_MIN_PEAK

__all__ = [
    "PEAK_RELATIVE_FLOOR",
    "PEAK_RESOLUTION_MAX",
    "PROBE_DISCARD",
    "PROBE_FRAMES",
    "RESPONSE_SIGMAS",
    "STATE_LOW_SNR",
    "STATE_NO_RESPONSE",
    "STATE_OK",
    "STATE_WEAK_PATH",
    "VERDICT_STATES",
    "WIDTH_RELATIVE_FLOOR",
    "WIDTH_RESOLUTION_MAX",
    "decide",
    "probe_policy",
]

#: Mounted frames after the write (LCOS flip) before the kept ones.
PROBE_DISCARD = 2
#: Measured kept-frame budget -- see the module docstring. ``--frames`` may
#: raise it, never lower it.
PROBE_FRAMES = 7


def probe_policy(frames: int) -> ProbePolicy:
    """Fixed-budget acquisition for an energy-ratio measurement."""
    return ProbePolicy(
        mode="frames",
        n_frames=max(int(frames), PROBE_FRAMES),
        n_discard=PROBE_DISCARD,
        combine="median",
    )


# ---------------------------------------------------------------------------
# 四态判定: 阈值 + ``decide``. 从 ``scripts/slm/probe_optical_path_validity.py``
# 原样搬来 (中文注释里的实测依据一并保留), 脚本再 import 回去 —— 判定只有这一份。
# ---------------------------------------------------------------------------

#: 离焦响应要算「显著」, 需要超过**合并标准误**的多少倍 sigma。
#:
#: 用 SEM (标准误) 而不是逐帧标准差: 比较的是两个**均值*, N 帧的均值比单帧准
#: sqrt(N) 倍。
RESPONSE_SIGMAS = 3.0
#: 相对变化的**物理下限**: 离焦 ±0.5 rad (Noll 4, 幅度 1 时孔径边缘 ±1.73 rad)
#: 必然把远场光斑的宽度改变 ≥20%、把峰值改变 ≥25%。低于这个量级的变化在物理上
#: 不该发生 —— 这条是「显著」的兜底判据 (只在测得准时才允许, 见
#: ``analyse_defocus_response``)。
WIDTH_RELATIVE_FLOOR = 0.20
PEAK_RELATIVE_FLOOR = 0.25
#: 「测得准」的分辨率上限: 宽度要能分辨 25% 的变化, 峰值要能分辨 50% 的变化。
#: 分辨率比这还差 ⇒ 就算离焦真的在起作用, 我们也**看不出来** ⇒ 信噪比不足,
#: 而不是「面板没调制」。
#:
#: 取值依据: 几何 sweep 拟合的是 ``w**2(c)``, 宽度读数的相对误差会**平方**地
#: 进到拟合里 —— 宽度 25% 的分辨率对应 w² 约 50% 的不确定度, 再叠上逐点噪声,
#: 曲率就拟不出来了。这正是 2026-10-22 那次 campaign 报「宽度不响应」时
#: 实际处的位置 (曝光 1-10 ms, 峰值 24-213 counts)。
WIDTH_RESOLUTION_MAX = 0.25
PEAK_RESOLUTION_MAX = 0.50
#: 判定状态的取值。
STATE_OK = "ok"
STATE_LOW_SNR = "low_snr"
STATE_NO_RESPONSE = "no_response"
#: 光斑可见、离焦也响应, 但光通量远低于本系统历史水平 ⇒ 光路被部分遮挡/衰减。
STATE_WEAK_PATH = "weak_path"
#: 四态的全集 (消费方用它校验读到的状态名, 不用猜字符串)。
VERDICT_STATES = (STATE_OK, STATE_LOW_SNR, STATE_WEAK_PATH, STATE_NO_RESPONSE)


def decide(
    ladder_summary: dict[str, Any],
    response: dict[str, Any],
    *,
    throughput: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """四态判定 + 依据 + 建议。

    判定顺序是有意的: 先看「有没有信号」, 再看「响应测不准还是根本没响应」,
    最后才看「响应正常但光通量远低于本系统历史水平」—— 前两态说的是「测不出来」,
    ``weak_path`` 说的是「测得出来, 但被测的已经不是原来那束光」。
    """
    reasons: list[str] = []
    advice: list[str] = []
    usable = ladder_summary["usable_window_ms"]
    best_peak = float(ladder_summary["best_usable_peak_counts"])
    noise = ladder_summary["noise_floor"]
    recommended = ladder_summary.get("recommended_exposure")

    if not usable:
        state = STATE_LOW_SNR
        reasons.append(
            "曝光阶梯里没有任何一档把光斑抬到噪声底之上 "
            f"(噪声底 {noise['threshold_counts']:.1f} counts @ "
            f"{noise['exposure_ms']:g} ms, 阶梯最高峰值 "
            f"{ladder_summary['max_ladder_peak_counts']:.1f} counts)"
        )
        advice.append(
            "把曝光阶梯整体抬高 (例如 --exposures 0.2,1,5,20,50,100)，或检查光源与光路"
        )
    else:
        reasons.append(
            f"可用曝光窗口 {', '.join(f'{v:g}' for v in usable)} ms "
            f"(峰值高于噪声底 {noise['threshold_counts']:.1f} counts 且未饱和)"
        )
        if recommended is not None:
            reasons.append(
                f"建议曝光 {recommended['exposure_ms']:g} ms (峰值 "
                f"{recommended['peak_counts']:.0f} counts"
                + ("" if recommended["in_target_band"] else ", 落在目标区间外")
                + ")"
            )

    responds = bool(response["responds"])
    width_resolution = float(response["width_resolution"])
    peak_resolution = float(response["peak_resolution"])
    precision_ok = bool(response["precision_ok"])
    if usable:
        if responds:
            state = STATE_OK if best_peak >= DEFAULT_MIN_PEAK else STATE_LOW_SNR
            reasons.append(
                "离焦有显著响应: 宽度最大变化 "
                f"{response['width_signal_px']:.2f} px "
                f"({response['width_relative_change']:.1%}, 出现在离焦 "
                f"{response['width_signal_at_rad']:+g} rad), 峰值最大变化 "
                f"{response['peak_signal_counts']:.1f} counts "
                f"({response['peak_relative_change']:.1%}, 出现在离焦 "
                f"{response['peak_signal_at_rad']:+g} rad); 判据 = "
                f"{RESPONSE_SIGMAS:g}×合并标准误 "
                f"(σ_FWHM={response['sigma_fwhm_px']:.2f} px, "
                f"σ_peak={response['sigma_peak_counts']:.1f} counts) "
                f"或相对变化下限 ({WIDTH_RELATIVE_FLOOR:.0%} / "
                f"{PEAK_RELATIVE_FLOOR:.0%})"
            )
        elif not precision_ok:
            state = STATE_LOW_SNR
            reasons.append(
                "离焦响应淹没在噪声里: 宽度只能分辨 "
                f"{width_resolution:.1%} 的变化 (上限 {WIDTH_RESOLUTION_MAX:.0%}), "
                f"峰值只能分辨 {peak_resolution:.1%} (上限 {PEAK_RESOLUTION_MAX:.0%}); "
                f"实测宽度只变了 {response['width_signal_px']:.2f} px "
                f"({response['width_relative_change']:.1%}, "
                f"σ={response['sigma_fwhm_px']:.2f} px), 峰值只变了 "
                f"{response['peak_signal_counts']:.1f} counts "
                f"({response['peak_relative_change']:.1%}, "
                f"σ={response['sigma_peak_counts']:.1f})"
            )
            advice.append(
                "这不是「光路坏」而是「测不准」: 把曝光提到峰值 100-200 counts "
                "再复测, 或在采集侧多帧平均 (曝光阶梯里挑更亮的那一档); "
                "几何 sweep 的宽度响应在当前信噪比下必然淹没"
            )
        else:
            state = STATE_NO_RESPONSE
            reasons.append(
                "光斑清晰可见 (峰值 "
                f"{response['baseline_peak_counts']:.0f} counts) 且测得准 "
                f"(宽度分辨率 {width_resolution:.1%} ≤ {WIDTH_RESOLUTION_MAX:.0%}, "
                f"峰值分辨率 {peak_resolution:.1%} ≤ {PEAK_RESOLUTION_MAX:.0%}) "
                "但离焦毫无响应: 宽度只变了 "
                f"{response['width_signal_px']:.2f} px "
                f"({response['width_relative_change']:.1%}), 峰值只变了 "
                f"{response['peak_signal_counts']:.1f} counts "
                f"({response['peak_relative_change']:.1%})"
            )
            advice.append(
                "排查顺序: 1) python -m ao_shaping.tools.slm.panel.diagnose 的三步链 "
                "(冻结 / 调制 / 线性) —— 面板没调制时离焦不会改变任何读数; "
                "2) 确认离焦真的加到了光束照到的那块面板上 (瞳面位置); "
                "3) 确认相机仍在焦面附近 (离焦幅度是否已把焦面推出视场)"
            )
    else:
        state = STATE_LOW_SNR

    # 光通量判定: 只覆盖 ``ok`` —— ``low_snr`` / ``no_response`` 已经指出了更根本的
    # 问题 (测不出来 / 没有调制), 再叠加一层「光也弱」只会把排查顺序搅乱。
    if state == STATE_OK and throughput and throughput.get("ratio") is not None:
        ratio = float(throughput["ratio"])
        if ratio < float(throughput["min_ratio"]):
            state = STATE_WEAK_PATH
            reasons.append(
                f"光通量只有参考的 {ratio:.0%}: 峰值 {throughput['counts_per_ms']:.2f} "
                f"counts/ms, 参考 {throughput['reference_counts_per_ms']:.2f} counts/ms "
                f"({throughput['reference_source']}, n={throughput['reference_n']})"
                f" ⇒ 衰减到约 1/{1.0 / max(ratio, 1e-6):.1f}"
            )
            advice.insert(
                0,
                "先查光路遮挡 / 滤镜 / 元件位置再继续: 被挡掉的是能量, 加大曝光只能把"
                "峰值抬回 100-200 counts 的目标区间, 挡掉的那部分回不来 —— "
                "把「加曝光后判 ok」当成已修好, 会把整段标定建在衰减后的光束上",
            )

    if state not in (STATE_OK, STATE_WEAK_PATH) and recommended is not None and usable:
        advice.append(
            f"下一次标定建议用曝光 {recommended['exposure_ms']:g} ms "
            f"(峰值 {recommended['peak_counts']:.0f} counts)"
        )

    return {
        "state": state,
        "reasons": reasons,
        "advice": advice,
        "exit_code": 0 if state == STATE_OK else 2,
    }
