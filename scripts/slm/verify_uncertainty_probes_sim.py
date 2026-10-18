"""sim 验收脚本: commit 080e0ac9 (issue #134) 的不确定性采样探针 A/B 对照。

在**同一 seed、同一预算**下跑两臂 model-in-loop, 只让"拒绝轮之后那几轮用哪种
探针"这一个变量不同:

* **arm A (blind)** -- ``uncertainty_on_rejection=False`` (历史路径): 拒绝后
  ``probe_count += escalate_probe_count``, 多出来的槽位用固定随机强相位填充。
* **arm B (R-86)** -- ``uncertainty_on_rejection=True``: 同样加同样的槽位, 但
  相位由 :class:`UncertaintyProbeSelector` 选出 (最大化远场预测方差)。

对照 #134 的四条验收:

1. 拒绝后下一轮 ``loss_after`` 回退更小 / 更快被 ``acceptance_verdict`` 接受;
2. 新探针满足约束集 (幅度带 / 带限 / 多样性), 种子不撞 train/holdout;
3. ``acceptance_verdict`` / trust region / ``max_rejection_streak`` 零改动 --
   本脚本只做运行期观测, 不替换任何一个被测函数 (三个包装器都只记录再转发);
4. 与 R-82 共用同一选择器模块 (本脚本只 import, 不复制其度量)。

机制层面的直接测量: 在**两臂探针的并集**算出的同一个 Fisher 后验下, 把两臂
的探针都送进选择器的 ``predictive_variance`` 比大小。调用选择器自己的公共方法
而不是自己重写一遍 ``U(p)`` 公式 —— 这个仓库反复被"重复实现漂移"咬伤, 度量
必须单一真源。后验也必须是共用的: 各用各的后验自证等于没比。

用法::

    python scripts/slm/verify_uncertainty_probes_sim.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from ao_shaping.optimizer.wfless import slm_model_in_loop as mod  # noqa: E402
from ao_shaping.optimizer.wfless.slm_model_in_loop import (  # noqa: E402
    SlmModelInLoopConfig,
    optimize_slm_model_in_loop,
)

#: 拒绝场景的构造方式: 把 Step A 的 loss 容差收到浮点噪声量级, 于是**任何**
#: 非严格下降的轮次都会被 ``loss_worsened`` 拒掉。这是 config 里已有的旋钮,
#: 不是 monkeypatch —— 被测的 ``acceptance_verdict`` 必须是真函数, 否则
#: "拒绝轮探针选择有没有用" 就无从谈起。
_TIGHT_LOSS_DELTA = 1e-12

#: 已知的台架像差 (Noll 序 -> 弧度), 让 Step A 有真信号可拟合。
#: `_SimBench` 的文档说明: 完美瞳面上没有像差可找, 拟合纯噪声, 每轮给出不同的
#: 随机系数, 接受守卫会正确地全部拒收 —— 一个正确但空洞的测试。
_SIM_ABERRATION = {
    4: 0.45,    # defocus
    6: -0.30,   # astig x
    8: 0.22,    # coma x
    11: -0.18,  # spherical
    14: 0.12,   # secondary astig x
}

_BASE = dict(
    cam_type="sim",
    region=32,
    cam_size=128,
    far_field_padding=4,
    n_rounds=4,
    probe_count=2,
    probe_spread=4.0,
    step_a_iterations=30,
    holdout_probe_count=2,
    step_b_iterations=20,
    target_side=64,
    n_orders=4,
    dtype="float32",
    device="cpu",
    seed=20261016,
    holdout_on_sim=True,          # 让 holdout_correlation 真的被测到
    min_holdout_correlation=0.0,  # 记录但不因它 abort (sim 本来就不 abort)
    max_rejection_streak=5,
    escalate_probe_count=2,
    sim_aberration=dict(_SIM_ABERRATION),
    acceptance_loss_delta=_TIGHT_LOSS_DELTA,
)


class _Instrumented:
    """包住三个纯函数, 观测"这一轮到底用了哪些探针 / 哪些种子 / 哪个优化器"。

    用的是本仓库测试里同一套 ``monkeypatch.setattr`` 惯用法, 不引入新机制。
    三个包装器都只记录再原样转发, 不改行为, 也不替换被测的
    ``acceptance_verdict`` / trust region / ``max_rejection_streak``。

    轮次归属靠**调用顺序**还原: 循环里是 ``_make_optimizer`` ->
    ``_resolve_probe_seeds`` -> (``_select_uncertainty_probes``), 所以优化器先
    暂存, 等 ``_resolve_probe_seeds`` 报出轮号再绑定。
    """

    def __init__(self) -> None:
        # round_index -> {"probes": [...], "optimizer": obj}
        self.selector_calls: dict[int, dict] = {}
        self.seed_splits: list[dict] = []
        self._pending_opt: object | None = None
        self._round_index: int = -1
        self._round_opt: object | None = None
        self._orig_selector = mod._select_uncertainty_probes
        self._orig_seeds = mod._resolve_probe_seeds
        self._orig_make_opt = mod._make_optimizer

    def make_opt(self, config, coefficients, source_amplitude):  # noqa: ANN001
        opt = self._orig_make_opt(config, coefficients, source_amplitude)
        self._pending_opt = opt
        return opt

    def seeds(self, seed, index, probe_count, holdout_probe_count, hook=None):  # noqa: ANN001
        train, holdout = self._orig_seeds(
            seed, index, probe_count, holdout_probe_count, hook
        )
        self._round_index = int(index)
        self._round_opt = self._pending_opt
        self.seed_splits.append(
            {"round": int(index), "train": list(train), "holdout": list(holdout)}
        )
        return train, holdout

    def selector(self, optimizer, config, seeds, *, prev_probes=None):  # noqa: ANN001
        probes = self._orig_selector(
            optimizer, config, seeds, prev_probes=prev_probes
        )
        self.selector_calls[self._round_index] = {
            "probes": [np.asarray(p, dtype=np.float64) for p in probes],
            "optimizer": self._round_opt if self._round_opt is not None else optimizer,
            "u": self._u_under_same_ruler(
                self._round_opt if self._round_opt is not None else optimizer,
                config, seeds, prev_probes,
                [np.asarray(p, dtype=np.float64) for p in probes],
            ),
        }
        return probes

    @staticmethod
    def _u_under_same_ruler(optimizer, config, seeds, prev_probes, probes):  # noqa: ANN001
        """在**同一后验**下, 给不确定性探针与"同样种子的盲随机探针"打同一把尺。

        后验必须就是选择器自己用来上升的那一个 (``prev_probes``; 没有时退化为
        各向同性先验), 而不是事后拼一个并集 —— 换了尺子, 量的就不是这个机制
        优化的那个目标, 而是一个新的、无人验证过的量。
        """
        from ao_shaping.algorithm.signal_processing.uncertainty_probe_selector import (
            UncertaintyProbeSelector,
        )
        from ao_shaping.optimizer.wfless.slm_model_in_loop import _probe_phase

        spread = float(config.probe_spread)
        sel = UncertaintyProbeSelector(
            optimizer,
            seed=int(seeds[0]) if seeds else int(config.seed),
            n_samples=int(config.uncertainty_probe_samples),
            ridge=float(config.uncertainty_probe_ridge),
            spread_cap=spread,
            spread_floor=0.75 * spread,
            diversity_weight=float(config.uncertainty_probe_diversity),
        )
        if prev_probes:
            sel.update_covariance(
                [np.asarray(p, dtype=np.float64) for p in prev_probes]
            )
        blind = [
            _probe_phase(int(config.region), spread, s,
                         smooth=config.cam_type != "sim")
            for s in seeds
        ]

        def score(batch):  # noqa: ANN001
            vals = []
            for probe in batch:
                leaf = torch.from_numpy(
                    np.ascontiguousarray(probe, dtype=np.float64)
                )
                leaf.requires_grad_(True)
                with torch.no_grad():
                    vals.append(float(sel.predictive_variance(leaf)))
            return float(np.mean(vals)) if vals else float("nan")

        # ---- 幅度对照 (关键) ----------------------------------------------
        # U(p) 随探针幅度单调上升: 幅度越大, 远场对系数扰动越敏感, 方差就越大。
        # 而选择器把不确定性探针**推到幅度下限** (std=3.0), 盲随机探针在名义
        # 幅度 (std=4.0)。于是直接比两者的 U 是在比幅度, 不是比"信息量",
        # ratio 会系统性地偏向盲随机。必须先把两者归一到**同一幅度**再比。
        target = float(np.mean([np.std(p) for p in probes]))

        def at_amplitude(batch, std: float) -> list[np.ndarray]:  # noqa: ANN001
            out = []
            for p in batch:
                s = float(np.std(p))
                out.append(p * (std / s) if s > 1e-12 else p)
            return out

        unc_norm = at_amplitude(probes, target)
        blind_norm = at_amplitude(blind, target)

        # ---- 带限对照 (第二个混杂因子) --------------------------------------
        # 选择器的探针**总是**过一遍高斯低通 (sigma=bandlimit_sigma), 而循环里
        # 的盲随机探针在 ``cam_type=="sim"`` 下是 ``smooth=False`` 的**全带**
        # 随机相位。全带瞳面相位产生对比度更高的远场散斑, 对系数扰动更敏感,
        # 于是 U 天然更高。所以只对齐幅度仍然是在比"带宽", 不是比"梯度上升
        # 找到了什么"。真正的对照是: **同一个被约束家族里的随机抽样** ——
        # 把盲探针也低通到同样的 sigma、再对齐同样的幅度, 此时剩下的差异才是
        # 梯度上升的功劳。
        from scipy import ndimage as _ndi

        smooth_blind = [
            _ndi.gaussian_filter(p, sigma=1.5) for p in blind_norm
        ]
        # 低通后 std 会掉, 再拉回同一幅度
        smooth_blind = at_amplitude(smooth_blind, target)

        return {
            "posterior": "prev_probes" if prev_probes else "isotropic_prior",
            "std_uncertainty": [float(np.std(p)) for p in probes],
            "std_blind": [float(np.std(p)) for p in blind],
            "matched_std": target,
            # 原始 (含幅度 + 带宽差) —— 保留以便看到 confound 有多大
            "u_uncertainty": score(probes),
            "u_blind_same_seeds": score(blind),
            # 幅度归一
            "u_uncertainty_matched": score(unc_norm),
            "u_blind_matched": score(blind_norm),
            # 幅度 + 带宽都归一 —— 唯一可用于结论的那一对
            "u_blind_bandmatched": score(smooth_blind),
        }

    def install(self) -> None:
        mod._make_optimizer = self.make_opt
        mod._resolve_probe_seeds = self.seeds
        mod._select_uncertainty_probes = self.selector

    def restore(self) -> None:
        mod._make_optimizer = self._orig_make_opt
        mod._resolve_probe_seeds = self._orig_seeds
        mod._select_uncertainty_probes = self._orig_selector


def _hf_energy_fraction(phase: np.ndarray) -> float:
    """**低频**能量占比: 谱中心 4x4 / 全谱。数值越大 = 相位越平滑。

    命名历史: 早先写作 ``_hf_energy_fraction`` 并标注"高频占比", 但它取的是
    ``f[lo:2*lo, lo:2*lo]`` 这个**中心**块, 那是低频。已用纯余弦标定
    (1 cycle -> 1.0, 4 cycles -> 0.0)。函数名保留以免牵动调用点, 语义与打印
    标签均已改正 —— 早先据此写下的"带限约束生效"那句结论方向是反的。
    """
    f = np.abs(np.fft.fftshift(np.fft.fft2(phase - phase.mean()))) ** 2
    total = float(f.sum())
    if total <= 0:
        return 0.0
    c = f.shape[0] // 2
    return float(f[c - 2:c + 2, c - 2:c + 2].sum()) / total


def _pair_cos(probes: list[np.ndarray]) -> list[float]:
    out = []
    for a in range(len(probes)):
        for b in range(a + 1, len(probes)):
            fa = probes[a].ravel() - probes[a].mean()
            fb = probes[b].ravel() - probes[b].mean()
            out.append(
                float(fa @ fb / ((np.linalg.norm(fa) * np.linalg.norm(fb)) + 1e-12))
            )
    return out


def run_arm(name: str, *, uncertainty_on_rejection: bool) -> dict:
    """跑一臂, 返回逐轮记录 + 该轮下发的探针 (不评分, 评分留给共后验的第二遍)。"""
    inst = _Instrumented()
    inst.install()
    try:
        result = optimize_slm_model_in_loop(
            SlmModelInLoopConfig(
                **{
                    **_BASE,
                    "uncertainty_on_rejection": uncertainty_on_rejection,
                    "uncertainty_probes": False,  # 只对照拒绝轮这条路径 (R-86)
                }
            )
        )
    finally:
        inst.restore()

    rounds = [r for r in result.recorder.history if r.get("stage") == "round"]
    per_round = []
    for i, row in enumerate(rounds):
        call = inst.selector_calls.get(i)
        if call is not None:
            probes = call["probes"]
            source = "uncertainty"
        else:
            # blind 臂没有 selector 调用; 按循环里同样的算术 + 同样的 smooth
            # 开关重建它下发的探针, 否则比的就不是同一批硬件指令了。
            from ao_shaping.optimizer.wfless.slm_model_in_loop import _probe_phase

            train_seeds = inst.seed_splits[i]["train"]
            smooth = _BASE["cam_type"] != "sim"
            probes = [
                _probe_phase(int(_BASE["region"]), float(_BASE["probe_spread"]),
                             s, smooth=smooth)
                for s in train_seeds
            ]
            source = "blind_fixed"
        per_round.append({
            "round": int(row["round"]),
            "probe_source": source,
            "probes": probes,
            "accepted": bool(row["accepted"]),
            "rejection_streak": int(row["rejection_streak"]),
            "probe_count": int(row["probe_count"]),
            "loss_before": float(row["loss_before"]),
            "loss_after": float(row["loss_after"]),
            "loss_delta": float(row["loss_after"] - row["loss_before"]),
            "holdout_correlation": float(row.get("holdout_correlation", float("nan"))),
            "probe_std": [float(np.std(p)) for p in probes],
            "probe_hf_fraction": [_hf_energy_fraction(p) for p in probes],
            "probe_pair_cos": _pair_cos(probes),
        })

    return {
        "arm": name,
        "uncertainty_on_rejection": uncertainty_on_rejection,
        "status": str(result.status),
        "best_score": float(result.best_score),
        "flat_score": float(result.flat_score),
        "rounds": per_round,
        "accepted_rounds": int(sum(r["accepted"] for r in per_round)),
        "seed_splits": inst.seed_splits,
        "selector_calls": len(inst.selector_calls),
        # 机制测量的原始记录 (含 U(p)) 单独挂一份: ``selector_calls`` 里存着
        # 优化器对象, 不能直接进 json, 也不该被 _jsonable 顺手抹掉。
        "selector_calls_detail": [
            {
                "round": int(i),
                "u": call["u"],
                "probe_std": [float(np.std(p)) for p in call["probes"]],
            }
            for i, call in sorted(inst.selector_calls.items())
        ],
    }


def _collect_u_scores(arms: list[dict]) -> list[dict]:
    """汇总包装器内测到的 ``U(p)``: 不确定性探针 vs 同种子的盲随机探针。

    尺子是选择器自己那把 (见 :meth:`_Instrumented._u_under_same_ruler`), 所以
    这个比值回答的是一个明确的问题: **同一后验下, 梯度上升选出的探针是否真的
    比同样种子的盲随机探针更"不确定"?** 若不是, 该机制在优化它宣称的东西。
    """
    rows = []
    for arm in arms:
        by_round = {c["round"]: c for c in arm["selector_calls_detail"]}
        for i, r in enumerate(arm["rounds"]):
            call = by_round.get(i)
            if call is None or call.get("u") is None:
                continue
            u = call["u"]
            rows.append({
                "round": r["round"],
                "posterior": u["posterior"],
                "std_uncertainty": u["std_uncertainty"],
                "std_blind": u["std_blind"],
                "matched_std": u["matched_std"],
                "u_uncertainty_raw": u["u_uncertainty"],
                "u_blind_raw": u["u_blind_same_seeds"],
                "ratio_raw": (
                    float(u["u_uncertainty"] / u["u_blind_same_seeds"])
                    if u["u_blind_same_seeds"] > 0 else float("inf")
                ),
                "u_uncertainty_matched": u["u_uncertainty_matched"],
                "u_blind_matched": u["u_blind_matched"],
                "ratio_matched": (
                    float(u["u_uncertainty_matched"] / u["u_blind_matched"])
                    if u["u_blind_matched"] > 0 else float("inf")
                ),
                "u_blind_bandmatched": u["u_blind_bandmatched"],
                "ratio_bandmatched": (
                    float(u["u_uncertainty_matched"] / u["u_blind_bandmatched"])
                    if u["u_blind_bandmatched"] > 0 else float("inf")
                ),
                "accepted": r["accepted"],
                "loss_delta": r["loss_delta"],
            })
    return rows


def _sign_flip_p(deltas: np.ndarray) -> float:
    """精确 sign-flip 置换检验 (双侧): "R-86 平均更好" 是不是运气。

    不假设正态 —— 8 个种子这么小, t 检验的前提根本不成立。直接枚举 2**n 个
    符号翻转, 没有任何分布假设。与本仓库其它 A/B 同一套想法 (见
    ``src/ml/zernike/eval_stats.py``); 这里按同一公式就地写一份最短版, 因为
    离线验收脚本不该把可选依赖 ``src/ml`` 拉进导入路径。计数边界用 ``>=``,
    故最小可达 p = 2/2**n。
    """
    deltas = np.asarray(deltas, dtype=np.float64)
    deltas = deltas[np.isfinite(deltas)]
    n = int(deltas.size)
    if n == 0:
        return 1.0
    observed = abs(float(deltas.mean()))
    signs = np.array(
        [[1.0 if (mask >> k) & 1 else -1.0 for k in range(n)]
         for mask in range(2 ** n)],
        dtype=np.float64,
    )
    stats = np.abs((signs @ deltas) / n)
    return float(np.sum(stats >= observed - 1e-15) / (2 ** n))


class _ForceBandlimit:
    """把选择器的 ``bandlimit_sigma`` 钉到给定值 (0 = 关掉带限)。

    选择器是在 ``_select_uncertainty_probes`` **函数体内部** import 的, 所以补丁
    要打在模块属性上才会生效。默认不动它 —— 只在 ``--selector-bandlimit`` 明确
    指定时才替换, 因为线上默认是 1.5, 而这一项恰恰是本轮要隔离的变量。
    """

    def __init__(self, sigma: float) -> None:
        self._sigma = float(sigma)
        self._orig: object | None = None

    def install(self) -> None:
        from ao_shaping.algorithm.signal_processing import (
            uncertainty_probe_selector as ups,
        )

        self._orig = ups.UncertaintyProbeSelector
        orig = self._orig
        # 闭包变量, **不是** ``self._sigma`` —— 在 ``_Pinned.__init__`` 里的
        # ``self`` 是新建的选择器实例, 拿不到外层包装器的属性。
        sigma = self._sigma

        class _Pinned(orig):  # type: ignore[misc, valid-type]
            def __init__(self, *a, **kw):  # noqa: ANN002, ANN003
                kw["bandlimit_sigma"] = sigma
                super().__init__(*a, **kw)

        ups.UncertaintyProbeSelector = _Pinned

    def restore(self) -> None:
        if self._orig is None:
            return
        from ao_shaping.algorithm.signal_processing import (
            uncertainty_probe_selector as ups,
        )

        ups.UncertaintyProbeSelector = self._orig
        self._orig = None


def _jsonable(obj):  # noqa: ANN001
    """去掉 ndarray / 非序列化对象, 供 json.dump。"""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items() if k != "probes"
                and k != "round_optimizers"}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return f"<ndarray {obj.shape}>"
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--selector-bandlimit", type=float, default=None,
        help=("钉住选择器的 bandlimit_sigma (0 = 关掉带限)。不指定则用线上"
              "默认 1.5。这一项是本轮要隔离的变量。"),
    )
    args = ap.parse_args()

    band = _ForceBandlimit(args.selector_bandlimit) if (
        args.selector_bandlimit is not None
    ) else None
    if band is not None:
        band.install()
        print(f"[bandlimit] 选择器 bandlimit_sigma 钉为 {args.selector_bandlimit}")
    try:
        return _main_inner(args)
    finally:
        if band is not None:
            band.restore()


def _main_inner(args) -> int:  # noqa: ANN001
    out_dir = ROOT / "data" / "debug" / "uncertainty_probes_sim"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    save_dir = out_dir / stamp
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("uncertainty-probe sim A/B  (commit 080e0ac9, issue #134)")
    print("=" * 78)
    print(f"base: seed={_BASE['seed']} rounds={_BASE['n_rounds']} "
          f"probe_count={_BASE['probe_count']} escalate={_BASE['escalate_probe_count']} "
          f"acceptance_loss_delta={_TIGHT_LOSS_DELTA:g}")

    arms = []
    for name, toggle in (("blind_fixed", False), ("uncertainty_r86", True)):
        print(f"\n--- arm {name} (uncertainty_on_rejection={toggle}) ---")
        arm = run_arm(name, uncertainty_on_rejection=toggle)
        arms.append(arm)
        for r in arm["rounds"]:
            print(
                f"  rd{r['round']} src={r['probe_source']:<12} "
                f"n={r['probe_count']} acc={str(r['accepted']):<5} "
                f"loss {r['loss_before']:.5f}->{r['loss_after']:.5f} "
                f"(d={r['loss_delta']:+.5f}) "
                f"holdout_r={r['holdout_correlation']:.4f} "
                f"std={min(r['probe_std']):.3f}-{max(r['probe_std']):.3f} "
                f"hf={max(r['probe_hf_fraction']):.4f} "
                f"maxcos={max(r['probe_pair_cos'] or [0.0]):+.3f}"
            )
        print(f"  status={arm['status']} accepted={arm['accepted_rounds']}"
              f"/{len(arm['rounds'])} selector_calls={arm['selector_calls']}")

    # 机制测量: 共后验下的 U(p)。优化器取 R-86 臂在那一轮的真实优化器
    # (只有该臂在那些轮真的走过选择器), 看不到就用 None。
    u_scores = _collect_u_scores(arms)
    print("\n" + "=" * 78)
    print("机制测量: 同一后验下, 不确定性探针 vs 同种子盲随机探针的 U(p)")
    print("  结论只看 ratio_matched (幅度归一); ratio_raw 混着幅度差, 不可用")
    print("=" * 78)
    if not u_scores:
        print("  (无 R-86 轮次: 本次两臂都没有触发不确定采样)")
    for row in u_scores:
        print(f"  rd{row['round']} [{row['posterior']}] "
              f"std unc={np.mean(row['std_uncertainty']):.2f} "
              f"blind={np.mean(row['std_blind']):.2f} "
              f"-> matched@{row['matched_std']:.2f}")
        print(f"      raw    : U_unc={row['u_uncertainty_raw']:.4e} "
              f"U_blind={row['u_blind_raw']:.4e} "
              f"ratio_raw={row['ratio_raw']:.3f} (幅度混杂, 不作结论)")
        print(f"      matched: U_unc={row['u_uncertainty_matched']:.4e} "
              f"U_blind={row['u_blind_matched']:.4e} "
              f"ratio={row['ratio_matched']:.3f}")
        print(f"      bandmat: U_blind={row['u_blind_bandmatched']:.4e} "
              f"ratio_bandmatched={row['ratio_bandmatched']:.3f} "
              f"{'OK >1' if row['ratio_bandmatched'] > 1.0 else 'NOT >1'}  "
              f"(acc={row['accepted']}, dloss={row['loss_delta']:+.5f})")

    # --- 验收 2: 种子不变量 -----------------------------------------------
    a_seeds = [(s["round"], tuple(s["train"]), tuple(s["holdout"]))
               for s in arms[0]["seed_splits"]]
    b_seeds = [(s["round"], tuple(s["train"]), tuple(s["holdout"]))
               for s in arms[1]["seed_splits"]]
    seeds_identical = a_seeds == b_seeds
    collisions = []
    for a in arms:
        for s in a["seed_splits"]:
            overlap = set(s["train"]) & set(s["holdout"])
            if overlap:
                collisions.append({"arm": a["arm"], "round": s["round"],
                                   "overlap": sorted(overlap)})
    print(f"\n种子不变量: 两臂种子逐轮一致={seeds_identical}; "
          f"train/holdout 碰撞={collisions or '无'}")

    # --- 验收 2: 约束集 ----------------------------------------------------
    floor, cap = 0.75 * float(_BASE["probe_spread"]), float(_BASE["probe_spread"])
    blind_hf, unc_hf = [], []
    blind_cos, unc_cos = [], []
    std_violations = []
    for a in arms:
        for r in a["rounds"]:
            for s in r["probe_std"]:
                if s > cap + 1e-6 or s < floor - 1e-6:
                    std_violations.append(
                        {"arm": a["arm"], "round": r["round"], "std": s})
            (blind_hf if a["arm"] == "blind_fixed" else unc_hf).extend(
                r["probe_hf_fraction"])
            (blind_cos if a["arm"] == "blind_fixed" else unc_cos).extend(
                r["probe_pair_cos"])
    print(f"幅度带 [{floor:.2f}, {cap:.2f}] 越界={std_violations or '无'}")
    print(f"低频占比(越大越平滑): blind mean={np.mean(blind_hf):.4f} "
          f"max={np.max(blind_hf):.4f} | "
          f"uncertainty mean={np.mean(unc_hf):.4f} max={np.max(unc_hf):.4f}")
    print(f"探针两两余弦: blind max={np.max(blind_cos):+.4f} | "
          f"uncertainty max={np.max(unc_cos):+.4f} (越接近 0 越不塌缩)")

    # --- 多种子扫描: 单 seed 的 A/B 没有统计功效 ---------------------------
    # 上面那组是把机制看清楚用的 (逐轮日志最全)。#134 的验收要的是"回退幅度更小
    # 或更快被接受"这种**比较性**结论, 一个 seed 只是一次抽样, 差值完全可能落在
    # 噪声里。所以另跑一组多种子配对扫描: 同一个 seed 同时喂两臂 (配对), 差值在
    # 配对内抵消难度的差异。
    print("\n" + "=" * 78)
    print("多种子配对扫描 (每个 seed 同时跑两臂)")
    print("=" * 78)
    sweep_seeds = [20261016, 7, 11, 42, 1234, 20260101, 555, 90210]
    sweep: list[dict] = []
    for sd in sweep_seeds:
        # 直接改模块级 ``_BASE`` 里的条目即可, 不需要 (也不能) 写 ``global``:
        # dict 是可变对象, ``_BASE["seed"] = ...`` 就地生效, 而在本函数里先读
        # ``_BASE`` 再写 ``global _BASE`` 会被解释器判为 SyntaxError。
        keep = dict(_BASE)
        try:
            _BASE["seed"] = int(sd)
            row: dict = {"seed": int(sd)}
            for name, toggle in (("blind", False), ("r86", True)):
                a = run_arm(f"{name}_s{sd}", uncertainty_on_rejection=toggle)
                pr = [r for r in a["rounds"] if r["rejection_streak"] > 0]
                row[name] = {
                    "status": a["status"],
                    "accepted": a["accepted_rounds"],
                    "rounds": len(a["rounds"]),
                    "post_rejection_rounds": len(pr),
                    "mean_dloss_post": (
                        float(np.mean([r["loss_delta"] for r in pr]))
                        if pr else None),
                    "mean_holdout_post": (
                        float(np.nanmean([r["holdout_correlation"] for r in pr]))
                        if pr else None),
                    "selector_calls": a["selector_calls"],
                }
            if row["blind"]["mean_dloss_post"] is not None and \
               row["r86"]["mean_dloss_post"] is not None:
                row["delta_dloss"] = (
                    row["r86"]["mean_dloss_post"] - row["blind"]["mean_dloss_post"]
                )
            sweep.append(row)
        finally:
            _BASE.clear()
            _BASE.update(keep)
        b, r = row["blind"], row["r86"]
        dd = row.get("delta_dloss")

        def fmt(v):  # noqa: ANN001
            return "n/a" if v is None else f"{v:+.6f}"

        print(f"  seed {sd:>8}: blind acc={b['accepted']}/{b['rounds']} "
              f"dloss={fmt(b['mean_dloss_post'])} "
              f"| r86 acc={r['accepted']}/{r['rounds']} "
              f"dloss={fmt(r['mean_dloss_post'])} "
              f"calls={r['selector_calls']}"
              f"{'' if dd is None else f'  paired ddloss={dd:+.6f}'}")

    usable = [s for s in sweep if s.get("delta_dloss") is not None]
    if usable:
        deltas = np.array([s["delta_dloss"] for s in usable], dtype=np.float64)
        neg = int(np.sum(deltas < 0))   # 负 = R-86 回退更小 = 更好
        print(f"\n  可配对的 seed: {len(deltas)}/{len(sweep)}")
        print(f"  paired ddloss (R-86 - blind): mean={deltas.mean():+.6f} "
              f"median={np.median(deltas):+.6f} sd={deltas.std(ddof=1) if len(deltas) > 1 else 0:.6f}")
        print(f"  R-86 回退更小的 seed: {neg}/{len(deltas)}")
        print(f"  精确 sign-flip 置换检验 p = "
              f"{_sign_flip_p(deltas):.4f} (双侧, {len(deltas)} 个样本, "
              f"最小可达 p = {2 / 2 ** len(deltas):.4f})")

    def _post_rejection(arm: dict) -> list[dict]:
        return [r for r in arm["rounds"] if r["rejection_streak"] > 0]

    post = {
        a["arm"]: {
            "rounds": len(_post_rejection(a)),
            "mean_loss_delta": (
                float(np.mean([r["loss_delta"] for r in _post_rejection(a)]))
                if _post_rejection(a) else None),
            "mean_holdout_correlation": (
                float(np.nanmean([r["holdout_correlation"]
                                   for r in _post_rejection(a)]))
                if _post_rejection(a) else None),
            "accepted_rounds": a["accepted_rounds"],
        }
        for a in arms
    }
    print("\n" + "=" * 78)
    print("A/B 汇总 (仅统计被拒轮之后的轮次, 即 R-86 实际生效的轮次)")
    print("=" * 78)
    print(f"{'arm':<18}{'rounds':>7}{'mean dloss':>14}{'mean holdout_r':>18}")
    for a in arms:
        s = post[a["arm"]]
        d, h = s["mean_loss_delta"], s["mean_holdout_correlation"]
        print(f"{a['arm']:<18}{s['rounds']:>7}"
              f"{('n/a' if d is None else f'{d:+.6f}'):>14}"
              f"{('n/a' if h is None else f'{h:.4f}'):>18}")

    summary = {
        "commit": "080e0ac9de2d7d453676d174ae4a1845acb77ee8",
        "issue": "#134 (R-86)",
        "timestamp_utc": stamp,
        "base_config": _BASE,
        "arms": arms,
        "post_rejection": post,
        "mechanism_u_scores": u_scores,
        "multiseed_sweep": sweep,
        "acceptance": {
            "seed_invariant_identical_across_arms": seeds_identical,
            "train_holdout_collisions": collisions,
            "spread_band": [floor, cap],
            "spread_violations": std_violations,
            "hf_fraction": {
                "blind_mean": float(np.mean(blind_hf)),
                "blind_max": float(np.max(blind_hf)),
                "uncertainty_mean": float(np.mean(unc_hf)),
                "uncertainty_max": float(np.max(unc_hf)),
            },
            "pair_cos_max": {
                "blind": float(np.max(blind_cos)),
                "uncertainty": float(np.max(unc_cos)),
            },
        },
    }
    (save_dir / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    np.savez_compressed(
        save_dir / "probes.npz",
        **{
            f"{a['arm']}_rd{r['round']}_p{k}": p
            for a in arms for r in a["rounds"] for k, p in enumerate(r["probes"])
        },
    )
    print(f"\n保存: {save_dir / 'summary.json'}")
    print(f"保存: {save_dir / 'probes.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())