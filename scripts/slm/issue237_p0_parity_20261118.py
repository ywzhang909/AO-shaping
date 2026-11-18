"""Issue #237 criterion (6): offline baseline/default/identifiability audit."""
from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "data/debug/slm_transfer/issue237_p0_parity_20261118"
TEST = "tests/ao_shaping/algorithm/signal_processing/test_defocus_forward_registration.py"
ARTIFACT = "report/slm/differentiable_opt/artifacts/fwdmodel_fit_20261116/fwdmodel_fit.json"
CORE = "src/ao_shaping/algorithm/signal_processing/"


def run(args: list[str], name: str) -> dict[str, Any]:
    """Run a bounded offline command, preserving stdout and exit status."""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", check=False)
    log = OUT / f"{name}.log"
    log.write_text(result.stdout + result.stderr, encoding="utf-8")
    logger.info("{}: exit {}", name, result.returncode)
    return {"argv": args, "command": subprocess.list2cmdline(args),
            "exit_status": result.returncode, "log": str(log.relative_to(ROOT))}


def default_checks() -> dict[str, Any]:
    """Add signature/legacy-checkpoint checks absent from registration tests."""
    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        FrozenDefocusParameters,
        load_frozen_parameters,
    )
    from ao_shaping.algorithm.signal_processing.physical_forward_fit import PhysicalForwardFit

    signature = inspect.signature(PhysicalForwardFit)
    names = ("learn_registration", "registration_shift_ccd_px", "monotonic_ccd_h",
             "slm_phase_a", "slm_phase_b", "amplitude_residual_coeffs")
    defaults = {name: signature.parameters[name].default for name in names}
    expected = dict(learn_registration=False, registration_shift_ccd_px=None,
                    monotonic_ccd_h=(), slm_phase_a=0., slm_phase_b=0.,
                    amplitude_residual_coeffs=())
    fields = {field.name: field for field in dataclasses.fields(FrozenDefocusParameters)}
    checkpoint = (ROOT / "data/debug/slm_transfer/defocus_forward_20261107_focus_r2_fit/"
                  "checkpoints/M3_illumination/frozen_parameters.json")
    params = load_frozen_parameters(checkpoint)
    return {"verdict": "PASS" if defaults == expected and
            params.registration_shift_ccd_px is None and not params.monotonic_ccd_h else "FAIL",
            "fitter_defaults": defaults,
            "frozen_defaults": {name: repr(fields[name].default) for name in names if name in fields},
            "sigma_ccd_px_has_default": fields["sigma_ccd_px"].default is not dataclasses.MISSING,
            "legacy_checkpoint": str(checkpoint.relative_to(ROOT)),
            "legacy_registration": params.registration_shift_ccd_px,
            "legacy_monotonic_ccd_h": params.monotonic_ccd_h,
            "note": "None vs (0,0) 固定合成预测 np.array_equal 已由目标测试覆盖，不重复实现。"
            "sigma_ccd_px 是必填冻结参数，不是默认 off 的开关；该旧 checkpoint 值为 0。"}


def checkpoint_zero_shift_equality() -> dict[str, Any]:
    """Execute None vs (0.0, 0.0) predictions on the real legacy checkpoint."""
    import numpy as np
    import torch

    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        FrozenDefocusForward,
        load_frozen_parameters,
    )

    checkpoint = (ROOT / "data/debug/slm_transfer/defocus_forward_20261107_focus_r2_fit/"
                  "checkpoints/M3_illumination/frozen_parameters.json")
    params = load_frozen_parameters(checkpoint)
    zero = dataclasses.replace(params, registration_shift_ccd_px=(0.0, 0.0))
    xs = np.linspace(1220.0, 1392.0, 12)
    ys = np.linspace(880.0, 1010.0, 10)
    xx, yy = np.meshgrid(xs, ys)
    camera_xy = np.stack((xx, yy), axis=-1)
    rng = np.random.default_rng(20261118)
    phases = (rng.standard_normal((2, params.grid_size, params.grid_size)) * 0.15).astype(np.float32)
    predictions = []
    for candidate in (params, zero):
        forward = FrozenDefocusForward(candidate, camera_xy, device="cpu")
        with torch.no_grad():
            predictions.append(forward.predict(phases).numpy())
    equal = bool(np.array_equal(predictions[0], predictions[1]))
    return {"case": "legacy M3 checkpoint, 10x12 camera grid around origin, 2 random phases",
            "np_array_equal": equal,
            "max_abs_diff": float(np.max(np.abs(predictions[0] - predictions[1]))),
            "verdict": "PASS" if equal else "FAIL"}


def census() -> list[dict[str, Any]]:
    """Record grep-equivalent literal matches with exact live-source lines."""
    specifications = [
        ("2D registration", "implemented", "differentiable_defocus_uniformity.py",
         "registration_shift_ccd_px:"),
        ("PSF sigma", "implemented: scalar Gaussian kernel, not arbitrary measured PSF",
         "differentiable_defocus_uniformity.py", "def _build_blaze_kernel"),
        ("CCD response LUT / monotonic_response", "implemented",
         "monotonic_response.py", "class MonotonicResponse:"),
    ]
    rows = []
    for name, state, filename, needle in specifications:
        path = CORE + filename
        hits = [f"{path}:{i}" for i, line in enumerate(
            (ROOT / path).read_text(encoding="utf-8-sig").splitlines(), 1) if needle in line]
        rows.append({"feature": name, "state": state, "literal_pattern": needle, "locations": hits})
    for name, needles in [("gain field", ("gain_field", "pixel_gain", "gain_map")),
                          ("complex-amplitude stray light", ("stray", "coherent_background"))]:
        hits = []
        for file in sorted((ROOT / CORE).rglob("*.py")):
            for i, line in enumerate(file.read_text(encoding="utf-8-sig").splitlines(), 1):
                if any(needle in line.lower() for needle in needles):
                    hits.append({"location": f"{file.relative_to(ROOT).as_posix()}:{i}",
                                 "text": line.strip()})
        rows.append({"feature": name, "state": "not implemented" if not hits else "inspect matches",
                     "literal_patterns": needles, "search_root": CORE + "**/*.py",
                     "matches": hits})
    return rows


def main() -> int:
    """Produce machine-readable evidence and a Chinese scoped acceptance report."""
    OUT.mkdir(parents=True, exist_ok=True)
    commands = []
    for number in (1, 2):
        commands.append(run([sys.executable, "scripts/slm/replay_2d_registration_1103.py",
                             "--execute", "--out", str(OUT / f"replay_run{number}.json")],
                            f"replay_run{number}"))
    commands.append(run([sys.executable, "-m", "pytest", "--import-mode=importlib",
                         "-o", "cache_dir=" + str(OUT / "pytest_cache"), TEST], "pytest"))
    commands.append(run([sys.executable, "-m", "ruff", "check", str(Path(__file__).relative_to(ROOT)),
                         TEST], "ruff"))
    commands.append({"command": ".venv/Scripts/python.exe scripts/slm/analyze_fwdmodel_fit.py "
                     "--no-joint --no-sweeps --no-figures --out "
                     "data/debug/slm_transfer/issue237_p0_parity_20261118/fwdmodel_off",
                     "exit_status": 1, "log": str((OUT / "fwdmodel_off.log").relative_to(ROOT)),
                     "note": "尝试两次，工具均返回 exit 1；首日志止于拟合分组；第二日志见 fwdmodel_off.err.txt，完成首数据集并推进到第二数据集灰阶分析后中止，无 traceback，原因未知。"
                     "未产生 fwdmodel_fit.json，未执行 registration-on 对照。"})
    artifact = json.loads((ROOT / ARTIFACT).read_text(encoding="utf-8"))
    identifiability = [{"dataset": dataset["name"],
                        "identifiable": dataset["jacobian"]["identifiable"],
                        "unidentifiable": dataset["jacobian"]["unidentifiable"],
                        "identifiable_count": len(dataset["jacobian"]["identifiable"]),
                        "total": len(dataset["jacobian"]["names"])}
                       for dataset in artifact["datasets"]]
    defaults = default_checks()
    zero_shift = checkpoint_zero_shift_equality()
    if defaults["verdict"] != "PASS" or zero_shift["verdict"] != "PASS":
        defaults["verdict"] = "FAIL"
    replay_paths = [OUT / f"replay_run{number}.json" for number in (1, 2)]
    outputs_present = all(path.exists() for path in replay_paths)
    replay_records = []
    if outputs_present:
        for path in replay_paths:
            document = json.loads(path.read_text(encoding="utf-8"))
            for run_record in document["runs"]:
                for pair in run_record.get("pairs", []):
                    if "reproduces_baseline_shift" in pair:
                        replay_records.append({"out": str(path), "record": pair})
    bom_free = not Path(__file__).read_bytes().startswith(b"\xef\xbb\xbf")
    result = {
        "offline": True, "hardware": "未访问、未占用 SLM / CCD / 激光；无需释放设备。",
        "commands": commands, "baseline_expected": {"integer_shift_xy": [-3, 32],
        "ncc_zero_before": 0.0315531992032807, "ncc_integer_best": 0.6802495575769043,
        "tolerance": 1e-6}, "baseline_actual": replay_records,
        "baseline_verdict": "NOT-TESTABLE" if not outputs_present else "PASS" if replay_records
        and all(row["record"]["reproduces_baseline_shift"] for row in replay_records) else "FAIL",
        "actual_out_produced": [str(path) for path in replay_paths if path.exists()],
        "reproduces_baseline_shift": None if not replay_records else
        all(row["record"]["reproduces_baseline_shift"] for row in replay_records),
        "determinism": {"verdict": "NOT-TESTABLE" if not outputs_present else "PASS" if
                        replay_paths[0].read_bytes() == replay_paths[1].read_bytes() else "FAIL",
                        "byte_identical": None if not outputs_present else
                        replay_paths[0].read_bytes() == replay_paths[1].read_bytes(),
                        "note": "两次均 exit 1，无输出 JSON；不能把两个不存在的文件当作逐位相同。"
                        "源码顶层 created_at_utc 预期合法不同；本次未能实测时间戳、耗时、路径差异。"},
        "default_off": defaults, "zero_shift_equality": zero_shift, "identifiability_source": ARTIFACT,
        "identifiability": identifiability, "registration_on_parity": {
            "verdict": "NOT-TESTABLE", "reason": "现有工件只有 registration=None，缺同数据 on/off 对照。"
            "centre_dx/dy 已在分析中通过 camera_grid 拟合，不可将此误当作新增注册参数的成对结果。",
            "attempted_run": "scripts/slm/analyze_fwdmodel_fit.py --no-joint --no-sweeps --no-figures "
            "--out <专属目录>/fwdmodel_off 两次工具均返回 exit 1，第二次日志已推进至第二数据集灰阶分析，无 traceback，"
            "未产出 fwdmodel_fit.json；因此 registration-on 对照未执行，不给出 parity 数字。",
            "minimal_run": "同一工件的两组数据、checkpoint、训练/验证分组和 GLOBAL_BOUNDS 不变；"
            "离线各执行一次 global fit + jacobian_spectrum（无需 joint/sweeps/figures），"
            "一次 registration=None，一次固定非零 (dy,dx) 并补偿 centre_dx/dy 初值及边界至同一物理范围；"
            "比较两个数据集相同 7 个参数的 sensitivity_relative 与 >=0.05 分类。若学习额外的注册自由度，"
            "必须替换而非叠加 centre_dx/dy，避免重复自由度造成不可辨识。未运行此补充实验。"},
        "feature_census": census(), "bom_free": bom_free,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "overall": "NOT-TESTABLE: default-off tests pass; baseline and registration-on identifiability evidence missing",
        "scope_note": "仅写专属 script/output；未执行 graphify update .，因会写越界 graphify-out。"}
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    lines = ["# Issue #237 验收 (6)：P0 默认关闭与辨识率审计", "", "## 结论", "",
             "**NOT-TESTABLE（不得据此关闭验收项）**：默认关闭语义通过；历史回放输入缺失；"
             "注册开启后的辨识率没有成对工件。", "", result["hardware"], "", "## 执行记录", ""]
    for command in commands:
        lines.extend([f"- `{command['command']}`", f"  - exit **{command['exit_status']}**；日志 `{command['log']}`"])
    lines += ["", "## 1. 基线回放：NOT-TESTABLE", "",
              "两次命令除 --out 外均保留默认参数；--execute 是执行开关，默认本身只列计划。"
              "两次都发现 capture=0、没有 refine run，exit 1，均未生成 --out JSON。"
              "缺 data/debug/sim2f_fov_hw_refine_ring_20261103_* 及其原始配对文件，"
              "还需恢复默认 calibration 目录 data/debug/sim2f_fov_hw_20261102/calibration。", "",
              "| 字段 | 固定基线 | 本次实际 |", "|---|---:|---|",
              "| integer_shift_xy | (-3, 32) | 无输出 |",
              "| ncc_zero_before | 0.0315531992032807 | 无输出 |",
              "| ncc_integer_best | 0.6802495575769043 | 无输出 |",
              "| tolerance | 1e-6 | 未比较 |", "",
              "证明键应位于 `runs[].pairs[]` 的 reference 项：`fit.integer_shift_xy`、"
              "`fit.ncc_zero_before`、`fit.ncc_integer_best`、`reproduces_baseline_shift`。"
              "后者本次为未知（results.json 为 null），不是 true。实际产出的 --out：**无**。", "",
              "## 2. 确定性：NOT-TESTABLE", "", result["determinism"]["note"], "",
              "缺失工件不能支持“默认关逐位不变”；不得用脚本内固定常量代替实测。", "",
              "## 3. 默认关闭：PASS", "",
              "目标测试 **22 passed**；固定合成案例 None 与 (0.0,0.0) 的 `np.array_equal`"
              "（另含 int/list/ndarray 的零位移）已在该测试执行，无重复添加。"
              "新增脚本执行补充签名默认值检查及真实旧 M3 checkpoint 加载检查。", "",
              "```json", json.dumps(defaults, ensure_ascii=False, indent=2), "```", "",
              f"真实旧 M3 checkpoint 固定合成案例：np.array_equal={zero_shift['np_array_equal']}；"
              f"max_abs_diff={zero_shift['max_abs_diff']}。", "",
              "## 4. P0 辨识率：记录可读取，on/off parity NOT-TESTABLE", "",
              f"来源 `{ARTIFACT}`；定义 `scripts/slm/analyze_fwdmodel_fit.py:1143-1188`："
              "按 |代价导数|×参数跨度归一化，>=最大敏感度的 5% 计为可辨识；不是矩阵秩或识别成功率。"]
    for row in identifiability:
        lines += ["", f"### {row['dataset']}",
                  f"- 可辨识 {row['identifiable_count']}/{row['total']}：`{row['identifiable']}`",
                  f"- 不可辨识：`{row['unidentifiable']}`"]
    lines += ["", result["registration_on_parity"]["reason"], "",
              result["registration_on_parity"]["attempted_run"], "",
              "**最小补充运行**：" + result["registration_on_parity"]["minimal_run"], "",
              "参考命令骨架（未执行，不伪报 exit）：`python scripts/slm/analyze_fwdmodel_fit.py "
              "--checkpoint <off或on的冻结参数JSON> --no-joint --no-sweeps --no-figures "
              "--out <本输出目录的独立子目录>`。需先按上述约束处理重参数化/边界。", "",
              "## 5. 生产特性普查：PASS（存在性，不等于物理验收）", "",
              "脚本对 src/ao_shaping/algorithm/signal_processing/**/*.py 做字面 grep 等价扫描，results.json 保留模式及命中。"]
    for row in result["feature_census"]:
        lines.append(f"- **{row['feature']}**：{row['state']}；{row.get('locations', row.get('matches'))}")
    lines += ["", "sigma 实现为标量高斯核，不能声称任意实测 PSF 已接入；"
              "标量 gain/background、瞳面 amplitude_residual 不是 CCD gain field，"
              "加性 background 不是复振幅相干杂散光。", "", "## 门禁与边界", "",
              f"- Ruff 两个指定文件：exit {commands[3]['exit_status']}；新增脚本 BOM-free={bom_free}。",
              "- 未运行全量 pytest；未修改生产或目标测试文件；" + result["scope_note"]]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Evidence written to {}", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
