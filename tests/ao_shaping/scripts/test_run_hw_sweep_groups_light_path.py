"""光路不通 ⇒ **中止整轮** (2026-10-22 12:07-12:24 实测)。

背景
----

第三轮硬件采集跑到 19/27 组时, 实验系统的 0 级光斑从「聚焦的 14px 光斑」变成
「一片弥散晕」 (实测 FWHM ~68px、峰值 33/255) —— 12:07-12:24 之间 SLM 出现过
USB -10002 (被拔插过)。此后每一组都报「光束没有到达相机」, 而**驱动仍然一组一组
往下跑**, 白烧设备时间。

驱动里早就有这一类 (:data:`KIND_LIGHT_PATH_BLOCKED`), 但**没接上**: 分类函数
的 kind 循环里不认识它 ⇒ 光路不通既不会中止整轮, 也不会在索引里留下根因。

这里钉住的契约:

* 「光路受阻 / 光束没有到达相机 / no ladder frame carries signal」⇒ 归类
  ``light_path_blocked``, 且**不在** ``WAITABLE_KINDS`` 里 (等是等不来的);
* 归类之后**立刻中止整轮** (退出码非 0, 剩下的组记 ``skipped``), 输出里说清
  **是哪一组**、**判定理由原文**、**该查哪三段光路**;
* 普通失败仍然照常跑下一组 (原行为不变);
* 焦面标定产物 ``slm1_<时间戳>.json`` 能被 adopt 判据认出来 (不重跑) —— 但
  「光路受阻时写的那份」不算数 (那一轮标定根本没完成)。

判据本身在探针脚本里 (``assess_spot``: 峰值停在读出噪声本底 **且** 帧里没有
局域光斑 ⇒ 受阻; 光到了但光斑过宽 ⇒ 只是没聚焦, 属「警告并继续」), 这里只测
驱动**认不认得**那个结论。

**不碰硬件**: 脚本按路径 import, 喂文本与临时目录。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "slm" / "run_hw_sweep_groups.py"


def _load_driver():
    """按路径加载驱动脚本 (``scripts/slm/`` 是脚本目录, 不是硬件包)。"""
    spec = importlib.util.spec_from_file_location(
        "_hw_sweep_driver_light_path", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


driver = _load_driver()

KIND_LIGHT_PATH_BLOCKED = driver.KIND_LIGHT_PATH_BLOCKED
KIND_SLM_ABSENT = driver.KIND_SLM_ABSENT
KIND_SLM_BUSY = driver.KIND_SLM_BUSY
WAITABLE_KINDS = driver.WAITABLE_KINDS
classify_device_unavailable = driver.classify_device_unavailable
check_adopt_dir = driver.check_adopt_dir


# ---------------------------------------------------------------------------
# 实测到的输出文本 (逐字取自失败轮的索引)
# ---------------------------------------------------------------------------
#: 离焦探针在光路受阻时打印的那一段 (退出码 4)。
BLOCKED_STDOUT = (
    "\n[中止] 光路受阻: 光束没有到达相机：检查 光束→SLM 面板 → 2f 透镜 → CCD 三段光路\n"
    "       基线帧里没有任何局域光斑 (峰值 1.24 counts, FWHM 68 px / 画幅 250 px), "
    "扫描只会记下噪声。\n"
    "       判定与理由已写进 out/run_config.json\n"
)

#: 泽尼克扫描的受阻提示 (退出码 3)。
BLOCKED_STDOUT_SWEEP = (
    "光路受阻: 光束没有到达相机：检查 光束→SLM 面板 → 2f 透镜 → CCD 三段光路\n"
    "  请先确认 光束→SLM 面板 → 2f 透镜 → CCD 三段光路。\n"
)

#: 曝光阶梯估不出上限 —— 画面里一个信号都没有 (同样的根因, 英文报错)。
BLOCKED_STDOUT_LADDER = (
    "ValueError: no ladder frame carries signal; cannot estimate a ceiling\n"
)

#: 「光到了但光斑过宽」⇒ 只是没聚焦, **不是**受阻 (警告并继续)。
NOT_BLOCKED_STDOUT = (
    "[警告] 基线帧有光但没有可评分的局域光斑 "
    "(光路有光但没有可评分的局域光斑（光斑过大或弥散）: "
    "先用焦面探测标定瞳面菲涅尔焦距); 这一轮仍然继续, "
    "但指标要按「光斑没聚焦」来解读\n"
)

#: SLM 被拔插 (设备问题, 与光路无关)。
SLM_ABSENT_STDERR = (
    "ao_shaping.drivers.slm.santec.driver.SantecError: 无法打开SLM #1 "
    "(错误码: -10002, 未找到USB设备，请检查设备电源和连接)\n"
)

#: 别的进程占着 SLM (设备被占, **可以**等)。
SLM_BUSY_STDERR = (
    "ao_shaping.drivers.slm.santec.driver.SantecError: 无法打开SLM #1 "
    "(错误码: -10003, USB设备已打开)\n"
)


class TestClassifyLightPathBlocked:
    """「光路不通」必须被认出来, 而且**不能等**。"""

    def test_blocked_text_is_classified_as_light_path_blocked(self):
        for text in (BLOCKED_STDOUT, BLOCKED_STDOUT_SWEEP, BLOCKED_STDOUT_LADDER):
            verdict = classify_device_unavailable(text)
            assert verdict is not None, text
            assert verdict.kind == KIND_LIGHT_PATH_BLOCKED, text

    def test_light_path_blocked_is_not_waitable(self):
        """🔴 光路不通要**立刻**中止整轮 —— 等是等不来的 (光不会自己回来)。"""
        assert KIND_LIGHT_PATH_BLOCKED not in WAITABLE_KINDS
        verdict = classify_device_unavailable(BLOCKED_STDOUT)
        assert verdict is not None
        assert verdict.waitable is False

    def test_the_reason_text_is_kept_verbatim(self):
        """判定理由**原文**要留在结果里 (它带着三段光路的清单与实测数字)。"""
        verdict = classify_device_unavailable(BLOCKED_STDOUT)
        assert verdict is not None
        assert "光路受阻" in verdict.reason_text
        assert "SLM" in verdict.reason_text

    def test_the_advice_names_the_three_segments(self):
        verdict = classify_device_unavailable(BLOCKED_STDOUT)
        assert verdict is not None
        advice = verdict.advice()
        assert "SLM" in advice and "2f" in advice and "CCD" in advice

    def test_no_device_error_code_is_attached(self):
        """输出里混着别的 Santec 码时**不挂**上去 —— 那会把根因指歪。"""
        verdict = classify_device_unavailable(SLM_ABSENT_STDERR + BLOCKED_STDOUT)
        assert verdict is not None
        assert verdict.kind == KIND_LIGHT_PATH_BLOCKED
        assert verdict.code is None


class TestOrdinaryFailuresAreNotBlocked:
    """普通失败**不能**被误判成光路不通。"""

    def test_an_ordinary_failure_is_not_device_unavailable(self):
        assert classify_device_unavailable("Traceback: ValueError: bad config") is None

    def test_a_broad_spot_is_only_a_warning(self):
        """光到了但光斑过宽 ⇒ 只是没聚焦, **警告并继续**, 不是受阻。"""
        assert classify_device_unavailable(NOT_BLOCKED_STDOUT) is None

    def test_slm_absent_stays_slm_absent(self):
        verdict = classify_device_unavailable(SLM_ABSENT_STDERR)
        assert verdict is not None
        assert verdict.kind == KIND_SLM_ABSENT
        assert verdict.code == -10002

    def test_slm_busy_stays_slm_busy(self):
        verdict = classify_device_unavailable(SLM_BUSY_STDERR)
        assert verdict is not None
        assert verdict.kind == KIND_SLM_BUSY
        assert verdict.waitable is True


# ---------------------------------------------------------------------------
# 焦面标定产物 (2026-10-22: 判据认不出它 ⇒ 每次续跑都重跑一遍标定)
# ---------------------------------------------------------------------------
def _write_focus_evidence(directory: Path, payload: dict) -> Path:
    """写一份焦面标定证据 (``slm1_<时间戳>.json``)。"""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "slm1_20261022_120000.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


class TestFocusProbeEvidence:
    """焦面标定的产物要被 adopt 判据认出来。"""

    def test_the_evidence_file_is_recognised(self, tmp_path):
        directory = tmp_path / "focus_calibration"
        _write_focus_evidence(
            directory,
            {"best_focal_mm": 1523.8, "path_blocked": False, "notes": []},
        )
        assert check_adopt_dir(directory) is None

    def test_a_blocked_evidence_is_not_adoptable(self, tmp_path):
        """光路受阻时焦面探测**照样**把证据写盘 ⇒ 那一轮标定没完成, 不能登记。"""
        directory = tmp_path / "focus_calibration"
        _write_focus_evidence(
            directory,
            {
                "best_focal_mm": None,
                "path_blocked": True,
                "blockage_reason": "光束没有到达相机：检查 光束→SLM 面板 → 2f 透镜 → CCD 三段光路",
            },
        )
        problem = check_adopt_dir(directory)
        assert problem is not None
        assert "光路受阻" in problem

    def test_resume_reruns_a_blocked_focus_group(self, tmp_path):
        """续跑**不许**把「受阻时写的证据」当成「已经跑过」。"""
        evidence = tmp_path / "focus_calibration"
        _write_focus_evidence(
            evidence,
            {"best_focal_mm": None, "path_blocked": True, "blockage_reason": "光束没有到达相机"},
        )
        probe = tmp_path / "fake_focus_probe.py"
        probe.write_text("print('focus ok')\n", encoding="utf-8")
        root = tmp_path / "groups"
        plan = tmp_path / "plan.json"
        plan.write_text(
            json.dumps(
                {
                    "root": str(root),
                    "groups": [
                        {
                            "group": "focus_calibration",
                            "variable": "f_v_mm",
                            "script": str(probe),
                            "args": ["--debug-dir", str(evidence)],
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        rc = driver.main(["--plan", str(plan), "--resume"])

        assert rc == 0
        groups = json.loads((root / "groups_manifest.json").read_text(encoding="utf-8"))
        assert groups["groups"][0]["status"] == "ok", "受阻的标定必须重跑, 不能 adopted"
        assert groups["groups"][0]["argv"], "这一组必须真的被跑起来"
        assert "focus ok" in groups["groups"][0]["stdout_tail"]

    def test_resume_adopts_a_successful_focus_group(self, tmp_path):
        """成功的那次标定 ⇒ 登记为 adopted, **不重跑**。"""
        evidence = tmp_path / "focus_calibration"
        _write_focus_evidence(
            evidence, {"best_focal_mm": 1523.8, "path_blocked": False}
        )
        probe = tmp_path / "fake_focus_probe.py"
        probe.write_text("print('focus ok')\n", encoding="utf-8")
        root = tmp_path / "groups"
        plan = tmp_path / "plan.json"
        plan.write_text(
            json.dumps(
                {
                    "root": str(root),
                    "groups": [
                        {
                            "group": "focus_calibration",
                            "variable": "f_v_mm",
                            "script": str(probe),
                            "args": ["--debug-dir", str(evidence)],
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        rc = driver.main(["--plan", str(plan), "--resume"])

        assert rc == 0
        groups = json.loads((root / "groups_manifest.json").read_text(encoding="utf-8"))
        assert groups["groups"][0]["status"] == "adopted"
        assert groups["groups"][0]["argv"] == []


# ---------------------------------------------------------------------------
# 端到端: 光路不通 ⇒ 中止整轮
# ---------------------------------------------------------------------------
def _plan_with_blocked_group(tmp_path: Path, blocked_text: str) -> tuple[Path, Path]:
    """一份计划: 第一组报「光路受阻」, 后面两组是无害命令。"""
    root = tmp_path / "groups"
    plan = tmp_path / "plan.json"
    groups = [
        {
            "group": "dz_scan",
            "variable": "dz_mm",
            "script": sys.executable,
            "args": [
                "-c",
                "import sys; sys.stderr.write(%r); sys.exit(4)" % blocked_text,
            ],
        },
        {"group": "zernike_farfield_nmax8", "variable": "n_max",
         "script": sys.executable, "args": ["-c", "print('after')"]},
        {"group": "zernike_farfield_nmax12", "variable": "n_max",
         "script": sys.executable, "args": ["-c", "print('after2')"]},
    ]
    plan.write_text(
        json.dumps({"root": str(root), "groups": groups}, ensure_ascii=False),
        encoding="utf-8",
    )
    return plan, root


class TestAbortOnLightPathBlocked:
    """光路不通 ⇒ 立刻中止整轮, 而不是把剩下的组也烧一遍。"""

    def test_the_round_is_aborted_and_the_rest_is_skipped(self, tmp_path):
        plan, root = _plan_with_blocked_group(tmp_path, BLOCKED_STDOUT)

        rc = driver.main(["--plan", str(plan)])

        assert rc != 0, "光路不通时整轮要报失败"
        groups = json.loads(
            (root / "groups_manifest.json").read_text(encoding="utf-8")
        )["groups"]
        assert [g["group"] for g in groups] == [
            "dz_scan",
            "zernike_farfield_nmax8",
            "zernike_farfield_nmax12",
        ]
        # 🔴 关键契约: 受阻的那组不是普通失败, 剩下的组也**不许**跑
        assert groups[0]["status"] == "light_path_blocked"
        assert groups[0]["light_path_blocked"]["kind"] == KIND_LIGHT_PATH_BLOCKED
        assert groups[1]["status"] == "skipped"
        assert groups[1]["skip_reason"] == "round_aborted"
        assert groups[2]["status"] == "skipped"

    def test_the_output_names_the_group_and_the_reason(self, tmp_path, capsys):
        plan, _root = _plan_with_blocked_group(tmp_path, BLOCKED_STDOUT)

        driver.main(["--plan", str(plan)])

        captured = capsys.readouterr()
        # 是哪一组
        assert "dz_scan" in captured.err
        # 判定理由原文
        assert "光路受阻" in captured.err
        # 该查哪三段光路
        assert "SLM" in captured.err and "2f" in captured.err and "CCD" in captured.err
        # 还能续跑
        assert "--resume" in captured.err

    def test_an_ordinary_failure_still_lets_the_round_continue(self, tmp_path):
        """「这一组坏了」≠「光路不通」: 前者照常跑下一组 (原行为不变)。"""
        root = tmp_path / "groups"
        plan = tmp_path / "plan.json"
        plan.write_text(
            json.dumps(
                {
                    "root": str(root),
                    "groups": [
                        {
                            "group": "boom",
                            "variable": "none",
                            "script": sys.executable,
                            "args": [
                                "-c",
                                "import sys; sys.stderr.write('ValueError: x'); sys.exit(3)",
                            ],
                        },
                        {
                            "group": "after",
                            "variable": "none",
                            "script": sys.executable,
                            "args": ["-c", "print('after')"],
                        },
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        rc = driver.main(["--plan", str(plan)])

        assert rc == 1
        groups = json.loads(
            (root / "groups_manifest.json").read_text(encoding="utf-8")
        )["groups"]
        assert groups[0]["status"] == "failed"
        assert groups[1]["status"] == "ok"
