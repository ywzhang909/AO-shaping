"""守卫：光路的历史配置不得作为**静默默认值**出现在脚本里。

背景（2026-10-23）：模型采样窗心、零点、焦点、检查点都是**逐会话测量**的；
把它们写成默认值会让"换光路后仍在用旧配置"这件事**无声发生**（当天实测：窗心偏
72–80 px，而所有指标照常输出）。本测试用 AST 检查两类不变量：

1. 「测量输入」参数必须 `required=True`（缺参即 argparse 报错，不许有历史默认）；
2. 允许保留历史默认的参数，脚本里**必须**存在对应的 `⚠` 警告文案（不许静默）。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "scripts" / "slm"

# 参数名 -> 脚本（相对 scripts/slm）
REQUIRED_INPUTS = {
    "direction_test_after_optics_change.py": ("--checkpoint", "--seed-panel", "--spot-xy", "--out"),
    "refit_model_after_optics_change.py": ("--dataset", "--out", "--checkpoint"),
    "run_focus_recalibration.py": ("--out",),
}

# 允许有历史默认、但必须带警告的脚本与警告关键字
WARNED_DEFAULTS = {
    "generate_defocus_differentiable_report.py": ("历史默认值",),
    "generate_recalibration_report.py": ("历史标定",),
}


def _flag_options(script: Path) -> dict[str, dict]:
    tree = ast.parse(script.read_text(encoding="utf-8"))
    found: dict[str, dict] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "add_argument":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        flag = node.args[0].value
        if not isinstance(flag, str) or not flag.startswith("--"):
            continue
        options = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        found[flag] = options
    return found


@pytest.mark.parametrize("script_name, flags", REQUIRED_INPUTS.items())
def test_measured_inputs_have_no_silent_default(script_name: str, flags: tuple[str, ...]):
    """测量类输入必须 required=True —— 历史值不许当默认。"""
    options = _flag_options(SCRIPTS / script_name)
    for flag in flags:
        assert flag in options, f"{script_name}: 缺少 {flag}"
        required = options[flag].get("required")
        is_required = isinstance(required, ast.Constant) and required.value is True
        assert is_required, f"{script_name}: {flag} 必须 required=True（不得有历史默认值）"


@pytest.mark.parametrize("script_name, keywords", WARNED_DEFAULTS.items())
def test_historical_defaults_are_never_silent(script_name: str, keywords: tuple[str, ...]):
    """可以留历史默认，但必须显式警告，否则等于静默沿用旧光路。"""
    text = (SCRIPTS / script_name).read_text(encoding="utf-8")
    assert "⚠" in text, f"{script_name}: 历史默认值必须带 ⚠ 警告"
    for keyword in keywords:
        assert keyword in text, f"{script_name}: 缺少对「{keyword}」的警告文案"


def test_no_measured_spot_literal_in_scripts():
    """任何脚本都不得把实测零点写死成字面量（如 1219/964、1418/966）。"""
    forbidden = ("spot_xy=(1219", "spot_xy=(1418", "1219, 964)", "1418, 966)")
    offenders = []
    for script in sorted(SCRIPTS.glob("*.py")):
        text = script.read_text(encoding="utf-8")
        for literal in forbidden:
            if literal in text:
                offenders.append(f"{script.name}: {literal}")
    assert not offenders, f"脚本里出现写死的实测零点: {offenders}"
