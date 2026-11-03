"""把 ``libs/`` 里的厂商 SDK 装进当前 venv (可重复执行)。

大恒 ``gxipy`` 原本只能靠 ``PYTHONPATH=libs`` 才 import; 桌面界面
(``python -m ao_shaping.gui.ccd.beam_quality``) 换个启动方式就没有那个变量, 于是
"终端里能跑、双击就 ImportError"。本脚本用**目录符号链接**把 ``libs/gxipy`` 挂进
venv 的 ``site-packages``, 再补 ``.pth`` 与 dist-info, 空 PYTHONPATH 也能 import。
用法: 不带参数 = 安装 (幂等); ``--check`` = 只报告, 缺失时非 0 退出。Windows 建符号链接需权限 (开发者模式或管理员); 缺权限直接报错, 不退化成复制。
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sysconfig
from pathlib import Path

LIB_DIR = Path(__file__).resolve().parents[2] / "libs"  # scripts/_common/ 的上两级
NAME = "gxipy"
INSTALLER_TAG = "link_vendor_sdks"  # 写进 dist-info/INSTALLER, 认领本脚本写的元数据


def _same(a: Path, b: Path) -> bool:
    """路径比较 (大小写归一: Windows 盘符/目录大小写不稳定)。"""
    return os.path.normcase(str(a)) == os.path.normcase(str(b))


def _version() -> str:
    """从 ``libs/gxipy/__init__.py`` 读 ``__version__`` (不 import: 那会加载原生 SDK)。"""
    init = LIB_DIR / NAME / "__init__.py"
    if not init.is_file():
        raise SystemExit(f"缺少厂商 SDK: {init}")
    match = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", init.read_text(encoding="utf-8"), re.MULTILINE)
    if match is None:
        raise SystemExit(f"{init} 里找不到 __version__")
    return match.group(1)


def _link_target(link: Path) -> Path:
    """解析符号链接目标 (相对链接按所在目录解释; 去掉 Windows 的长路径前缀)。"""
    raw = os.readlink(link)  # Windows 常给 \\?\D:\... 或 \\?\UNC\server\share
    raw = "\\\\" + raw[8:] if raw.startswith("\\\\?\\UNC\\") else raw.removeprefix("\\\\?\\")
    target = Path(raw)
    return (target if target.is_absolute() else link.parent / target).resolve()


def _link(site: Path) -> bool:
    """建 ``site-packages/gxipy -> libs/gxipy`` 目录符号链接; True = 本次新建。"""
    source, link = (LIB_DIR / NAME).resolve(), site / NAME
    if link.is_symlink():
        if _same(_link_target(link), source):
            return False
        raise SystemExit(f"{link} 指向 {_link_target(link)}, 不是 {source}; 不动别人的链接")
    if link.exists():
        # policy: 真实目录 = 别人 (pip/uv) 装的, 绝不删除
        raise SystemExit(f"{link} 已是真实目录 (pip/uv 装的), 不覆盖; 请先 python -m pip uninstall {NAME}")
    try:
        os.symlink(source, link, target_is_directory=True)
    except OSError as exc:
        raise SystemExit(
            f"创建符号链接失败 ({exc}): {link} -> {source}\n"
            "Windows 请打开「开发者模式」(设置 → 系统 → 开发者选项) 或在管理员终端重跑。"
        ) from exc
    return True


def _write_metadata(site: Path, version: str) -> Path:
    """删掉本脚本写过的旧元数据, 再写 ``.pth`` 与 dist-info; 返回 dist-info 路径。"""
    for old in site.glob(f"{NAME}-*.dist-info"):
        tag = old / "INSTALLER"
        if tag.is_file() and tag.read_text(encoding="utf-8").strip() == INSTALLER_TAG:
            shutil.rmtree(old, ignore_errors=True)
    for old in site.glob(f"__editable__.{NAME}-*.pth"):
        old.unlink()
    dist_info = site / f"{NAME}-{version}.dist-info"
    dist_info.mkdir(parents=True, exist_ok=True)
    (site / f"__editable__.{NAME}-{version}.pth").write_text(f"{LIB_DIR}\n", encoding="utf-8")
    files = {
        "METADATA": f"Metadata-Version: 2.1\nName: {NAME}\nVersion: {version}\n"
        f"Summary: Daheng Imaging Python SDK, symlinked from libs/{NAME}\n",
        "INSTALLER": f"{INSTALLER_TAG}\n",
        "RECORD": "",
        "WHEEL": "Wheel-Version: 1.0\nGenerator: link_vendor_sdks\n"
        "Root-Is-Purelib: true\nTag: py3-none-any\n",
    }
    for filename, text in files.items():
        (dist_info / filename).write_text(text, encoding="utf-8")
    return dist_info


def _missing(site: Path, version: str) -> list[str]:
    """缺失/链错的安装件 (空列表 = 就绪)。"""
    link = site / NAME
    ok = link.is_symlink() and _same(_link_target(link), (LIB_DIR / NAME).resolve())
    missing = [] if ok else [f"{link} 不是指向 {LIB_DIR / NAME} 的符号链接"]
    rest = [site / f"__editable__.{NAME}-{version}.pth", site / f"{NAME}-{version}.dist-info" / "METADATA"]
    return missing + [str(path) for path in rest if not path.exists()]


def main(argv: list[str] | None = None) -> int:
    """安装, 或 (``--check``) 仅检查厂商 SDK 链接。"""
    parser = argparse.ArgumentParser(description="把 libs/ 里的厂商 SDK 链进当前 venv")
    parser.add_argument("--check", action="store_true", help="只检查, 未安装时非 0 退出")
    args = parser.parse_args(argv)
    site = Path(sysconfig.get_paths()["purelib"])  # Windows: <sys.prefix>/Lib/site-packages
    version = _version()
    if args.check:
        missing = _missing(site, version)
        print("\n".join(f"缺失: {item}" for item in missing) or f"{NAME} {version} 已就绪: {site / NAME}")
        return 1 if missing else 0
    created = _link(site)
    print(f"{NAME} {version} {'新建符号链接' if created else '已安装'}: {site / NAME} -> {_link_target(site / NAME)}")
    print(f"dist-info: {_write_metadata(site, version)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
