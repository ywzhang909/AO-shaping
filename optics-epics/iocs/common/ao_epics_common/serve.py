"""ao_epics_common.serve - 通用 IOC 启动入口。

用法(从 IOC 目录):
    python -m ao_epics_common.serve config/ioc.yaml my_ioc.SLMIoc

其中 my_ioc 是 IOC 的 src 模块(需在 PYTHONPATH 中),SLMIoc 是其 PVGroup 子类。
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

from .ioc_runner import run_ioc


def main() -> None:
    if len(sys.argv) < 3:
        print(
            "用法: python -m ao_epics_common.serve <ioc.yaml> <module.PVGroupClass>",
            file=sys.stderr,
        )
        sys.exit(2)
    ioc_yaml = Path(sys.argv[1])
    # module.PVGroupClass:类名必须是最后一段(模块可含点,如 src.slm_ioc)
    module_path, _, class_name = sys.argv[2].rpartition(".")

    # 确保 IOC 根目录在 sys.path(便于 import src.*)
    ioc_root = ioc_yaml.resolve().parent.parent
    if str(ioc_root) not in sys.path:
        sys.path.insert(0, str(ioc_root))

    module = importlib.import_module(module_path)
    group_class = getattr(module, class_name)
    # 位置参数已被本入口消费:从 sys.argv 移除,剩余参数(如 --prefix)
    # 经 argv 显式传给 ioc_arg_parser,避免其解析到已消费的位置参数
    remaining = sys.argv[3:]
    sys.argv = sys.argv[:1] + remaining
    run_ioc(ioc_yaml, group_class, argv=remaining or None)


if __name__ == "__main__":
    main()
