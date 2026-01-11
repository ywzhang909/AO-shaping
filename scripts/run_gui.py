#!/usr/bin/env python3
"""
AO系统GUI启动脚本
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.ao_shaping.gui.main_window import main

if __name__ == "__main__":
    main()</content>
</xai:function_call">创建GUI启动脚本。现在测试GUI是否能正常运行。