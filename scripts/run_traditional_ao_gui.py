#!/usr/bin/env python3
"""
传统AO系统GUI启动脚本
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.ao_shaping.gui.traditional_ao_main import main

if __name__ == "__main__":
    main()