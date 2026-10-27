#!/usr/bin/env python
"""自动检测 GPU 并安装合适的 PyTorch 版本"""

from __future__ import annotations
import subprocess
from loguru import logger
import sys


def get_gpu_capability():
    """检测当前 GPU 的 compute capability"""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_capability(0)
    except ImportError:
        pass
    return None


def install_torch(cuda_version: str):
    """安装指定 CUDA 版本的 PyTorch"""
    index = f"https://download.pytorch.org/whl/{cuda_version}"
    logger.info(f"Installing PyTorch with {cuda_version}...")
    subprocess.run(
        [sys.executable, "-m", "uv", "pip", "install", "torch", "torchvision",
         "--index", index],
        check=True
    )
    logger.info(f"Installed torch with {cuda_version}")


def main():
    cap = get_gpu_capability()

    if cap is None:
        logger.info("CUDA not available, installing CPU version")
        install_torch("cpu")
        return

    major, minor = cap
    sm_version = major * 10 + minor

    logger.info(f"Detected GPU compute capability: sm_{major}{minor} ({sm_version})")

    if sm_version >= 120:
        # RTX 50xx (Blackwell)
        install_torch("cu130")
    elif sm_version >= 90:
        # RTX 30xx, 40xx (Ampere, Hopper)
        install_torch("cu124")
    else:
        # GTX 10xx, RTX 20xx 等旧显卡
        install_torch("cu124")
        logger.info("Note: Old GPU detected. Using cu124 (may have reduced performance)")


if __name__ == "__main__":
    main()