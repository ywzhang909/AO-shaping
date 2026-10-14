#!/bin/bash
# setup_pytorch.sh - 根据 GPU 自动安装合适的 PyTorch 版本
# 使用方式: bash scripts/setup_pytorch.sh

# 检测 GPU
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "Unknown")
echo "Detected GPU: $GPU_NAME"

# 根据 GPU 型号判断
if echo "$GPU_NAME" | grep -q "RTX 50\|RTX 509\|RTX 508\|RTX 507"; then
    # RTX 50xx 系列 (Blackwell)
    echo "Installing PyTorch for RTX 50xx (CUDA 13.0)..."
    uv pip install torch torchvision --index https://download.pytorch.org/whl/cu130
elif echo "$GPU_NAME" | grep -q "GTX 10\|GTX 16\|RTX 20\|RTX 30\|RTX 40"; then
    # 旧显卡 - 使用 CUDA 12.4
    echo "Installing PyTorch for older GPU (CUDA 12.4)..."
    uv pip install torch torchvision --index https://download.pytorch.org/whl/cu124
else
    echo "Unknown GPU, installing CUDA 12.4 version..."
    uv pip install torch torchvision --index https://download.pytorch.org/whl/cu124
fi

echo "Done. Testing CUDA..."
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
