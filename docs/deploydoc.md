# 部署文档

**项目名称:** AO-shaping  
**文档版本:** 1.0.0  
**创建日期:** 2026-03-05  
**最后更新:** 2026-03-05

---

## 目录

1. [系统要求](#1-系统要求)
2. [硬件连接](#2-硬件连接)
3. [软件安装](#3-软件安装)
4. [环境配置](#4-环境配置)
5. [首次运行](#5-首次运行)
6. [网络配置](#6-网络配置)
7. [故障排除](#7-故障排除)
8. [维护升级](#8-维护升级)

---

## 1. 系统要求

### 1.1 硬件要求

| 组件 | 最低要求 | 推荐配置 |
|-----|---------|---------|
| CPU | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 |
| 内存 | 8 GB | 16 GB 或更高 |
| 存储 | 50 GB SSD | 100 GB SSD |
| GPU | NVIDIA GTX 1060 | NVIDIA RTX 3070 或更高 |
| 显示器 | 1920x1080 | 2560x1440 |

### 1.2 软件要求

| 软件 | 版本要求 | 备注 |
|-----|---------|------|
| 操作系统 | Windows 10/11 或 Ubuntu 20.04+ | 64 位 |
| Python | 3.13+ | - |
| CUDA | 11.8+ | GPU 加速必需 |
| cuDNN | 8.6+ | 随 CUDA 安装 |

### 1.3 硬件设备支持

| 设备类型 | 型号 | 驱动要求 |
|---------|------|---------|
| SLM | Santec SLM-200 | Santec SDK / PyVISA |
| DM | NLight | 串口/USB |
| WFS | Thorlabs WFS | Thorlabs SDK |
| CCD | Mii / 大恒 | 对应 SDK |

---

## 2. 硬件连接

### 2.1 设备连接图

```
┌─────────────────────────────────────────────────────────────────┐
│                        AO 系统硬件连接图                          │
└─────────────────────────────────────────────────────────────────┘

    ┌──────────┐         ┌──────────┐         ┌──────────┐
    │   PC     │         │   SLM    │         │    DM    │
    │          │◄───────►│          │◄───────►│          │
    │  Python  │  PCIe/   │ Santec   │   串口   │  NLight  │
    │   AO     │   USB    │  SLM-200 │   RS232  │   DM     │
    └────┬─────┘         └──────────┘         └──────────┘
         │
         │                    ┌──────────┐
         │                    │    WFS   │
         │◄───────────────────►│          │
         │      Camera Link   │ Thorlabs │
         │                    │   WFS    │
         │                    └──────────┘
         │
         │                    ┌──────────┐
         │                    │   CCD    │
         └───────────────────►│          │
                USB/以太网     │  MiiCam  │
                              │   /     │
                              │ 大恒相机 │
                              └──────────┘
```

### 2.2 连接步骤

#### 2.2.1 SLM (Santec SLM-200)

```bash
# 1. 连接电源
# 2. 使用 USB/PCIe 线连接 PC
# 3. 安装 Santec SDK
# 4. 验证连接
```

**检查连接:**
```python
from ao_shaping.drivers.slm import SantecSLM200

slm = SantecSLM200()
slm.open()
print(slm.get_status())
slm.close()
```

#### 2.2.2 DM (NLight)

```bash
# 1. 连接 RS232 串口线
# 2. 设置串口参数: 波特率 115200, 8N1
# 3. 连接电源
```

**检查连接:**
```python
from ao_shaping.drivers.dm import NLightDM

dm = NLightDM()
dm.open()
print(dm.get_info())
dm.close()
```

#### 2.2.3 WFS (Thorlabs)

```bash
# 1. 使用 Camera Link 线连接
# 2. 安装 Thorlabs WFS SDK
# 3. 连接电源
```

**检查连接:**
```python
from ao_shaping.drivers.wfs import ThorlabsWFS

wfs = ThorlabsWFS()
wfs.open()
wavefront = wfs.get_wavefront()
print(f"RMS: {wavefront.rms}")
wfs.close()
```

#### 2.2.4 CCD (MiiCam)

```bash
# 1. 使用 USB 3.0 线连接
# 2. 安装 MiiCam SDK
# 3. 安装对应的 .dll/.so 文件
```

**检查连接:**
```python
from ao_shaping.drivers.ccd import MiiCAM

ccd = MiiCAM()
ccd.open()
image = ccd.capture()
print(f"图像尺寸: {image.shape}")
ccd.close()
```

---

## 3. 软件安装

### 3.1 安装 Python 3.13

```bash
# Windows
# 下载安装包: https://www.python.org/downloads/
# 勾选 "Add Python to PATH"

# Linux (Ubuntu)
sudo apt update
sudo apt install python3.13 python3.13-dev python3.13-venv

# 验证
python3.13 --version
```

### 3.2 安装 uv

```bash
# Windows (PowerShell)
irm get.uv.io | iex

# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# 验证
uv --version
```

### 3.3 克隆项目

```bash
# 克隆仓库
git clone https://github.com/[organization]/AO-shaping.git
cd AO-shaping
```

### 3.4 安装依赖

```bash
# 使用 uv 安装所有依赖
uv sync

# 或手动安装
uv pip install -e .
```

### 3.5 安装 GPU 支持

```bash
# 检查 CUDA 版本
nvidia-smi

# 安装 PyTorch (CUDA 11.8+)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

### 3.6 安装硬件 SDK

#### Santec SLM SDK

```bash
# 联系 Santec 获取 SDK
# 安装 SDK 并配置环境变量

# Windows: 设置 SANTEC_SLM_PATH
# Linux: 设置 LD_LIBRARY_PATH
```

#### Thorlabs WFS SDK

```bash
# 从 Thorlabs 官网下载 WFS SDK
# 安装并配置
```

#### NI-DAQmx

```bash
# Windows: 下载 NI-DAQmx
# Linux: 使用 Linux 支持版本
```

---

## 4. 环境配置

### 4.1 创建配置文件

创建 `config.yaml` 或使用默认配置：

```yaml
# config.yaml
hardware:
  slm:
    model: "Santec SLM-200"
    resolution: [1920, 1080]
    connection: "visa"
    
  dm:
    model: "NLight"
    n_actuators: 97
    port: "COM3"
    
  wfs:
    model: "Thorlabs WFS"
    
  ccd:
    model: "MiiCAM"
    resolution: [512, 512]

optimization:
  wf:
    max_iterations: 50
    convergence: 0.001
    
  wfless:
    max_iterations: 100
    n_patterns: 50

logging:
  level: "INFO"
  file: "ao_shaping.log"
```

### 4.2 环境变量

```bash
# Windows (PowerShell)
$env:AO_SHAPING_CONFIG = "path/to/config.yaml"
$env:PYTHONPATH = "path/to/AO-shaping"

# Linux
export AO_SHAPING_CONFIG="path/to/config.yaml"
export PYTHONPATH="path/to/AO-shaping"
```

### 4.3 路径配置

```python
# 配置数据存储路径
import os

# 数据目录
DATA_DIR = "//10.10.0.53/storage/AO_data"

# 模型保存路径
MODEL_DIR = "./models"

# 日志目录
LOG_DIR = "./logs"
```

---

## 5. 首次运行

### 5.1 验证安装

```python
# 测试导入
python -c "
import ao_shaping
print('ao_shaping version:', ao_shaping.__version__)

# 测试子模块
from ao_shaping.drivers import device_base
from ao_shaping.algorithm import adam
from ao_shaping.wf import DM_wfs
from ao_shaping.wfless import gready_cam
from ao_shaping.utils import wavefront_calc

print('All modules imported successfully!')
"
```

### 5.2 测试硬件连接

```python
# test_hardware.py
from ao_shaping.drivers.slm import SantecSLM200
from ao_shaping.drivers.dm import NLightDM
from ao_shaping.drivers.wfs import ThorlabsWFS
from ao_shaping.drivers.ccd import MiiCAM

def test_all_hardware():
    """测试所有硬件连接"""
    
    print("测试 SLM...")
    try:
        with SantecSLM200() as slm:
            print("  SLM: OK")
    except Exception as e:
        print(f"  SLM: FAILED - {e}")
    
    print("测试 DM...")
    try:
        with NLightDM() as dm:
            print("  DM: OK")
    except Exception as e:
        print(f"  DM: FAILED - {e}")
    
    print("测试 WFS...")
    try:
        with ThorlabsWFS() as wfs:
            print("  WFS: OK")
    except Exception as e:
        print(f"  WFS: FAILED - {e}")
    
    print("测试 CCD...")
    try:
        with MiiCAM() as ccd:
            print("  CCD: OK")
    except Exception as e:
        print(f"  CCD: FAILED - {e}")

if __name__ == "__main__":
    test_all_hardware()
```

### 5.3 运行示例

```bash
# 运行示例脚本
python examples/basic_wavefront_correction.py

# 或使用 Streamlit UI
streamlit run scripts/streamlit_visualizer.py
```

---

## 6. 网络配置

### 6.1 数据存储挂载

```bash
# Linux - 挂载 CIFS
sudo mount -t cifs \
    -o user=tifo,password=TIFO1234,uid=1000,gid=1000,iocharset=utf8,vers=3.0 \
    //10.10.0.53/storage/AO_data \
    /path/to/AO-shaping/data

# 添加到 /etc/fstab (持久化)
//10.10.0.53/storage/AO_data /path/to/AO-shaping/data cifs user=tifo,password=TIFO1234,uid=1000,gid=1000,iocharset=utf8,vers=3.0 0 0

# Windows - 映射网络驱动器
# 在文件资源管理器中: 工具 -> 映射网络驱动器
# 输入: \\10.10.0.53\storage\AO_data
```

### 6.2 防火墙配置

```bash
# Ubuntu - 允许必要端口
sudo ufw allow 22    # SSH
sudo ufw allow 8501   # Streamlit
sudo ufw enable
```

### 6.3 远程访问

```bash
# 使用 SSH 隧道访问 Streamlit
ssh -L 8501:localhost:8501 user@server
```

---

## 7. 故障排除

### 7.1 常见错误

#### 7.1.1 导入错误

**问题:** `ModuleNotFoundError: No module named 'ao_shaping'`

**解决:**
```bash
# 重新安装项目
uv pip install -e .

# 或添加到 PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/AO-shaping"
```

#### 7.1.2 GPU 检测失败

**问题:** `RuntimeError: CUDA not available`

**解决:**
```bash
# 检查 CUDA 安装
nvidia-smi

# 检查 PyTorch CUDA 支持
python -c "import torch; print(torch.cuda.is_available())"

# 重新安装 PyTorch CUDA 版本
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

#### 7.1.3 硬件连接失败

**问题:** 无法连接 SLM/DM/WFS/CCD

**解决:**
```bash
# 1. 检查物理连接
# 2. 检查设备管理器 (Windows) / lsusb (Linux)
# 3. 检查驱动是否安装
# 4. 检查串口/COM 端口号是否正确

# 列出可用串口 (Python)
import serial.tools.list_ports
ports = serial.tools.list_ports.comports()
for port in ports:
    print(f"{port.device}: {port.description}")
```

#### 7.1.4 VISA 设备未找到

**问题:** `VisaIOError: VI_ERROR_RSRC_NFOUND`

**解决:**
```bash
# 1. 检查 NI-VISA 是否安装
# 2. 检查设备是否开机
# 3. 使用 NI-MAX 查看设备列表

# Python 中列出 VISA 资源
import pyvisa
rm = pyvisa.ResourceManager()
print(rm.list_resources())
```

### 7.2 性能问题

#### 7.2.1 内存不足

**解决:**
```python
# 减少批处理大小
optimizer = GreedyCAM(slm, ccd, batch_size=10)  # 默认 50
```

#### 7.2.2 GPU 显存不足

**解决:**
```python
# 减少模型大小或使用 CPU
import torch
torch.cuda.empty_cache()
```

### 7.3 日志调试

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 使用 loguru
from loguru import logger
logger.add("debug.log", level="DEBUG")
```

---

## 8. 维护升级

### 8.1 定期维护

| 任务 | 频率 | 说明 |
|-----|------|------|
| 依赖更新 | 每月 | `uv pip install --upgrade` |
| 数据备份 | 每周 | 备份重要数据到异地 |
| 日志清理 | 每周 | 清理超过 30 天的日志 |
| 硬件检查 | 每月 | 检查设备连接和校准 |

### 8.2 版本升级

```bash
# 1. 拉取最新代码
git pull origin master

# 2. 更新依赖
uv sync

# 3. 运行测试
pytest

# 4. 启动服务
streamlit run scripts/streamlit_visualizer.py
```

### 8.3 备份与恢复

#### 8.3.1 配置文件备份

```bash
# 备份配置
cp config.yaml config.yaml.backup

# 备份数据
rsync -av data/ backup/data/
```

#### 8.3.2 恢复

```bash
# 恢复配置
cp config.yaml.backup config.yaml

# 恢复数据
rsync -av backup/data/ data/
```

---

## 附录

### A. 快速检查清单

- [ ] Python 3.13+ 已安装
- [ ] uv 已安装
- [ ] 项目已克隆
- [ ] 依赖已安装 (`uv sync`)
- [ ] GPU 驱动已安装
- [ ] 硬件已连接
- [ ] 数据目录已挂载
- [ ] 配置文件已创建
- [ ] 硬件测试通过
- [ ] Streamlit 可启动

### B. 联系方式

- 技术支持: support@example.com
- 问题报告: https://github.com/[organization]/AO-shaping/issues

### C. 参考资料

- [项目文档](structure.md)
- [使用文档](usagedoc.md)
- [Git 文档](gitdoc.md)
- [PyTorch 文档](https://pytorch.org/docs/)
- [PyVISA 文档](https://pyvisa.readthedocs.io/)

---

*本文档由 AI 自动生成，最后更新于 2026-03-05*
