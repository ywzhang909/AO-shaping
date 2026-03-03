"""
加载 DOE1_phase.mat 文件并转换为 SLM 可用的 10-bit CSV 格式
"""
import numpy as np
from scipy.io import loadmat
from pathlib import Path


def convert_to_10bit(phase_array: np.ndarray, bits: int = 10) -> np.ndarray:
    """
    将相位数组转换为 10-bit 灰阶值 (0 ~ 2^bits-1)

    参数:
        phase_array: 输入相位数组
        bits: 位深度，默认 10

    返回:
        10-bit 灰阶数组 (0 ~ 1023)
    """
    max_val = (2 ** bits - 1)
    phase_min, phase_max = phase_array.min(), phase_array.max()
    if phase_max > phase_min:
        normalized = (phase_array - phase_min) / (phase_max - phase_min)
    else:
        normalized = np.zeros_like(phase_array)
    gray_10bit = (normalized * max_val).astype(np.uint16)
    return gray_10bit


def save_10bit_csv(gray_array: np.ndarray, path: Path, bits: int = 10) -> None:
    """
    保存 10-bit 灰阶值为 CSV 文件（SLM 标准格式）

    CSV 格式:
    - 第一行是列标题: Y/X,0,1,2,...
    - 每行第一个值是行号(Y), 后面是该行的灰度值
    """
    max_val = (2 ** bits - 1)
    gray_clipped = np.clip(gray_array, 0, max_val).astype(np.uint16)

    rows, cols = gray_clipped.shape
    with open(path, 'w') as f:
        header = ['Y/X'] + [str(i) for i in range(cols)]
        f.write(','.join(header) + '\n')
        for y in range(rows):
            row_data = [str(y)] + [str(v) for v in gray_clipped[y, :]]
            f.write(','.join(row_data) + '\n')


def mat_to_slm_csv(mat_path: Path, output_path: Path = None, bits: int = 10) -> Path:
    """
    将 MATLAB .mat 文件转换为 SLM 可用的 CSV 文件

    参数:
        mat_path: 输入 .mat 文件路径
        output_path: 输出 CSV 文件路径，默认为与输入同名
        bits: 位深度，默认 10

    返回:
        输出 CSV 文件路径
    """
    # 加载 mat 文件
    mat_data = loadmat(mat_path)

    # 查找相位数据变量
    phase_key = None
    for key in mat_data.keys():
        if not key.startswith('__'):
            phase_key = key
            break

    phase_data = mat_data[phase_key]
    print(f"已加载: {mat_path}")
    print(f"  变量名: {phase_key}")
    print(f"  形状: {phase_data.shape}")
    print(f"  原始范围: [{phase_data.min():.4f}, {phase_data.max():.4f}]")

    # 转换为 10-bit
    gray_10bit = convert_to_10bit(phase_data, bits=bits)
    print(f"  转换后范围: [{gray_10bit.min()}, {gray_10bit.max()}]")

    # 确定输出路径
    if output_path is None:
        output_path = mat_path.with_suffix('.csv')

    # 保存为 CSV
    save_10bit_csv(gray_10bit, output_path, bits=bits)
    print(f"已保存: {output_path}")

    return output_path


if __name__ == "__main__":
    # 获取脚本所在目录，构建数据目录路径
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data" / "calibration" / "SLM整形参数1"

    # 转换 DOE1_phase.mat
    mat_path = data_dir / "DOE1_phase.mat"
    output_path = data_dir / "DOE1_phase_10bit.csv"
    mat_to_slm_csv(mat_path, output_path, bits=10)

    # 转换 DOE2_phase.mat（如果需要）
    mat_path2 = data_dir / "DOE2_phase.mat"
    if mat_path2.exists():
        output_path2 = data_dir / "DOE2_phase_10bit.csv"
        print()
        mat_to_slm_csv(mat_path2, output_path2, bits=10)
