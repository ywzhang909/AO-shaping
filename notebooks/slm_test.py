# %%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# SLM 分辨率 (根据 CSV 文件分析得出)
RESOLUTION = (1920, 1200)  # 宽 x 高 (从原始 diagonal-grating.csv 检测到)
Bits = 10  # 或 8


def load_csv(path) -> np.ndarray:
    """
    加载 SLM 相位 CSV 文件

    CSV 格式:
    - 第一行是列标题: Y/X,0,1,2,...
    - 每行第一个值是行号(Y), 后面是该行的相位值
    - 相位值范围: 0 ~ 2^Bits-1
    """
    with open(path, 'r') as f:
        # 读取第一行，解析列数
        header = f.readline().strip().split(',')
        cols = len(header) - 1  # 减去 Y/X 列

    # 使用 numpy 加载数据 (跳过第一行标题，第一列是行号)
    data = np.loadtxt(path, delimiter=',', skiprows=1, usecols=range(1, cols + 1))
    return data


def generate_csv(img: np.ndarray, path: str, bits: int = Bits) -> None:
    """
    生成 SLM 相位 CSV 文件

    参数:
        img: 输入的相位图像 (可以是浮点数或整数)
        path: 输出 CSV 文件路径
        bits: 位深度 (8 或 10)
    """
    # scale to the right resolution first
    target_height, target_width = RESOLUTION[1], RESOLUTION[0]

    # 使用简单的最近邻或双线性插值调整大小
    from scipy.ndimage import zoom
    if img.shape != (target_height, target_width):
        zoom_y = target_height / img.shape[0]
        zoom_x = target_width / img.shape[1]
        img_scaled = zoom(img, (zoom_y, zoom_x), order=1)  # 双线性插值
    else:
        img_scaled = img

    # scalerize (0~2^Bits-1) ->(0, 2pi)
    max_val = (2 ** bits - 1)//2

    # 归一化到 0-max_val 范围
    if img_scaled.dtype == np.float32 or img_scaled.dtype == np.float64:
        img_min, img_max = img_scaled.min(), img_scaled.max()
        if img_max > img_min:
            img_scaled = (img_scaled - img_min) / (img_max - img_min) * max_val
        else:
            img_scaled = np.zeros_like(img_scaled)

    # 量化到整数
    img_quantized = np.clip(img_scaled, 0, max_val).astype(np.uint16)

    # save to file
    rows, cols = img_quantized.shape

    with open(path, 'w') as f:
        # 写入标题行
        header = ['Y/X'] + [str(i) for i in range(cols)]
        f.write(','.join(header) + '\n')

        # 写入数据行
        for y in range(rows):
            row_data = [str(y)] + [str(v) for v in img_quantized[y, :]]
            f.write(','.join(row_data) + '\n')


def save_image(img: np.ndarray, path: str) -> None:
    """
    保存图像为 BMP 格式

    参数:
        img: 相位图像 (0~2^Bits-1)
        path: 输出文件路径 (.bmp)
    """
    # 归一化到 0-255 用于显示
    max_val = 2 ** Bits - 1
    img_display = (img / max_val * 255).astype(np.uint8)

    # 保存为 BMP
    Image.fromarray(img_display).convert('L').save(path)


def show(img: np.ndarray, title: str = "SLM Phase Pattern"):
    """
    显示 SLM 相位图案
    """
    plt.figure(figsize=(12, 8))
    plt.imshow(img, cmap='gray', vmin=0, vmax=2**Bits-1)
    plt.colorbar(label=f'Phase Value (0-{2**Bits-1})')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout()
    plt.show()


# %% 相位图案生成函数

def generate_focus(focal_length: float, wavelength: float = 532e-9, pixel_size: float = 8e-6) -> np.ndarray:
    """
    生成聚焦相位图案 (抛物面相位)

    参数:
        focal_length: 焦距 (米)
        wavelength: 波长 (米), 默认 532nm
        pixel_size: 像素大小 (米), 默认 8um

    返回:
        相位图案 (0~2^Bits-1)
    """
    height, width = RESOLUTION[1], RESOLUTION[0]
    max_val = 2 ** Bits - 1

    # 创建坐标网格
    x = np.arange(width) - width // 2
    y = np.arange(height) - height // 2
    X, Y = np.meshgrid(x, y)

    # 计算半径 (像素)
    R2 = X**2 + Y**2

    # 抛物面相位: phi = (pi / lambda / f) * r^2
    # 转换为 SLM 灰度值
    phase = (np.pi / wavelength / focal_length) * (R2 * pixel_size**2)

    # 包裹到 0~2π 并映射到 0~max_val
    phase_wrapped = np.mod(phase, 2 * np.pi)
    img = (phase_wrapped / (2 * np.pi) * max_val).astype(np.uint16)

    return img


def generate_checkerboard(period: int = 100) -> np.ndarray:
    """
    生成棋盘格相位图案

    参数:
        period: 棋盘格周期 (像素)

    返回:
        相位图案 (0 或 max_val)
    """
    height, width = RESOLUTION[1], RESOLUTION[0]
    max_val = 2 ** Bits - 1

    # 创建棋盘格
    y = np.arange(height) // period
    x = np.arange(width) // period
    X, Y = np.meshgrid(x, y)

    # 黑白交替
    checker = (X + Y) % 2
    img = (checker * max_val).astype(np.uint16)

    return img


def generate_binary_grating(b:int=2, a:int=3, direction: str = 'horizontal') -> np.ndarray:
    """
    生成 01 光栅 (二元光栅)

    参数:
        period: 光栅周期 (像素)
        direction: 'horizontal' 或 'vertical'

    返回:
        相位图案 (0 或 max_val//2， pi)
    """
    height, width = RESOLUTION[1], RESOLUTION[0]
    max_val = (2 ** Bits - 1)//2

    if direction == 'horizontal':
        # 水平光栅
        y = np.arange(height)
        grating = np.where(y%(a+b)<b, 0, max_val)
        img = np.tile(grating[:, np.newaxis], (1, width))
    else:
        # 垂直光栅
        x = np.arange(width)
        grating = np.where(x%(a+b)<b, 0, max_val)
        img = np.tile(grating[np.newaxis, :], (height, 1))

    return (img * max_val).astype(np.uint16)


def generate_blazed_grating(period: int = 100, direction: str = 'horizontal') -> np.ndarray:
    """
    生成闪耀光栅 (锯齿状相位分布)

    闪耀光栅具有线性变化的锯齿状相位分布，可以将光衍射到特定级次，
    提高衍射效率。

    参数:
        period: 光栅周期 (像素)
        direction: 'horizontal' 或 'vertical', 默认为 'horizontal'

    返回:
        相位图案 (0 ~ max_val//2, 对应相位 0 ~ π)
    """
    height, width = RESOLUTION[1], RESOLUTION[0]
    # 最大值为正常最大值的一半 (对应相位 2π)
    max_val = (2 ** Bits - 1)

    if direction == 'horizontal':
        # 水平方向闪耀光栅 (相位沿垂直方向变化)
        y = np.arange(height)
        # 生成锯齿波形: 0 -> max_val (对应相位 0 -> π)
        grating = np.astype(y % period, np.float32) / period
        grating = grating * max_val
        img = np.tile(grating[:, np.newaxis], (1, width))
    else:
        # 垂直方向闪耀光栅 (相位沿水平方向变化)
        x = np.arange(width)
        # 生成锯齿波形: 0 -> max_val (对应相位 0 -> π)
        grating = (np.astype(x % period, np.float32) / period * max_val)
        img = np.tile(grating[np.newaxis, :], (height, 1))

    return img.astype(np.uint16)


def generate_microlens_array(lens_size: int = 200, focal_length: float = 0.1,
                              wavelength: float = 532e-9, pixel_size: float = 8e-6) -> np.ndarray:
    """
    生成微透镜阵列相位图案

    参数:
        lens_size: 单个微透镜的大小 (像素)
        focal_length: 焦距 (米)
        wavelength: 波长 (米)
        pixel_size: 像素大小 (米)

    返回:
        相位图案
    """
    height, width = RESOLUTION[1], RESOLUTION[0]
    max_val = 2 ** Bits - 1

    # 创建单个透镜的相位图案
    x = np.arange(lens_size) - lens_size // 2
    y = np.arange(lens_size) - lens_size // 2
    X, Y = np.meshgrid(x, y)
    R2 = X**2 + Y**2

    # 抛物面相位
    phase = (np.pi / wavelength / focal_length) * (R2 * pixel_size**2)
    phase_wrapped = np.mod(phase, 2 * np.pi)
    lens_pattern = (phase_wrapped / (2 * np.pi) * max_val).astype(np.uint16)

    # 平铺成阵列
    n_y = height // lens_size + 1
    n_x = width // lens_size + 1

    array = np.tile(lens_pattern, (n_y, n_x))

    # 裁剪到目标大小
    img = array[:height, :width]

    return img


def generate_turbulence_screen(Cn2: float = 1e-14, L: float = 1000,
                                wavelength: float = 532e-9, pixel_size: float = 8e-6,
                                screen_size: float = None) -> np.ndarray:
    """
    生成大气湍流相位屏 (基于 Kolmogorov 谱)

    参数:
        Cn2: 折射率结构常数 (m^(-2/3))
        L: 传输距离 (米)
        wavelength: 波长 (米)
        pixel_size: 像素大小 (米)
        screen_size: 屏的物理大小 (米), 默认根据分辨率计算

    返回:
        相位图案
    """
    height, width = RESOLUTION[1], RESOLUTION[0]
    max_val = 2 ** Bits - 1

    if screen_size is None:
        screen_size = max(height, width) * pixel_size

    # 创建频率网格
    kx = 2 * np.pi * np.fft.fftfreq(width, pixel_size)
    ky = 2 * np.pi * np.fft.fftfreq(height, pixel_size)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    K[0, 0] = 1e-10  # 避免除以零

    # Kolmogorov 谱: Phi(k) = 0.033 * Cn2 * k^(-11/3)
    # 相位屏功率谱: W_phi(k) = 2 * pi * k^2 * L * Phi(k)
    power_spectrum = 2 * np.pi * K**2 * L * 0.033 * Cn2 * K**(-11/3)

    # 生成随机相位
    random_phase = np.random.randn(height, width) + 1j * np.random.randn(height, width)

    # 在频域应用功率谱
    screen_fft = np.sqrt(power_spectrum) * random_phase

    # 逆 FFT 得到相位屏
    phase_screen = np.real(np.fft.ifft2(screen_fft))

    # 归一化并映射到 0~max_val
    phase_screen = (phase_screen - phase_screen.min()) / (phase_screen.max() - phase_screen.min()) * max_val

    return phase_screen.astype(np.uint16)


def generate_zernike(n: int, m: int, amplitude: float = 1.0, radius: float = None) -> np.ndarray:
    """
    生成 Zernike 多项式相位图案

    参数:
        n: 径向阶数
        m: 角向阶数
        amplitude: 振幅 (单位: 波长)
        radius: 圆形孔径半径 (像素), 默认为短边的一半

    返回:
        相位图案
    """
    height, width = RESOLUTION[1], RESOLUTION[0]
    max_val = 2 ** Bits - 1

    if radius is None:
        radius = min(height, width) // 2

    # 创建归一化坐标
    x = (np.arange(width) - width // 2) / radius
    y = (np.arange(height) - height // 2) / radius
    X, Y = np.meshgrid(x, y)

    # 转换为极坐标
    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X)

    # 只在圆内计算
    mask = R <= 1.0

    # 计算 Zernike 多项式
    from scipy.special import factorial

    def zernike_radial(n, m, r):
        """Zernike 径向多项式"""
        R = np.zeros_like(r)
        for k in range((n - abs(m)) // 2 + 1):
            coef = ((-1)**k * factorial(n - k)) / \
                   (factorial(k) * factorial((n + abs(m)) // 2 - k) * factorial((n - abs(m)) // 2 - k))
            R += coef * r**(n - 2*k)
        return R

    # 计算 Zernike 多项式
    if m >= 0:
        Z = zernike_radial(n, m, R) * np.cos(m * Theta)
    else:
        Z = zernike_radial(n, -m, R) * np.sin(-m * Theta)

    # 应用圆形孔径
    Z = Z * mask

    # 转换为相位 (单位: 2π)
    phase = Z * amplitude * 2 * np.pi

    # 包裹并映射到 0~max_val
    phase_wrapped = np.mod(phase, 2 * np.pi)
    img = (phase_wrapped / (2 * np.pi) * max_val).astype(np.uint16)

    return img


# %% 测试代码
if __name__ == "__main__":
    P = 8

    print(f'周期为{P}像素')
    # 测试 5: 生成聚焦相位
    # print("\n生成聚焦相位...")
    # focus_phase = generate_focus(focal_length=0.5, wavelength=532e-9)
    # show(focus_phase, "Focus Phase (f=0.5m)")
    # generate_csv(focus_phase, r"data\pattern\focus-phase.csv")
    # print(r"已生成: data\pattern\focus-phase.csv")

    # # 测试 6: 生成棋盘格
    # print("\n生成棋盘格...")
    # checker = generate_checkerboard(period=P)
    # # show(checker, "Checkerboard (period=150)")
    # generate_csv(checker, r"data\pattern\checkerboard.csv")
    # print(r"已生成: data\pattern\checkerboard.csv")

    # # 测试 7: 生成 01 光栅
    print("\n生成 01 光栅...")
    _b = int(P/2) 
    _a = P-_b
    binary_grating_h = generate_binary_grating(b=_b, a=_a, direction='horizontal')
    # show(binary_grating_h, "Binary Grating (horizontal)")
    generate_csv(binary_grating_h, r"data\pattern\binary-grating-h.csv")
    print(r"已生成: data\pattern\binary-grating-h.csv")

    binary_grating_v = generate_binary_grating(b=_b, a=_a, direction='vertical')
    # show(binary_grating_v, "Binary Grating (vertical)")
    generate_csv(binary_grating_v, r"data\pattern\binary-grating-v.csv")
    print(r"已生成: data\pattern\binary-grating-v.csv")

    # 测试 8: 生成闪耀光栅
    print("\n生成闪耀光栅...")
    blazed_grating_h = generate_blazed_grating(period=P, direction='horizontal')
    # show(blazed_grating_h, "Blazed Grating (horizontal)")
    generate_csv(blazed_grating_h, f"data/pattern/{P}-blazed-grating-h.csv")
    print(f"已生成: data/pattern/{P}-blazed-grating-h.csv")

    blazed_grating_v = generate_blazed_grating(period=P, direction='vertical')
    # show(blazed_grating_v, "Blazed Grating (vertical)")
    generate_csv(blazed_grating_v, f"data/pattern/{P}-blazed-grating-v.csv")
    print(f"已生成: data/pattern/{P}-blazed-grating-v.csv")

    # # 测试 9: 生成微透镜阵列
    # print("\n生成微透镜阵列...")
    # microlens = generate_microlens_array(lens_size=200, focal_length=0.1)
    # show(microlens, "Microlens Array")
    # generate_csv(microlens, r"data\pattern\microlens-array.csv")
    # print(r"已生成: data\pattern\microlens-array.csv")

    # # 测试 10: 生成湍流屏
    # print("\n生成湍流相位屏...")
    # turbulence = generate_turbulence_screen(Cn2=1e-14, L=1000)
    # show(turbulence, "Turbulence Screen")
    # generate_csv(turbulence, r"data\pattern\turbulence-screen.csv")
    # print(r"已生成: data\pattern\turbulence-screen.csv")

    # # 测试 11: 生成 Zernike 相位
    # print("\n生成 Zernike 相位...")
    # # 常见的 Zernike 模式: (n=2, m=0) 是离焦, (n=2, m=±2) 是像散
    # zernike_defocus = generate_zernike(n=2, m=0, amplitude=2.0)
    # show(zernike_defocus, "Zernike Defocus (n=2, m=0)")
    # generate_csv(zernike_defocus, r"data\pattern\zernike-defocus.csv")
    # print(r"已生成: data\pattern\zernike-defocus.csv")

    # zernike_astig = generate_zernike(n=2, m=2, amplitude=1.0)
    # show(zernike_astig, "Zernike Astigmatism (n=2, m=2)")
    # generate_csv(zernike_astig, r"data\pattern\zernike-astigmatism.csv")
    # print(r"已生成: data\pattern\zernike-astigmatism.csv")

    # print("\n所有测试完成!")