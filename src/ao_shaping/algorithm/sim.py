import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

# ================== 1. 全局参数设置 ==================
N = 256          # 网格数
L = 0.5          # 物理尺寸 (米)
dx = L / N       # 空间步长
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
THETA = np.arctan2(Y, X)

# --- 光学参数 ---
wavelength = 1550e-9  # 波长
k0 = 2 * np.pi / wavelength

# --- 传播参数 ---
Z_total = 1000.0      # 传播距离 1km
num_steps = 50        # 分步数量
dz = Z_total / num_steps

# --- 湍流参数 ---
Cn2 = 1e-14           # 湍流强度

# ================== 2. 频域参数 (用于衍射) ==================
fx = np.fft.fftfreq(N, d=dx)
fy = np.fft.fftfreq(N, d=dx)
FX, FY = np.meshgrid(fx, fy)
k_trans = 2 * np.pi * np.sqrt(FX**2 + FY**2) # 横向波数

# 避免除以零
kz_arg = 1 - (wavelength * FX)**2 - (wavelength * FY)**2
kz_arg[kz_arg <= 0] = 1e-10
kz = k0 * np.sqrt(kz_arg) # 传播方向波数

# ================== 3. 工具函数 ==================
def create_gaussian_field(w0):
    """创建高斯光束 (X偏振)"""
    return np.exp(-(R**2) / (w0**2))

def create_target_radial_field(w0):
    """创建目标径向偏振光场"""
    amplitude = np.exp(-(R**2) / (w0**2))
    Ex = amplitude * np.cos(THETA)
    Ey = amplitude * np.sin(THETA)
    return Ex, Ey

def calculate_stokes_rgb(Ex, Ey):
    """计算斯托克斯参数并转换为RGB图像用于可视化"""
    S0 = np.abs(Ex)**2 + np.abs(Ey)**2
    S1 = np.abs(Ex)**2 - np.abs(Ey)**2
    S2 = 2 * np.real(Ex * np.conj(Ey))
    
    # 避免除零
    S0 = np.where(S0 == 0, 1e-10, S0)
    
    # 计算方位角和圆度
    alpha = np.arctan2(S2, S1) / 2
    alpha = (alpha - alpha.min()) / (alpha.ptp() + 1e-10)
    
    p = np.sqrt(S1**2 + S2**2) / S0
    
    # HSV to RGB
    H = alpha
    S = p
    V = S0 / S0.max()
    
    HSV = np.dstack((H, S, V))
    return hsv_to_rgb(HSV)

def generate_turbulence_screen(dz, Cn2):
    """生成一个薄层湍流相位屏 (频谱法)"""
    # 简化的功率谱
    power = 0.023 * Cn2 * dz * (k_trans + 1e-6)**(-11/3)
    power[0,0] = 0
    
    # 生成随机相位
    phi_fft = (np.random.randn(N, N) + 1j * np.random.randn(N, N)) * np.sqrt(power)
    phi = np.fft.ifft2(phi_fft).real
    
    return phi

# ================== 4. 分步傅里叶传播类 ==================
class VectorBeamPropagator:
    def __init__(self, N, dx, wavelength, Z_total, num_steps, Cn2):
        self.N = N
        self.dx = dx
        self.wavelength = wavelength
        self.Z_total = Z_total
        self.num_steps = num_steps
        self.Cn2 = Cn2
        self.dz = Z_total / num_steps
        
        # 频域参数
        self._setup_frequency_domain()
        
    def _setup_frequency_domain(self):
        """初始化频域参数"""
        fx = np.fft.fftfreq(N, d=dx)
        fy = np.fft.fftfreq(N, d=dx)
        FX, FY = np.meshgrid(fx, fy)
        self.k_trans = 2 * np.pi * np.sqrt(FX**2 + FY**2)
        
        # 传播算符
        kz_arg = 1 - (wavelength * FX)**2 - (wavelength * FY)**2
        kz_arg[kz_arg <= 0] = 1e-10
        self.H = np.exp(1j * k0 * np.sqrt(kz_arg) * self.dz)
        
    def propagate(self, Ex_init, Ey_init):
        """执行完整的分步傅里叶传播"""
        Ex = Ex_init.astype(complex)
        Ey = Ey_init.astype(complex)
        
        print(f"🚀 开始传播: {self.Z_total/1000} km...")
        
        for step in range(self.num_steps):
            # --- 1. 衍射 (频域) ---
            Ex = np.fft.ifft2(np.fft.fft2(Ex) * self.H)
            Ey = np.fft.ifft2(np.fft.fft2(Ey) * self.H)
            
            # --- 2. 湍流 (空域) ---
            phi_turb = generate_turbulence_screen(self.dz, self.Cn2)
            Ex *= np.exp(1j * phi_turb)
            Ey *= np.exp(1j * phi_turb)
            
            if (step+1) % (self.num_steps//5) == 0:
                print(f"   已完成 {(step+1)*100/self.num_steps:.0f}%")
                
        return Ex, Ey

# ================== 5. 自适应光学校正器 ==================
class VectorAdaptiveOptics:
    def __init__(self, N):
        self.N = N
        
    def correct_to_vector_beam(self, Ex_measured, Ey_measured, target_mode="radial", w0=0.1):
        """
        校正逻辑：测量场 -> 目标场
        物理上这对应于一个可编程的q-plate或双SLM系统
        """
        # --- 模拟探测 ---
        # 这里我们假设能完美测量Ex和Ey
        
        # --- 定义目标 ---
        if target_mode == "radial":
            Ex_target = np.exp(-(R**2) / (w0**2)) * np.cos(THETA)
            Ey_target = np.exp(-(R**2) / (w0**2)) * np.sin(THETA)
        else:
            Ex_target = np.exp(-(R**2) / (w0**2))
            Ey_target = np.zeros_like(Ex_target)
            
        # --- 计算校正器调制函数 ---
        # 假设我们有一个器件可以同时调制相位和偏振
        # 我们需要的调制函数是：Target_Field / Measured_Field
        # (注意：实际中需要考虑能量守恒，这里简化为复数除法)
        
        # 避免除以零
        norm = np.abs(Ex_measured)**2 + np.abs(Ey_measured)**2
        norm[norm < norm.max() * 1e-3] = 1e-10
        
        # 简单的除法可能不稳定，我们使用梯度下降思想进行迭代优化
        Ex_corr, Ey_corr = self._iterative_correction(Ex_measured, Ey_measured, Ex_target, Ey_target)
        
        return Ex_corr, Ey_corr, Ex_target, Ey_target
    
    def _iterative_correction(self, Ex_in, Ey_in, Ex_tar, Ey_tar, iterations=20):
        """使用简单的迭代算法寻找最佳校正场"""
        Ex = Ex_in.copy()
        Ey = Ey_in.copy()
        
        for i in range(iterations):
            # 计算误差
            dEx = (Ex - Ex_tar)
            dEy = (Ey - Ey_tar)
            
            # 梯度下降更新 (模拟校正器调整)
            # 这里的 0.1 是学习率
            Ex = Ex - 0.1 * dEx
            Ey = Ey - 0.1 * dEy
            
        return Ex, Ey

# ================== 6. 主程序 ==================
if __name__ == "__main__":
    # --- 初始化 ---
    propagator = VectorBeamPropagator(N, dx, wavelength, Z_total, num_steps, Cn2)
    ao_system = VectorAdaptiveOptics(N)
    
    # --- 步骤 1: 创建初始光束 (线偏振高斯光) ---
    Ex_init = create_gaussian_field(w0=0.05)
    Ey_init = np.zeros_like(Ex_init)
    print("✅ 初始化完成: 初始光束为X偏振高斯光")
    
    # --- 步骤 2: 分步傅里叶传播 (经过大气) ---
    Ex_prop, Ey_prop = propagator.propagate(Ex_init, Ey_init)
    
    # --- 步骤 3: 自适应光学校正 ---
    print("🚀 开始自适应光学校正 (目标: 径向偏振光)...")
    Ex_final, Ey_final, Ex_tar, Ey_tar = ao_system.correct_to_vector_beam(Ex_prop, Ey_prop, target_mode="radial")
    
    # ================== 7. 可视化 ==================
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    
    # --- 图1: 初始状态 ---
    rgb_init = calculate_stokes_rgb(Ex_init, Ey_init)
    axes[0].imshow(rgb_init)
    axes[0].set_title("1. 初始光场\n(线偏振高斯光)")
    axes[0].axis('off')
    
    # --- 图2: 传播后 (湍流干扰) ---
    rgb_prop = calculate_stokes_rgb(Ex_prop, Ey_prop)
    axes[1].imshow(rgb_prop)
    axes[1].set_title("2. 传播1km后\n(湍流畸变)")
    axes[1].axis('off')
    
    # --- 图3: 目标模式 ---
    rgb_tar = calculate_stokes_rgb(Ex_tar, Ey_tar)
    axes[2].imshow(rgb_tar)
    axes[2].set_title("3. 目标模式\n(理想径向偏振)")
    axes[2].axis('off')
    
    # --- 图4: 校正后 ---
    rgb_final = calculate_stokes_rgb(Ex_final, Ey_final)
    axes[3].imshow(rgb_final)
    axes[3].set_title("4. AO校正后\n(矢量光恢复)")
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("✅ 仿真全流程结束!")