import os
from typing import Tuple, Any

import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

from ao_shaping.drivers import CameraStreamManager, NlightDM

Far_Cam_ID = int(os.environ.get('Far_Cam_ID', '1'))
Near_Cam_ID = int(os.environ.get('Near_Cam_ID', '0'))

# TODO 添加近场图像

class LaserCastEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ansi', 'rgb_array']}
    '''
    激光环境。

    参数:
    '''
    def __init__(self, max_iter, target_power=10_000, r_bucket=5, img_size:Tuple[int,int]=(250,250), history_len:int=8, render_mode='human', img_noise:bool=False) -> None:
        super().__init__()
        
        self.cam = CameraStreamManager(cam_id=Far_Cam_ID, exposure_time_ms=70, skip_sampling=False)
        self.dm = NlightDM(keep_when_exit=True)
        # 初始化 DM 设备
        # 初始化相机设备
        self.cam.initialize()
        center = (665, 415)
        # 重置相机窗口以确保质心在图像中心
        self.img_size, _ = self.cam.reset_window(size=img_size, center=center)
        self.dm.initialize()

        self.action_dim = self.dm.DM_Num-1
        
        self.history_len = history_len
        
        self.r_bucket = r_bucket
        self._ideal_power = 255*(2*r_bucket)**2
        self.target_power = target_power if target_power else self._ideal_power
        self.max_iter = max_iter
        
        self.wighted_mask = 0
        self.action_space:spaces.Box = spaces.Box(low=-5, high=5, shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "vector": spaces.Box(low=-1, high=1, shape=(self.history_len, self.dm.DM_Num), dtype=np.float32),
            "image": spaces.Box(low=0, high=2^8-1, shape=(self.history_len, *self.img_size), dtype=np.uint8),
            "powers": spaces.Box(low=0, high=2, shape=(self.history_len, 1), dtype=np.float32)
        })
        self.img_noise = img_noise

        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
    def step(self, action:np.ndarray)->Tuple[dict, float, bool, bool, dict]:
        """
        执行一个动作并返回相应的观测、奖励、是否完成、是否达到最大迭代次数以及信息。

        参数:
        action (np.ndarray): 要执行的动作。

        返回:
        Tuple[dict, float, bool, bool, dict]: 包含观测、奖励、是否完成、是否达到最大迭代次数以及信息的元组。
        """
        # 创建一个与self.v形状相同的零数组dv
        dv = np.zeros((self.dm.DM_Num,), dtype=float)
        dv[1:] = action.astype(float)
        # 计算新的电压值v，通过将当前电压self.v与dv相加，并将结果限制在-299到499之间
        _v = np.clip(self.v + dv, -300, 499)
        
        self.dm.send_voltages(_v, 0.002)
        self.img = self.cam.get_numpy_image()[np.newaxis,:,:]
        if self.img_noise:
            noise = np.random.normal(0, 10, self.img.shape).astype(self.img.dtype)
            self.img = np.clip(self.img+noise, 0, 255).astype(self.img.dtype)
            
        _power = self.calc_pib(self.img)
        
        if self.img_noise:
            reward = (_power - self.history_powers[-1])
            
            if _power>=self.target_power:
                reward = _power * self._time_penalty * (self.max_iter - self.iter)
            
            # 如果电压超出范围或最小电压变化小于1，则给予负奖励
            forbiden_cond = any([
                (np.min(self.v + dv) < -150 or np.max(self.v + dv) > 200),
                self.iter>=self.max_iter,
                np.max(self.img) < 20
            ])
            if forbiden_cond:
                reward = -10 * (self.target_power - _power) * self._time_penalty * (self.max_iter - self.iter)**2
        else:
            reward = _power
        
        # 更新历史电压、功率和强度数组
        self.history_votages = np.roll(self.history_votages, -1, axis=0)
        self.history_votages[-1,:] = _v
        self.history_powers = np.roll(self.history_powers, -1, axis=0)
        self.history_powers[-1] = _power
        self.history_intensity = np.roll(self.history_intensity, -1, axis=0)
        self.history_intensity[-1,:,:] = self.img[0,:,:]
        
        self.v = _v
        self.last_power = _power
        self.iter += 1

        return self.step_obs, reward, _power>=self.target_power, forbiden_cond, self.step_info

    def reset(self, *, seed=None, options=None) -> Tuple[Any, dict]:
        super().reset(seed=seed, options=options)
        if seed:
            np.random.seed(seed)
        
        self.init_env()
        self.iter = 0
        self._time_penalty = abs(self.target_power / self.max_iter)
        
        return self.step_obs, self.step_info
    
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    @property
    def step_obs(self):
        scaled_voltages = (self.history_votages + self.v_low) / (self.v_high-self.v_low) - 0.5
        return {
            "vector": scaled_voltages,
            "image": self.history_intensity,
            "powers": self.history_powers / self._ideal_power
        }
        
    @property
    def step_info(self)->dict:
        return {'J':self.last_power, 'iter': self.iter, 'v':self.v}
        
    @property
    def v_low(self):
        return self.action_space.low[0]

    @property
    def v_high(self):
        return self.action_space.high[0]
    
    def init_env(self) -> None:
        """
        初始化环境。

        此方法执行以下操作：
        1. 初始化 DM（变形镜）设备。
        2. 设置初始电压为零并发送到 DM。
        3. 初始化相机设备。
        4. 获取初始图像并计算其质心。
        5. 重置相机窗口以确保质心在图像中心。
        6. 获取初始图像并计算初始功率。
        7. 初始化历史电压、功率和强度数组。

        返回:
            dict: 包含初始化后的观测值的字典。
        """
        # 设置初始电压为零
        self.v = np.random.rand(self.dm.DM_Num,) * (self.v_high-self.v_low) + self.v_low
        self.v[0] = 0
        self.init_v = self.v.copy()
        self.dm.send_voltages(self.v, 0.1)

        _img = self.cam.get_numpy_image(10)
        self.center = np.unravel_index(np.argmax(_img), _img.shape)[::-1]
        c_x, c_y = self.center
        xv, yv = np.ogrid[:self.img_size[0], :self.img_size[1]]
        self.imgmesh_dist = np.sqrt((xv-c_x) ** 2 + (yv-c_y) ** 2)
        
        self.img = np.expand_dims(_img, 0)
        
        self.last_power = self.calc_pib(self.img)
        self.init_power = self.last_power
        
        self.history_votages = np.zeros((self.history_len, self.dm.DM_Num), dtype=np.float32)
        self.history_powers = np.ones((self.history_len, 1), dtype=np.float32) * self.init_power
        self.history_intensity = np.repeat(self.img, self.history_len, axis=0)
        
        self.init_obs = self.step_obs.copy()
        
    def calc_pib(self, img:np.ndarray)->float:
        bucket_mask = self.imgmesh_dist <= self.r_bucket
        return np.sum(img[0][bucket_mask]).astype(float)
    
    def calc_target(self) -> float:
        return np.sum(self.wighted_mask*self.img) / (self.img_size[0]*self.img_size[1])
    
    def render(self)->Any:
        if not self.window and self.render_mode == 'human':
            pygame.init()
            self.window = pygame.display.set_mode(self.img_size)
            self.clock = time.time()
        
        if self.render_mode == 'human':    
            canvas = pygame.surfarray.make_surface(self.img[0].transpose())
            h_r = self.r_bucket//2
            pygame.draw.rect(canvas, (255, 0, 0), (self.img_size[0]//2-h_r, self.img_size[1]//2-h_r, 2*h_r, 2*h_r), 1)
            pygame.display.set_caption(f'fps={self.iter / max(time.time()-self.clock, 1.0)} {self.step_info}')
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            
        elif self.render_mode == 'ansi':
            gain = self.last_power - self.init_power
            output_str = f'{self.last_power=} {gain=}'
            return output_str
            
        elif self.render_mode == 'rgb_array':
            return self.img.copy()

# ================== 1. 物理仿真核心 ==================
class VectorWaveOpticsSim:
    def __init__(self, N=64, L=0.1, wavelength=1550e-9, Z=1000.0, Cn2=1e-14):
        self.N = N
        self.L = L
        self.wavelength = wavelength
        self.Z = Z
        self.Cn2 = Cn2
        self.dx = L / N
        self.k0 = 2 * np.pi / wavelength

        # 网格
        x = np.linspace(-L/2, L/2, N)
        y = np.linspace(-L/2, L/2, N)
        X, Y = np.meshgrid(x, y)
        self.R = np.sqrt(X**2 + Y**2)
        self.THETA = np.arctan2(Y, X)
        self.X, self.Y = X, Y

        # 频域参数 (角谱法)
        fx = np.fft.fftfreq(N, d=self.dx)
        fy = np.fft.fftfreq(N, d=self.dx)
        FX, FY = np.meshgrid(fx, fy)
        self.k_trans = 2 * np.pi * np.sqrt(FX**2 + FY**2)
        
        kz_arg = 1 - (wavelength * FX)**2 - (wavelength * FY)**2
        kz_arg[kz_arg <= 0] = 1e-10
        self.propagator = np.exp(1j * self.k0 * np.sqrt(kz_arg) * Z)

        # 湍流相位屏 (固定路径)
        self.turb_phase = self._generate_turbulence()

    def _generate_turbulence(self):
        # 简化的湍流相位
        power = (self.k_trans + 1e-6)**(-11/3)
        power[0,0] = 0
        phi_fft = (np.random.randn(self.N, self.N) + 1j * np.random.randn(self.N, self.N)) * np.sqrt(power)
        phi = np.fft.ifft2(phi_fft).real
        return (phi - np.mean(phi)) * 2

    def diffract(self, Ex, Ey):
        """角谱法传播"""
        Ux = np.fft.ifft2(np.fft.fft2(Ex) * self.propagator)
        Uy = np.fft.ifft2(np.fft.fft2(Ey) * self.propagator)
        return Ux, Uy

    def add_turbulence(self, Ex, Ey):
        """施加湍流"""
        return Ex * np.exp(1j * self.turb_phase), Ey * np.exp(1j * self.turb_phase)

    def create_target_radial(self, w0_factor=5):
        """创建目标径向偏振光"""
        w0 = self.L / w0_factor
        amplitude = np.exp(-(self.R**2) / (w0**2))
        Ex = amplitude * np.cos(self.THETA)
        Ey = amplitude * np.sin(self.THETA)
        return Ex, Ey
    
    @staticmethod
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

# ================== 2. 强化学习环境 ==================
class VectorAOTurbulenceEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, N=64, max_steps=50, actuator_grid=8):
        super(VectorAOTurbulenceEnv, self).__init__()
        
        self.N = N
        self.max_steps = max_steps
        self.actuator_grid = actuator_grid
        self.step_count = 0
        
        # --- 物理引擎 ---
        self.physics = VectorWaveOpticsSim(N=N, Cn2=1e-14)
        
        # --- 动作空间: 扩展为两个控制通道 ---
        # 动作维度: [Phase_Voltages (actuator_grid^2),  Polarization_Angles (actuator_grid^2)]
        # Phase: 控制镜面高度 (相位延迟)
        # Polarization: 控制局部偏振旋转角度 (如液晶轴向)
        action_dim = (actuator_grid * actuator_grid) * 2
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )
        
        # --- 状态空间: 增加偏振态作为观测 ---
        # 通道 0: 强度 (Intensity)
        # 通道 1: 相位 (Phase)
        # 通道 2: 偏振角 (Polarization Angle)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(3, N, N), dtype=np.uint8
        )
        
        self._init_state()

    def _init_state(self):
        self.Ex = np.ones((self.N, self.N), dtype=complex)
        self.Ey = np.zeros((self.N, self.N), dtype=complex)
        self.target_Ex = None
        self.target_Ey = None
        self.done = False

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.step_count = 0
        self._init_state()
        
        # --- 初始化目标 ---
        self.target_Ex, self.target_Ey = self.physics.create_target_radial()
        
        # --- 初始扰动: 经过湍流传播 ---
        init_phase = np.random.uniform(-np.pi, np.pi, (self.N, self.N))
        self.Ex = np.exp(1j * init_phase)
        self.Ey = np.zeros_like(self.Ex)
        
        # 传播一步增加真实性
        self.Ex, self.Ey = self.physics.diffract(self.Ex, self.Ey)
        self.Ex, self.Ey = self.physics.add_turbulence(self.Ex, self.Ey)
        
        return self._get_obs(), {}

    def step(self, action):
        if self.done:
            raise RuntimeError("环境已结束")

        self.step_count += 1
        
        # --- 1. 动作解码 (拆分相位和偏振控制) ---
        mid_idx = len(action) // 2
        action_phase = action[:mid_idx]      # 前半部分: 相位控制
        action_polar = action[mid_idx:]      # 后半部分: 偏振控制

        # --- 2. 生成控制面型 ---
        # 插值动作到全分辨率
        phase_screen = self._interpolate_action(action_phase)
        # 偏振旋转角: 将动作映射到 -pi/2 ~ pi/2
        delta_theta = self._interpolate_action(action_polar) * np.pi 
        
        # --- 3. 执行校正 (核心物理模型) ---
        # 模拟一个可编程的矢量光学器件 (如双SLM或q-plate)
        # 这里我们使用一个简化的模型: 局部坐标旋转
        
        # 获取当前的偏振态角度
        # Jones 矢量旋转: J_out = R(-theta) * J_in * R(theta)
        # 简化模型: 直接调制偏振方向
        amp = np.sqrt(np.abs(self.Ex)**2 + np.abs(self.Ey)**2)
        current_theta = np.angle(self.Ex) # 简化表示
        
        # 新的偏振方向 = 原始方向 + 控制增量
        new_theta = current_theta + delta_theta
        
        # 重新合成 Ex, Ey (假设保持为线偏振，但方向改变)
        self.Ex = amp * np.cos(new_theta)
        self.Ey = amp * np.sin(new_theta)
        
        # 同时应用相位校正 (校正波前畸变)
        self.Ex = self.Ex * np.exp(1j * phase_screen)
        self.Ey = self.Ey * np.exp(1j * phase_screen)
        
        # --- 4. 计算奖励 ---
        reward, info = self._calculate_reward()
        
        # --- 5. 终止条件 ---
        self.done = self.step_count >= self.max_steps

        # --- 6. 获取新状态 ---
        obs = self._get_obs()
        
        return obs, reward, self.done, False, info

    def _interpolate_action(self, action_vec):
        """将低维动作插值到高维网格"""
        grid_size = self.actuator_grid
        action_grid = action_vec.reshape((grid_size, grid_size))
        
        # 使用 OpenCV 进行双线性插值 (如果没有 cv2，可以用 scipy)
        try:
            import cv2
            screen = cv2.resize(action_grid, (self.N, self.N), interpolation=cv2.INTER_LINEAR)
        except ImportError:
            from scipy.ndimage import zoom
            zoom_factor = self.N / grid_size
            screen = zoom(action_grid, zoom_factor, order=1)
        
        return screen

    def _calculate_reward(self):
        """计算奖励: 结合强度和偏振匹配度"""
        # --- 1. 偏振保真度 (主要奖励) ---
        # 计算当前矢量场与目标径向场的重叠
        overlap_x = np.vdot(self.Ex, self.target_Ex)
        overlap_y = np.vdot(self.Ey, self.target_Ey)
        energy_current = np.vdot(self.Ex, self.Ex) + np.vdot(self.Ey, self.Ey)
        energy_target = np.vdot(self.target_Ex, self.target_Ex) + np.vdot(self.target_Ey, self.target_Ey)
        
        fidelity = (np.abs(overlap_x)**2 + np.abs(overlap_y)**2) / (energy_current * energy_target + 1e-10)
        
        # --- 2. 模式质量 (OAM 或 径向纯度) ---
        # 对于径向偏振，Ex 和 Ey 应该与 cos(theta), sin(theta) 高度相关
        cos_t = np.cos(self.physics.THETA)
        sin_t = np.sin(self.physics.THETA)
        purity_x = np.corrcoef(self.Ex.real.flatten(), cos_t.flatten())[0,1]**2
        purity_y = np.corrcoef(self.Ey.real.flatten(), sin_t.flatten())[0,1]**2
        mode_purity = (purity_x + purity_y) / 2
        
        # --- 3. 综合奖励 ---
        # 早期侧重于模式，后期侧重于保真度
        reward = 0.7 * fidelity + 0.3 * mode_purity

        info = {
            "fidelity": fidelity,
            "mode_purity": mode_purity,
            "step": self.step_count
        }
        
        return float(reward), info

    def _get_obs(self):
        """生成包含强度、相位、偏振角的观测"""
        intensity = np.abs(self.Ex)**2 + np.abs(self.Ey)**2
        intensity = ((intensity - intensity.min()) / (intensity.ptp() + 1e-10) * 255).astype(np.uint8)
        
        phase = np.angle(self.Ex)
        phase = ((phase - np.min(phase)) / (np.ptp(phase) + 1e-10) * 255).astype(np.uint8)
        
        # 偏振角 (由 Ex 和 Ey 计算)
        # 对于线偏振，偏振角 alpha = 0.5 * arctan2(2*Re(Ex*Ey*), |Ex|^2 - |Ey|^2)
        try:
            S1 = 2 * np.real(self.Ex * np.conj(self.Ey))
            S0 = np.abs(self.Ex)**2 + np.abs(self.Ey)**2
            # 避免除零
            S0 = np.clip(S0, 1e-10, None)
            alpha = 0.5 * np.arctan2(S1, (np.abs(self.Ex)**2 - np.abs(self.Ey)**2))
            alpha = ((alpha - np.min(alpha)) / (np.ptp(alpha) + 1e-10) * 255).astype(np.uint8)
        except:
            alpha = np.zeros_like(phase, dtype=np.uint8)

        # Stack to channels
        obs = np.stack([intensity, phase, alpha], axis=0) # (3, N, N)
        return obs

    def render(self, mode='human'):
        if mode == 'human':
            plt.cla()
            intensity = np.abs(self.Ex)**2 + np.abs(self.Ey)**2
            plt.imshow(intensity, cmap='hot', extent=[-self.physics.L/2, self.physics.L/2, -self.physics.L/2, self.physics.L/2])
            plt.colorbar(label='Intensity')
            plt.title(f"Step {self.step_count}")
            plt.xlabel("X (m)")
            plt.ylabel("Y (m)")
            plt.pause(0.01)
        elif mode == 'rgb_array':
            intensity = np.abs(self.Ex)**2 + np.abs(self.Ey)**2
            return np.stack([intensity, intensity, intensity], axis=-1)

# ================== 3. 测试运行 ==================
if __name__ == "__main__":
    env = VectorAOTurbulenceEnv(N=64, max_steps=20, actuator_grid=8)
    
    print(f"动作空间: {env.action_space.shape}") # 应为 (128,)
    print(f"状态空间: {env.observation_space.shape}") # 应为 (3, 64, 64)
    
    obs, _ = env.reset()
    total_reward = 0
    
    plt.figure(figsize=(10, 5))
    for i in range(50):
        # 随机策略
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        if i % 10 == 0:
            print(f"Step {i}, Reward: {reward:.3f}, Fidelity: {info['fidelity']:.3f}")
        
        if i % 50 == 0:
            env.render('human')
        
        if done:
            print(f"--- Episode End. Total Reward: {total_reward:.3f} ---")
            obs, _ = env.reset()
            total_reward = 0
            break

    plt.show()
    env.close()
    
    
