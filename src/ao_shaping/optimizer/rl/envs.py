import os
import time
from typing import Tuple, Any, Dict

import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import time

from ao_shaping.drivers import CameraStreamManager, NlightDM

from ao_shaping.drivers.sim.compat import (
    TraditionalAOSystem, AOConfig,
)


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


class TraditionalAOEnv(gym.Env):
    """
    改进版传统自适应光学仿真环境
    
    针对强化学习训练优化：
    - 使用成形奖励（shaped reward）促进收敛
    - 添加进步奖励（progress reward）
    - 适度的动作惩罚
    - 归一化的观测空间
    """
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, 
                 N: int = 64,
                 max_steps: int = 50,
                 n_actuators: int = 4,
                 n_subapertures: int = 4,
                 reward_type: str = 'shaped',
                 Cn2: float = 1e-14,
                 render_mode: str = None):
        """
        初始化TraditionalAOEnv
        
        参数:
            N: 网格大小
            max_steps: 每个episode的最大步数
            n_actuators: 变形镜驱动器数量（每边）
            n_subapertures: WFS子孔径数量
            reward_type: 奖励类型 ['shaped', 'strehl', 'progress']
            Cn2: 折射率结构常数
            render_mode: 渲染模式
        """
        super(TraditionalAOEnv, self).__init__()
        
        self.N = N
        self.max_steps = max_steps
        self.n_actuators = n_actuators
        self.n_subapertures = n_subapertures
        self.reward_type = reward_type
        self.Cn2 = Cn2
        self.render_mode = render_mode
        self.step_count = 0
        
        # 物理参数
        self.L = 0.1  # 孔径尺寸 (m)
        self.wavelength = 1550e-9  # 波长 (m)
        self.dx = self.L / N
        
        # 初始化设备
        self._init_devices()
        
        # 动作空间: DM电压 (归一化到[-1, 1])
        self.action_dim = n_actuators * n_actuators
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(self.action_dim,), 
            dtype=np.float32
        )
        
        # 观测空间: 包含历史信息的字典
        # image: 强度图像 (N, N) -> 归一化到 [0, 1]
        # slopes: 波前斜率 (2 * n_subapertures^2,)
        # history: 历史观测 (history_len, 3) 包含strehl, rms, power
        self.history_len = 8
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0.0, high=1.0,
                shape=(N, N),
                dtype=np.float32
            ),
            "slopes": spaces.Box(
                low=-10.0, high=10.0,
                shape=(2 * n_subapertures ** 2,),
                dtype=np.float32
            ),
            "history": spaces.Box(
                low=-1.0, high=2.0,
                shape=(self.history_len, 3),
                dtype=np.float32
            ),
            "voltages": spaces.Box(
                low=-1.0, high=1.0,
                shape=(self.action_dim,),
                dtype=np.float32
            )
        })
        
        # 初始化状态
        self._init_state()
        
        # 用于进度奖励
        self.prev_strehl = 0.0
        self.prev_rms = 1.0
        self.best_strehl = 0.0
        
        # 渲染相关
        self.window = None
        self.clock = None
    
    def _init_devices(self):
        """初始化仿真设备"""
        self.config = AOConfig(
            N=self.N,
            L=self.L,
            wavelength=self.wavelength,
            Cn2=self.Cn2,
            dm_actuators=self.n_actuators,
            subapertures=self.n_subapertures,
            propagation_distance=1000.0
        )
        self.ao_system = TraditionalAOSystem(self.config)

    def _init_state(self):
        """初始化状态"""
        self.step_count = 0
        self.history = np.zeros((self.history_len, 3), dtype=np.float32)
        self.current_voltages = np.zeros(self.action_dim, dtype=np.float32)
        self._reset_sim()

    
    def _reset_sim(self):
        """使用仿真模块重置"""
        result = self.ao_system.reset()
        self.current_image = result['image'].astype(np.float32) / 65535.0
        self.current_slopes = result['slopes'].astype(np.float32)
        self.prev_strehl = result['strehl']
        self.prev_rms = np.sqrt(np.mean(self.ao_system.turbulence.phase_screen**2))
        self.best_strehl = self.prev_strehl
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        
        self._init_state()
        
        return self._get_obs(), self._get_info()
    
    def step(self, action: np.ndarray):
        """执行一步"""
        if self.step_count >= self.max_steps:
            return self._get_obs(), 0.0, True, False, self._get_info()
        
        self.step_count += 1
        
        # 应用动作 (带平滑)
        action = np.clip(action, -1, 1)
        self.current_voltages = action.copy()
        self._step_sim(action)

        # 计算奖励
        reward = self._calculate_reward()
        
        # 更新历史
        self._update_history()
        
        # 检查终止条件
        done = self.step_count >= self.max_steps
        
        return self._get_obs(), reward, done, False, self._get_info()
    
    def _step_sim(self, action):
        """使用仿真模块执行一步"""
        result = self.ao_system.step(action)
        self.current_image = result['image'].astype(np.float32) / 65535.0
        self.current_slopes = result['slopes'].astype(np.float32)
        # 保存从step返回的性能指标
        self.current_strehl = result['strehl']
        self.current_power = result['power']
    
    def _calculate_reward(self) -> float:
        """计算奖励 - 改进版，解决收敛问题"""
        # 获取当前性能
        
        # 使用湍流相位屏计算，不使用不存在的E_corrected
        phase = self.ao_system.turbulence.phase_screen
        strehl = np.exp(-np.std(phase)**2) if self.step_count > 0 else self.prev_strehl
        rms = np.sqrt(np.mean(phase**2)) if self.step_count > 0 else self.prev_rms

        
        if self.reward_type == 'shaped':
            # 成形奖励：组合多种信号
            # 1. Strehl比奖励 (主要)
            strehl_reward = strehl * 10.0  # 缩放到合理范围
            
            # 2. 进步奖励 (鼓励改善)
            strehl_improvement = max(0, strehl - self.prev_strehl) * 20.0
            
            # 3. RMS奖励 (低RMS好)
            rms_reward = (1.0 - min(rms / 2.0, 1.0)) * 5.0
            
            # 4. 动作正则化 (惩罚大动作)
            action_penalty = -0.01 * np.mean(np.abs(self.current_voltages))
            
            # 5. 稳定性奖励 (保持好性能)
            stability_bonus = 0.0
            if strehl > self.best_strehl:
                stability_bonus = (strehl - self.best_strehl) * 10.0
                self.best_strehl = strehl
            
            reward = strehl_reward + strehl_improvement + rms_reward + action_penalty + stability_bonus
            
        elif self.reward_type == 'strehl':
            # 仅Strehl奖励
            reward = strehl * 10.0 + max(0, strehl - self.prev_strehl) * 5.0
            
        elif self.reward_type == 'progress':
            # 进步奖励
            reward = (strehl - self.prev_strehl) * 50.0
        else:
            reward = strehl * 10.0
        
        # 更新previous值
        self.prev_strehl = strehl
        self.prev_rms = rms
        
        return float(reward)
    
    def _update_history(self):
        """更新历史记录"""
        # 使用从step返回的性能指标
        strehl = self.current_strehl
        rms = np.sqrt(np.mean(self.ao_system.turbulence.phase_screen**2))
        power = self.current_power

        # 滚动历史
        self.history = np.roll(self.history, -1, axis=0)
        self.history[-1] = [strehl, rms, power]
    
    def _get_obs(self) -> Dict:
        """获取观测"""
        return {
            "image": self.current_image.astype(np.float32),
            "slopes": self.current_slopes.astype(np.float32),
            "history": self.history.astype(np.float32),
            "voltages": self.current_voltages.astype(np.float32)
        }
    
    def _get_info(self) -> Dict:
        """获取信息"""
        strehl = self.prev_strehl
        rms = np.sqrt(np.mean(self.ao_system.turbulence.phase_screen**2))

        return {
            "strehl": float(strehl),
            "rms": float(rms),
            "step": self.step_count,
            "best_strehl": float(self.best_strehl)
        }
    
    def render(self, mode='human'):
        """渲染环境"""
        if mode == 'human':
            self._render()
        elif mode == 'rgb_array':
            return (self.current_image * 255).astype(np.uint8)
    
    def _render(self):
        """备用渲染实现"""
        intensity = (self.current_image * 255).astype(np.uint8)
        
        if not self.window:
            pygame.init()
            self.window = pygame.display.set_mode((self.N * 2, self.N))
            self.clock = pygame.time.Clock()
        
        surf = pygame.surfarray.make_surface(intensity)
        self.window.blit(surf, (0, 0))
        
        # 绘制信息
        font = pygame.font.Font(None, 24)
        info = self._get_info()
        info_text = f"Step: {info['step']}, Strehl: {info['strehl']:.3f}, RMS: {info['rms']:.4f}"
        text_surf = font.render(info_text, True, (255, 255, 255))
        self.window.blit(text_surf, (10, 10))
        
        pygame.display.flip()
        self.clock.tick(30)
    
    def close(self):
        """关闭环境"""
        if self.window:
            pygame.quit()
            self.window = None


class SimTurbulenceAOEnv(gym.Env):
    """Gymnasium environment for turbulence AO using simulation optics.

    Optical chain:
    Laser -> Deformable Mirror -> Focusing Lens -> Atmospheric Turbulence
    -> CCD + Shack-Hartmann-like Sensor.

    The environment uses :class:`TraditionalAOSystem` from
    ``ao_shaping.drivers.sim.compat`` for wave propagation and sensor synthesis.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        n_grid: int = 128,
        n_actuators: int = 8,
        n_subapertures: int = 8,
        max_steps: int = 100,
        history_len: int = 8,
        cn2: float = 1e-14,
        wavelength: float = 1550e-9,
        aperture_size: float = 0.1,
        propagation_distance: float = 1000.0,
        pib_radius: int = 4,
        pib_target: float | None = None,
        time_penalty: float = 0.02,
        goal_gain: float = 0.15,
        action_penalty: float = 0.002,
        saturation_penalty: float = 0.01,
    ) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.history_len = history_len
        self.step_count = 0
        self.pib_radius = pib_radius
        self._configured_pib_target = pib_target
        self.pib_target = pib_target
        self.time_penalty = time_penalty
        self.goal_gain = goal_gain
        self.action_penalty = action_penalty
        self.saturation_penalty = saturation_penalty

        self.config = AOConfig(
            N=n_grid,
            L=aperture_size,
            wavelength=wavelength,
            Cn2=cn2,
            dm_actuators=n_actuators,
            subapertures=n_subapertures,
            propagation_distance=propagation_distance,
        )
        self.ao_system = TraditionalAOSystem(self.config)

        self.action_dim = n_actuators * n_actuators
        self.action_space = spaces.Box(
            low=-0.05,
            high=0.05,
            shape=(self.action_dim,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Dict(
            {
                "ccd": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(history_len, n_grid, n_grid),
                    dtype=np.float32,
                ),
                "hartmann_slopes": spaces.Box(
                    low=-50.0,
                    high=50.0,
                    shape=(history_len, 2 * n_subapertures ** 2),
                    dtype=np.float32,
                ),
                "dm_signal": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(history_len, self.action_dim),
                    dtype=np.float32,
                ),
                "metrics": spaces.Box(
                    low=np.array([0.0, 0.0, -1.0, -1.0, 0.0], dtype=np.float32),
                    high=np.array([1.0, 5.0, 2.0, 2.0, 1.0], dtype=np.float32),
                    shape=(5,),
                    dtype=np.float32,
                ),
            }
        )

        self._initial_strehl = 0.0
        self._initial_power = 0.0
        self._initial_rms = 0.0
        self._initial_pib = 0.0
        self._last_strehl = 0.0
        self._last_power = 0.0
        self._last_rms = 0.0
        self._last_ccd: np.ndarray | None = None
        self._last_slopes: np.ndarray | None = None
        self._ccd_history: np.ndarray | None = None
        self._slopes_history: np.ndarray | None = None
        self._dm_history: np.ndarray | None = None
        self._last_pib = 0.0
        self._best_pib = 0.0
        self._center = (n_grid // 2, n_grid // 2)
        yy, xx = np.ogrid[:n_grid, :n_grid]
        self._pib_mask = (
            (xx - self._center[1]) ** 2 + (yy - self._center[0]) ** 2
        ) <= pib_radius ** 2

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed, options=options)
        if seed is not None:
            np.random.seed(seed)

        self.step_count = 0
        self.ao_system.set_dm_voltages(np.zeros(self.action_dim, dtype=float))
        result = self.ao_system.reset()
        self._sync_from_result(result)
        self._initial_strehl = self._last_strehl
        self._initial_power = self._last_power
        self._initial_rms = self._last_rms
        self._initial_pib = self._last_pib
        self._init_history()
        self._best_pib = self._last_pib
        if self._configured_pib_target is None:
            self.pib_target = self._initial_pib * (1.0 + self.goal_gain)
        else:
            self.pib_target = self._configured_pib_target
        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray):
        self.step_count += 1
        clipped_action = np.clip(action, self.action_space.low, self.action_space.high)
        prev_pib = self._last_pib
        prev_strehl = self._last_strehl
        result = self.ao_system.step(clipped_action.astype(float))
        self._sync_from_result(result)
        self._update_history()

        pib_scale = max(self._initial_pib, 1.0)
        strehl_delta = self._last_strehl - prev_strehl
        pib_delta = (self._last_pib - prev_pib) / pib_scale
        pib_gain = (self._last_pib - self._initial_pib) / pib_scale
        strehl_gain = self._last_strehl - self._initial_strehl
        pib_ratio = self._last_pib / max(self.pib_target, 1.0)

        reward = 0.0
        reward += 3.0 * np.tanh(12.0 * pib_delta)
        reward += 2.0 * np.tanh(8.0 * strehl_delta)
        reward += 1.5 * np.tanh(6.0 * pib_gain)
        reward += 1.0 * np.tanh(10.0 * strehl_gain)

        if self._last_pib > self._best_pib:
            reward += 0.5 * np.tanh(8.0 * ((self._last_pib - self._best_pib) / pib_scale))

        action_cost = self.action_penalty * float(np.mean(np.square(clipped_action)))
        saturation = float(np.mean(np.abs(self.ao_system.dm_voltages)))
        saturation_cost = self.saturation_penalty * max(0.0, saturation - 0.6)
        step_cost = self.time_penalty
        success_bonus = 5.0 if pib_ratio >= 1.0 else 0.0
        reward = reward + success_bonus - action_cost - saturation_cost - step_cost

        self._best_pib = max(self._best_pib, self._last_pib)
        terminated = self._last_pib >= self.pib_target
        truncated = self.step_count >= self.max_steps and not terminated
        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    def _sync_from_result(self, result: Dict[str, Any]) -> None:
        image = result["image"].astype(np.float32)
        image_norm = image / max(float(np.max(image)), 1.0)

        phase = self.ao_system.turbulence.phase_screen
        self._last_rms = float(np.sqrt(np.mean(phase ** 2)))
        self._last_strehl = float(result["strehl"])
        self._last_power = float(result["power"])
        self._last_pib = float(np.sum(image[self._pib_mask]))
        self._last_ccd = image_norm
        self._last_slopes = result["slopes"].astype(np.float32)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        if self._ccd_history is None or self._slopes_history is None or self._dm_history is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        return {
            "ccd": self._ccd_history.astype(np.float32),
            "hartmann_slopes": self._slopes_history.astype(np.float32),
            "dm_signal": self._dm_history.astype(np.float32),
            "metrics": np.array(
                [
                    self._last_strehl,
                    self._last_rms,
                    (self._last_pib - self._initial_pib) / max(self._initial_pib, 1.0),
                    (self._best_pib - self._initial_pib) / max(self._initial_pib, 1.0),
                    float(np.sqrt(np.mean(np.square(self.ao_system.dm_voltages)))),
                ],
                dtype=np.float32,
            ),
        }

    def _get_info(self) -> Dict[str, float | int | str]:
        return {
            "strehl": self._last_strehl,
            "rms": self._last_rms,
            "power": self._last_power,
            "pib": self._last_pib,
            "best_pib": self._best_pib,
            "initial_pib": self._initial_pib,
            "pib_ratio": self._last_pib / max(self.pib_target, 1.0),
            "pib_target": float(self.pib_target),
            "step": self.step_count,
        }

    def _init_history(self) -> None:
        if self._last_ccd is None or self._last_slopes is None:
            raise RuntimeError("Environment state is not initialized.")

        self._ccd_history = np.repeat(
            self._last_ccd[np.newaxis, :, :], self.history_len, axis=0
        )
        self._slopes_history = np.repeat(
            np.clip(self._last_slopes / 10.0, -5.0, 5.0)[np.newaxis, :],
            self.history_len,
            axis=0,
        )
        dm_now = self.ao_system.dm_voltages.astype(np.float32)
        self._dm_history = np.repeat(
            dm_now[np.newaxis, :], self.history_len, axis=0
        )

    def _update_history(self) -> None:
        if self._ccd_history is None or self._slopes_history is None or self._dm_history is None:
            self._init_history()
            return

        self._ccd_history = np.roll(self._ccd_history, -1, axis=0)
        self._ccd_history[-1] = self._last_ccd

        self._slopes_history = np.roll(self._slopes_history, -1, axis=0)
        self._slopes_history[-1] = np.clip(self._last_slopes / 10.0, -5.0, 5.0)

        self._dm_history = np.roll(self._dm_history, -1, axis=0)
        self._dm_history[-1] = self.ao_system.dm_voltages.astype(np.float32)


# ================== 6. 测试运行 ==================
if __name__ == "__main__":
    # 测试TraditionalAOEnv
    print("测试TraditionalAOEnv...")
    
    env = TraditionalAOEnv(
        N=64,
        max_steps=20,
        n_actuators=4,
        n_subapertures=4,
        reward_type='shaped',
        seed=42
    )
    
    print(f"动作空间形状: {env.action_space.shape}")
    print(f"观测空间: {env.observation_space}")
    
    obs, info = env.reset()
    print(f"初始Strehl比: {info['strehl']:.4f}")
    print(f"初始RMS: {info['rms']:.6f}")
    
    total_reward = 0
    for i in range(50):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        if i % 10 == 0:
            print(f"Step {i}, Reward: {reward:.3f}, Strehl: {info['strehl']:.4f}, RMS: {info['rms']:.6f}")
        
        if done:
            print(f"--- Episode End. Total Reward: {total_reward:.3f} ---")
            break
    
    env.close()
    print("测试完成!")
