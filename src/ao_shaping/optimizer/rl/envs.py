import os
import time
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



    
    
