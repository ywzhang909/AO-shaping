from typing import Tuple, Any
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from glob import glob
import time
from copy import deepcopy

import gymnasium as gym
from gymnasium import spaces

import torch as th
from torch import nn
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.logger import Image, Figure

from drivers import NlightDM, WFSManager, MlaRes

import matplotlib.pyplot as plt
import pygame
import numpy as np
import pandas as pd


def schedule_lr_delta(rms):
    '''
    Schedule the learning rate and disturbance voltage based on the RMS of the wavefront
    
    Args:
        rms (float): RMS of the wavefront
    
    Returns:
        tuple: A tuple containing the learning rate (lr) and delta (disturb voltage).
    '''
    if rms > 0.3:
        return 2, 3
    elif rms > 0.25:
        return 1.5, 2
    elif rms > 0.2:
        return 1.1, 1.2
    elif rms > 0.15:
        return 1, 1
    elif rms > 0.11:
        return 0.9, 0.9
    elif rms > 0.08:
        return 0.8, 0.8
    else:
        return 0.7, 0.7


class WFS():
    
    def __init__(self, pupil_diameter):
        self.wfs = WFSManager(MlaRes.Res768, use_custom_ref=True, high_speed=True, pupil_diameter=pupil_diameter)

    def calc_j(self):
        self.wfs.take_image()
        wf, statics = self.wfs.get_wavefront()
        return wf, statics


class LaserProEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ansi', 'rgb_array']}
    '''
    激光环境。

    参数:
    
    
    '''
    def __init__(self, max_iter=2000, target_rms=0.01, action_delta=10, history_len:int=5,
                 render_mode='human', pupil_diameter=2.24, wfs_res='768') -> None:
        '''
        初始化激光环境
        
        Args:
            max_iter: 最大迭代次数
            target_rms: 目标RMS值
            action_delta: 动作变化范围
            history_len: 历史记录长度
            render_mode: 渲染模式
            pupil_diameter: 光瞳直径
            wfs_res: 波前传感器分辨率
        '''
        
        super().__init__()
        
        # 初始化参数
        self.pupil_diameter = pupil_diameter
        self.wfs_res = wfs_res
        
        # 创建WFS和DM实例
        self.wfs = WFS(pupil_diameter)
        self.dm = NlightDM(keep_when_exit=True)

        self.action_dim = self.dm.DM_Num-1
        
        self.history_len = history_len
        self.target_rms = target_rms
        self.max_iter = max_iter
        
        # 定义动作空间
        self.action_space:spaces.Box = spaces.Box(low=-action_delta, high=action_delta, shape=(self.action_dim,), dtype=np.float32)
        
        # 定义观测空间
        # 注意：这里需要根据实际图像大小设置，暂时使用占位符
        self.img_size = (192, 192)  # 默认图像大小
        self.observation_space = spaces.Dict({
            "voltage": spaces.Box(low=-1, high=1, shape=(self.history_len, self.dm.DM_Num), dtype=np.float32),
            "wavefront": spaces.Box(low=0, high=2*np.pi, shape=(self.history_len, *self.img_size), dtype=np.float32),
            "rms": spaces.Box(low=0, high=2, shape=(self.history_len, 1), dtype=np.float32)
        })
        
        # 显示参数
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        # 环境状态变量
        self.iter = 0
        self.v = np.zeros(self.dm.DM_Num, dtype=np.float32)
        
        # 历史记录
        self.history_voltages = np.zeros((self.history_len, self.dm.DM_Num), dtype=np.float32)
        self.history_rms = np.zeros((self.history_len, 1), dtype=np.float32)
        self.history_wavefront = np.zeros((self.history_len, *self.img_size), dtype=np.float32)
        
        # 初始状态
        self.init_rms = 0.0
        self.init_voltages = np.zeros(self.dm.DM_Num, dtype=np.float32)
        
        # 学习率和扰动参数
        self.current_lr = 1.0
        self.current_delta = 1.0
        
        # 算法类型
        self.algorithm_type = "rl"  # 默认使用强化学习
        
    def _initialize_devices(self) -> None:
        """
        初始化设备
        """
        # 初始化 DM 设备
        self.dm.initialize()
    
    def reset(self, *, seed=None, options=None) -> Tuple[Any, dict]:
        super().reset(seed=seed, options=options)
        if seed:
            np.random.seed(seed)
        
        # 初始化设备
        self._initialize_devices()
        
        # 设置初始电压
        self.v = np.random.rand(self.dm.DM_Num,) * (self.v_high-self.v_low) + self.v_low
        self.dm.send_voltages(self.v, 0.01)
        wavefront, statistics = self.wfs.calc_j()
        
        # 根据初始RMS值设置学习率和扰动参数
        self.current_lr, self.current_delta = schedule_lr_delta(statistics['rms'])
        
        # 初始化历史记录
        self.history_voltages = np.zeros((self.history_len, self.dm.DM_Num), dtype=np.float32)
        self.history_rms = np.ones((self.history_len, 1), dtype=np.float32) * statistics['rms']
        self.history_wavefront = np.repeat(wavefront[np.newaxis, ...], self.history_len, axis=0)
        
        # 记录初始状态
        self.init_rms = statistics['rms']
        self.init_voltages = deepcopy(self.v)
        
        self.iter = 0
        return self._get_obs(), self._get_info()
    
    def set_algorithm(self, algorithm_type: str):
        """
        设置优化算法类型
        
        Args:
            algorithm_type (str): 算法类型，可以是 "rl", "adam", "greedy" 等
        """
        self.algorithm_type = algorithm_type
    
    def _apply_action(self, action: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        应用动作并获取新的波前和统计信息
        
        Args:
            action (np.ndarray): 要应用的动作
            
        Returns:
            Tuple[np.ndarray, dict]: 波前数据和统计信息
        """
        # 根据算法类型应用动作
        if self.algorithm_type == "rl":
            # 强化学习方式：直接应用动作
            dv = np.zeros((self.dm.DM_Num,), dtype=float)
            dv[1:] = action.astype(float)
            _v = self.v + dv
        elif self.algorithm_type == "adam":
            # Adam优化方式：使用学习率缩放动作
            dv = np.zeros((self.dm.DM_Num,), dtype=float)
            dv[1:] = action.astype(float) * self.current_lr
            _v = self.v + dv
        else:
            # 默认方式：直接应用动作
            dv = np.zeros((self.dm.DM_Num,), dtype=float)
            dv[1:] = action.astype(float)
            _v = self.v + dv
            
        self.dm.send_voltages(_v, 0.002)
        wavefront, statistics = self.wfs.calc_j()
        
        # 更新学习率和扰动参数
        self.current_lr, self.current_delta = schedule_lr_delta(statistics['rms'])
        
        return wavefront, statistics
    
    def render(self)->Any:
        if self.window ==None and self.render_mode == 'human':
            pygame.init()
            self.window = pygame.display.set_mode(self.img_size)
            self.clock = time.time()
        
        if self.render_mode == 'human':    
            canvas = pygame.surfarray.make_surface(self.last_wavefront)
            h_r = 5//2  # 默认r_bucket为5
            pygame.draw.rect(canvas, (255, 0, 0), (self.img_size[0]//2-h_r, self.img_size[1]//2-h_r, 2*h_r, 2*h_r), 1)
            pygame.display.set_caption(f'fps={self.iter / max(time.time()-self.clock, 1.0)}')
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            
        elif self.render_mode == 'ansi':
            gain = self.init_rms - self.last_rms
            output_str = f'last_rms={self.last_rms} gain={gain}'
            return output_str
            
        elif self.render_mode == 'rgb_array':
            return self.last_wavefront
        
    def step(self, action:np.ndarray)->Tuple[dict, float, bool, bool, dict]:
        """
        执行一个动作并返回相应的观测、奖励、是否完成、是否达到最大迭代次数以及信息。

        参数:
        action (np.ndarray): 要执行的动作。

        返回:
        Tuple[dict, float, bool, bool, dict]: 包含观测、奖励、是否完成、是否达到最大迭代次数以及信息的元组。
        """
        # 检查约束条件
        _v_test = self.v + np.pad(action.astype(float), (1, 0))  # 添加第一个元素为0
        forbidden_cond = any([
                not np.all(-180 < _v_test < 180),
                self.iter >= self.max_iter,
            ])
            
        if forbidden_cond:
            reward = -1e10
            wavefront = np.zeros_like(self.history_wavefront[-1])
            statistics = {'rms': 2.0}  # 设置一个较大的RMS值作为惩罚
        else:
            wavefront, statistics = self._apply_action(action)
            time_reward_bonus = (self.max_iter-self.iter) / self.max_iter
            reward = -statistics['rms']
            if statistics['rms'] < self.target_rms:
                reward += time_reward_bonus
            
        # 更新历史记录
        self.history_voltages = np.roll(self.history_voltages, -1, axis=0)
        self.history_voltages[-1,:] = self.v + np.pad(action.astype(float), (1, 0))
        self.history_rms = np.roll(self.history_rms, -1, axis=0)
        self.history_rms[-1] = statistics['rms']
        self.history_wavefront = np.roll(self.history_wavefront, -1, axis=0)
        self.history_wavefront[-1,:,:] = wavefront
        
        self.v = self.v + np.pad(action.astype(float), (1, 0))
        self.iter += 1
        
        done = self.iter >= self.max_iter
        truncated = forbidden_cond

        return self._get_obs(), reward, done, truncated, self._get_info()
    
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    @property
    def last_voltage(self)->np.ndarray:
        return self.history_voltages[-1]
    
    @property
    def last_rms(self)->float:
        return self.history_rms[-1]
    
    @property
    def last_wavefront(self):
        return self.history_wavefront[-1]
    
    @property
    def v_low(self):
        return self.action_space.low[0]

    @property
    def v_high(self):
        return self.action_space.high[0]
    
    def _get_obs(self):
        """
        获取当前观测值
        """
        scaled_voltages = (self.history_voltages - self.v_low) / (self.v_high-self.v_low) - 0.5
        return {
            "voltage": scaled_voltages.astype(np.float32),
            "wavefront": self.history_wavefront.astype(np.float32),
            "rms": self.history_rms.astype(np.float32)
        }

    def _get_info(self)->dict:
        """
        获取当前步骤信息
        """
        return {'rms': self.last_rms, 'iter': self.iter, 'v': self.v}


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        
        self.best_rms = None
    
    def _on_step(self, **kwargs):
        if not self.best_rms:
            self.best_rms = self.training_env.get_attr('init_rms')
             
        if self.n_calls >= 0:
            info = self.locals["infos"]
            done = self.locals["dones"][0]
            
            if done:
                cam_img = self.training_env.render(mode="rgb_array")
                self.logger.record('wavefront/image', Image(cam_img, 'HW'), exclude=('stdout', 'log', 'json', 'csv'))
                
                figure = plt.figure()
                power_history = [r['rms'] for r in info[::-1]]
                figure.add_subplot().plot(power_history)
                self.logger.record('rms/value', info[0]['rms'])
                self.logger.record('rms/mean', np.mean(power_history))
                self.logger.record('rms/figure', Figure(figure, True), exclude=('stdout', 'log', 'json', 'csv'))
                plt.close()
                
                voltages = info[0]['v']
                figure = plt.figure()
                figure.add_subplot().plot(voltages)
                ax = figure.add_subplot()
                _ = ax.imshow(np.stack([r['v'] for r in info], axis=0), aspect='auto', interpolation='nearest')
                self.logger.record('voltages/figure', Figure(figure, close=True), exclude=('stdout', 'log', 'json', 'csv'))
                plt.close()
                
                max_iter = info[0]['iter']
                self.logger.record('iter/value', max_iter)
                
                self.best_power = None
        return True
    
    def _on_rollout_end(self):
        return super()._on_rollout_end()


class LSTMEncoder(nn.Module):
    def __init__(self, feature_dim, hidden_dim,num_layers=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rnn = nn.LSTM(feature_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        
    def forward(self, x):
        x,_ = self.rnn(x)
        return x[:,-1,:]


class CustomCombineImageAndVetorExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        extractor = {}
        
        for key, subspace in observation_space.spaces.items():
            if key == 'wavefront':  # 修改为wavefront而不是image
                extractor[key] = self._build_img_extractor(subspace, features_dim)
            elif key == 'voltage':
                extractor[key] = nn.Sequential(
                    LSTMEncoder(subspace.shape[1], 128, num_layers=1),
                    nn.Linear(128, features_dim),
                    nn.ReLU())
            elif key == 'rms':
                extractor[key] = nn.Sequential(
                    LSTMEncoder(subspace.shape[1], 4, num_layers=1),
                    nn.Linear(4, 1),
                    nn.ReLU())
        
        self.extractors = nn.ModuleDict(extractor)
        self._features_dim = 2*features_dim  + 1

    def forward(self, observations: th.Tensor) -> th.Tensor:
        encoder_list = []
        for key, extractor in self.extractors.items():
            encoder_list.append(extractor(observations[key]))
        return th.cat(encoder_list, dim=1)
    
    @staticmethod
    def _build_img_extractor(observation_space, features_dim):
        n_input_channels = observation_space.shape[0]
        cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        
        return nn.Sequential(cnn, linear)


# 训练代码
train = True
device = 'cuda' if th.cuda.is_available() else 'cpu'
img_size = (192, 192)
env = LaserProEnv(render_mode='rgb_array', max_iter=200, pupil_diameter=2.24)  # 移除不存在的参数
policy_kwargs = dict(
    features_extractor_class=CustomCombineImageAndVetorExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)
agent = SAC('MultiInputPolicy', env, 
            verbose=1, device=device, tensorboard_log='logs/sac_tensorboard',
            use_sde=True, use_sde_at_warmup=True,
            learning_starts=200, buffer_size=3000, batch_size=100, learning_rate=0.004,
            policy_kwargs=policy_kwargs)

if train:
    # ckpt_paths = glob('ckpts/rl/rl_model_*_steps.zip')
    # latest_ckpt_path = list(sorted(ckpt_paths, key=lambda x: os.path.getmtime(x)))[-1]
    # print(latest_ckpt_path)
    # agent.load(latest_ckpt_path, print_system_info=False)
    # agent.load_replay_buffer('ckpts/rl/rl_model_replay_buffer_800_steps.pkl', truncate_last_traj=True)
    callbacks = CallbackList([
        TensorboardCallback(),
        CheckpointCallback(save_freq=1000, save_path='ckpts/rl', save_replay_buffer=False, verbose=1)
    ])
    agent.learn(50000, progress_bar=True, callback=callbacks, log_interval=1)

    agent.save('laser_agent')
    agent.save_replay_buffer('laser_buffer')

else:
    ckpt_paths = glob('ckpts/rl/rl_model_*_steps.zip')
    latest_ckpt_path = list(sorted(ckpt_paths, key=lambda x: os.path.getmtime(x)))[-1]
    print(latest_ckpt_path)
    agent.load(latest_ckpt_path, env=env)
    
    env = LaserProEnv(render_mode='human', max_iter=500, target_rms=0.01)  # 修改类名为LaserProEnv
    obs,_ = env.reset()
    env.render()
    history = []
    
    s_time = time.time()
    while True:
        action,_ = agent.predict(obs, deterministic=True)
        obs, reward, done, trunk, info = env.step(action)
        # info = info[0]
        # trunk = info['TimeLimit.truncated']
        history.append(info['rms'])  # 修改为rms而不是J
        env.render()
        if done or trunk:
            time_cost = time.time() - s_time
            env.reset()
            plt.plot(history)
            plt.title(f'{trunk=} {done=}')
            plt.show()
            plt.close()
            
            break