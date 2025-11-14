from typing import Tuple, Any
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from glob import glob
import time

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

from drivers import Thorlab_WFS as WFS, MlaRes, NlightDM as DM

import matplotlib.pyplot as plt
import numpy as np


class WFSLaserCastEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ansi', 'rgb_array']}
    '''
    激光投射环境，使用波前传感器(WFS)读取RMS值。

    参数:
    '''
    def __init__(self, max_iter, target_rms=None, history_len:int=10, render_mode='human') -> None:
        super().__init__()
        
        # 使用WFSManager替代CameraStreamManager
        self.wfs = WFS(MlaRes.Res768, use_custom_ref=False, high_speed=True, pupil_diameter=2.8)
        self.dm = DM(keep_when_exit=True)

        self.action_dim = self.dm.DM_Num-1
        
        self.history_len = history_len
        self.max_iter = max_iter
        
        # 设置目标RMS值
        self.target_rms = target_rms if target_rms else 0.1  # 默认目标RMS值
        
        # 动作空间：DM电压调整值
        self.action_space:spaces.Box = spaces.Box(low=-5, high=5, shape=(self.action_dim,), dtype=np.float32)
        
        # 观察空间：包括历史电压、波前数据和RMS值
        self.observation_space = spaces.Dict({
            "vector": spaces.Box(low=-1, high=1, shape=(self.history_len, self.dm.DM_Num), dtype=np.float32),
            "wavefront": spaces.Box(low=-10, high=10, shape=(self.history_len, 32, 32), dtype=np.float32),  # 假设波前数据是32x32
            "rms_values": spaces.Box(low=0, high=10, shape=(self.history_len, 1), dtype=np.float32)
        })

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
        # 计算新的电压值v，通过将当前电压self.v与dv相加，并将结果限制在-300到499之间
        _v = np.clip(self.v + dv, -300, 499)
        
        self.dm.send_voltages(_v, 0.002)
        
        # 使用WFS获取波前数据和RMS值
        self.wfs.take_image()
        wf, statics = self.wfs.get_wavefront()
        _rms = statics['rms']
        
        # 优化奖励函数：奖励更快的RMS下降速度并防止性能劣化
        # 1. 基础奖励：当前RMS值越小奖励越高
        base_reward = -_rms
        
        # 2. 下降速度奖励：如果RMS下降则给予额外奖励
        rms_improvement = self.last_rms - _rms
        improvement_reward = 0.0
        if rms_improvement > 0:
            # 奖励更快的下降速度，下降越多奖励越高
            improvement_reward = rms_improvement * 10.0
        else:
            # 如果RMS上升（性能劣化），给予惩罚
            improvement_reward = rms_improvement * 20.0  # 更大的惩罚系数
        
        # 3. 相对改进奖励：相对于初始RMS的改进
        relative_improvement = self.init_rms - _rms
        relative_reward = relative_improvement * 5.0  # 适当的比例系数
        
        # 4. 奖励变化奖励：鼓励持续改进
        current_reward = base_reward + improvement_reward + relative_reward
        reward_change = current_reward - self.last_reward
        reward_change_reward = 0.0
        if reward_change > 0:
            # 如果奖励在增加，给予额外奖励
            reward_change_reward = reward_change * 2.0
        else:
            # 如果奖励在减少，给予轻微惩罚
            reward_change_reward = reward_change * 5.0
        
        # 组合奖励
        reward = current_reward + reward_change_reward
        
        # 更新最后奖励值
        self.last_reward = current_reward
        
        # 如果RMS达到目标值或超过最大迭代次数，则结束
        done = _rms <= self.target_rms
        truncated = self.iter >= self.max_iter
        
        # 更新历史数据数组
        self.history_voltages = np.roll(self.history_voltages, -1, axis=0)
        self.history_voltages[-1,:] = _v
        self.history_rms = np.roll(self.history_rms, -1, axis=0)
        self.history_rms[-1] = _rms
        self.history_wavefront = np.roll(self.history_wavefront, -1, axis=0)
        self.history_wavefront[-1,:,:] = wf
        
        self.v = _v
        self.last_rms = _rms
        self.iter += 1

        return self.step_obs, reward, done, truncated, self.step_info

    def reset(self, *, seed=None, options=None) -> Tuple[Any, dict]:
        super().reset(seed=seed, options=options)
        if seed:
            np.random.seed(seed)
        
        self.init_env()
        self.iter = 0
        
        return self.step_obs, self.step_info
    
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    @property
    def step_obs(self):
        scaled_voltages = (self.history_voltages + self.v_low) / (self.v_high-self.v_low) - 0.5
        return {
            "vector": scaled_voltages,
            "wavefront": self.history_wavefront,
            "rms_values": self.history_rms
        }
        
    @property
    def step_info(self)->dict:
        return {'RMS':self.last_rms, 'iter': self.iter, 'v':self.v}
        
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
        2. 初始化 WFS（波前传感器）设备。
        3. 设置初始电压并发送到 DM。
        4. 获取初始波前数据和RMS值。
        5. 初始化历史电压、RMS和波前数据数组。
        """
        # 初始化 DM 设备
        self.dm.initialize()
        
        # 初始化 WFS 设备
        self.wfs.initialize()
        
        # 设置初始电压
        self.v = np.random.rand(self.dm.DM_Num,) * (self.v_high-self.v_low) + self.v_low
        self.init_v = self.v.copy()
        self.dm.send_voltages(self.v, 0.01)
        
        # 获取初始波前数据
        self.wfs.take_image()
        wf, statics = self.wfs.get_wavefront()
        self.last_rms = statics['rms']
        self.init_rms = self.last_rms
        
        # 初始化历史数据数组
        self.history_voltages = np.zeros((self.history_len, self.dm.DM_Num), dtype=np.float32)
        self.history_rms = np.ones((self.history_len, 1), dtype=np.float32) * self.init_rms
        self.history_wavefront = np.repeat(wf[np.newaxis, :, :], self.history_len, axis=0)
        
        self.init_obs = self.step_obs.copy()
        
        # 初始化奖励相关属性
        self.last_reward = 0.0
        
    def render(self)->Any:
        if self.render_mode == 'ansi':
            output_str = f'{self.last_rms=}'
            return output_str
        elif self.render_mode == 'rgb_array':
            # 将波前数据转换为图像形式返回
            wf_data = self.history_wavefront[-1]
            # 归一化到0-255范围
            wf_normalized = ((wf_data - wf_data.min()) / (wf_data.max() - wf_data.min()) * 255).astype(np.uint8)
            return wf_normalized[np.newaxis, :, :]
        # human模式不实现可视化显示


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        
        self.history = []
        self.best_rms = float('inf')
    
    def _on_step(self, **kwargs):
        if self.n_calls >= 0:
            info:dict = self.locals["infos"][0]
            self.history.append(info)
            done = self.locals["dones"][0]
            
            # 每隔一定步数记录一次wavefront图像
            if self.n_calls % 10 == 0:  # 每10步记录一次
                # 获取当前环境的wavefront数据
                wavefront_data = self.training_env.get_attr('history_wavefront')[0]
                # 记录最新的wavefront图像
                if len(wavefront_data) > 0:
                    latest_wf = wavefront_data[-1]  # 获取最新的wavefront
                    figure = plt.figure()
                    ax = figure.add_subplot()
                    im = ax.imshow(latest_wf, cmap='viridis')
                    ax.set_title(f'Wavefront at step {self.n_calls}')
                    plt.colorbar(im, ax=ax)
                    self.logger.record(f'wavefront/step_{self.n_calls}', Figure(figure, close=True), exclude=('stdout', 'log', 'json', 'csv'))
                    plt.close()
            
            if done:
                rms = self.history[-1]['RMS']
                figure = plt.figure()
                rms_history = [r['RMS'] for r in self.history]
                figure.add_subplot().plot(rms_history)
                self.logger.record('rms/value', rms)
                self.logger.record('rms/mean', np.mean(rms_history))
                self.logger.record('rms/figure', Figure(figure, True), exclude=('stdout', 'log', 'json', 'csv'))
                plt.close()
                
                voltages = self.training_env.get_attr('v')[0]
                figure = plt.figure()
                figure.add_subplot().plot(voltages)
                ax = figure.add_subplot()
                _ = ax.imshow(np.stack([r['v'] for r in self.history], axis=0), aspect='auto', interpolation='nearest')
                self.logger.record('voltages/figure', Figure(figure, close=True), exclude=('stdout', 'log', 'json', 'csv'))
                plt.close()
                
                # 记录wavefront变化的动画图
                wavefront_history = self.training_env.get_attr('history_wavefront')[0]
                if len(wavefront_history) > 0:
                    figure = plt.figure(figsize=(10, 6))
                    # 显示最新的几个wavefront图像
                    num_display = min(6, len(wavefront_history))
                    for i in range(num_display):
                        ax = figure.add_subplot(2, 3, i+1)
                        idx = -(num_display - i)  # 从倒数第num_display个开始显示
                        wf = wavefront_history[idx]
                        im = ax.imshow(wf, cmap='viridis')
                        ax.set_title(f'Step {len(wavefront_history) + idx}')
                        plt.colorbar(im, ax=ax)
                    self.logger.record('wavefront/evolution', Figure(figure, close=True), exclude=('stdout', 'log', 'json', 'csv'))
                    plt.close()
                
                max_iter = self.history[-1]['iter']
                self.logger.record('iter/value', max_iter)
                
                self.history = []
                self.best_rms = float('inf')
        return True


class LSTMEncoder(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_layers=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rnn = nn.LSTM(feature_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        
    def forward(self, x):
        x,_ = self.rnn(x)
        return x[:,-1,:]

    
class CustomCombineWFAndVectorExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        extractor = {}
        
        for key, subspace in observation_space.spaces.items():
            if key == 'wavefront':
                # 波前数据的特征提取器
                extractor[key] = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(subspace.shape[1] * subspace.shape[2], features_dim),
                    nn.ReLU())
            elif key == 'vector':
                extractor[key] = nn.Sequential(
                    LSTMEncoder(subspace.shape[1], 128, num_layers=1),
                    nn.Linear(128, features_dim),
                    nn.ReLU())
            elif key == 'rms_values':
                extractor[key] = nn.Sequential(
                    LSTMEncoder(subspace.shape[1], 4, num_layers=1),
                    nn.Linear(4, 1),
                    nn.ReLU())
        
        self.extractors = nn.ModuleDict(extractor)
        self._features_dim = 2*features_dim + 1

    def forward(self, observations: th.Tensor) -> th.Tensor:
        encoder_list = []
        for key, extractor in self.extractors.items():
            encoder_list.append(extractor(observations[key]))
        return th.cat(encoder_list, dim=1)


# 训练模式
train = True
device = 'cuda' if th.cuda.is_available() else 'cpu'

# 创建环境
env = WFSLaserCastEnv(render_mode='rgb_array', max_iter=200)

# 策略网络参数
policy_kwargs = dict(
    features_extractor_class=CustomCombineWFAndVectorExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)

# 创建SAC agent
agent = SAC('MultiInputPolicy', env, 
            verbose=1, device=device, tensorboard_log='logs/sac_tensorboard_wfs',
            use_sde=True, use_sde_at_warmup=True,
            learning_starts=200, buffer_size=3000, batch_size=100, learning_rate=0.004,
            policy_kwargs=policy_kwargs)

if train:
    # 回调函数
    callbacks = CallbackList([
        TensorboardCallback(),
        CheckpointCallback(save_freq=1000, save_path='ckpts/rl_wfs', save_replay_buffer=False, verbose=1)
    ])
    
    # 开始训练
    agent.learn(50000, progress_bar=True, callback=callbacks, log_interval=1)

    # 保存模型
    agent.save('wfs_laser_agent')
    agent.save_replay_buffer('wfs_laser_buffer')

else:
    # 加载最新的检查点
    ckpt_paths = glob('ckpts/rl_wfs/rl_model_*_steps.zip')
    if ckpt_paths:
        latest_ckpt_path = list(sorted(ckpt_paths, key=lambda x: os.path.getmtime(x)))[-1]
        print(latest_ckpt_path)
        agent.load(latest_ckpt_path, env=env)
        
        # 创建测试环境
        env = WFSLaserCastEnv(render_mode='ansi', max_iter=500, target_rms=0.05)
        obs,_ = env.reset()
        history = []
        
        s_time = time.time()
        while True:
            action,_ = agent.predict(obs, deterministic=True)
            obs, reward, done, trunk, info = env.step(action)
            history.append(info['RMS'])
            
            if done or trunk:
                time_cost = time.time() - s_time
                env.reset()
                plt.plot(history)
                plt.title(f'{trunk=} {done=}')
                plt.show()
                plt.close()
                break