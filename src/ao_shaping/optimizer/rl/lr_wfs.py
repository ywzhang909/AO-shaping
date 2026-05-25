from typing import Any
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
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.logger import Image, Figure

from ao_shaping.drivers import NlightDM, WFSManager
from ao_shaping.drivers.wfs.thorlab_wfs import MlaRes

import matplotlib.pyplot as plt
import numpy as np


class WFS():

    def __init__(self, pupil_diameter):
        self.wfs = WFSManager(MlaRes.Res768, use_custom_ref=True, high_speed=True, pupil_diameter=pupil_diameter)

    def calc_j(self):
        self.wfs.take_image()
        wf, statics = self.wfs.get_wavefront()
        return wf, statics


class LaserCastEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ansi', 'rgb_array']}
    '''
    激光环境。

    参数:
    
    
    '''
    def __init__(self, max_iter=2000, target_rms=0.01, action_delta=10, history_len:int=5, render_mode='human') -> None:
        '''
        '''

        super().__init__()

        self.wfs = WFS()
        self.dm = NlightDM(keep_when_exit=True)

        self.action_dim = self.dm.DM_Num-1

        self.history_len = history_len
        self.target_rms = target_rms
        self.max_iter = max_iter

        self.action_space:spaces.Box = spaces.Box(low=-action_delta, high=action_delta, shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "voltage": spaces.Box(low=-1, high=1, shape=(self.history_len, self.dm.DM_Num), dtype=np.float32),
            "image": spaces.Box(low=0, high=2*np.pi, shape=(self.history_len, *self.img_size), dtype=np.uint8),
            "rms": spaces.Box(low=0, high=2, shape=(self.history_len, 1), dtype=np.float32)
        })

        # display params
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def init_env(self) -> None:
        """
        初始化环境。

        返回:
            dict: 包含初始化后的观测值的字典。
        """
        # 初始化 DM 设备
        self.dm.initialize()
        # 设置初始电压为零
        self.v = np.random.rand(self.dm.DM_Num,) * (self.v_high-self.v_low) + self.v_low
        self.dm.send_voltages(self.v, 0.01)
        wavefront, stastistics = self.wfs.calc_j()

        self.history_votages = np.zeros((self.history_len, self.dm.DM_Num), dtype=np.float32)
        self.history_rms = np.ones((self.history_len, 1), dtype=np.float32) * stastistics['rms']
        self.history_wavefront = np.repeat(wavefront, self.history_len, axis=0)

        # 记录随机初始环境
        self.init_rms = self.last_rms.copy()
        self.init_voltages = deepcopy(self.last_voltage)

    def render(self)->Any:
        import pygame
        if self.window is None and self.render_mode == 'human':
            pygame.init()
            self.window = pygame.display.set_mode(self.img_size)
            self.clock = time.time()

        if self.render_mode == 'human':
            canvas = pygame.surfarray.make_surface(self.last_wavefront)
            h_r = self.r_bucket//2
            pygame.draw.rect(canvas, (255, 0, 0), (self.img_size[0]//2-h_r, self.img_size[1]//2-h_r, 2*h_r, 2*h_r), 1)
            pygame.display.set_caption(f'fps={self.iter / max(time.time()-self.clock, 1.0)} {self.step_info}')
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

        elif self.render_mode == 'ansi':
            gain = self.init_rms - self.init_rms
            output_str = f'{self.last_rms=} {gain=}'
            return output_str

        elif self.render_mode == 'rgb_array':
            return self.last_wavefront

    def step(self, action:np.ndarray)->tuple[dict, float, bool, bool, dict]:
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
        _v = self.v + dv
        forbiden_cond = any([
                not np.all(-180 < _v < 180),
                self.iter>=self.max_iter,
            ])
        if forbiden_cond:
            reward = -1e10
        else:
            self.dm.send_voltages(_v, 0.002)
            time_reward_bonus = (self.max_iter-self.iter) / self.max_iter
            wavefront, stastistics = self.wfs.calc_j()
            reward = -stastistics['rms']
            if stastistics['rms'] < self.target_rms:
                reward += time_reward_bonus
            self.wavefront = wavefront

        # 更新历史电压、功率和强度数组
        self.history_votages = np.roll(self.history_votages, -1, axis=0)
        self.history_votages[-1,:] = _v
        self.history_rms = np.roll(self.history_rms, -1, axis=0)
        self.history_rms[-1] = stastistics['rms']
        self.history_wavefront = np.roll(self.history_wavefront, -1, axis=0)
        self.history_wavefront[-1,:,:] = self.wavefront

        self.v = _v
        self.iter += 1

        return self.step_obs, reward, self.iter>=self.max_iter, forbiden_cond, self.step_info

    def reset(self, *, seed=None, options=None) -> tuple[Any, dict]:
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
    def last_voltage(self)->np.ndarray[float]:
        return self.history_votages[-1]

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

    @property
    def step_obs(self):
        scaled_voltages = (self.history_votages + self.v_low) / (self.v_high-self.v_low) - 0.5
        return {
            "voltage": scaled_voltages,
            "wavefront": self.history_wavefront,
            "rms": self.history_rms
        }

    @property
    def step_info(self)->dict:
        return {'rms':self.last_rms, 'iter': self.iter, 'v':self.v}


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
                power_history = [r['J'] for r in info[::-1]]
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
            if key == 'image':
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


train = True
device = 'cuda' if th.cuda.is_available() else 'cpu'
img_size = (192, 192)
env = LaserCastEnv(render_mode='rgb_array', max_iter=200, img_size=img_size, img_noise=True, r_bucket=5)
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

    env = LaserCastEnv(render_mode='human', max_iter=500, target_power=1.8e6)
    obs,_ = env.reset()
    env.render()
    history = []

    s_time = time.time()
    while True:
        action,_ = agent.predict(obs, deterministic=True)
        obs, reward, done, trunk, info = env.step(action)
        # info = info[0]
        # trunk = info['TimeLimit.truncated']
        history.append(info['J'])
        env.render()
        if done or trunk:
            time_cost = time.time() - s_time
            env.reset()
            plt.plot(history)
            plt.title(f'{trunk=} {done=}')
            plt.show()
            plt.close()

            break

