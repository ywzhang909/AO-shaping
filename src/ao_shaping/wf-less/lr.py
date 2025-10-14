from typing import Tuple, Any
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from glob import glob
import time
import matplotlib.pyplot as plt
import pygame
import numpy as np

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

from ao_shaping.drivers import CameraStreamManager, NlightDM

# TODO 添加近场图像

class LaserCastEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ansi', 'rgb_array']}
    '''
    激光投射环境。

    参数:
    
    
    '''
    def __init__(self, max_iter, target_power=10_000, r_bucket=5, img_size:Tuple[int,int]=(250,250), history_len:int=8, render_mode='human', img_noise:bool=False) -> None:
        super().__init__()
        
        self.cam = CameraStreamManager(cam_id=0, explosure_time=60, skip_sampling=False)
        self.dm = NlightDM(keep_when_exit=True)
        # 初始化 DM 设备
        # 初始化相机设备
        self.cam.initialize()
        center = (681, 410)
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
        c_x, c_y = self.center[0], self.center[1]
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
        if self.window ==None and self.render_mode == 'human':
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


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        
        self.history = []
        self.best_power = -1
    
    def _on_step(self, **kwargs):
        if self.best_power < 0:
            self.best_power = self.training_env.get_attr('init_power')[0]
             
        if self.n_calls >= 0:
            info:dict = self.locals["infos"][0]
            self.history.append(info)
            done = self.locals["dones"][0]
        
            # if info['J'] > self.best_power and self.best_power>0:
            #     obs = self.training_env.get_attr('init_obs')[0]
            #     action = self.training_env.get_attr('v')[0][1:]
            #     next_obs = self.training_env.get_attr('step_obs')[0]
            #     power = info['J']
            #     reward = np.sum(self.training_env.get_attr('wighted_mask')[0] * self.training_env.get_attr('img')[0])\
            #         + 10 * (self.training_env.get_attr('max_iter')[0])**2 * (power - self.training_env.get_attr('init_power')[0][0])
            #     self.model.replay_buffer.add(obs, next_obs, action, reward, done, {})
            # '''File "D:\Downloads\Libs\conda\py312\Lib\site-packages\stable_baselines3\common\buffers.py", line 640, in add
            #     self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])
            #     ~~~~~~~~~~~~~^^^^^^^^^^
            # ValueError: could not broadcast input array from shape (0,) into shape (1,)'''
            
            if done:
                cam_img = self.training_env.render(mode="rgb_array")[0,:,:]
                self.logger.record('intensity/image', Image(cam_img, 'HW'), exclude=('stdout', 'log', 'json', 'csv'))
                
                power = self.history[-1]['J']
                figure = plt.figure()
                power_history = [r['J'] for r in self.history]
                figure.add_subplot().plot(power_history)
                self.logger.record('power/value', power)
                self.logger.record('power/mean', np.mean(power_history))
                self.logger.record('power/figure', Figure(figure, True), exclude=('stdout', 'log', 'json', 'csv'))
                plt.close()
                
                voltages = self.training_env.get_attr('v')[0]
                figure = plt.figure()
                figure.add_subplot().bar(np.arange(voltages.shape[0]), voltages)
                ax = figure.add_subplot()
                _ = ax.imshow(np.stack([r['v'] for r in self.history], axis=0), aspect='auto', interpolation='nearest')
                self.logger.record('voltages/figure', Figure(figure, close=True), exclude=('stdout', 'log', 'json', 'csv'))
                plt.close()
                
                max_iter = self.history[-1]['iter']
                self.logger.record('iter/value', max_iter)
                
                self.history = []
                self.best_power = -1
        return True
    
    def _on_rollout_end(self):
        return super()._on_rollout_end()
    
    
class BinomialActionNoise(ActionNoise):
    def __init__(self, sigma:float, sample_len:int, dtype:np.dtype) -> None:
        self._sigma = sigma
        self._dtype = dtype
        self._sample_len = sample_len
        super().__init__()

    def __call__(self) -> np.ndarray:
        return np.array([np.random.binomial(1, 0.5, (self._sample_len,))*2.0-1]).astype(self._dtype)
    
    
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
            elif key == 'vector':
                extractor[key] = nn.Sequential(
                    LSTMEncoder(subspace.shape[1], 128, num_layers=1),
                    nn.Linear(128, features_dim),
                    nn.ReLU())
            elif key == 'powers':
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
env = LaserCastEnv(render_mode='rgb_array', target_power=10_000, max_iter=200, img_size=img_size, img_noise=True, r_bucket=5)
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
    ckpt_paths = glob('ckpts/rl/rl_model_*_steps.zip')
    if len(ckpt_paths) > 0:
        latest_ckpt_path = list(sorted(ckpt_paths, key=lambda x: os.path.getmtime(x)))[-1]
        print(latest_ckpt_path)
        agent.load(latest_ckpt_path, env=env)
    callbacks = CallbackList([
        TensorboardCallback(),
        CheckpointCallback(save_freq=1000, save_path='ckpts/rl', save_replay_buffer=False, verbose=1)
    ])
    agent.learn(10_200, progress_bar=True, callback=callbacks, log_interval=1)
    # agent.save('laser_agent')

else:
    ckpt_paths = glob('ckpts/rl/rl_model_*_steps.zip')
    latest_ckpt_path = list(sorted(ckpt_paths, key=lambda x: os.path.getmtime(x)))[-1]
    print(latest_ckpt_path)
    agent.load(latest_ckpt_path, env=env)
    
    env = LaserCastEnv(render_mode='human', max_iter=500)
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
        
