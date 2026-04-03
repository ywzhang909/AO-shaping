
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from glob import glob
import time
import matplotlib.pyplot as plt

import numpy as np

import torch as th
from torch import nn
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.logger import Image, Figure
from gymnasium import spaces

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
env = LaserCastEnv(render_mode='rgb_array', target_power=8_000, max_iter=200, img_size=img_size, img_noise=True, r_bucket=5)
policy_kwargs = dict(
    features_extractor_class=CustomCombineImageAndVetorExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)
agent = SAC('MultiInputPolicy', env, 
            verbose=1, device=device, tensorboard_log='logs/sac_tensorboard',
            use_sde=True, use_sde_at_warmup=True,
            learning_starts=1_000, buffer_size=9_000, batch_size=128, learning_rate=0.008,
            policy_kwargs=policy_kwargs)

if train:
    ckpt_paths = glob('ckpts/rl/rl_model_*_steps.zip')
    if len(ckpt_paths) > 0:
        latest_ckpt_path = list(sorted(ckpt_paths, key=lambda x: os.path.getmtime(x)))[-1]
        print(latest_ckpt_path)
        agent = agent.load(latest_ckpt_path, env=env)
    callbacks = CallbackList([
        TensorboardCallback(),
        CheckpointCallback(save_freq=1000, save_path='ckpts/rl', save_replay_buffer=False, verbose=1)
    ])
    agent.learn(100_200, progress_bar=True, callback=callbacks, log_interval=1)
    # agent.save('laser_agent')

else:
    ckpt_paths = glob('ckpts/rl/rl_model_*_steps.zip')
    latest_ckpt_path = list(sorted(ckpt_paths, key=lambda x: os.path.getmtime(x)))[-1]
    print(latest_ckpt_path)
    agent.load(latest_ckpt_path, env=env)
    
    env = LaserCastEnv(render_mode='human', max_iter=200)
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
        
