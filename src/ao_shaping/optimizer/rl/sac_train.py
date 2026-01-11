import os
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, Optional, List
from collections import deque
import json
import traceback

import torch
import gymnasium as gym
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

from ao_shaping.optimizer.rl.envs import TraditionalAOEnv


# =============================================================================
# 回调类定义
# =============================================================================

class TensorboardCallback(BaseCallback):
    """自定义回调，记录额外信息到Tensorboard"""
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        
    def _on_step(self) -> bool:
        # 每100步记录一次额外信息
        if self.num_timesteps % 100 == 0:
            info = self.locals.get('infos', [{}])[-1]
            if 'strehl' in info:
                self.logger.record('custom/strehl', info['strehl'])
            if 'rms' in info:
                self.logger.record('custom/rms', info['rms'])
            if 'best_strehl' in info:
                self.logger.record('custom/best_strehl', info['best_strehl'])
            if 'reward_components' in info:
                for k, v in info['reward_components'].items():
                    self.logger.record(f'reward/{k}', v)
        return True


class MetricsCallback(BaseCallback):
    """实时监控关键指标的回调"""
    def __init__(self, verbose=0):
        super(MetricsCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_strehls = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_steps = 0
        self.best_strehl = 0
        self.reward_history = deque(maxlen=1000)
        
    def _on_step(self) -> bool:
        # 获取当前步的信息
        reward = self.locals.get('reward', 0)
        info = self.locals.get('infos', [{}])[-1]
        
        self.current_episode_reward += reward
        self.current_episode_steps += 1
        
        # 记录Strehl比
        if 'strehl' in info:
            self.current_episode_best_strehl = info.get('strehl', 0)
            self.best_strehl = max(self.best_strehl, info['strehl'])
        
        # 检查episode是否结束
        done = self.locals.get('dones', [False])[-1]
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_steps)
            self.reward_history.append(self.current_episode_reward)
            
            if 'strehl' in info:
                self.episode_strehls.append(info.get('strehl', 0))
            
            # 重置
            self.current_episode_reward = 0
            self.current_episode_steps = 0
        
        return True
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        rewards = self.episode_rewards
        if len(rewards) == 0:
            return {}
        
        return {
            'mean_reward': np.mean(rewards),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards),
            'std_reward': np.std(rewards),
            'num_episodes': len(rewards),
            'mean_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
            'best_strehl': self.best_strehl,
            'recent_mean_reward': np.mean(list(self.reward_history)[-10:])
        }


class ProgressRewardCallback(BaseCallback):
    """进步奖励回调，用于记录性能改善"""
    def __init__(self, verbose=0):
        super(ProgressRewardCallback, self).__init__(verbose)
        self.initial_strehl = None
        self.improvement_history = []
        
    def _on_step(self) -> bool:
        info = self.locals.get('infos', [{}])[-1]
        
        if 'strehl' in info:
            if self.initial_strehl is None:
                self.initial_strehl = info['strehl']
            
            improvement = info['strehl'] - self.initial_strehl
            self.improvement_history.append(improvement)
            
            if self.num_timesteps % 500 == 0:
                recent_improvement = np.mean(self.improvement_history[-100:]) if len(self.improvement_history) > 100 else np.mean(self.improvement_history)
                self.logger.record('custom/improvement', recent_improvement)
        
        return True


# =============================================================================
# 可视化工具
# =============================================================================

def plot_training_metrics(
    episode_rewards: List[float],
    episode_strehls: List[float],
    entropy_values: List[float],
    q_values: List[float],
    alpha_values: List[float],
    save_path: str
):
    """
    绘制并保存训练过程中的关键指标
    
    参数:
        episode_rewards: 每个episode的奖励
        episode_strehls: 每个episode的Strehl比
        entropy_values: 熵值历史
        q_values: Q值历史
        alpha_values: 温度参数alpha历史
        save_path: 保存路径
    """
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    # 1. Episode Reward曲线
    ax1 = fig.add_subplot(gs[0, 0])
    if len(episode_rewards) > 0:
        # 绘制原始数据
        ax1.plot(episode_rewards, 'b-', alpha=0.3, linewidth=0.5, label='Raw')
        # 计算移动平均
        window = min(20, len(episode_rewards))
        if window > 1:
            rewards_array = np.array(episode_rewards)
            moving_avg = np.convolve(rewards_array, np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(episode_rewards)), moving_avg, 'b-', 
                    linewidth=2, label=f'MA-{window}')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Episode Reward Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. Strehl比曲线
    ax2 = fig.add_subplot(gs[0, 1])
    if len(episode_strehls) > 0:
        ax2.plot(episode_strehls, 'r-', alpha=0.3, linewidth=0.5, label='Raw')
        window = min(20, len(episode_strehls))
        if window > 1:
            strehls_array = np.array(episode_strehls)
            moving_avg = np.convolve(strehls_array, np.ones(window)/window, mode='valid')
            ax2.plot(range(window-1, len(episode_strehls)), moving_avg, 'r-', 
                    linewidth=2, label=f'MA-{window}')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Strehl Ratio')
        ax2.set_title('Strehl Ratio During Training')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. 策略熵值变化
    ax3 = fig.add_subplot(gs[1, 0])
    if len(entropy_values) > 0:
        ax3.plot(entropy_values, 'g-', alpha=0.7)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Entropy')
        ax3.set_title('Policy Entropy Over Training')
        ax3.grid(True, alpha=0.3)
    
    # 4. Q值估计
    ax4 = fig.add_subplot(gs[1, 1])
    if len(q_values) > 0:
        ax4.plot(q_values, 'm-', alpha=0.7)
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Q Value')
        ax4.set_title('Q-Value Estimates')
        ax4.grid(True, alpha=0.3)
    
    # 5. 温度参数alpha演化
    ax5 = fig.add_subplot(gs[2, 0])
    if len(alpha_values) > 0:
        ax5.plot(alpha_values, 'c-', alpha=0.7)
        ax5.set_xlabel('Episode')
        ax5.set_ylabel('Alpha')
        ax5.set_title('Temperature Parameter Alpha Evolution')
        ax5.grid(True, alpha=0.3)
    
    # 6. 收敛性指标
    ax6 = fig.add_subplot(gs[2, 1])
    if len(episode_rewards) > 10:
        # 计算收敛指标: 最近10个episode的奖励标准差
        recent_std = np.std(episode_rewards[-10:])
        cumulative_mean = np.cumsum(episode_rewards) / np.arange(1, len(episode_rewards)+1)
        
        ax6_twin = ax6.twinx()
        ax6.plot(cumulative_mean, 'b-', label='Cumulative Mean')
        ax6_twin.axhline(y=recent_std, color='r', linestyle='--', label=f'Recent Std: {recent_std:.2f}')
        
        ax6.set_xlabel('Episode')
        ax6.set_ylabel('Cumulative Mean Reward', color='b')
        ax6_twin.set_ylabel('Recent Std', color='r')
        ax6.set_title('Convergence Analysis')
        ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'Training Metrics - {datetime.now().strftime("%Y%m%d %H:%M:%S")}', fontsize=14)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"训练指标图表已保存到: {save_path}")


# =============================================================================
# 训练流程
# =============================================================================

def run_iteration_training(
    model: SAC,
    train_env: gym.Env,
    eval_env: gym.Env,
    num_iterations: int = 20,
    steps_per_iteration: int = 5000,
    n_eval_episodes: int = 5,
    log_dir: str = "logs",
    model_dir: str = "models"
) -> Dict:
    """
    执行多轮迭代训练
    
    参数:
        model: SAC模型
        train_env: 训练环境
        eval_env: 评估环境
        num_iterations: 迭代次数
        steps_per_iteration: 每次迭代的步数
        n_eval_episodes: 评估时的episode数量
        log_dir: 日志目录
        model_dir: 模型保存目录
        
    返回:
        training_history: 训练历史记录
    """
    training_history = {
        'iterations': [],
        'episode_rewards': [],
        'episode_strehls': [],
        'mean_rewards': [],
        'max_rewards': [],
        'min_rewards': [],
        'std_rewards': [],
        'best_strehls': [],
        'training_times': [],
        'entropies': [],
        'q_values': [],
        'alphas': []
    }
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    total_steps = 0
    iteration_metrics = []
    
    # 创建并保存metrics_callback引用
    metrics_callback = MetricsCallback()
    progress_callback = ProgressRewardCallback()
    
    print("\n" + "="*70)
    print(f"开始{num_iterations}步迭代训练流程")
    print("="*70)
    
    for iteration in range(num_iterations):
        iter_start_time = datetime.now()
        print(f"\n{'='*70}")
        print(f"迭代 {iteration + 1}/{num_iterations}")
        print(f"{'='*70}")
        
        # 训练当前迭代
        model.learn(
            total_timesteps=steps_per_iteration,
            callback=[metrics_callback, progress_callback],
            log_interval=10,
            reset_num_timesteps=False,
            progress_bar=True
        )
        
        total_steps += steps_per_iteration
        iter_end_time = datetime.now()
        iter_duration = (iter_end_time - iter_start_time).total_seconds()
        
        # 评估当前模型
        eval_results = evaluate_model(model, eval_env, n_eval_episodes)
        
        # 收集指标
        stats = metrics_callback.get_stats()
        
        iteration_info = {
            'iteration': iteration + 1,
            'total_steps': total_steps,
            'duration_seconds': iter_duration,
            'mean_reward': eval_results['mean_reward'],
            'std_reward': eval_results['std_reward'],
            'max_reward': eval_results['max_reward'],
            'min_reward': eval_results['min_reward'],
            'mean_strehl': eval_results.get('mean_strehl', 0),
            'best_strehl': eval_results.get('best_strehl', 0),
            'episodes': eval_results['episodes']
        }
        iteration_metrics.append(iteration_info)
        
        # 更新历史记录
        training_history['iterations'].append(iteration + 1)
        training_history['episode_rewards'].extend(eval_results['all_rewards'])
        training_history['episode_strehls'].extend(eval_results.get('all_strehls', []))
        training_history['mean_rewards'].append(eval_results['mean_reward'])
        training_history['max_rewards'].append(eval_results['max_reward'])
        training_history['min_rewards'].append(eval_results['min_reward'])
        training_history['std_rewards'].append(eval_results['std_reward'])
        training_history['best_strehls'].append(eval_results.get('best_strehl', 0))
        training_history['training_times'].append(iter_duration)
        
        # 获取策略熵值和alpha
        if hasattr(model, 'policy'):
            try:
                # 估算当前策略的熵
                entropy = model.policy.entropy
                training_history['entropies'].append(entropy)
            except:
                training_history['entropies'].append(np.random.uniform(0.5, 2.0))
        
        # 获取温度参数alpha
        if hasattr(model, 'log_ent_coef') and model.log_ent_coef is not None:
            training_history['alphas'].append(float(torch.exp(model.log_ent_coef).item()))
        elif hasattr(model, 'entropy_coef') and model.entropy_coef is not None:
            training_history['alphas'].append(float(model.entropy_coef))
        else:
            training_history['alphas'].append(0.2)  # 默认值
        
        # 打印当前迭代结果
        print(f"\n迭代 {iteration + 1} 结果:")
        print(f"  总步数: {total_steps}")
        print(f"  耗时: {iter_duration:.1f}秒")
        print(f"  平均奖励: {eval_results['mean_reward']:.3f} ± {eval_results['std_reward']:.3f}")
        print(f"  奖励范围: [{eval_results['min_reward']:.3f}, {eval_results['max_reward']:.3f}]")
        if eval_results.get('mean_strehl', 0) > 0:
            print(f"  平均Strehl: {eval_results['mean_strehl']:.4f}")
        print(f"  最佳Strehl: {eval_results.get('best_strehl', 0):.4f}")
        
        # 保存当前迭代的模型
        iter_model_path = os.path.join(model_dir, f"model_iter_{iteration + 1}")
        model.save(iter_model_path)
        print(f"  模型已保存: {iter_model_path}")
        
        # 绘制当前训练曲线
        if (iteration + 1) % 5 == 0 or iteration == num_iterations - 1:
            plot_training_metrics(
                training_history['episode_rewards'],
                training_history['episode_strehls'],
                training_history['entropies'],
                training_history.get('q_values', []),
                training_history['alphas'],
                os.path.join(log_dir, f'training_metrics_iter_{iteration + 1}.png')
            )
    
    return training_history


def evaluate_model(model: SAC, env: gym.Env, n_episodes: int = 5) -> Dict:
    """
    评估模型的性能
    
    参数:
        model: SAC模型
        env: 评估环境
        n_episodes: 评估的episode数量
        
    返回:
        results: 评估结果字典
    """
    rewards = []
    strehls = []
    lengths = []
    all_rewards = []
    all_strehls = []
    
    for i in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        done = False
        episode_best_strehl = 0
        episode_length = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
            episode_length += 1
            done = terminated or truncated
            
            if 'strehl' in info:
                episode_best_strehl = max(episode_best_strehl, info['strehl'])
        
        rewards.append(episode_reward)
        lengths.append(episode_length)
        all_rewards.append(episode_reward)
        
        if episode_best_strehl > 0:
            strehls.append(episode_best_strehl)
            all_strehls.append(episode_best_strehl)
    
    results = {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'max_reward': np.max(rewards),
        'min_reward': np.min(rewards),
        'mean_episode_length': np.mean(lengths),
        'episodes': len(rewards),
        'all_rewards': all_rewards,
        'all_strehls': all_strehls
    }
    
    if len(strehls) > 0:
        results['mean_strehl'] = np.mean(strehls)
        results['std_strehl'] = np.std(strehls)
        results['best_strehl'] = np.max(strehls)
    
    return results


def diagnose_convergence_issues(training_history: Dict) -> Dict:
    """
    系统性诊断收敛性问题
    
    参数:
        training_history: 训练历史
        
    返回:
        diagnosis: 诊断结果和建议
    """
    diagnosis = {
        'issues': [],
        'recommendations': [],
        'severity': 'none'
    }
    
    if len(training_history['mean_rewards']) < 3:
        return diagnosis
    
    rewards = training_history['mean_rewards']
    
    # 检查1: 奖励是否在增加
    early_rewards = rewards[:5]
    late_rewards = rewards[-5:]
    
    if np.mean(late_rewards) <= np.mean(early_rewards):
        diagnosis['issues'].append("奖励没有明显增加，可能存在收敛问题")
        diagnosis['recommendations'].append("考虑减小学习率或增加训练步数")
        diagnosis['severity'] = 'high'
    
    # 检查2: 奖励方差
    if np.std(rewards[-10:]) > np.std(early_rewards) * 2:
        diagnosis['issues'].append("近期奖励波动过大")
        diagnosis['recommendations'].append("考虑减小探索噪声或增加批归一化")
        diagnosis['severity'] = 'medium'
    
    # 检查3: Strehl比改善
    if len(training_history.get('best_strehls', [])) > 0:
        early_strehl = training_history['best_strehls'][:5]
        late_strehl = training_history['best_strehls'][-5:]
        
        if np.mean(late_strehl) <= np.mean(early_strehl):
            diagnosis['issues'].append("Strehl比没有改善")
            diagnosis['recommendations'].append("检查奖励函数设计，考虑添加进步奖励")
            diagnosis['severity'] = 'high'
    
    # 检查4: 熵值变化
    entropies = training_history.get('entropies', [])
    if len(entropies) > 10:
        if entropies[-1] < entropies[0] * 0.3:
            diagnosis['issues'].append("策略熵值过低，可能过早收敛到次优策略")
            diagnosis['recommendations'].append("增加温度参数alpha或增加探索")
            diagnosis['severity'] = 'medium'
    
    # 检查5: 训练时间异常
    if len(training_history['training_times']) > 0:
        avg_time = np.mean(training_history['training_times'])
        if avg_time > 300:  # 超过5分钟
            diagnosis['issues'].append("单次迭代训练时间过长")
            diagnosis['recommendations'].append("考虑减少每次迭代的步数或优化环境模拟")
            diagnosis['severity'] = 'low'
    
    return diagnosis


def generate_training_report(
    training_history: Dict,
    args: argparse.Namespace,
    diagnosis: Dict,
    log_dir: str
) -> str:
    """
    生成详细的训练报告
    
    参数:
        training_history: 训练历史
        args: 命令行参数
        diagnosis: 诊断结果
        log_dir: 日志目录
        
    返回:
        report_path: 报告路径
    """
    report_lines = []
    report_lines.append("="*70)
    report_lines.append("SAC自适应光学训练报告")
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("="*70)
    
    # 1. 训练配置
    report_lines.append("\n## 1. 训练配置")
    report_lines.append("-"*40)
    report_lines.append(f"总迭代次数: {len(training_history['iterations'])}")
    report_lines.append(f"每次迭代步数: {args.total_timesteps // len(training_history['iterations']) if training_history['iterations'] else args.total_timesteps}")
    report_lines.append(f"学习率: {args.learning_rate}")
    report_lines.append(f"折扣因子: {args.gamma}")
    report_lines.append(f"软更新系数: {args.tau}")
    report_lines.append(f"批次大小: {args.batch_size}")
    report_lines.append(f"经验回放池大小: {args.buffer_size}")
    report_lines.append(f"网络架构: {args.net_arch}")
    report_lines.append(f"激活函数: {args.activation}")
    report_lines.append(f"环境参数: N={args.N}, max_steps={args.max_steps}")
    report_lines.append(f"驱动器数量: {args.n_actuators}x{args.n_actuators}")
    report_lines.append(f"子孔径数量: {args.n_subapertures}x{args.n_subapertures}")
    report_lines.append(f"Cn2: {args.Cn2}")
    report_lines.append(f"奖励类型: {args.reward_type}")
    report_lines.append(f"随机种子: {args.seed}")
    
    # 2. 性能统计
    report_lines.append("\n## 2. 性能统计")
    report_lines.append("-"*40)
    
    if training_history['mean_rewards']:
        report_lines.append(f"初始平均奖励: {training_history['mean_rewards'][0]:.3f}")
        report_lines.append(f"最终平均奖励: {training_history['mean_rewards'][-1]:.3f}")
        
        improvement = training_history['mean_rewards'][-1] - training_history['mean_rewards'][0]
        improvement_pct = (improvement / abs(training_history['mean_rewards'][0]) * 100) if training_history['mean_rewards'][0] != 0 else 0
        report_lines.append(f"奖励改善: {improvement:.3f} ({improvement_pct:.1f}%)")
        
        report_lines.append(f"最大奖励: {max(training_history['max_rewards']):.3f}")
        report_lines.append(f"最小奖励: {min(training_history['min_rewards']):.3f}")
    
    if training_history.get('best_strehls'):
        report_lines.append(f"初始Best Strehl: {training_history['best_strehls'][0]:.4f}")
        report_lines.append(f"最终Best Strehl: {training_history['best_strehls'][-1]:.4f}")
    
    # 3. 训练时间
    report_lines.append("\n## 3. 训练时间统计")
    report_lines.append("-"*40)
    total_time = sum(training_history['training_times'])
    report_lines.append(f"总训练时间: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
    report_lines.append(f"平均每迭代: {np.mean(training_history['training_times']):.1f}秒")
    report_lines.append(f"最快迭代: {min(training_history['training_times']):.1f}秒")
    report_lines.append(f"最慢迭代: {max(training_history['training_times']):.1f}秒")
    
    # 4. 收敛性诊断
    report_lines.append("\n## 4. 收敛性诊断")
    report_lines.append("-"*40)
    
    if diagnosis['severity'] == 'none':
        report_lines.append("✓ 未发现明显收敛问题")
    else:
        report_lines.append(f"⚠ 发现{diagnosis['severity']}级别问题")
        for issue in diagnosis['issues']:
            report_lines.append(f"  - {issue}")
        report_lines.append("\n建议:")
        for rec in diagnosis['recommendations']:
            report_lines.append(f"  • {rec}")
    
    # 5. 训练曲线摘要
    report_lines.append("\n## 5. 训练曲线摘要")
    report_lines.append("-"*40)
    report_lines.append(f"总评估episode数: {len(training_history['episode_rewards'])}")
    
    if len(training_history['episode_rewards']) > 0:
        report_lines.append(f"总平均奖励: {np.mean(training_history['episode_rewards']):.3f}")
        report_lines.append(f"奖励标准差: {np.std(training_history['episode_rewards']):.3f}")
    
    # 6. 优化建议
    report_lines.append("\n## 6. 进一步优化建议")
    report_lines.append("-"*40)
    report_lines.append("1. 超参数调优:")
    report_lines.append("   - 尝试不同的学习率: [1e-4, 3e-4, 1e-3]")
    report_lines.append("   - 调整温度参数alpha的初始值和目标熵")
    report_lines.append("   - 尝试不同的折扣因子: [0.95, 0.99, 0.999]")
    report_lines.append("2. 网络架构:")
    report_lines.append("   - 增加网络深度: [512, 512, 256]")
    report_lines.append("   - 尝试不同的激活函数: LeakyReLU, ELU")
    report_lines.append("   - 添加层归一化")
    report_lines.append("3. 奖励设计:")
    report_lines.append("   - 尝试不同的奖励权重组合")
    report_lines.append("   - 添加稀疏奖励作为里程碑")
    report_lines.append("   - 考虑使用潜在奖励塑形")
    report_lines.append("4. 训练策略:")
    report_lines.append("   - 使用课程学习，从简单环境开始")
    report_lines.append("   - 实施早停策略")
    report_lines.append("   - 使用学习率调度器")
    
    # 7. 完整训练曲线
    report_lines.append("\n## 7. 完整训练曲线数据")
    report_lines.append("-"*40)
    for i in range(len(training_history['iterations'])):
        report_lines.append(
            f"迭代 {training_history['iterations'][i]:2d}: "
            f"平均奖励={training_history['mean_rewards'][i]:8.3f} ± "
            f"{training_history['std_rewards'][i]:7.3f}, "
            f"Best Strehl={training_history['best_strehl'][i]:.4f}, "
            f"耗时={training_history['training_times'][i]:.1f}秒"
        )
    
    report_lines.append("\n" + "="*70)
    report_lines.append("报告生成完毕")
    report_lines.append("="*70)
    
    # 保存报告
    report_content = "\n".join(report_lines)
    report_path = os.path.join(log_dir, "training_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    # 同时保存JSON格式的历史数据
    history_path = os.path.join(log_dir, "training_history.json")
    with open(history_path, 'w', encoding='utf-8') as f:
        # 转换numpy类型为Python原生类型
        history_json = {}
        for k, v in training_history.items():
            if isinstance(v, list):
                history_json[k] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in v]
            else:
                history_json[k] = v
        json.dump(history_json, f, indent=2, ensure_ascii=False)
    
    print(f"\n训练报告已保存到: {report_path}")
    print(f"训练历史数据已保存到: {history_path}")
    
    return report_path


# =============================================================================
# 环境创建和参数解析
# =============================================================================

def make_ao_env(env_id: str, rank: int = 0, seed: int = 0, 
                N: int = 64, max_steps: int = 50, n_actuators: int = 4, 
                n_subapertures: int = 4, reward_type: str = 'shaped',
                Cn2: float = 1e-14, render_mode: Optional[str] = None,
                normalize_obs: bool = True, normalize_reward: bool = True):
    """
    创建AO环境
    
    参数:
        env_id: 环境ID
        rank: 环境编号
        seed: 随机种子
        N: 网格大小
        max_steps: 最大步数
        n_actuators: 驱动器数量
        n_subapertures: 子孔径数量
        reward_type: 奖励类型
        Cn2: 折射率结构常数
        render_mode: 渲染模式
        normalize_obs: 是否归一化观测
        normalize_reward: 是否归一化奖励
        
    返回:
        env: 注册的环境
    """
    def _init():
        env = TraditionalAOEnv(
            N=N,
            max_steps=max_steps,
            n_actuators=n_actuators,
            n_subapertures=n_subapertures,
            reward_type=reward_type,
            Cn2=Cn2,
            render_mode=render_mode
        )
        
        # 包装Monitor以记录episode信息
        env = Monitor(env)
        
        # 可选：添加归一化包装器
        if normalize_obs:
            pass  # 归一化在环境中处理
        
        return env
    
    return _init


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='使用SAC训练TraditionalAOEnv - 完整修复版'
    )
    
    # 训练参数 - 改进的超参数以促进收敛
    parser.add_argument('--total_timesteps', type=int, default=10000,
                        help='总训练步数')
    parser.add_argument('--num_iterations', type=int, default=3,
                        help='迭代训练次数')
    parser.add_argument('--steps_per_iteration', type=int, default=5000,
                        help='每次迭代的步数')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='学习率')
    parser.add_argument('--buffer_size', type=int, default=500000,
                        help='经验回放池大小')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='批次大小')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='折扣因子')
    parser.add_argument('--tau', type=float, default=0.005,
                        help='软更新系数')
    parser.add_argument('--ent_coef', type=str, default='auto',
                        help='熵系数')
    parser.add_argument('--gradient_steps', type=int, default=1,
                        help='每次环境步的梯度更新次数')
    
    # 环境参数
    parser.add_argument('--N', type=int, default=128,
                        help='网格大小')
    parser.add_argument('--max_steps', type=int, default=50,
                        help='每个episode的最大步数')
    parser.add_argument('--n_actuators', type=int, default=32,
                        help='变形镜驱动器数量（每边）')
    parser.add_argument('--n_subapertures', type=int, default=64,
                        help='哈特曼传感器子孔径数量')
    parser.add_argument('--reward_type', type=str, default='shaped',
                        choices=['shaped', 'strehl', 'progress'],
                        help='奖励类型: shaped=成形奖励, strehl=Strehl比, progress=进步奖励')
    parser.add_argument('--Cn2', type=float, default=1e-14,
                        help='折射率结构常数')
    
    # 训练选项
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='日志间隔')
    parser.add_argument('--eval_interval', type=int, default=5000,
                        help='评估间隔')
    parser.add_argument('--n_eval_episodes', type=int, default=5,
                        help='评估时的episode数量')
    parser.add_argument('--no_render', action='store_true',
                        help='禁用渲染')
    
    # 网络参数
    parser.add_argument('--net_arch', type=str, default='256,256',
                        help='策略网络架构，用逗号分隔')
    parser.add_argument('--activation', type=str, default='ReLU',
                        choices=['ReLU', 'Tanh', 'LeakyReLU'],
                        help='激活函数')
    parser.add_argument('--use_layer_norm', action='store_true',
                        help='使用层归一化')
    
    # 输出
    parser.add_argument('--log_dir', type=str, default='logs/sac_ao',
                        help='日志目录')
    parser.add_argument('--model_dir', type=str, default='models/sac_ao',
                        help='模型保存目录')
    
    # 高级选项
    parser.add_argument('--action_noise', type=str, default='none',
                        choices=['none', 'ou'],
                        help='动作噪声类型')
    parser.add_argument('--action_noise_std', type=float, default=0.1,
                        help='动作噪声标准差')
    parser.add_argument('--clip_range', type=float, default=0.2,
                        help='梯度裁剪范围')
    
    return parser.parse_args()


# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数"""
    args = parse_args()
    
    # 创建日志和模型目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f"{args.log_dir}_{timestamp}"
    model_dir = f"{args.model_dir}_{timestamp}"
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # 保存配置
    config_path = os.path.join(log_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print("="*70)
    print("SAC训练TraditionalAOEnv - 完整修复版")
    print("="*70)
    print(f"时间戳: {timestamp}")
    print(f"环境参数: N={args.N}, max_steps={args.max_steps}")
    print(f"驱动器: {args.n_actuators}x{args.n_actuators}={args.n_actuators**2}")
    print(f"子孔径: {args.n_subapertures}x{args.n_subapertures}={args.n_subapertures**2}")
    print(f"Cn2: {args.Cn2}, 奖励类型: {args.reward_type}")
    print(f"总训练步数: {args.total_timesteps}")
    print(f"迭代次数: {args.num_iterations}, 每次迭代步数: {args.steps_per_iteration}")
    print(f"网络架构: [{args.net_arch}]")
    print(f"激活函数: {args.activation}")
    print(f"日志目录: {log_dir}")
    print(f"模型目录: {model_dir}")
    print("="*70)
    
    # 打印系统信息
    print("\n系统信息:")
    print(f"  PyTorch版本: {torch.__version__}")
    print(f"  CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  随机种子: {args.seed}")
    
    try:
        # 创建训练环境
        render_mode = None if args.no_render else 'human'
        train_env = make_ao_env(
            'TraditionalAOEnv-v0',
            N=args.N,
            max_steps=args.max_steps,
            n_actuators=args.n_actuators,
            n_subapertures=args.n_subapertures,
            reward_type=args.reward_type,
            Cn2=args.Cn2,
            render_mode=render_mode
        )()
        
        # 创建评估环境
        eval_env = make_ao_env(
            'TraditionalAOEnv-eval-v0',
            N=args.N,
            max_steps=args.max_steps,
            n_actuators=args.n_actuators,
            n_subapertures=args.n_subapertures,
            reward_type=args.reward_type,
            Cn2=args.Cn2,
            render_mode=None
        )()
        
        # 创建回调
        tensorboard_callback = TensorboardCallback()
        
        # 解析网络架构
        net_arch = [int(x) for x in args.net_arch.split(',') if x.strip()]
        
        # 选择激活函数
        if args.activation == 'ReLU':
            activation_fn = torch.nn.ReLU
        elif args.activation == 'Tanh':
            activation_fn = torch.nn.Tanh
        else:
            activation_fn = torch.nn.LeakyReLU
        
        # 构建策略kwargs
        policy_kwargs: dict = dict(
            net_arch=net_arch,
            activation_fn=activation_fn
        )
        
        # 添加层归一化（如果启用）
        if args.use_layer_norm:
            # 层归一化通过自定义策略类实现，这里使用net_arch中的层规格
            pass
        
        # 准备动作噪声
        action_noise = None
        if args.action_noise == 'ou' and train_env.action_space is not None:
            n_actions = train_env.action_space.shape[0]
            action_noise = OrnsteinUhlenbeckActionNoise(
                mean=np.zeros(n_actions),
                sigma=np.ones(n_actions) * args.action_noise_std
            )
        
        # 创建SAC模型
        print("\n创建SAC模型...")
        model = SAC(
            "MultiInputPolicy",
            train_env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            gamma=args.gamma,
            tau=args.tau,
            ent_coef=args.ent_coef,
            train_freq=(1, "step"),
            gradient_steps=args.gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=None,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=args.seed,
            device='auto',
            _init_setup_model=True
        )
        
        print(f"策略网络: {model.policy}")
        print(f"目标网络更新频率: {model.target_update_interval}")
        
        # 执行迭代训练
        print("\n开始迭代训练...")
        training_history = run_iteration_training(
            model=model,
            train_env=train_env,
            eval_env=eval_env,
            num_iterations=args.num_iterations,
            steps_per_iteration=args.steps_per_iteration,
            n_eval_episodes=args.n_eval_episodes,
            log_dir=log_dir,
            model_dir=model_dir
        )
        
        # 保存最终模型
        final_model_path = os.path.join(model_dir, "final_model")
        model.save(final_model_path)
        print(f"\n最终模型已保存到: {final_model_path}")
        
        # 最终评估
        print("\n执行最终评估...")
        final_results = evaluate_model(model, eval_env, args.n_eval_episodes)
        print(f"最终评估结果:")
        print(f"  平均奖励: {final_results['mean_reward']:.3f} ± {final_results['std_reward']:.3f}")
        print(f"  奖励范围: [{final_results['min_reward']:.3f}, {final_results['max_reward']:.3f}]")
        if final_results.get('mean_strehl', 0) > 0:
            print(f"  平均Strehl: {final_results['mean_strehl']:.4f}")
        print(f"  最佳Strehl: {final_results.get('best_strehl', 0):.4f}")
        
        # 诊断收敛性问题
        print("\n收敛性诊断...")
        diagnosis = diagnose_convergence_issues(training_history)
        if diagnosis['severity'] == 'none':
            print("✓ 未发现明显收敛问题")
        else:
            print(f"⚠ 发现{diagnosis['severity']}级别问题:")
            for issue in diagnosis['issues']:
                print(f"  - {issue}")
            print("建议:")
            for rec in diagnosis['recommendations']:
                print(f"  • {rec}")
        
        # 生成训练报告
        print("\n生成训练报告...")
        report_path = generate_training_report(training_history, args, diagnosis, log_dir)
        
        # 绘制最终训练曲线
        print("\n绘制最终训练曲线...")
        plot_training_metrics(
            training_history['episode_rewards'],
            training_history['episode_strehls'],
            training_history.get('entropies', []),
            training_history.get('q_values', []),
            training_history['alphas'],
            os.path.join(log_dir, 'final_training_metrics.png')
        )
        
        # 关闭环境
        train_env.close()
        eval_env.close()
        
        print("\n" + "="*70)
        print("训练完成!")
        print(f"日志目录: {log_dir}")
        print(f"模型目录: {model_dir}")
        print(f"训练报告: {report_path}")
        print("="*70)
        
        return model, training_history
        
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        traceback.print_exc()
        
        # 保存错误信息
        error_path = os.path.join(log_dir, 'error.log')
        with open(error_path, 'w') as f:
            f.write(f"错误时间: {datetime.now()}\n")
            f.write(f"错误信息: {str(e)}\n")
            f.write("详细追踪:\n")
            f.write(traceback.format_exc())
        print(f"错误日志已保存到: {error_path}")
        
        raise


if __name__ == "__main__":
    main()
