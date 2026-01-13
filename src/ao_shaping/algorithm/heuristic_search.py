from typing import Callable
import numpy as np
from abc import ABC, abstractmethod


class HeuristicSearchBase(ABC):
    """
    启发式搜索算法基类
    """
    
    def __init__(self, dim: int, bounds: tuple[float, float]):
        """
        初始化启发式搜索算法
        
        Args:
            dim: 搜索空间维度
            bounds: 搜索边界 (min, max)，默认为 (-100, 100)
        """
        self.dim = dim
        self.bounds = bounds
        self.best_solution = None
        self.best_fitness = float('inf')
        self.iteration = 0
        
    @abstractmethod
    def search_step(self, fitness_func: Callable[[np.ndarray], float]) -> tuple[np.ndarray, float]:
        """
        执行一次搜索步骤
        
        Args:
            fitness_func: 适应度函数
            
        Returns:
            (solution, fitness): 解和对应的适应度值
        """
        pass
    
    def optimize(self, fitness_func: Callable[[np.ndarray], float], max_iter: int = 1000) -> tuple[np.ndarray, float]:
        """
        运行优化过程
        
        Args:
            fitness_func: 适应度函数
            max_iter: 最大迭代次数
            
        Returns:
            (best_solution, best_fitness): 最优解和对应的适应度值
        """
        for _ in range(max_iter):
            solution, fitness = self.search_step(fitness_func)
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = solution.copy()
        
        assert self.best_solution is not None, "优化未找到任何解"
        return self.best_solution, self.best_fitness


class ParticleSwarmOptimization(HeuristicSearchBase):
    """
    粒子群优化算法 (PSO)
    """
    
    def __init__(self, dim: int, bounds: tuple[float, float], n_particles: int = 30,
                 w: float = 0.7, c1: float = 1.5, c2: float = 1.5):
        """
        初始化粒子群优化算法
        
        Args:
            dim: 搜索空间维度
            n_particles: 粒子数量
            bounds: 搜索边界 (min, max)
            w: 惯性权重
            c1: 个体学习因子
            c2: 社会学习因子
        """
        super().__init__(dim, bounds)
        self.n_particles = n_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        # 初始化粒子群
        self.positions = np.random.uniform(bounds[0], bounds[1], (n_particles, dim))
        self.velocities = np.random.uniform(-1, 1, (n_particles, dim))
        self.personal_best_positions = self.positions.copy()
        self.personal_best_fitness = np.full(n_particles, float('inf'))
        
        # 全局最优
        self.global_best_position = None
        self.global_best_fitness = float('inf')
    
    def search_step(self, fitness_func: Callable[[np.ndarray], float]) -> tuple[np.ndarray, float]:
        """
        执行一次PSO搜索步骤
        """
        self.iteration += 1
        
        # 第一次迭代需要初始化适应度
        if self.iteration == 1:
            for i in range(self.n_particles):
                fitness = fitness_func(self.positions[i])
                self.personal_best_fitness[i] = fitness
                self.personal_best_positions[i] = self.positions[i].copy()
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = self.positions[i].copy()
        
        for i in range(self.n_particles):
            # 计算新速度
            r1, r2 = np.random.rand(), np.random.rand()
            self.velocities[i] = (
                self.w * self.velocities[i] +
                self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i]) +
                self.c2 * r2 * (self.global_best_position - self.positions[i])
            )
            
            # 更新位置
            self.positions[i] += self.velocities[i]
            
            # 边界处理
            self.positions[i] = np.clip(self.positions[i], self.bounds[0], self.bounds[1])
            
            # 计算适应度
            fitness = fitness_func(self.positions[i])
            
            # 更新个体最优
            if fitness < self.personal_best_fitness[i]:
                self.personal_best_fitness[i] = fitness
                self.personal_best_positions[i] = self.positions[i].copy()
                
                # 更新全局最优
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = self.positions[i].copy()
        
        assert self.global_best_position is not None, "PSO未找到任何解"
        return self.global_best_position.copy(), self.global_best_fitness


class GeneticAlgorithm(HeuristicSearchBase):
    """
    遗传算法 (GA)
    """
    
    def __init__(self, dim: int, bounds: tuple[float, float], population_size: int = 50,
                 crossover_rate: float = 0.8, mutation_rate: float = 0.1):
        """
        初始化遗传算法
        
        Args:
            dim: 搜索空间维度
            population_size: 种群大小
            bounds: 搜索边界 (min, max)
            crossover_rate: 交叉概率
            mutation_rate: 变异概率
        """
        super().__init__(dim, bounds)
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        
        # 初始化种群
        self.population = np.random.uniform(bounds[0], bounds[1], (population_size, dim))
        self.fitness_values = np.full(population_size, float('inf'))
        
    def selection(self) -> np.ndarray:
        """
        锦标赛选择
        """
        tournament_size = 3
        selected = []
        
        for _ in range(self.population_size):
            # 随机选择锦标赛参与者
            indices = np.random.choice(self.population_size, tournament_size)
            tournament_fitness = self.fitness_values[indices]
            winner_idx = indices[np.argmin(tournament_fitness)]
            selected.append(self.population[winner_idx].copy())
            
        return np.array(selected)
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        均匀交叉
        """
        if np.random.rand() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
            
        child1, child2 = parent1.copy(), parent2.copy()
        
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                child1[i], child2[i] = child2[i], child1[i]
                
        return child1, child2
    
    def mutation(self, individual: np.ndarray) -> np.ndarray:
        """
        高斯变异
        """
        mutated = individual.copy()
        
        for i in range(self.dim):
            if np.random.rand() < self.mutation_rate:
                # 添加高斯噪声
                mutated[i] += np.random.normal(0, (self.bounds[1] - self.bounds[0]) * 0.1)
                
        # 边界处理
        mutated = np.clip(mutated, self.bounds[0], self.bounds[1])
        return mutated
    
    def search_step(self, fitness_func: Callable[[np.ndarray], float]) -> tuple[np.ndarray, float]:
        """
        执行一次遗传算法搜索步骤
        """
        self.iteration += 1
        
        # 第一次迭代需要初始化适应度
        if self.iteration == 1:
            for i in range(self.population_size):
                self.fitness_values[i] = fitness_func(self.population[i])
            # 更新最优解
            best_idx = np.argmin(self.fitness_values)
            self.best_fitness = self.fitness_values[best_idx]
            self.best_solution = self.population[best_idx].copy()
        
        # 评估当前种群
        for i in range(self.population_size):
            self.fitness_values[i] = fitness_func(self.population[i])
            
        # 更新最优解
        best_idx = np.argmin(self.fitness_values)
        if self.fitness_values[best_idx] < self.best_fitness:
            self.best_fitness = self.fitness_values[best_idx]
            self.best_solution = self.population[best_idx].copy()
            
        # 选择
        selected_population = self.selection()
        
        # 交叉和变异生成新一代
        new_population = []
        for i in range(0, self.population_size, 2):
            parent1 = selected_population[i]
            parent2 = selected_population[(i + 1) % self.population_size]
            
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutation(child1)
            child2 = self.mutation(child2)
            
            new_population.extend([child1, child2])
            
        self.population = np.array(new_population[:self.population_size])
        
        return self.best_solution.copy(), self.best_fitness


class SimulatedAnnealing(HeuristicSearchBase):
    """
    模拟退火算法 (SA)
    """
    
    def __init__(self, dim: int, bounds: tuple[float, float],
                 initial_temperature: float = 1000, cooling_rate: float = 0.95, min_temperature: float = 1e-8):
        """
        初始化模拟退火算法
        
        Args:
            dim: 搜索空间维度
            bounds: 搜索边界 (min, max)
            initial_temperature: 初始温度
            cooling_rate: 冷却速率
            min_temperature: 最小温度
        """
        super().__init__(dim, bounds)
        self.temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        
        # 初始化当前解
        self.current_solution = np.random.uniform(bounds[0], bounds[1], dim)
        self.current_fitness = float('inf')
        
    def generate_neighbor(self, solution: np.ndarray) -> np.ndarray:
        """
        生成邻域解
        """
        neighbor = solution.copy()
        
        # 随机选择一个维度进行扰动
        idx = np.random.randint(0, self.dim)
        perturbation = np.random.normal(0, (self.bounds[1] - self.bounds[0]) * 0.1)
        neighbor[idx] += perturbation
        
        # 边界处理
        neighbor = np.clip(neighbor, self.bounds[0], self.bounds[1])
        return neighbor
    
    def acceptance_probability(self, old_fitness: float, new_fitness: float) -> float:
        """
        计算接受概率
        """
        if new_fitness < old_fitness:
            return 1.0
        return np.exp((old_fitness - new_fitness) / self.temperature)
    
    def search_step(self, fitness_func: Callable[[np.ndarray], float]) -> tuple[np.ndarray, float]:
        """
        执行一次模拟退火搜索步骤
        """
        self.iteration += 1
        
        # 初始化第一个解
        if self.iteration == 1:
            self.current_fitness = fitness_func(self.current_solution)
            self.best_solution = self.current_solution.copy()
            self.best_fitness = self.current_fitness
            return self.current_solution.copy(), self.current_fitness
        
        # 生成邻域解
        neighbor = self.generate_neighbor(self.current_solution)
        neighbor_fitness = fitness_func(neighbor)
        
        # 判断是否接受新解
        if (neighbor_fitness < self.current_fitness or
            np.random.rand() < self.acceptance_probability(self.current_fitness, neighbor_fitness)):
            self.current_solution = neighbor
            self.current_fitness = neighbor_fitness
            
            # 更新全局最优解
            if neighbor_fitness < self.best_fitness:
                self.best_solution = neighbor.copy()
                self.best_fitness = neighbor_fitness
        
        # 降温
        self.temperature = max(self.temperature * self.cooling_rate, self.min_temperature)
        
        return self.best_solution.copy(), self.best_fitness


class DifferentialEvolution(HeuristicSearchBase):
    """
    差分进化算法 (DE)
    """
    
    def __init__(self, dim: int, bounds: tuple[float, float], population_size: int = 50,
                 F: float = 0.8, CR: float = 0.9):
        """
        初始化差分进化算法
        
        Args:
            dim: 搜索空间维度
            population_size: 种群大小
            bounds: 搜索边界 (min, max)
            F: 差分权重
            CR: 交叉概率
        """
        super().__init__(dim, bounds)
        self.population_size = population_size
        self.F = F
        self.CR = CR
        
        # 初始化种群
        self.population = np.random.uniform(bounds[0], bounds[1], (population_size, dim))
        self.fitness_values = np.full(population_size, float('inf'))
        
    def search_step(self, fitness_func: Callable[[np.ndarray], float]) -> tuple[np.ndarray, float]:
        """
        执行一次差分进化搜索步骤
        """
        self.iteration += 1
        
        # 第一次迭代需要初始化适应度
        if self.iteration == 1:
            for i in range(self.population_size):
                self.fitness_values[i] = fitness_func(self.population[i])
            # 更新最优解
            best_idx = np.argmin(self.fitness_values)
            self.best_fitness = self.fitness_values[best_idx]
            self.best_solution = self.population[best_idx].copy()
        
        # 评估当前种群
        for i in range(self.population_size):
            self.fitness_values[i] = fitness_func(self.population[i])
            
        # 更新最优解
        best_idx = np.argmin(self.fitness_values)
        if self.fitness_values[best_idx] < self.best_fitness:
            self.best_fitness = self.fitness_values[best_idx]
            self.best_solution = self.population[best_idx].copy()
            
        # 变异和交叉生成新一代
        new_population = []
        for i in range(self.population_size):
            # 选择三个不同的个体
            indices = [j for j in range(self.population_size) if j != i]
            a, b, c = np.random.choice(indices, 3, replace=False)
            
            # 变异
            mutant = self.population[a] + self.F * (self.population[b] - self.population[c])
            mutant = np.clip(mutant, self.bounds[0], self.bounds[1])
            
            # 交叉
            trial = self.population[i].copy()
            j_rand = np.random.randint(0, self.dim)
            for j in range(self.dim):
                if np.random.rand() < self.CR or j == j_rand:
                    trial[j] = mutant[j]
                    
            # 选择
            trial_fitness = fitness_func(trial)
            if trial_fitness < self.fitness_values[i]:
                new_population.append(trial)
            else:
                new_population.append(self.population[i])
                
        self.population = np.array(new_population)
        
        return self.best_solution.copy(), self.best_fitness