"""
元启发式搜索算法实现
包含粒子群算法(PSO)、Runge-Kutta优化器(RUN)和遗传算法(GA)
"""

import numpy as np
from typing import Callable, Tuple, Optional
import random


class PSO:
    """
    粒子群优化算法 (Particle Swarm Optimization)
    """
    
    def __init__(self, 
                 dimensions: int,
                 bounds: Tuple[np.ndarray, np.ndarray],
                 objective_func: Callable,
                 num_particles: int = 30,
                 max_iter: int = 100,
                 w: float = 0.729,  # 惯性权重
                 c1: float = 1.494,  # 个体学习因子
                 c2: float = 1.494,  # 社会学习因子
                 verbose: bool = False):
        """
        初始化PSO算法
        
        Args:
            dimensions: 优化问题的维度
            bounds: 搜索空间的上下界 (min_bounds, max_bounds)
            objective_func: 目标函数
            num_particles: 粒子数量
            max_iter: 最大迭代次数
            w: 惯性权重
            c1: 个体学习因子
            c2: 社会学习因子
            verbose: 是否显示进度
        """
        self.dimensions = dimensions
        self.bounds = bounds
        self.objective_func = objective_func
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.verbose = verbose
        
        # 初始化粒子群
        self.positions = np.random.uniform(
            low=bounds[0], high=bounds[1], size=(num_particles, dimensions)
        )
        self.velocities = np.random.uniform(
            low=-1, high=1, size=(num_particles, dimensions)
        )
        
        # 计算初始适应度
        self.fitness = np.array([
            self.objective_func(pos) for pos in self.positions
        ])
        
        # 初始化个体最优和全局最优
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_fitness = np.copy(self.fitness)
        
        # 全局最优
        best_idx = np.argmin(self.fitness)
        self.global_best_position = self.positions[best_idx]
        self.global_best_fitness = self.fitness[best_idx]
    
    def update_velocity_position(self):
        """更新粒子的速度和位置"""
        r1, r2 = np.random.random((2, self.num_particles, self.dimensions))
        
        # 更新速度
        cognitive_velocity = self.c1 * r1 * (
            self.personal_best_positions - self.positions
        )
        social_velocity = self.c2 * r2 * (
            self.global_best_position - self.positions
        )
        
        self.velocities = (
            self.w * self.velocities +
            cognitive_velocity +
            social_velocity
        )
        
        # 更新位置
        self.positions += self.velocities
        
        # 边界处理
        for i in range(self.dimensions):
            self.positions[:, i] = np.clip(
                self.positions[:, i],
                self.bounds[0][i],
                self.bounds[1][i]
            )
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        执行PSO优化
        
        Returns:
            全局最优解和对应的适应度值
        """
        for iteration in range(self.max_iter):
            # 计算当前所有粒子的适应度
            for i in range(self.num_particles):
                fitness = self.objective_func(self.positions[i])
                self.fitness[i] = fitness
                
                # 更新个体最优
                if fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best_positions[i] = np.copy(self.positions[i])
                    
                    # 更新全局最优
                    if fitness < self.global_best_fitness:
                        self.global_best_fitness = fitness
                        self.global_best_position = np.copy(self.positions[i])
            
            # 更新速度和位置
            self.update_velocity_position()
            
            if self.verbose and iteration % 10 == 0:
                print(f"PSO Iteration {iteration}: Best fitness = {self.global_best_fitness}")
        
        return self.global_best_position, self.global_best_fitness


class RUN:
    """
    Runge-Kutta优化器 (Runge-Kutta Optimizer)
    一种基于Runge-Kutta方法的元启发式优化算法
    """
    
    def __init__(self, 
                 dimensions: int,
                 bounds: Tuple[np.ndarray, np.ndarray],
                 objective_func: Callable,
                 population_size: int = 30,
                 max_iter: int = 100,
                 verbose: bool = False):
        """
        初始化RUN算法
        
        Args:
            dimensions: 优化问题的维度
            bounds: 搜索空间的上下界 (min_bounds, max_bounds)
            objective_func: 目标函数
            population_size: 种群大小
            max_iter: 最大迭代次数
            verbose: 是否显示进度
        """
        self.dimensions = dimensions
        self.bounds = bounds
        self.objective_func = objective_func
        self.population_size = population_size
        self.max_iter = max_iter
        self.verbose = verbose
        
        # 初始化种群
        self.population = np.random.uniform(
            low=bounds[0], high=bounds[1], size=(population_size, dimensions)
        )
        
        # 计算初始适应度
        self.fitness = np.array([
            self.objective_func(ind) for ind in self.population
        ])
        
        # 找到最佳个体
        best_idx = np.argmin(self.fitness)
        self.best_solution = self.population[best_idx]
        self.best_fitness = self.fitness[best_idx]
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        执行RUN优化
        
        Returns:
            最优解和对应的适应度值
        """
        for iteration in range(self.max_iter):
            # 计算步长因子
            t = (iteration + 1) / self.max_iter
            alpha = np.exp(t - 1)  # 动态步长因子
            
            new_population = np.copy(self.population)
            
            for i in range(self.population_size):
                # 随机选择两个不同的个体
                idx1, idx2 = random.sample(
                    [j for j in range(self.population_size) if j != i], 2
                )
                
                # Runge-Kutta步骤
                # 计算k1, k2, k3, k4
                x = self.population[i]
                x1 = self.population[idx1]
                x2 = self.population[idx2]
                
                # 基于最优解和随机个体的扰动
                r1, r2, r3 = np.random.random(3)
                
                # Runge-Kutta-like更新
                k1 = self.objective_func_direction(x, x1, x2, r1, r2)
                k2 = self.objective_func_direction(
                    x + alpha * k1 / 2, x1, x2, r1 * 0.8, r2 * 0.9
                )
                k3 = self.objective_func_direction(
                    x + alpha * k2 / 2, x1, x2, r1 * 0.9, r2 * 0.8
                )
                k4 = self.objective_func_direction(
                    x + alpha * k3, x1, x2, r1 * 0.7, r2 * 0.7
                )
                
                # 更新个体
                new_x = x + alpha * (k1 + 2 * k2 + 2 * k3 + k4) / 6
                new_x = self.apply_bounds(new_x)
                
                new_population[i] = new_x
            
            # 计算新种群的适应度
            new_fitness = np.array([
                self.objective_func(ind) for ind in new_population
            ])
            
            # 选择操作：保留更好的个体
            for i in range(self.population_size):
                if new_fitness[i] < self.fitness[i]:
                    self.population[i] = new_population[i]
                    self.fitness[i] = new_fitness[i]
                    
                    # 更新全局最优
                    if new_fitness[i] < self.best_fitness:
                        self.best_fitness = new_fitness[i]
                        self.best_solution = np.copy(new_population[i])
            
            if self.verbose and iteration % 10 == 0:
                print(f"RUN Iteration {iteration}: Best fitness = {self.best_fitness}")
        
        return self.best_solution, self.best_fitness
    
    def objective_func_direction(self, x, x1, x2, r1, r2):
        """
        计算目标函数的方向向量
        """
        # 简化的方向计算，实际实现可能需要更复杂的逻辑
        direction = (x1 - x) * r1 + (x2 - x) * r2
        return direction
    
    def apply_bounds(self, x):
        """
        应用边界约束
        """
        for i in range(self.dimensions):
            x[i] = np.clip(x[i], self.bounds[0][i], self.bounds[1][i])
        return x


class GeneticAlgorithm:
    """
    遗传算法 (Genetic Algorithm)
    """
    
    def __init__(self,
                 dimensions: int,
                 bounds: Tuple[np.ndarray, np.ndarray],
                 objective_func: Callable,
                 population_size: int = 50,
                 max_iter: int = 100,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.02,
                 selection_method: str = 'tournament',  # 'roulette', 'tournament'
                 verbose: bool = False):
        """
        初始化遗传算法
        
        Args:
            dimensions: 优化问题的维度
            bounds: 搜索空间的上下界 (min_bounds, max_bounds)
            objective_func: 目标函数
            population_size: 种群大小
            max_iter: 最大迭代次数
            crossover_rate: 交叉率
            mutation_rate: 变异率
            selection_method: 选择方法
            verbose: 是否显示进度
        """
        self.dimensions = dimensions
        self.bounds = bounds
        self.objective_func = objective_func
        self.population_size = population_size
        self.max_iter = max_iter
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.selection_method = selection_method
        self.verbose = verbose
        
        # 初始化种群
        self.population = np.random.uniform(
            low=bounds[0], high=bounds[1], size=(population_size, dimensions)
        )
        
        # 计算初始适应度
        self.fitness = np.array([
            self.objective_func(ind) for ind in self.population
        ])
        
        # 找到最佳个体
        best_idx = np.argmin(self.fitness)
        self.best_solution = self.population[best_idx]
        self.best_fitness = self.fitness[best_idx]
    
    def selection(self) -> np.ndarray:
        """
        选择操作
        """
        if self.selection_method == 'tournament':
            return self.tournament_selection()
        else:
            return self.roulette_wheel_selection()
    
    def tournament_selection(self, tournament_size: int = 3) -> np.ndarray:
        """
        锦标赛选择
        """
        selected = np.zeros_like(self.population)
        
        for i in range(self.population_size):
            # 随机选择锦标赛参与者
            tournament_indices = np.random.choice(
                self.population_size, size=tournament_size, replace=False
            )
            
            # 选择适应度最好的个体
            tournament_fitness = self.fitness[tournament_indices]
            winner_idx = tournament_indices[np.argmin(tournament_fitness)]
            selected[i] = self.population[winner_idx]
        
        return selected
    
    def roulette_wheel_selection(self) -> np.ndarray:
        """
        轮盘赌选择
        """
        # 计算选择概率（适应度越小越好，所以需要转换）
        max_fitness = np.max(self.fitness)
        inverted_fitness = max_fitness - self.fitness + 1e-10
        probabilities = inverted_fitness / np.sum(inverted_fitness)
        
        selected_indices = np.random.choice(
            self.population_size,
            size=self.population_size,
            p=probabilities
        )
        
        return self.population[selected_indices]
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        交叉操作
        """
        if np.random.random() < self.crossover_rate:
            # 算术交叉
            alpha = np.random.random()
            child1 = alpha * parent1 + (1 - alpha) * parent2
            child2 = (1 - alpha) * parent1 + alpha * parent2
        else:
            child1, child2 = parent1.copy(), parent2.copy()
        
        return child1, child2
    
    def mutation(self, individual: np.ndarray) -> np.ndarray:
        """
        变异操作
        """
        mutated = individual.copy()
        
        for i in range(self.dimensions):
            if np.random.random() < self.mutation_rate:
                # 高斯变异
                sigma = (self.bounds[1][i] - self.bounds[0][i]) * 0.1
                mutated[i] += np.random.normal(0, sigma)
                
                # 边界处理
                mutated[i] = np.clip(
                    mutated[i], self.bounds[0][i], self.bounds[1][i]
                )
        
        return mutated
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        执行遗传算法优化
        
        Returns:
            最优解和对应的适应度值
        """
        for iteration in range(self.max_iter):
            # 选择
            selected_population = self.selection()
            
            # 交叉和变异
            new_population = np.zeros_like(self.population)
            
            for i in range(0, self.population_size, 2):
                parent1 = selected_population[i]
                parent2 = selected_population[
                    (i + 1) % self.population_size
                ]
                
                child1, child2 = self.crossover(parent1, parent2)
                
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                
                new_population[i] = child1
                if i + 1 < self.population_size:
                    new_population[i + 1] = child2
            
            # 计算新种群的适应度
            new_fitness = np.array([
                self.objective_func(ind) for ind in new_population
            ])
            
            # 精英保留策略
            combined_pop = np.vstack([self.population, new_population])
            combined_fitness = np.hstack([self.fitness, new_fitness])
            
            # 选择最好的个体
            sorted_indices = np.argsort(combined_fitness)
            best_indices = sorted_indices[:self.population_size]
            
            self.population = combined_pop[best_indices]
            self.fitness = combined_fitness[best_indices]
            
            # 更新全局最优
            best_idx = np.argmin(self.fitness)
            if self.fitness[best_idx] < self.best_fitness:
                self.best_fitness = self.fitness[best_idx]
                self.best_solution = self.population[best_idx].copy()
            
            if self.verbose and iteration % 10 == 0:
                print(f"GA Iteration {iteration}: Best fitness = {self.best_fitness}")
        
        return self.best_solution, self.best_fitness


def optimize_pib_heuristic(
    center,
    epochs,
    algorithm='PSO',  # 'PSO', 'RUN', 'GA'
    r_bucket=0,
    delta: float = 1,
    lr: float = 0,
    exposure_time_ms: int = 80,
    shrink_iter: int = 0,
    shrink_ratio: float = 0.9,
    cam_id=0,
    show: bool = False,
    init_v=[],
    cam_size=250,
    target_max_brightness=40,
    dm_unit_mask=None,
    dm_neibor_diff=200,
    dm_max_voltage=None,
    dm_min_voltage=None,
    **kwargs
):
    """
    使用元启发式算法优化PIB
    """
    from ao_shaping.drivers import CameraStreamManager, NlightDM
    from ao_shaping.utils import ImageVoltagesDisplay, logger, Recorder
    from ao_shaping.utils.spots_calc import centroid, radius
    from ao_shaping.algorithm.target_func import ImageTargetFunc

    # 相机参数
    CAM_SAMPLE_ITER = 1
    ADVISE_EXPOSURE_TIME_BRIGHTNESS = int(255/3)
    TEST_EXPOSURE_TIME_BRIGHTNESS = 220
    IDEAL_SPOT_RADIUS = int(6)  # 默认值

    delta = abs(delta)
    epochs = int(epochs)

    recorder = Recorder(mark="pib_heuristic", mode="max")

    with CameraStreamManager(cam_id=cam_id, exposure_time_ms=exposure_time_ms, skip_sampling=False) as cam, \
            NlightDM(keep_when_exit=True, max_neibor_diff=dm_neibor_diff, max_voltage=dm_max_voltage, min_voltage=dm_min_voltage) as dm:
        
        if dm_unit_mask is None:
            dm_unit_mask = dm.default_dm_unit_mask
            if dm_unit_mask[0]:
                logger.warning("dm_unit_mask[0] is True, which means the first unit is active.")

        if init_v is None or len(init_v) == 0:
            _init_v = np.zeros(dm.DM_Num, dtype=np.float64)
        else:
            _init_v = np.array(init_v)
        
        dm.send_voltages(_init_v, 0.5)

        _img = cam.autoset_exposure_time_ms(target_max_brightness=TEST_EXPOSURE_TIME_BRIGHTNESS)

        def intellij_center(img):
            (h, w) = img.shape
            margin = int(IDEAL_SPOT_RADIUS)
            center = centroid(np.where(img > np.max(img[:max(int(h//50), 2), :max(int(w//50), 2)]), 1, 0))
            (cx, cy) = center
            if np.all(img[cy-margin: cy+margin, cx-margin: cx+margin] >= np.max(img) * 0.4):
                center = centroid(img)
            return center

        if center is None:
            center = intellij_center(_img)
        elif isinstance(center, str):
            _img = cam.get_numpy_image(10)
            if center == "mass":
                center = centroid(_img)
            elif center == 'max':
                center = np.unravel_index(np.argmax(_img), _img.shape)[::-1]
            elif center == 'shape':
                (h, w) = _img.shape
                center = centroid(
                    np.where(_img > np.max(_img[:max(int(h//50), 2), :max(int(w//50), 2)]), 1, 0))
            else:
                raise ValueError(f"known center: {center}")
        else:
            center = center

        logger.info(f"Centroid: {center}, Max brightness: {np.max(_img)} @ {cam.exposure_time}ms")

        img_size = (cam_size, cam_size)
        img_size, center = cam.reset_window(center, img_size)

        if exposure_time_ms > 0:
            cam.exposure_time = exposure_time_ms
            init_img = cam.get_numpy_image(CAM_SAMPLE_ITER)
        elif 0 < target_max_brightness < 255 and target_max_brightness > 0:
            init_img = cam.autoset_exposure_time_ms(
                target_max_brightness=target_max_brightness, twice_valid=True)
        else:
            init_img = cam.autoset_exposure_time_ms(
                target_max_brightness=ADVISE_EXPOSURE_TIME_BRIGHTNESS, twice_valid=True)
        
        logger.debug(f"Inital Image Max brightness: {np.max(init_img)} @ {cam.exposure_time}ms")
        img_size = init_img.shape[::-1]

        if r_bucket <= 0:
            _w, _h = img_size
            target_func = ImageTargetFunc.build_from_init_image(init_img)
            r_bucket = target_func.radius(init_img, energy=0.99)
            r_bucket = min(r_bucket, cam_size//2) * shrink_ratio
            _fix_bucket = False
        else:
            _fix_bucket = True

        if show:
            window = ImageVoltagesDisplay(img_size)
            window.init_window()

        target_func = ImageTargetFunc.build_from_init_image(init_img)
        
        def calc_pib(img):
            # 根据环围半径找出边缘梯度最大的阶数
            r, nr = target_func.avg_radius(img, moment=0.5)
            return -r, -nr

        def test_pib(img):
            return target_func.pib(img, IDEAL_SPOT_RADIUS)[1]

        j, pib_ratio = calc_pib(init_img)

        # 定义目标函数（最小化负PIB以最大化PIB）
        def objective_function(voltages):
            # 确保电压在范围内
            clipped_voltages = np.clip(voltages, dm.min_voltage, dm.max_voltage)
            dm.send_voltages(clipped_voltages)
            img = cam.get_numpy_image(CAM_SAMPLE_ITER)
            pib_value, _ = calc_pib(img)
            # 返回负值因为优化器最小化目标函数
            return -pib_value

        # 定义搜索边界
        lower_bounds = np.full(dm.DM_Num, dm.min_voltage)
        upper_bounds = np.full(dm.DM_Num, dm.max_voltage)
        bounds = (lower_bounds, upper_bounds)

        # 选择优化算法
        if algorithm == 'PSO':
            optimizer = PSO(
                dimensions=dm.DM_Num,
                bounds=bounds,
                objective_func=objective_function,
                num_particles=min(30, dm.DM_Num),
                max_iter=epochs // 10,  # 减少迭代次数以适应实际硬件操作
                verbose=True
            )
        elif algorithm == 'RUN':
            optimizer = RUN(
                dimensions=dm.DM_Num,
                bounds=bounds,
                objective_func=objective_function,
                population_size=min(30, dm.DM_Num),
                max_iter=epochs // 10,
                verbose=True
            )
        elif algorithm == 'GA':
            optimizer = GeneticAlgorithm(
                dimensions=dm.DM_Num,
                bounds=bounds,
                objective_func=objective_function,
                population_size=min(50, dm.DM_Num * 2),
                max_iter=epochs // 10,
                verbose=True
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # 执行优化
        best_solution, best_fitness = optimizer.optimize()

        # 发送最终电压
        dm.send_voltages(best_solution)
        final_img = cam.get_numpy_image(CAM_SAMPLE_ITER)
        final_pib = test_pib(final_img)

        # 记录结果
        recorder.append({
            "J": -best_fitness,  # 转换回正PIB值
            "pib": final_pib,
            "_p%": final_pib,  # 简化的PIB比例
            "_max_r": r_bucket,
            "_v": best_solution,
            "_img": final_img,
            "_diff": 0,
            "lr": lr,
            "r": r_bucket,
            "delta": delta,
            "_epoch": epochs,
            "exp_t": cam.exposure_time,
            "max_brt": np.max(final_img),
            "_grad": np.zeros_like(best_solution),
        })

        if show:
            window.close()

        return recorder