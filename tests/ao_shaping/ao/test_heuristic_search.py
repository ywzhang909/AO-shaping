import numpy as np
from ao_shaping.algorithm.heuristic_search import (
    ParticleSwarmOptimization,
    GeneticAlgorithm,
    SimulatedAnnealing,
    DifferentialEvolution,
)

def test_sphere_function(x):
    """测试函数：球函数（Sphere Function）"""
    return np.sum(x**2)

def test_rosenbrock_function(x):
    """测试函数：Rosenbrock函数"""
    return np.sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def test_algorithms():
    """测试所有启发式搜索算法"""
    print("Testing heuristic search algorithms...")
    
    # 测试维度
    dim = 10
    bounds = (-5, 5)
    
    # 测试粒子群优化算法
    print("\n1. Testing Particle Swarm Optimization (PSO)...")
    pso = ParticleSwarmOptimization(dim, n_particles=30, bounds=bounds)
    best_solution, best_fitness = pso.optimize(test_sphere_function, max_iter=100)
    print(f"   Best fitness: {best_fitness:.6f}")
    print(f"   Best solution: {best_solution[:5]}...")  # 只显示前5个元素
    
    # 测试遗传算法
    print("\n2. Testing Genetic Algorithm (GA)...")
    ga = GeneticAlgorithm(dim, population_size=50, bounds=bounds)
    best_solution, best_fitness = ga.optimize(test_sphere_function, max_iter=100)
    print(f"   Best fitness: {best_fitness:.6f}")
    print(f"   Best solution: {best_solution[:5]}...")  # 只显示前5个元素
    
    # 测试模拟退火算法
    print("\n3. Testing Simulated Annealing (SA)...")
    sa = SimulatedAnnealing(dim, bounds=bounds)
    best_solution, best_fitness = sa.optimize(test_sphere_function, max_iter=1000)
    print(f"   Best fitness: {best_fitness:.6f}")
    print(f"   Best solution: {best_solution[:5]}...")  # 只显示前5个元素
    
    # 测试差分进化算法
    print("\n4. Testing Differential Evolution (DE)...")
    de = DifferentialEvolution(dim, population_size=50, bounds=bounds)
    best_solution, best_fitness = de.optimize(test_sphere_function, max_iter=100)
    print(f"   Best fitness: {best_fitness:.6f}")
    print(f"   Best solution: {best_solution[:5]}...")  # 只显示前5个元素
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    test_algorithms()