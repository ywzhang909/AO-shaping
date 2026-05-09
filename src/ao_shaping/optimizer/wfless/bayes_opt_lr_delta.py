import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import warnings
warnings.filterwarnings('ignore')

from ao_shaping.optimizer.wfless.pib import optimize_pib
from ao_shaping.drivers import CameraStreamManager
from ao_shaping.utils import logger


def objective_function(lr, delta, center=None, epochs=100, exposure_time_ms=80, cam_id=0):
    """
    目标函数：用于贝叶斯优化，返回负的PIB值（因为我们要最大化PIB）
    
    Args:
        lr (float): 学习率
        delta (float): 步长
        center: 图像中心位置
        epochs (int): 优化迭代次数
        exposure_time_ms (int): 曝光时间（毫秒）
        cam_id: 相机ID
        
    Returns:
        float: 负的PIB值（用于最小化优化）
    """
    try:
        # 设置默认参数
        if center is None:
            # 获取图像中心
            with CameraStreamManager(cam_id=cam_id, exposure_time_ms=exposure_time_ms) as cam:
                _img = cam.get_numpy_image(10)
                h, w = _img.shape
                # 计算质心
                from ao_shaping.utils.spots_calc import centroid
                center = centroid(np.where(_img > np.max(_img[:max(int(h//50), 2), :max(int(w//50), 2)]), 1, 0))

        # 运行优化
        recorder = optimize_pib(
            center=center,
            epochs=epochs,
            delta=abs(delta),
            lr=lr,
            exposure_time_ms=exposure_time_ms,
            shrink_iter=0,
            shrink_ratio=0.9,
            cam_id=cam_id,
            show=False,
            init_v=[],
            cam_size=200,
            dm_unit_mask=None,
            dm_neibor_diff=200,
            dm_max_voltage=None,
            dm_min_voltage=None
        )

        # 获取最佳PIB值
        best_iter, (max_j_id, max_j) = recorder.get_best_iter()
        pib_value = best_iter["pib"]

        # 返回负值，因为我们使用最小化优化器来最大化PIB
        return -pib_value

    except Exception as e:
        logger.error(f"优化过程中出现错误: {e}")
        # 返回一个较大的正值表示失败
        return 1e6


def bayesian_optimize_lr_delta(
    n_calls=30,
    lr_bounds=(0.1, 5.0),
    delta_bounds=(0.1, 5.0),
    center=None,
    epochs=100,
    exposure_time_ms=80,
    cam_id=0
):
    """
    使用贝叶斯优化来寻找最优的学习率和delta参数
    
    Args:
        n_calls (int): 优化调用次数
        lr_bounds (tuple): 学习率的范围 (min, max)
        delta_bounds (tuple): delta的范围 (min, max)
        center: 图像中心位置
        epochs (int): 每次评估的迭代次数
        exposure_time_ms (int): 曝光时间（毫秒）
        cam_id: 相机ID
        
    Returns:
        dict: 包含最优参数和结果的字典
    """
    # 定义搜索空间
    dimensions = [
        Real(*lr_bounds, name='lr'),
        Real(*delta_bounds, name='delta')
    ]

    # 创建目标函数的包装器
    @use_named_args(dimensions)
    def wrapped_objective(lr, delta):
        return objective_function(
            lr=lr,
            delta=delta,
            center=center,
            epochs=epochs,
            exposure_time_ms=exposure_time_ms,
            cam_id=cam_id
        )

    # 执行贝叶斯优化
    result = gp_minimize(
        func=wrapped_objective,
        dimensions=dimensions,
        n_calls=n_calls,
        random_state=42,
        n_initial_points=10
    )

    # 返回结果
    return {
        'best_params': {
            'lr': result.x[0],
            'delta': result.x[1]
        },
        'best_score': -result.fun,  # 转换回正的PIB值
        'optimization_result': result
    }


def grid_search_lr_delta(
    lr_values=[0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
    delta_values=[0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
    center=None,
    epochs=100,
    exposure_time_ms=80,
    cam_id=0
):
    """
    网格搜索学习率和delta参数
    
    Args:
        lr_values (list): 学习率候选值列表
        delta_values (list): delta候选值列表
        center: 图像中心位置
        epochs (int): 每次评估的迭代次数
        exposure_time_ms (int): 曝光时间（毫秒）
        cam_id: 相机ID
        
    Returns:
        dict: 包含最优参数和结果的字典
    """
    best_score = -np.inf
    best_params = {'lr': None, 'delta': None}
    results = []

    total_combinations = len(lr_values) * len(delta_values)
    current_combination = 0

    for lr in lr_values:
        for delta in delta_values:
            current_combination += 1
            print(f"测试参数组合 {current_combination}/{total_combinations}: lr={lr}, delta={delta}")

            try:
                score = -objective_function(
                    lr=lr,
                    delta=delta,
                    center=center,
                    epochs=epochs,
                    exposure_time_ms=exposure_time_ms,
                    cam_id=cam_id
                )

                results.append({
                    'lr': lr,
                    'delta': delta,
                    'score': score
                })

                if score > best_score:
                    best_score = score
                    best_params = {'lr': lr, 'delta': delta}

                print(f"  得分: {score:.4f}")

            except Exception as e:
                print(f"  错误: {e}")
                results.append({
                    'lr': lr,
                    'delta': delta,
                    'score': -np.inf,
                    'error': str(e)
                })

    return {
        'best_params': best_params,
        'best_score': best_score,
        'all_results': results
    }


if __name__ == "__main__":
    # 示例用法
    print("开始贝叶斯优化...")

    # 贝叶斯优化
    bayes_result = bayesian_optimize_lr_delta(
        n_calls=20,
        lr_bounds=(0.1, 5.0),
        delta_bounds=(0.1, 5.0),
        epochs=50,  # 减少迭代次数以加快测试
        exposure_time_ms=60
    )

    print("贝叶斯优化结果:")
    print(f"最优学习率: {bayes_result['best_params']['lr']:.4f}")
    print(f"最优delta: {bayes_result['best_params']['delta']:.4f}")
    print(f"最优PIB值: {bayes_result['best_score']:.4f}")

    print("\n" + "="*50 + "\n")

    # 网格搜索
    print("开始网格搜索...")
    grid_result = grid_search_lr_delta(
        lr_values=[0.5, 1.0, 1.5, 2.0, 2.5],
        delta_values=[0.5, 1.0, 1.5, 2.0, 2.5],
        epochs=50,  # 减少迭代次数以加快测试
        exposure_time_ms=60
    )

    print("网格搜索结果:")
    print(f"最优学习率: {grid_result['best_params']['lr']}")
    print(f"最优delta: {grid_result['best_params']['delta']}")
    print(f"最优PIB值: {grid_result['best_score']:.4f}")
