import click
import json
import numpy as np
from datetime import datetime
from pathlib import Path

from ao_shaping.optimizer.wfless.bayes_opt_lr_delta import (
    bayesian_optimize_lr_delta,
    grid_search_lr_delta
)
from ao_shaping.utils.file import gen_date_dir, gen_file_path_uuid, logger


@click.command()
@click.option("-d", "--root_dir", default="data", help="数据保存根目录 (default: data)")
@click.option("-e", "--epochs", default=100, help="每次评估的优化迭代次数 (default: 100)")
@click.option("--exposure_time_ms", default=60, help="相机曝光时间(毫秒) (default: 60)")
@click.option("--cam_id", default=0, help="相机ID (default: 0)")
@click.option("-n", "--n_calls", default=30, help="贝叶斯优化调用次数 (default: 30)")
@click.option("--lr_min", default=0.1, help="学习率最小值 (default: 0.1)")
@click.option("--lr_max", default=5.0, help="学习率最大值 (default: 5.0)")
@click.option("--delta_min", default=0.1, help="delta最小值 (default: 0.1)")
@click.option("--delta_max", default=5.0, help="delta最大值 (default: 5.0)")
@click.option("--method", type=click.Choice(['bayes', 'grid']), default='bayes', 
              help="优化方法: bayes(贝叶斯优化) 或 grid(网格搜索) (default: bayes)")
@click.option("--grid_lr_steps", default=5, help="网格搜索中学习率的步数 (default: 5)")
@click.option("--grid_delta_steps", default=5, help="网格搜索中delta的步数 (default: 5)")
@click.option("--debug", is_flag=True, help="是否开启调试模式 (default: False)")
def run(root_dir, epochs, exposure_time_ms, cam_id, n_calls, lr_min, lr_max, 
        delta_min, delta_max, method, grid_lr_steps, grid_delta_steps, debug):
    """使用贝叶斯优化寻找最优的学习率和delta参数"""
    
    # 记录配置
    config = {
        'root_dir': root_dir,
        'epochs': epochs,
        'exposure_time_ms': exposure_time_ms,
        'cam_id': cam_id,
        'n_calls': n_calls,
        'lr_bounds': [lr_min, lr_max],
        'delta_bounds': [delta_min, delta_max],
        'method': method,
        'grid_lr_steps': grid_lr_steps,
        'grid_delta_steps': grid_delta_steps,
        'debug': debug
    }
    
    logger.info(f"开始使用{method}方法优化学习率和delta参数")
    logger.info(f"配置: {config}")
    
    try:
        if method == 'bayes':
            # 贝叶斯优化
            result = bayesian_optimize_lr_delta(
                n_calls=n_calls,
                lr_bounds=(lr_min, lr_max),
                delta_bounds=(delta_min, delta_max),
                epochs=epochs,
                exposure_time_ms=exposure_time_ms,
                cam_id=cam_id
            )
        else:
            # 网格搜索
            lr_values = np.linspace(lr_min, lr_max, grid_lr_steps).tolist()
            delta_values = np.linspace(delta_min, delta_max, grid_delta_steps).tolist()
            
            result = grid_search_lr_delta(
                lr_values=lr_values,
                delta_values=delta_values,
                epochs=epochs,
                exposure_time_ms=exposure_time_ms,
                cam_id=cam_id
            )
        
        # 输出结果
        best_params = result['best_params']
        best_score = result['best_score']
        
        click.echo(f"\n优化完成!")
        click.echo(f"最优学习率 (lr): {best_params['lr']:.4f}")
        click.echo(f"最优delta: {best_params['delta']:.4f}")
        click.echo(f"最优PIB值: {best_score:.4f}")
        
        # 保存结果
        save_dir = Path(root_dir) / "bayes_opt" / datetime.now().strftime("%Y%m%d")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        config_file = save_dir / "config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
        
        # 保存结果
        result_file = save_dir / "result.json"
        result_data = {
            'best_params': best_params,
            'best_score': float(best_score),
            'timestamp': datetime.now().isoformat()
        }
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=4)
        
        # 如果是贝叶斯优化，保存更多详细信息
        if method == 'bayes' and 'optimization_result' in result:
            opt_result_file = save_dir / "optimization_result.json"
            opt_result = result['optimization_result']
            
            # 提取关键信息
            opt_data = {
                'x': [float(x) for x in opt_result.x],
                'fun': float(opt_result.fun),
                'success': opt_result.success,
                'message': opt_result.message,
                'nit': opt_result.nit,
                'models': len(opt_result.models) if hasattr(opt_result, 'models') else 0
            }
            
            with open(opt_result_file, 'w', encoding='utf-8') as f:
                json.dump(opt_data, f, ensure_ascii=False, indent=4)
        
        # 如果是网格搜索，保存所有结果
        if method == 'grid' and 'all_results' in result:
            all_results_file = save_dir / "all_results.json"
            with open(all_results_file, 'w', encoding='utf-8') as f:
                json.dump(result['all_results'], f, ensure_ascii=False, indent=4)
        
        click.echo(f"\n结果已保存到: {save_dir}")
        
    except Exception as e:
        logger.error(f"优化过程中出现错误: {e}")
        click.echo(f"错误: {e}")
        raise


if __name__ == "__main__":
    run()