import os
import json
import re
from datetime import datetime

import click
import coredumpy

import numpy as np
import matplotlib.pyplot as plt

from ao_shaping.drivers import CameraStreamManager, NlightDM
from ao_shaping.algorithm.heuristic_search import DMOptimizer
from ao_shaping.utils import Recorder
from ao_shaping.utils.file import gen_file_path_uuid, gen_date_dir, get_init_V_by_rms, logger
from ao_shaping.utils.display import plot_funcs
from ao_shaping.utils.spots_calc import centroid, radius


def parse_tuple(ctx, param, value):
    """解析元组格式的参数，支持 'x,y' 或 '(x,y)' 格式"""
    if value is None:
        return None
    if value.lower() in ["mass", "max", "shape"]:
        return value.lower()
    # 移除空格和括号
    s_clean = re.sub(r'[()\s]', '', str(value))
    try:
        parts = s_clean.split(',')
        if len(parts) != 2:
            raise ValueError("Must have exactly two integers")
        x, y = map(int, parts)
        return (x, y)
    except Exception as e:
        raise click.BadParameter(
            f"Invalid center format: {value}. Expected formats: 'x,y' or '(x,y)'"
        )


@click.command()
@click.option("-d", "--root_dir", default="data", help="数据保存根目录 (default: data)")
@click.option("-f", "--load_file", default='rms', help="加载优化结果文件 (default: None), 若为'rms'，则使用RMS优化结果初始化")
@click.option("--cam_id", default=lambda: os.environ.get('Far_Cam_ID', 0), help="远场光斑CCD设备ID (default: Far_Cam_ID/0)")
@click.option("-c", "--center", callback=parse_tuple, default="mass", help="场光斑CCD中心位置 (example: 665,403)")
@click.option("-t","--exposure_time_ms", default=60, help="远场光斑CCD曝光时间 (毫秒) (default: 60)")
@click.option("-e", "--epochs", default=4_000, help="优化迭代次数 (default: 4000)")
@click.option("-r", "--r_bucket", default=0, help="渲染半径桶大小 (default: 0，环围半径)")
@click.option("--delta", default=2, help="优化步长 (default: 2)")
@click.option("--lr", default=0.0, help="优化学习率 (default: None，表示基于环围半径动态学习率衰减)")
@click.option("--weight_decay", default=0.0, help="权重衰减 (default: 0.0)")
@click.option("--shrink_iter", default=300, help="优化迭代次数后收缩半径桶和步长 (default: 300)")
@click.option("--shrink_ratio", default=0.8, help="收缩半径桶和步长比例 (default: 0.8)")
@click.option("-s", "--cam_size", default=200, help="相机开窗大小 (default: 200*200)")
@click.option("-b", "--target_max_brightness", default=90, help="目标最大亮度值 (default: 90)")
@click.option("--method", default="pso", help="启发式搜索方法 (pso, ga, sa, de) (default: pso)")
@click.option("--debug", is_flag=True, help="是否开启调试模式 (default: False)")
@click.option("--show", is_flag=True, help="显示远场光斑CCD图像和优化历史 (default: False)")
def run(root_dir, load_file, cam_id, center, exposure_time_ms, epochs, r_bucket,
        delta, lr, weight_decay, shrink_iter, shrink_ratio, cam_size, target_max_brightness, method, debug, show):
    """启发式搜索优化器"""
    
    # 处理初始电压
    if load_file.lower() == 'rms':
        init_v = get_init_V_by_rms()
    elif load_file:
        last_v = np.loadtxt(load_file)
        init_v = last_v.tolist()
    else:
        init_v = []

    config = {
        'root_dir': root_dir,
        'load_file': load_file,
        'cam_id': cam_id,
        'center': center,
        'exposure_time_ms': exposure_time_ms,
        'target_max_brightness': target_max_brightness,
        'epochs': epochs,
        'r_bucket': r_bucket,
        'delta': delta,
        'lr': lr,
        'weight_decay': weight_decay,
        'shrink_iter': shrink_iter,
        'shrink_ratio': shrink_ratio,
        'cam_size': cam_size,
        'method': method,
        'debug': debug,
        'show': show
    }
    logger.info(config)

    dm_unit_mask = np.ones(64, dtype=bool)
    dm_unit_mask[0] = False
    # dm_unit_mask[38:] = False
    
    recorder = Recorder(mark="pib", mode="max")
    
    with CameraStreamManager(cam_id=cam_id, exposure_time_ms=exposure_time_ms, skip_sampling=False) as cam,\
            NlightDM(keep_when_exit=True, max_neibor_diff=400, max_voltage=300, min_voltage=-200) as dm:
        
        if init_v is None or len(init_v) == 0:
            _init_v = np.zeros(dm.DM_Num, dtype=np.float64)
        else:
            _init_v = np.array(init_v)
        dm.send_voltages(_init_v, 1)

        cam.autoset_exposure_time_ms(n_sample=10, target_max_brightness=255, threshold=0.2)
        _img = cam.get_numpy_image(10)
        if center is None:
            h,w = _img.shape
            # TODO: 如果环围半径较小，使用质心而非形心;如果中间存在空洞使用形心，否则质心
            center = centroid(np.where(_img > np.max(_img[:max(int(h//50),2),:max(int(w//50),2)])
                                       , 1, 0))
        elif isinstance(center, str):
            _img = cam.get_numpy_image(10)
            if center == "mass":
                center = centroid(_img)
            elif center == 'max':
                center = np.unravel_index(np.argmax(_img), _img.shape)[::-1]
            elif center == 'shape':
                center = centroid(
                    np.where(_img > np.max(_img[:max(int(h//50),2),:max(int(w//50),2)]), 1, 0))
                
            else:
                raise ValueError(f"known center: {center}")

        else:
            center = center
        logger.info(f"Centroid: {center}, Max brightness: {np.max(_img)} @ {cam.exposure_time}ms")

        img_size = (cam_size, cam_size)
        img_size, _ = cam.reset_window(center, img_size)
        if 0<target_max_brightness<255:
            init_img = cam.autoset_exposure_time_ms(
                n_sample=5, target_max_brightness=target_max_brightness)
        else:
            cam.reset_exposure_time(exposure_time_ms)
            init_img = cam.get_numpy_image(5)
        logger.debug(f"Inital Image Max brightness: {np.max(init_img)} @ {cam.exposure_time}ms")
        img_size = init_img.shape[::-1]
        xv, yv = np.ogrid[-img_size[0]//2:img_size[0]//2, -img_size[1]//2:img_size[1]//2]

        if r_bucket <= 0:
            r_bucket = radius(init_img, center=center, energy=0.9) * shrink_ratio
        
        imgmesh_dist = ((xv) ** 2 + (yv) ** 2).transpose()
        dist = np.sqrt(imgmesh_dist)
        bucket_mask = dist <= r_bucket
        pib_mask = dist <= 7  # 默认理想光斑半径为7
        
        # 创建启发式搜索优化器
        optimizer = DMOptimizer(
            dim=dm.DM_Num,  # 变形镜单元数量
            method=method,
            dm_unit_mask=dm_unit_mask,
            dm_bounds=(dm.min_voltage, dm.max_voltage)
        )
        
        # 定义适应度函数
        def fitness_func(voltages):
            # 发送电压到变形镜
            dm.send_voltages(voltages)
            # 获取图像
            img = cam.get_numpy_image(5)
            # 计算PIB值（我们希望最大化PIB，所以返回负值）
            pib = np.sum(img[pib_mask]).astype(float)
            return -pib  # 负值因为我们是最小化问题
        
        # 初始化记录器
        init_pib = np.sum(init_img[pib_mask]).astype(float)
        recorder.append({
            "J": init_pib,
            "pib": init_pib,
            "_p%": init_pib / np.sum(init_img[init_img>2]).astype(float),
            "_max_r": r_bucket,
            "_v": _init_v,
            "_img": init_img,
            "_diff": 0,
            "lr": lr,
            "r": r_bucket,
            "delta": delta,
            "_epoch": 0,
            "exp_t": cam.exposure_time,
            "max_brt": np.max(init_img),
        })
        
        best_solution = _init_v.copy()
        best_fitness = -init_pib  # 负值因为我们是最小化问题
        
        # 运行优化
        for epoch in range(1, epochs+1):
            # 运行一步优化
            solution, fitness = optimizer.optimizer.search_step(fitness_func)
            
            # 更新最优解
            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = solution.copy()
                
            # 获取当前图像用于记录
            current_img = cam.get_numpy_image(5)
            current_pib = np.sum(current_img[pib_mask]).astype(float)
            
            # 记录日志
            log = {
                "J": -fitness,  # 转换回正值
                "_p%": current_pib / np.sum(current_img[current_img>2]).astype(float),
                "_max_r": r_bucket,
                "pib": current_pib,
                "_diff": 0,  # 在启发式搜索中没有明确的梯度差异
                "lr": lr,
                "r": r_bucket,
                "delta": delta,
                "_epoch": epoch,
                "_v": solution,
                "_img": current_img,
                "exp_t": cam.exposure_time,
                "max_brt": np.max(current_img),
            }
            recorder.append(log)
            
            if epoch % 100 == 0:
                logger.info(f"Epoch {epoch}: PIB = {current_pib:.4f}, Fitness = {-fitness:.4f}")
        
        # 保存结果
        res_df = recorder.dataframe
        saved_dir = f'{root_dir}/flatten_voltages/{datetime.now().strftime("%Y%m%d")}'
        os.makedirs(saved_dir, exist_ok=True)
        recorder.save_best(saved_dir, target="_v",
                          process_fn=lambda x: np.around(x).astype(int), fmt="%d")
            
        if debug:
            save_dir = gen_date_dir(f'{root_dir}/heuristic_search')
            saved_file_name = gen_file_path_uuid(save_dir, 'pkl')
            res_df.to_pickle(saved_file_name, compression='zip')

            # 保存配置
            with open(saved_file_name.with_suffix('.json'), 'w', encoding='utf8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)

            # 创建图表
            fig, ax = plt.subplots(2, 2, figsize=(12, 8))
            # init image
            plot_funcs["img"](res_df.iloc[0]["_img"], ax[0, 0], f"Init Image, pib={res_df.iloc[0]['pib']:.3f}")
            # best image
            best_iter, (max_j_id, max_j) = recorder.get_best_iter()
            axim = plot_funcs["img"](res_df.iloc[max_j_id]["_img"], ax[0, 1], f"Best Image, pib={max_j:.3f}")
            cbar = fig.colorbar(axim, ax=[ax[0, 0], ax[0, 1]], orientation='horizontal')
            # pib history
            plot_funcs["pib_history"](res_df["pib"], ax[1, 0])
            # best voltages plot bar
            plot_funcs["voltages"](best_iter["_v"], ax[1, 1], "Best Voltages")
            
            plt.savefig(saved_file_name.with_suffix('.png'))
            plt.close()
        
        click.echo(f"启发式搜索优化完成，最优PIB值: {-best_fitness:.4f}")
        

if __name__ == "__main__":
    try:
        coredumpy.patch_except(directory='logs/debug/error')
    except:
        logger.error("coredumpy初始化失败")
        pass  # 忽略coredumpy初始化错误
    run()