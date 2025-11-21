import click
from pathlib import Path

from ao_shaping.wf_runner import run as wf_run
from ao_shaping.axis_beam_runner import run as axis_beam_run
from ao_shaping.combined_runner import run as combined_run


@click.group()
@click.option('--debug', is_flag=True, help='是否开启调试模式')
@click.option('--dir', default='data', help='数据保存根目录 (default: data)')
@click.option("--show", is_flag=True, help="显示远场光斑CCD图像和优化历史 (default: False)")
@click.pass_context
def main(ctx, debug, dir, show):
    """AO Shaping 统一入口程序"""
    ctx.ensure_object(dict)
    ctx.obj['debug'] = debug
    ctx.obj['dir'] = Path(dir)
    ctx.obj['show'] = show

@main.command()
@click.option("-e", "--epochs", default=20_000, help="优化迭代次数 (default: 20000)")
@click.option("-r", "--wfs_res", default='768', help="WFS分辨率 (default: 768)")
@click.option("-p", "--pupil_diameter", default=2.7, help="瞳孔直径 (default: 2.7)")
@click.option("-t", "--early_stop_threshold", default=0.0, help="早停阈值 (default: 0.0)")
@click.pass_context
def wf(ctx, epochs, wfs_res, pupil_diameter, early_stop_threshold):
    """波前优化器"""
    debug = ctx.obj['debug']
    dir = ctx.obj['dir']
    show = ctx.obj['show']
    wf_run(dir=str(dir), epochs=epochs, wfs_res=wfs_res, pupil_diameter=pupil_diameter, 
           early_stop_threshold=early_stop_threshold, debug=debug, show=show)


@main.command()
@click.option("-f", "--load_file", default=None, help="加载优化结果文件 (default: None)")
@click.option("--cam_id", default=lambda: 0, help="远场光斑CCD设备ID (default: 0)")
@click.option("-c", "--center", default=None, help="场光斑CCD中心位置 (default: None)")
@click.option("-t","--exposure_time_ms", default=800, help="远场光斑CCD曝光时间 (毫秒) (default: 60)")
@click.option("-e", "--epochs", default=4_000, help="优化迭代次数 (default: 4000)")
@click.option("-r", "--r_bucket", default=18, help="渲染半径桶大小 (default: 18)")
@click.option("--delta", default=2, help="优化步长 (default: 2)")
@click.option("--lr", default=2, help="优化学习率 (default: 2)")
@click.option("--weight_decay", default=0.0, help="权重衰减 (default: 0.0)")
@click.option("--shrink_iter", default=300, help="优化迭代次数后收缩半径桶和步长 (default: 300)")
@click.option("--shrink_ratio", default=0.8, help="收缩半径桶和步长比例 (default: 0.8)")
@click.option("-s", "--cam_size", default=200, help="相机开窗大小 (default: 200*200)")
@click.pass_context
def pib(ctx, load_file, cam_id, center, exposure_time_ms, epochs, r_bucket, 
                       delta, lr, weight_decay, shrink_iter, shrink_ratio, cam_size):
    """轴向光束优化器"""
    debug = ctx.obj['debug']
    root_dir = ctx.obj['dir']
    show = ctx.obj['show']

    axis_beam_run(root_dir=str(root_dir), load_file=load_file, cam_id=cam_id, center=center, 
                  exposure_time_ms=exposure_time_ms, epochs=epochs, r_bucket=r_bucket, 
                  delta=delta, lr=lr, weight_decay=weight_decay, shrink_iter=shrink_iter, 
                  shrink_ratio=shrink_ratio, cam_size=cam_size, debug=debug, show=show)


@main.command()
@click.option("-f", "--load_file", default=None, help="加载优化结果文件 (default: None)")
@click.option("-e", "--epochs", default=8_000, help="优化迭代次数 (default: 8000)")
@click.option("-R", "--wfs_res", type=click.Choice(['768', '512']), default='768', help="WFS分辨率 (default: 768)")
@click.option("-p", "--pupil_diameter", default=2.7, help="瞳孔直径 (default: 2.7)")
@click.option("-c", "--cam_id", default=lambda: 0, help="远场光斑CCD设备ID (default: 0)")
@click.option("-t", "--exposure_time_ms", default=500, help="远场光斑CCD曝光时间 (毫秒) (default: 500)")
@click.option("-s", "--cam_size", default=160, help="相机开窗大小 (default: 160)")
@click.option("-r", "--rms_threshold", default=0.12, help="RMS阈值 (default: 0.12)")
@click.option("-u", "--dm_unit_mask", type=click.Choice(['all','inner','outer']), default='all', help="DM单元掩码 (default: all)")
@click.pass_context
def combine(ctx, load_file, epochs, wfs_res, pupil_diameter, cam_id, exposure_time_ms, cam_size, rms_threshold, dm_unit_mask):
    """组合优化器（先波前优化，再轴向光束优化）"""
    debug = ctx.obj['debug']
    dir = ctx.obj['dir']
    combined_run(dir=str(dir), load_file=load_file, epochs=epochs, wfs_res=wfs_res, pupil_diameter=pupil_diameter, 
                 cam_id=cam_id, exposure_time_ms=exposure_time_ms, cam_size=cam_size, rms_threshold=rms_threshold, 
                 dm_unit_mask=dm_unit_mask, debug=debug)


if __name__ == "__main__":
    import coredumpy
    coredumpy.patch_except(directory='logs/debug/error')
    main()