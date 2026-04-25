import click

from ao_shaping.wf_runner import run as wf_run
from ao_shaping.axis_beam_runner import run as axis_beam_run
from ao_shaping.pipeline_runner import run as pipeline_run


@click.group()
def main():
    """AO Shaping 统一入口程序"""
    pass


@click.command('zernike-matrix')
@click.option('--n-max', default=10, help='Zernike最大阶数')
@click.option('--magnitude', default=0.5, help='扰动幅度 (波长)')
@click.option('--n-averages', 'n_averages', default=3, help='每次WFS读取次数 (M)')
@click.option('--n-cycles', 'n_cycles', default=1, help='正负交替循环次数 (N)')
@click.option('--wait-time', 'wait_time', default=0.1, help='等待时间 (秒)')
@click.option('--output', 'output_path', default='data/zernike_response_matrix', help='输出文件路径')
@click.option('--slm-number', 'slm_number', default=1, help='SLM设备编号')
@click.option('--wavelength', default=1064, help='工作波长 (nm)')
@click.option('--no-inverses', 'compute_inverses', default=True, flag_value=False, help='不计算逆矩阵')
@click.option('--no-excluded-piston', 'excluded_piston', default=True, flag_value=False, help='不排除piston模式')
def zernike_matrix_run(
    n_max: int,
    magnitude: float,
    n_averages: int,
    n_cycles: int,
    wait_time: float,
    output_path: str,
    slm_number: int,
    wavelength: int,
    compute_inverses: bool,
    excluded_piston: bool,
):
    """校准Zernike响应矩阵 (增强版)

    支持 N 次正负交替循环测量 + M 次 WFS 读取取平均 + 方差跟踪 + 逆矩阵计算。
    """
    from ao_shaping.optimizer.wf.zernike_response_matrix import (
        calibrate_zernike_response_matrix,
        save_zernike_response_matrix,
        DEFAULT_N_MAX,
        DEFAULT_MAGNITUDE,
        DEFAULT_N_AVERAGES,
        DEFAULT_N_CYCLES,
        DEFAULT_WAIT_TIME,
    )
    from ao_shaping.drivers.slm import ZernikeSLM
    from ao_shaping.drivers import ThorlabWFS

    n_max = n_max or DEFAULT_N_MAX
    magnitude = magnitude or DEFAULT_MAGNITUDE
    n_averages = n_averages or DEFAULT_N_AVERAGES
    n_cycles = n_cycles or DEFAULT_N_CYCLES
    wait_time = wait_time or DEFAULT_WAIT_TIME

    with ZernikeSLM(slm_number=slm_number, wavelength=wavelength, n_max=n_max) as zslm:
        with ThorlabWFS() as wfs:
            result = calibrate_zernike_response_matrix(
                zslm=zslm,
                wfs=wfs,
                n_max=n_max,
                magnitude=magnitude,
                n_averages=n_averages,
                n_cycles=n_cycles,
                wait_time=wait_time,
                excluded_piston=excluded_piston,
                compute_inverses=compute_inverses,
                verbose=True,
            )

    save_zernike_response_matrix(result, output_path, include_inverses=compute_inverses)
    click.echo(f"响应矩阵已保存到: {output_path}")
    click.echo(f"矩阵形状: {result.matrix.shape}")
    click.echo(f"平均方差: {result.mean_variance:.6f}")
    click.echo(f"最大方差: {result.max_variance:.6f}")
    if result.condition_number is not None:
        click.echo(f"条件数: {result.condition_number:.2e}")


# 将wf_run命令添加到main组中
main.add_command(wf_run, name='wf')

# 将axis_beam_run命令添加到main组中
main.add_command(axis_beam_run, name='pib')

# 将combined_run命令添加到main组中
main.add_command(pipeline_run, name='pipeline')

# 添加zernike-matrix命令
main.add_command(zernike_matrix_run, name='zernike-matrix')