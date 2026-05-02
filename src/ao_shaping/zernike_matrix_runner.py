import click
import coredumpy

from ao_shaping.utils import logger
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
from ao_shaping.drivers.wfs.thorlab_wfs import WFSManager
from ao_shaping.utils.matrix_utils import calc_n_zernike_terms
from ao_shaping.utils.display import ZernikeCalibrationDisplay


@click.command('zernike-matrix')
@click.option('--n-max', default=10, help='Zernike最大阶数')
@click.option('--magnitude', default=0.5, help='扰动幅度 (波长)')
@click.option('--n-averages', 'n_averages', default=3, help='每次WFS读取次数 (M)')
@click.option('--n-cycles', 'n_cycles', default=1, help='正负交替循环次数 (N)')
@click.option('--wait-time', 'wait_time', default=0.1, help='等待时间 (秒)')
@click.option('--output', 'output_path', default='data/zernike_response_matrix', help='输出文件路径')
@click.option('--slm-number', 'slm_number', default=1, help='SLM设备编号')
@click.option('--shift-x', 'shift_x', type=int, default=0, help='SLM X方向平移像素 (正=右, 负=左)')
@click.option('--shift-y', 'shift_y', type=int, default=0, help='SLM Y方向平移像素 (正=下, 负=上)')
@click.option('--wavelength', default=1064, help='工作波长 (nm)')
@click.option('--no-inverses', 'compute_inverses', default=True, flag_value=False, help='不计算逆矩阵')
@click.option('--no-excluded-piston', 'excluded_piston', default=True, flag_value=False, help='不排除piston模式')
@click.option('--display/--no-display', default=False, help='显示实时pygame显示')
def run(
    n_max: int,
    magnitude: float,
    n_averages: int,
    n_cycles: int,
    wait_time: float,
    output_path: str,
    slm_number: int,
    shift_x: int,
    shift_y: int,
    wavelength: int,
    compute_inverses: bool,
    excluded_piston: bool,
    display: bool,
):
    """获取Zernike响应矩阵

    支持 N 次正负交替循环测量 + M 次 WFS 读取取平均 + 方差跟踪 + 逆矩阵计算。
    """
    n_max = n_max or DEFAULT_N_MAX
    magnitude = magnitude or DEFAULT_MAGNITUDE
    n_averages = n_averages or DEFAULT_N_AVERAGES
    n_cycles = n_cycles or DEFAULT_N_CYCLES
    wait_time = wait_time or DEFAULT_WAIT_TIME

    # Calculate n_slm_terms before calibration
    n_slm_terms = calc_n_zernike_terms(n_max) - (1 if excluded_piston else 0)

    # Conditionally create display
    ui_display = None
    if display:
        ui_display = ZernikeCalibrationDisplay(n_wfs_terms=66, n_slm_terms=n_slm_terms)

    try:
        with ZernikeSLM(slm_number=slm_number, wavelength=wavelength, n_max=n_max, shift_x=shift_x, shift_y=shift_y) as zslm:
            with WFSManager() as wfs:
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
                    display=ui_display,
                    verbose=True,
                )

        save_zernike_response_matrix(result, output_path, include_inverses=compute_inverses)
        click.echo(f"响应矩阵已保存到: {output_path}")
        click.echo(f"矩阵形状: {result.matrix.shape}")
        click.echo(f"平均方差: {result.mean_variance:.6f}")
        click.echo(f"最大方差: {result.max_variance:.6f}")
        if result.condition_number is not None:
            click.echo(f"条件数: {result.condition_number:.2e}")
    finally:
        if ui_display is not None:
            ui_display.close()


if __name__ == "__main__":
    try:
        coredumpy.patch_except(directory='logs/debug/error')
    except Exception:
        logger.error("coredumpy初始化失败")
    run()
