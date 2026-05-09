from time import sleep

import click
from typing import Literal
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from loguru import logger

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
from ao_shaping.drivers.wfs.thorlab_wfs import WFSManager, MlaRes
from ao_shaping.utils.matrix_utils import calc_n_zernike_terms
from ao_shaping.utils.display import ZernikeCalibrationDisplay
from ao_shaping.utils.cli_helpers import parse_tuple, setup_coredumpy, get_timestamp_str


class DitheredReference:
    """Dithered reference measurement for SLM phase averaging.

    Applies sub-wavelength random phase dithering to average out
    pixelation steps and liquid crystal local relaxation errors.
    """

    def __init__(
        self,
        slm,
        dither_amp: float = 0.03,
        n_dither: int = 30,
        wait_time: float = 0.05,
    ):
        """Initialize dithered reference.

        Args:
            slm: ZernikeSLM instance
            dither_amp: Dithering amplitude in wavelength units (0.02-0.05 typical)
            n_dither: Number of dithering samples to average
            wait_time: Wait time after loading phase (seconds)
        """
        self.slm = slm
        self._slm = slm._slm
        self.dither_amp = dither_amp
        self.n_dither = n_dither
        self.wait_time = wait_time

    def measure(self, wfs, base_phase: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
        """Measure reference slopes with dithering average.

        Args:
            wfs: WFS instance with get_spot_deviation method
            base_phase: Base phase to add dither to (None = zero flat)

        Returns:
            tuple: (s_ref, diagnostics)
                - s_ref: Median reference slopes (flattened concat of dev_x and dev_y)
                - diagnostics: dict with snr, std, n_samples
        """
        if base_phase is None:
            h, w = self._slm.Panel_Res[1], self._slm.Panel_Res[0]
            base_phase = np.zeros((h, w), dtype=np.float64)

        slopes_list = []
        for i in range(self.n_dither):
            noise = np.random.randn(*base_phase.shape)
            noise = gaussian_filter(noise, sigma=20)
            noise = noise / np.std(noise) * self.dither_amp * 2 * np.pi

            phase_rad = (base_phase + noise) % (2 * np.pi)
            phase_gray = self._slm._phase_to_gray(phase_rad)
            self._slm.display_data(phase_gray)
            sleep(self.wait_time)

            dev_x, dev_y = wfs.get_spot_deviation(cancel_tile=False)
            s = np.concatenate([dev_x.flatten(), dev_y.flatten()])
            slopes_list.append(s)

        slopes_arr = np.array(slopes_list)
        s_ref = np.median(slopes_arr, axis=0)
        std = np.std(slopes_arr, axis=0)

        snr_linear = np.linalg.norm(s_ref) / (np.linalg.norm(std) + 1e-10)
        snr_db = 20 * np.log10(snr_linear + 1e-10)

        diagnostics = {
            "n_samples": self.n_dither,
            "dither_amp": self.dither_amp,
            "snr_linear": float(snr_linear),
            "snr_db": float(snr_db),
            "std_mean": float(np.mean(std)),
        }

        logger.info(f"Dithered reference SNR: {snr_db:.1f} dB (n={self.n_dither})")

        return s_ref, diagnostics


@click.command('zernike-matrix')
@click.pass_context
@click.option('--n-max', default=10, help='Zernike最大阶数')
@click.option('--magnitude', default=0.5, help='扰动幅度 (波长, 0=自动优化)')
@click.option('--n-averages', 'n_averages', default=3, help='每次WFS读取次数 (M)')
@click.option('--n-cycles', 'n_cycles', default=1, help='正负交替循环次数 (N)')
@click.option('--wait-time', 'wait_time', default=0.1, help='等待时间 (秒)')
@click.option('--output', 'output_path', default='data/zernike_response_matrix', help='输出文件路径')
@click.option('--slm-number', 'slm_number', default=1, help='SLM设备编号')
@click.option('--shift-x', 'shift_x', type=int, default=0, help='SLM X方向平移像素 (正=右, 负=左)')
@click.option('--shift-y', 'shift_y', type=int, default=0, help='SLM Y方向平移像素 (正=下, 负=上)')
@click.option('--wavelength', default=1064, help='工作波长 (nm)')
@click.option('--mla-index', 'mla_index', type=click.Choice(['512', '540', '600', '768', '1280']), default='512', help='MLA分辨率 (512, 540, 600, 768, 1280)')
@click.option('--exp-time', 'exp_time', type=float, default=0.0, help='曝光时间 (ms, 0=自动)')
@click.option('--auto-exposure/--no-auto-exposure', 'auto_exposure', default=True, help='启用WFS自动曝光 (默认开启)')
@click.option('--high-speed', 'high_speed', is_flag=True, default=False, help='启用高速模式')
@click.option('--use-custom-ref', 'use_custom_ref', is_flag=True, default=False, help='使用自定义参考文件')
@click.option('--pupil-diameter', 'pupil_diameter', type=float, default=2.0, help='瞳孔直径 (mm)')
@click.option('--pupil-center', callback=parse_tuple, default="(0,0)", help='瞳孔中心坐标 (默认: (0,0))')
@click.option('--no-inverses', 'compute_inverses', default=True, flag_value=False, help='不计算逆矩阵')
@click.option('--no-excluded-piston', 'excluded_piston', default=True, flag_value=False, help='不排除piston模式')
@click.option('--excluded-tip-tilt', 'excluded_tip_tilt', default=False, flag_value=True, help='排除tip/tilt模式 (Z2, Z3)')
@click.option('--display/--no-display', default=False, help='显示实时pygame显示')
@click.option('--debug', 'debug', is_flag=True, default=None, help='启用调试模式 (保存原始测量数据)')
@click.option('--auto-optimize/--no-auto-optimize', 'auto_optimize_amplitude', default=True, help='自动优化扰动幅度 (magnitude=0时)')
@click.option('--optimize-n-avg', 'optimize_n_avg', default=10, help='幅度优化时的WFS读取次数')
@click.option('--n-magnitudes', 'n_magnitudes', default=0, help='自动生成N个不同扰动幅度并分别保存 (0=禁用)')
@click.option('--dither/--no-dither', 'use_dither', default=False, help='使用亚波长抖动平均测量参考斜率')
@click.option('--dither-amp', 'dither_amp', default=0.03, help='抖动幅度 [λ], 建议0.02-0.05')
@click.option('--n-dither', 'n_dither', default=30, help='抖动样本数')
def run(
    ctx: click.Context,
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
    mla_index: Literal['512', '540', '600', '768', '1280'],
    exp_time: float,
    auto_exposure: bool,
    high_speed: bool,
    use_custom_ref: bool,
    pupil_diameter: float,
    pupil_center: tuple,
    compute_inverses: bool,
    excluded_piston: bool,
    excluded_tip_tilt: bool,
    display: bool,
    debug: bool | None,
    auto_optimize_amplitude: bool,
    optimize_n_avg: int,
    n_magnitudes: int,
    use_dither: bool,
    dither_amp: float,
    n_dither: int,
):
    """获取Zernike响应矩阵

    支持 N 次正负交替循环测量 + M 次 WFS 读取取平均 + 方差跟踪 + 逆矩阵计算。

    多幅度模式 (--n-magnitudes N):
        自动生成N个不同扰动幅度 (0.1到0.8λ)，分别标定并保存。

    调试模式 (--debug):
        保存每次测量的原始数据:
        - SLM相位图 (灰度值, 已应用shift)
        - WFS deviation数据
        - WFS Zernike系数 (averaging前)
    """
    n_max = n_max or DEFAULT_N_MAX
    magnitude = magnitude or DEFAULT_MAGNITUDE
    n_averages = n_averages or DEFAULT_N_AVERAGES
    n_cycles = n_cycles or DEFAULT_N_CYCLES
    wait_time = wait_time or DEFAULT_WAIT_TIME

    # Determine debug flag: use explicit value, or inherit from parent context
    if debug is None:
        debug = ctx.parent.obj.get("debug", False) if ctx.parent else False

    # Handle auto_exposure: when enabled, set exp_time to 0.0 to trigger auto-exposure
    effective_exp_time = 0.0 if auto_exposure else exp_time

    # Convert mla_index string to MlaRes enum
    mla_index_enum = MlaRes.from_str(mla_index)

    # Debug data saving: create callback if debug mode is enabled
    debug_data_callback = None
    debug_data_dir = None
    if debug:
        debug_data_dir = Path(output_path) / f"debug_{get_timestamp_str()}"
        debug_data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Debug mode enabled, saving raw data to: {debug_data_dir}")

        def debug_callback(
            mode_index: int,
            cycle: int,
            sample: int,
            slm_phase: np.ndarray,
            shift_x: int,
            shift_y: int,
            deviation_x: np.ndarray,
            deviation_y: np.ndarray,
            zernike_coeffs: np.ndarray,
            is_plus: bool,
        ) -> None:
            """Save debug data for each measurement."""
            sign_str = "plus" if is_plus else "minus"
            mode_dir = debug_data_dir / f"mode_{mode_index:03d}" / f"cycle_{cycle}" / sign_str
            mode_dir.mkdir(parents=True, exist_ok=True)

            np.save(mode_dir / f"sample_{sample:03d}_slm_phase.npy", slm_phase)

            if deviation_x is not None and len(deviation_x) > 0:
                np.save(mode_dir / f"sample_{sample:03d}_deviation_x.npy", deviation_x)
                np.save(mode_dir / f"sample_{sample:03d}_deviation_y.npy", deviation_y)

            if zernike_coeffs is not None and len(zernike_coeffs) > 0:
                np.save(mode_dir / f"sample_{sample:03d}_zernike_coeffs.npy", zernike_coeffs)

            import json
            meta = {
                "mode_index": mode_index,
                "cycle": cycle,
                "sample": sample,
                "shift_x": shift_x,
                "shift_y": shift_y,
                "is_plus": is_plus,
            }
            with open(mode_dir / f"sample_{sample:03d}_meta.json", "w") as f:
                json.dump(meta, f)

        debug_data_callback = debug_callback

    # Calculate n_slm_terms and n_wfs_terms before calibration
    n_remove = (1 if excluded_piston else 0) + (2 if excluded_tip_tilt else 0)
    n_slm_terms = calc_n_zernike_terms(n_max) - n_remove
    n_wfs_terms = calc_n_zernike_terms(n_max) - n_remove

    # Conditionally create display
    ui_display = None
    if display:
        ui_display = ZernikeCalibrationDisplay(n_wfs_terms=n_wfs_terms, n_slm_terms=n_slm_terms)

    try:
        with ZernikeSLM(slm_number=slm_number, wavelength=wavelength, n_max=n_max, shift_x=shift_x, shift_y=shift_y) as zslm:
            with WFSManager(
                mla_index=mla_index_enum,
                exp_time=effective_exp_time,
                high_speed=high_speed,
                use_custom_ref=use_custom_ref,
                pupil_diameter=pupil_diameter,
                pupil_center=pupil_center,
            ) as wfs:
                dither_diagnostics = None
                if not use_custom_ref:
                    zslm.set_flat()
                    sleep(0.5)

                    if use_dither:
                        click.echo(f"Measuring dithered reference: amp={dither_amp}, n={n_dither}")
                        dither = DitheredReference(
                            slm=zslm,
                            dither_amp=dither_amp,
                            n_dither=n_dither,
                            wait_time=wait_time,
                        )
                        s_ref, dither_diagnostics = dither.measure(wfs)
                        click.echo(f"Dithered ref SNR: {dither_diagnostics['snr_db']:.1f} dB")
                    else:
                        wfs.save_user_ref()
                        wfs.load_user_ref()

                magnitudes_to_run = []
                if n_magnitudes > 0:
                    magnitudes_to_run = np.linspace(0.1, 0.8, n_magnitudes).round(3).tolist()
                    click.echo(f"Multi-magnitude mode: running {n_magnitudes} calibrations with magnitudes {magnitudes_to_run}")
                elif magnitude == 0:
                    magnitudes_to_run = [None]
                else:
                    magnitudes_to_run = [magnitude]

                results = []
                for mag in magnitudes_to_run:
                    effective_magnitude = mag
                    if mag is None:
                        effective_magnitude = 0.0

                    mag_suffix = f"_mag{mag}" if mag is not None else "_auto"
                    mag_output_path = f"{output_path}{mag_suffix}"

                    click.echo(f"\n=== Calibration run: magnitude={effective_magnitude} ===")

                    result = calibrate_zernike_response_matrix(
                        zslm=zslm,
                        wfs=wfs,
                        n_max=n_max,
                        magnitude=effective_magnitude,
                        n_averages=n_averages,
                        n_cycles=n_cycles,
                        wait_time=wait_time,
                        excluded_piston=excluded_piston,
                        excluded_tip_tilt=excluded_tip_tilt,
                        compute_inverses=compute_inverses,
                        display=ui_display,
                        verbose=True,
                        debug_data_callback=debug_data_callback,
                        auto_optimize_amplitude=auto_optimize_amplitude,
                        optimize_n_avg=optimize_n_avg,
                    )

                    save_zernike_response_matrix(result, mag_output_path, include_inverses=compute_inverses)
                    click.echo(f"Saved: {mag_output_path}.h5")
                    results.append((mag, result))

                if len(results) == 1:
                    result = results[0][1]
                    click.echo(f"\n响应矩阵已保存到: {output_path}")
                else:
                    click.echo(f"\n多幅度标定完成: {len(results)} 组")
                    for mag, res in results:
                        click.echo(f"  mag={mag}: mean_var={res.mean_variance:.6f}, shape={res.matrix.shape}")

                click.echo(f"矩阵形状: {result.matrix.shape}")
                click.echo(f"平均方差: {result.mean_variance:.6f}")
                click.echo(f"最大方差: {result.max_variance:.6f}")
                click.echo(f"排除piston: {result.excluded_piston}, 排除tip/tilt: {result.excluded_tip_tilt}")
                if result.condition_number is not None:
                    click.echo(f"条件数: {result.condition_number:.2e}")
                if debug_data_dir is not None:
                    click.echo(f"调试数据已保存到: {debug_data_dir}")
    finally:
        if ui_display is not None:
            ui_display.close()


if __name__ == "__main__":
    setup_coredumpy()
    run()
