from pathlib import Path

import click
import numpy as np
from loguru import logger

from ao_shaping.drivers import MlaRes, ThorlabWFS
from ao_shaping.drivers.slm import ZernikeSLM
from ao_shaping.utils.cli_helpers import (
    _get_wfs_res,
    get_date_dir_name,
    parse_tuple,
    setup_coredumpy,
)
from ao_shaping.utils.zernike_calc import calc_n_zernike_terms


def measure_rms(wfs: ThorlabWFS) -> float:
    """Measure wavefront RMS from WFS."""
    wfs.take_image(3)
    wf, statics = wfs.get_wavefront(cancel_tile=False)
    return statics.get('rms', np.inf)


def measure_defocus_coefficient(wfs: ThorlabWFS, remove_tilt: bool = False) -> float:
    """Measure Zernike defocus coefficient from WFS.
    
    Args:
        wfs: ThorlabWFS instance
        remove_tilt: Whether to remove tilt before fitting
        
    Returns:
        Defocus coefficient (in wavelengths)
    """
    wfs.take_image(5)
    wf, statics = wfs.get_wavefront(cancel_tile=remove_tilt)
    
    if 'zernike_coeffs' in statics:
        zernike_coeffs = statics['zernike_coeffs']
        if len(zernike_coeffs) > 4:
            return float(zernike_coeffs[4])
    
    return 0.0


def search_offset_by_defocus(
    slm: ZernikeSLM,
    wfs: ThorlabWFS,
    defocus_amplitude: float = 5.0,
    search_range: int = 100,
    search_step: int = 10,
    n_samples: int = 3,
) -> tuple[int, int, dict]:
    """Search for optimal SLM XY offset using defocus method.
    
    Apply Zernike defocus to SLM, measure defocus on WFS at different
    XY offsets, find the offset where WFS measures maximum defocus.
    
    Args:
        slm: ZernikeSLM instance
        wfs: ThorlabWFS instance
        defocus_amplitude: Zernike defocus amplitude (wavelengths)
        search_range: Search range in pixels (+/-)
        search_step: Step size in pixels
        n_samples: Number of samples per position for noise averaging
        
    Returns:
        (best_x, best_y, results_dict)
    """
    click.echo(f"\n{'='*60}")
    click.echo("SLM Offset Search (Defocus Method)")
    click.echo(f"Defocus amplitude: {defocus_amplitude}")
    click.echo(f"Search range: +/-{search_range} pixels, step: {search_step}")
    click.echo(f"Samples per position: {n_samples}")
    click.echo(f"{'='*60}\n")
    
    baseline_defocus = measure_defocus_coefficient(wfs)
    click.echo(f"Baseline WFS defocus: {baseline_defocus:.4f}")
    
    results = []
    best_defocus = -np.inf
    best_x, best_y = 0, 0
    
    for offset_y in range(-search_range, search_range + 1, search_step):
        for offset_x in range(-search_range, search_range + 1, search_step):
            slm.set_shift(offset_x, offset_y)
            
            slm.send_zernike({(2, 0): defocus_amplitude})
            
            defocus_measurements = []
            for _ in range(n_samples):
                d = measure_defocus_coefficient(wfs)
                defocus_measurements.append(d)
            
            avg_defocus = np.mean(defocus_measurements)
            std_defocus = np.std(defocus_measurements)
            
            results.append({
                'offset_x': offset_x,
                'offset_y': offset_y,
                'avg_defocus': avg_defocus,
                'std_defocus': std_defocus,
            })
            
            click.echo(f"Offset ({offset_x:3d}, {offset_y:3d}): defocus={avg_defocus:.4f}±{std_defocus:.4f}")
            
            if avg_defocus > best_defocus:
                best_defocus = avg_defocus
                best_x, best_y = offset_x, offset_y
    
    slm.set_shift(0, 0)
    slm.send_zernike({(2, 0): 0})
    
    click.echo(f"\n{'='*60}")
    click.echo(f"Best offset: ({best_x}, {best_y}) with defocus={best_defocus:.4f}")
    click.echo(f"{'='*60}\n")
    
    return best_x, best_y, {
        'baseline_defocus': baseline_defocus,
        'best_defocus': best_defocus,
        'results': results,
    }


def optimize_defocus(
    slm: ZernikeSLM,
    wfs: ThorlabWFS,
    offset_x: int,
    offset_y: int,
    defocus_range: float = 10.0,
    defocus_step: float = 0.5,
    n_samples: int = 3,
) -> tuple[float, float, dict]:
    """Search for optimal defocus value that minimizes wavefront RMS.
    
    At the given XY offset, search through different defocus amplitudes
    and find the one that gives minimum RMS (flattest wavefront).
    
    Args:
        slm: ZernikeSLM instance
        wfs: ThorlabWFS instance
        offset_x: SLM X offset
        offset_y: SLM Y offset
        defocus_range: Search range for defocus amplitude (+/-)
        defocus_step: Step size for defocus search
        n_samples: Number of samples per position
        
    Returns:
        (best_defocus, min_rms, results_dict)
    """
    click.echo(f"\n{'='*60}")
    click.echo("Phase 2: Defocus Optimization")
    click.echo(f"At offset: ({offset_x}, {offset_y})")
    click.echo(f"Defocus range: +/-{defocus_range}, step: {defocus_step}")
    click.echo(f"{'='*60}\n")
    
    slm.set_shift(offset_x, offset_y)
    
    baseline_rms = measure_rms(wfs)
    click.echo(f"Baseline RMS (no defocus): {baseline_rms:.4f}")
    
    results = []
    best_rms = np.inf
    best_defocus = 0.0
    
    defocus_values = np.arange(-defocus_range, defocus_range + defocus_step, defocus_step)
    
    for defocus_amp in defocus_values:
        slm.send_zernike({(2, 0): float(defocus_amp)})
        
        rms_measurements = []
        for _ in range(n_samples):
            rms = measure_rms(wfs)
            rms_measurements.append(rms)
        
        avg_rms = np.mean(rms_measurements)
        std_rms = np.std(rms_measurements)
        
        results.append({
            'defocus': defocus_amp,
            'rms': avg_rms,
            'std': std_rms,
        })
        
        click.echo(f"Defocus={defocus_amp:6.2f}: RMS={avg_rms:.4f}±{std_rms:.4f}")
        
        if avg_rms < best_rms:
            best_rms = avg_rms
            best_defocus = defocus_amp
    
    slm.send_zernike({(2, 0): 0})
    
    click.echo(f"\n{'='*60}")
    click.echo(f"Best defocus: {best_defocus:.2f} (RMS: {best_rms:.4f})")
    click.echo(f"{'='*60}\n")
    
    return float(best_defocus), float(best_rms), {
        'baseline_rms': baseline_rms,
        'best_rms': best_rms,
        'results': results,
    }


def search_offset_by_defocus_with_optimization(
    slm: ZernikeSLM,
    wfs: ThorlabWFS,
    defocus_amplitude: float = 5.0,
    search_range: int = 100,
    search_step: int = 10,
    n_samples: int = 3,
    defocus_range: float = 10.0,
    defocus_step: float = 0.5,
) -> tuple[int, int, float, float, dict]:
    """Two-phase search: XY offset + optimal defocus.
    
    Phase 1: Find optimal XY offset using defocus method
    Phase 2: Find optimal defocus value that minimizes RMS
    
    Args:
        slm: ZernikeSLM instance
        wfs: ThorlabWFS instance
        defocus_amplitude: Defocus amplitude for phase 1
        search_range: XY search range
        search_step: XY search step
        n_samples: Samples per position
        defocus_range: Defocus search range for phase 2
        defocus_step: Defocus search step for phase 2
        
    Returns:
        (best_x, best_y, best_defocus, min_rms, results_dict)
    """
    click.echo(f"\n{'='*60}")
    click.echo("SLM Offset & Defocus Optimization")
    click.echo("=" * 60)
    click.echo("Phase 1: XY Offset Search")
    click.echo(f"{'='*60}\n")
    
    baseline_defocus = measure_defocus_coefficient(wfs)
    baseline_rms = measure_rms(wfs)
    click.echo(f"Baseline - defocus: {baseline_defocus:.4f}, RMS: {baseline_rms:.4f}")
    
    results = {'phase1': [], 'phase2': {}}
    best_x, best_y = 0, 0
    best_phase1_defocus = -np.inf
    
    for offset_y in range(-search_range, search_range + 1, search_step):
        for offset_x in range(-search_range, search_range + 1, search_step):
            slm.set_shift(offset_x, offset_y)
            slm.send_zernike({(2, 0): defocus_amplitude})
            
            defocus_measurements = []
            for _ in range(n_samples):
                d = measure_defocus_coefficient(wfs)
                defocus_measurements.append(d)
            
            avg_defocus = np.mean(defocus_measurements)
            
            results['phase1'].append({
                'offset_x': offset_x,
                'offset_y': offset_y,
                'defocus': avg_defocus,
            })
            
            if avg_defocus > best_phase1_defocus:
                best_phase1_defocus = avg_defocus
                best_x, best_y = offset_x, offset_y
    
    click.echo(f"\nPhase 1 complete: best offset=({best_x}, {best_y}), defocus={best_phase1_defocus:.4f}")
    
    click.echo(f"\n{'='*60}")
    click.echo("Phase 2: Defocus Optimization")
    click.echo(f"{'='*60}\n")
    
    best_defocus, min_rms, phase2_results = optimize_defocus(
        slm=slm,
        wfs=wfs,
        offset_x=best_x,
        offset_y=best_y,
        defocus_range=defocus_range,
        defocus_step=defocus_step,
        n_samples=n_samples,
    )
    
    results['phase2'] = phase2_results
    
    slm.set_shift(0, 0)
    slm.send_zernike({(2, 0): 0})
    
    click.echo(f"\n{'='*60}")
    click.echo("Final Results:")
    click.echo(f"  XY Offset: ({best_x}, {best_y})")
    click.echo(f"  Optimal Defocus: {best_defocus:.2f}")
    click.echo(f"  Minimum RMS: {min_rms:.4f}")
    click.echo(f"  RMS Improvement: {baseline_rms - min_rms:.4f}")
    click.echo(f"{'='*60}\n")
    
    return best_x, best_y, best_defocus, min_rms, results


def search_offset_by_vortex(
    slm: ZernikeSLM,
    wfs: ThorlabWFS,
    vortex_charge: int = 1,
    search_range: int = 50,
    search_step: int = 5,
    n_samples: int = 3,
) -> tuple[int, int, dict]:
    """Search for optimal SLM XY offset using vortex phase method.
    
    Apply vortex (OAM) phase to SLM, the phase singularity creates a
    dark region in the center. Search for XY offset where the dark
    region is most pronounced (minimum intensity at center).
    
    Args:
        slm: ZernikeSLM instance
        wfs: ThorlabWFS instance
        vortex_charge: Topological charge of vortex (1, 2, 3, ...)
        search_range: Search range in pixels (+/-)
        search_step: Step size in pixels
        n_samples: Number of samples per position
        
    Returns:
        (best_x, best_y, results_dict)
    """
    click.echo(f"\n{'='*60}")
    click.echo("SLM Offset Search (Vortex Method)")
    click.echo(f"Vortex charge: {vortex_charge}")
    click.echo(f"Search range: +/-{search_range} pixels, step: {search_step}")
    click.echo(f"{'='*60}\n")
    
    height, width = slm._zernike_dm.resolution
    center_y, center_x = height // 2, width // 2
    
    y_coords, x_coords = np.meshgrid(
        np.arange(height) - center_y,
        np.arange(width) - center_x,
        indexing='ij'
    )
    
    rho = np.sqrt(x_coords**2 + y_coords**2)
    theta = np.arctan2(y_coords, x_coords)
    
    vortex_phase = vortex_charge * theta
    vortex_pattern = (vortex_phase / (2 * np.pi) * 255).astype(np.uint8)
    
    wfs.take_image(5)
    wf, baseline_statics = wfs.get_wavefront(cancel_tile=False)
    baseline_intensity = baseline_statics.get('total_intensity', 1e6)
    
    click.echo(f"Baseline WFS intensity: {baseline_intensity:.2f}")
    
    results = []
    best_metric = -np.inf
    best_x, best_y = 0, 0
    
    for offset_y in range(-search_range, search_range + 1, search_step):
        for offset_x in range(-search_range, search_range + 1, search_step):
            slm.set_shift(offset_x, offset_y)
            slm._slm.display_data(vortex_pattern)
            
            intensity_measurements = []
            for _ in range(n_samples):
                wfs.take_image(3)
                wf, statics = wfs.get_wavefront(cancel_tile=False)
                intensity = statics.get('total_intensity', 1e6)
                intensity_measurements.append(intensity)
            
            avg_intensity = np.mean(intensity_measurements)
            
            metric = baseline_intensity - avg_intensity
            
            results.append({
                'offset_x': offset_x,
                'offset_y': offset_y,
                'avg_intensity': avg_intensity,
                'metric': metric,
            })
            
            click.echo(f"Offset ({offset_x:3d}, {offset_y:3d}): intensity={avg_intensity:.2f}, metric={metric:.2f}")
            
            if metric > best_metric:
                best_metric = metric
                best_x, best_y = offset_x, offset_y
    
    slm.set_shift(0, 0)
    slm._slm.display_data(np.zeros((height, width), dtype=np.uint8))
    
    click.echo(f"\n{'='*60}")
    click.echo(f"Best offset: ({best_x}, {best_y}) with metric={best_metric:.2f}")
    click.echo(f"{'='*60}\n")
    
    return best_x, best_y, {
        'baseline_intensity': baseline_intensity,
        'best_metric': best_metric,
        'results': results,
    }


@click.command()
@click.option("-d", "--dir", default="data", help="数据保存根目录 (default: data)")
@click.option("-r", "--wfs_res", default='1024', help="WFS分辨率 (default: 1024)")
@click.option("-p", "--pupil_diameter", default=2.7, help="瞳孔直径 (default: 2.7)")
@click.option("-c", "--pupil_center", callback=parse_tuple, default="(0,0)", help="瞳孔中心坐标 (default: (0,0))")
@click.option("--wavelength", default=532, help="SLM波长 (nm, default: 532)")
@click.option("--slm-number", default=1, help="SLM设备编号 (default: 1)")
@click.option("--remove-tilt", is_flag=True, help="移除波前测量中的倾斜项")
@click.option("--method", type=click.Choice(['defocus', 'vortex']), default='defocus', help="搜索方法 (default: defocus)")
@click.option("--defocus-amp", default=5.0, help="离焦幅度用于偏移搜索(波长, default: 5.0)")
@click.option("--vortex-charge", default=1, help="涡旋电荷数 (default: 1)")
@click.option("--search-range", default=50, help="XY搜索范围像素 (default: 50)")
@click.option("--search-step", default=10, help="XY搜索步长像素 (default: 10)")
@click.option("--defocus-range", default=10.0, help="离焦优化范围(波长, default: 10.0)")
@click.option("--defocus-step", default=0.5, help="离焦优化步长 (default: 0.5)")
@click.option("--n-samples", default=3, help="每个位置采样次数 (default: 3)")
@click.option("--save-csv", is_flag=True, help="保存结果到CSV")
def run(dir, wfs_res, pupil_diameter, pupil_center, wavelength, slm_number,
        remove_tilt, method, defocus_amp, vortex_charge, search_range,
        search_step, n_samples, defocus_range, defocus_step, save_csv):
    """SLM XY偏移自动搜索工具
    
    使用两种方法搜索最优SLM XY偏移:
    - defocus: 使用Zernike离焦相位,WFS测量离焦值最大化
    - vortex: 使用涡旋相位,寻找相位奇点
    
    示例:
        python -m ao_shaping.runners.slm_offset_runner --method defocus --search-range 50
    """
    with (
        ZernikeSLM(
            slm_number=slm_number,
            wavelength=wavelength,
            n_max=4,
            shift_x=0,
            shift_y=0,
        ) as slm,
        ThorlabWFS(
            _get_wfs_res(wfs_res),
            use_custom_ref=False,
            high_speed=True,
            pupil_diameter=pupil_diameter,
            pupil_center=pupil_center,
        ) as wfs,
    ):
        if method == 'defocus':
            best_x, best_y, best_defocus, min_rms, results = search_offset_by_defocus_with_optimization(
                slm=slm,
                wfs=wfs,
                defocus_amplitude=defocus_amp,
                search_range=search_range,
                search_step=search_step,
                n_samples=n_samples,
                defocus_range=defocus_range,
                defocus_step=defocus_step,
            )
            click.echo(f"最优偏移量: shift_x={best_x}, shift_y={best_y}")
            click.echo(f"最优离焦值: {best_defocus:.2f} 波长")
            click.echo(f"最小RMS: {min_rms:.4f}")
            click.echo(f"\n建议: 使用参数 --shift-x {best_x} --shift-y {best_y} 运行优化器")
            if best_defocus != 0:
                click.echo(f"或固定离焦: --shift-x {best_x} --shift-y {best_y} 并在优化器中设置初始离焦系数")
        else:
            best_x, best_y, results = search_offset_by_vortex(
                slm=slm,
                wfs=wfs,
                vortex_charge=vortex_charge,
                search_range=search_range,
                search_step=search_step,
                n_samples=n_samples,
            )
            click.echo(f"最优偏移量: shift_x={best_x}, shift_y={best_y}")
            click.echo(f"\n建议: 使用参数 --shift-x {best_x} --shift-y {best_y} 运行优化器")
        
        if save_csv:
            import csv
            save_dir = Path(dir) / "slm_offset" / get_date_dir_name()
            save_dir.mkdir(parents=True, exist_ok=True)
            
            if method == 'defocus':
                csv_path = save_dir / f"{method}_phase1_results.csv"
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=results['phase1'][0].keys())
                    writer.writeheader()
                    writer.writerows(results['phase1'])
                click.echo(f"XY偏移搜索结果已保存: {csv_path}")
                
                if results['phase2']['results']:
                    csv_path = save_dir / f"{method}_phase2_defocus.csv"
                    with open(csv_path, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=results['phase2']['results'][0].keys())
                        writer.writeheader()
                        writer.writerows(results['phase2']['results'])
                    click.echo(f"离焦优化结果已保存: {csv_path}")
            else:
                csv_path = save_dir / f"{method}_offset_results.csv"
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=results['results'][0].keys())
                    writer.writeheader()
                    writer.writerows(results['results'])
                click.echo(f"结果已保存: {csv_path}")


if __name__ == "__main__":
    setup_coredumpy()
    run()