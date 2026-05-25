import click

from ao_shaping.runners.wf_runner import run as wf_run
from ao_shaping.runners.axis_beam_runner import run as axis_beam_run
from ao_shaping.runners.pipeline_runner import run as pipeline_run
from ao_shaping.runners.zernike_matrix_runner import run as zernike_matrix_run
from ao_shaping.runners.rms_zernike_runner import run as rms_zernike_run
from ao_shaping.runners.ga_zernike_runner import run as ga_zernike_run
from ao_shaping.runners.combined_runner import run as combined_run


@click.group()
def main():
    """AO Shaping 统一入口程序"""
    pass


# 将wf_run命令添加到main组中
main.add_command(wf_run, name='wf')

# 将axis_beam_run命令添加到main组中
main.add_command(axis_beam_run, name='pib')

# 将combined_run命令添加到main组中
main.add_command(pipeline_run, name='pipeline')

# 添加zernike-matrix命令
main.add_command(zernike_matrix_run, name='zernike-matrix')

# 添加rms-zernike命令 (Zernike RMS优化)
main.add_command(rms_zernike_run, name='rms-zernike')

# 添加ga-zernike命令 (遗传算法Zernike优化)
main.add_command(ga_zernike_run, name='ga-zernike')

# 添加combined命令 (AdaMOD综合PIB优化器)
main.add_command(combined_run, name='combined')


if __name__ == '__main__':
    main()