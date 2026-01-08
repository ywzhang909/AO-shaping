import click
# import dotenv

from ao_shaping.wf_runner import run_spgd_rms, run_heuristic_rms
from ao_shaping.axis_beam_runner import run_spgd_pib, run_heuristic_pib
from ao_shaping.pipeline_runner import run as pipeline_run


@click.group()
def main():
    """AO Shaping 统一入口程序"""
    pass

main.add_command(run_spgd_rms, name='wf')

main.add_command(run_spgd_pib, name='pib')

main.add_command(pipeline_run, name='pipeline')

main.add_command(run_heuristic_pib, name='heuristic-pib')

main.add_command(run_heuristic_rms, name='heuristic-rms')

if __name__ == "__main__":
    import coredumpy
    coredumpy.patch_except(directory='logs/debug/error')
    main()