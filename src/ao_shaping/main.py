import click
# import dotenv

from ao_shaping.wf_runner import run as wf_run
from ao_shaping.axis_beam_runner import run as axis_beam_run
from ao_shaping.pipeline_runner import run as pipeline_run


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

if __name__ == "__main__":
    import coredumpy
    coredumpy.patch_except(directory='logs/debug/error')
    main()