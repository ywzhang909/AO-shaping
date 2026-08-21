# =============================================================================
# ioc-slm - Santec SLM-200 软 IOC(Windows Host 原生运行)
#
# 硬件:Santec SLM-200,USB/FTDI,SLMFunc.dll(ctypes)
# 驱动:复用 src/ao_shaping/drivers/slm/santec/driver.py 的 SantecSLM200
# 关键安全规则(AGENTS.md):
#   1) 灰度 RAW 路径:平坦相位用 np.full((h,w),gray,dtype=np.uint16),
#      禁止经 create_phase_from_array()(它把输入当弧度)。
#   2) 内存槽轮换:连续 write_phase+display_memory 到同一槽位是 no-op,
#      必须轮换(默认 itertools.cycle([3,4,5]))。
#
# 运行(Windows Host):
#   . ..\..\environment.ps1
#   $env:PYTHONPATH = "..\common;$env:PYTHONPATH"
#   python -m ao_epics_common.serve config\ioc.yaml src.slm_ioc.SLMIoc
#
# 验证(本机):
#   python -c "import epics; print(epics.get_pv('SLM-01:Wavelength'))"
# 验证(WSL,经 CA Gateway):
#   caget SLM-01:Wavelength
# =============================================================================
