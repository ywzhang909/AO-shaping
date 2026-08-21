# =============================================================================
# iocs/common - 公共软 IOC 框架(纯 Python + caproto)
#
# 依赖:caproto>=1.0, numpy, pyyaml
# 安装:
#   pip install caproto numpy pyyaml
#
# 结构:
#   ao_epics_common/        可导入的公共包
#       __init__.py         导出全部公共组件
#       ioc_config.py       ioc.yaml 声明式配置加载/校验
#       ioc_runner.py       caproto 服务端运行器(run_ioc)
#       device_manager.py   设备生命周期管理(open/close/状态)
#       slm_rules.py        SLM 安全规则(槽位轮换、灰度 RAW、波形校验)
#       serve.py            通用启动入口
#
# 单元测试:
#   python -m pytest iocs/common/tests/
# =============================================================================
