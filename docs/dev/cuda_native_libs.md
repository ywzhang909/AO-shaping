# CUDA 原生库入口硬规则（外迁自根 `AGENTS.md` §CUDA 原生库入口）

> 详细 cuBLAS DLL 绑定数据与实测表见 [docs/environment.md](../../docs/environment.md) §cuBLAS DLL 与 PATH。
> 门禁：`tests/ao_shaping/test_native_cuda_import_gate.py`；逃生开关 `AO_SKIP_CUBLAS_PRELOAD`。

## 规则

1. **禁止**新增「import 期就绑定 CUDA 原生库」的入口：`src/` 下任何模块级（含模块级 `try:` / 类体内）的 `import cupy`、`import OOPAO`，其**之前**必须调用 `ao_shaping._cublas_preload.pin_cublas()`。
2. **禁止**把这些 import 放进 `ao_shaping/__init__.py`（违反 `drivers/AGENTS.md` 的「import 不加载原生库、不落盘」契约）。
3. 拿不准就**延迟 import**（放函数内）：CuPy 只是可选加速时，最干净的做法是让它根本不在 import 期出现（`utils/image/spots_calc.py` 的 4 个 `*_cupy` 函数即此类）。
4. **已实测不成立、禁止再当修法提出**：`.venv` 隔离与 `uv run` 不清理 `PATH`；`os.add_dll_directory(torch/lib)` 对 cupy 裸名加载无效；从 `PATH` 删 CUDA 目录后 scoop 那份仍在；只钉 torch 的 `cublasLt`（不钉 `cublas`）会让 cupy 全线报错。
5. 根治只有一条：**环境里只留一套 CUDA minor**，且它必须是完整集合（含 cusolver/cufft/curand）；torch 只自带 cublas/cublasLt/cudart 三个 `*64_13.dll`，永远构不成完整集合。本机已采用 `uv sync --all-groups --extra torch-cu132`；换机器时按本机 `PATH` 首位 toolkit 的 minor 选 extra，uv 侧对照见 `pyproject.toml` 的 extras 注释。
