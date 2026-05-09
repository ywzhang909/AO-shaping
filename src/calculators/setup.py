from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# 定义扩展模块
extensions = [
    Extension(
        "adam_cython",
        ["adam.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "target_func_cython",
        ["target_func.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

setup(
    name="ao_shaping_cython_extensions",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': 3,
            'binding': True  # 这有助于生成更好的类型提示
        },
        annotate=True  # 生成HTML注解文件用于调试
    ),
    zip_safe=False,
)
