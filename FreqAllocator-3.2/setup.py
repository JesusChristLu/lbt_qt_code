from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize([
        Extension(
            "err_model_c",              # 编译后的模块名称
            sources=[r"F:\OneDrive\vs experiment\FreqAllocator-3.2\freq_allocator\model\err_model_c.pyx"],# Cython 源文件
            include_dirs=[numpy.get_include()],  # 包含 numpy 的头文件目录
            # 可能需要的其他选项
        )
    ]),
    install_requires=[
        'numpy',      # 确保 numpy 在安装依赖中列出
        'networkx',   # 确保 networkx 在安装依赖中列出
        'scipy'       # 如果使用了 scipy.optimize，也要列出
        # 其他可能的依赖项
    ]
)