from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='scan',
    ext_modules=[CUDAExtension('scan', [
        'scan.cpp',
        'scan_cuda.cu',
    ])],
    cmdclass={'build_ext': BuildExtension},
    extra_compile_args={'nvcc': '-gencode=arch=compute_75,code=sm_75'},
)
