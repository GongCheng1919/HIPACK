import subprocess
import os
from setuptools import setup, Extension,Command
from torch.utils import cpp_extension
from typing import Dict, List, Optional, Union, Tuple
# from torch.utils.cpp_extension import CppExtension, BuildExtension

class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        os.system('rm -vrf ./build')
        os.system('rm -vrf ./dist')
        os.system('rm -vrf ./*.egg-info')
        os.system('rm -vrf ./*.so')

raw_build_extensions = cpp_extension.BuildExtension.build_extensions

def new_build_extensions(self):
    raw_build_extensions(self)
    print("###BuildExtensions: ",self.extensions[0].extra_compile_args)

cpp_extension.BuildExtension.build_extensions = new_build_extensions

raw_write_ninja_file_and_compile_objects = cpp_extension._write_ninja_file_and_compile_objects

def new_write_ninja_file_and_compile_objects(sources: List[str],
        objects,
        cflags,
        post_cflags,
        cuda_cflags,
        cuda_post_cflags,
        cuda_dlink_post_cflags,
        build_directory: str,
        verbose: bool,
        with_cuda: Optional[bool]):
    print("#########cflags: ",cflags)
    print("#########post_cflags: ",post_cflags)
    print("#########cuda_cflags: ",cuda_cflags)
    print("#########cuda_post_cflags: ",cuda_post_cflags)
    print("#########cuda_dlink_post_cflags: ",cuda_dlink_post_cflags)
    exclude_flags = ['-g', '-fwrapv',"-O2", "-pthread", "-Wno-unused-result", "-Wsign-compare"]
    # 如果pst_cflags中有-O1,-O2,-O3,-Ofast之类的标志，则去除-O2
    cflags = [arg for arg in cflags if arg not in exclude_flags]
    raw_write_ninja_file_and_compile_objects(sources, objects, cflags, post_cflags, cuda_cflags, cuda_post_cflags, cuda_dlink_post_cflags, build_directory, verbose, with_cuda)

cpp_extension._write_ninja_file_and_compile_objects = new_write_ninja_file_and_compile_objects


raw_write_ninja_file_to_build_library = cpp_extension._write_ninja_file_to_build_library

def new_write_ninja_file_to_build_library(name,
        sources: List[str],
        extra_cflags,
        extra_cuda_cflags,
        extra_ldflags,
        extra_include_paths,
        build_directory: str,
        verbose: bool,
        with_cuda: Optional[bool],
        is_standalone: bool = False):
    print("#########extra_cflags: ",extra_cflags)
    print("#########extra_ldflags: ",extra_ldflags)
    # exclude_flags = ['-g', '-fwrapv',"-O2", "-pthread", "-Wno-unused-result", "-Wsign-compare"]
    # 如果pst_cflags中有-O1,-O2,-O3,-Ofast之类的标志，则去除-O2
    raw_write_ninja_file_to_build_library(name, sources, extra_cflags, extra_cuda_cflags, extra_ldflags, extra_include_paths, build_directory, verbose, with_cuda, is_standalone)

cpp_extension._write_ninja_file_to_build_library = new_write_ninja_file_to_build_library

class CustomBuildExtension(cpp_extension.BuildExtension):
    def build_extensions(self):
        # 自定义编译选项
        for extension in self.extensions:
            extension.extra_compile_args = ['-fopenmp', '-march=native', 
							  '-fconcepts-ts', '-std=c++17', '-Ofast', '-fstrict-overflow']  # 示例：只使用这些选项
            # 根据需要添加或删除特定的编译选项
            # 例如，移除'-g'和'-fwrapv'
            new_compile_args = [arg for arg in extension.extra_compile_args if arg not in ['-g', '-fwrapv',"-O2", "-pthread", "-Wno-unused-result", "-Wsign-compare"]]
            extension.extra_compile_args = new_compile_args

        # 调用父类的build_extensions方法来继续构建过程
        super().build_extensions()

# 设置环境变量以使用g++作为编译器
os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

# direct_conv2d_path = f"{os.environ['HOME']}/workspace/DirectConv/ours"
direct_conv2d_path = f"{os.environ['HOME']}/HIPACK/src"
machine_name = subprocess.check_output(['uname', '-m']).decode('utf-8').strip()

setup(name='direct_conv2d',
      version='1.0.7',  # 添加版本信息
      description='A direct convolution 2D extension for PyTorch.',  # 简短描述
      long_description=open('../README.md').read(),  # 长描述，从README.md文件读取
      long_description_content_type='text/markdown',  # 指定长描述内容的类型，这里是Markdown
      ext_modules=[cpp_extension.CppExtension(
          'direct_conv2d', ['./directconv2d.cpp'],
          extra_compile_args=['-fopenmp', '-march=native', "-DENABLE_OPENMP",
							  '-fconcepts-ts', '-std=c++17', '-Ofast', '-fstrict-overflow',
							  f'-I/usr/include/{machine_name}-linux-gnu/', 
                              f"{direct_conv2d_path}"],
          extra_link_args=['-lgomp','-lcpuinfo', 
						            '-lclog', '-fconcepts-ts',
						            '-lopenblas'],
          include_dirs=[f'/usr/include/{machine_name}-linux-gnu/',
                        f"{direct_conv2d_path}"],
          library_dirs=[f'/usr/include/{machine_name}-linux-gnu/']
      )],
      cmdclass={'build_ext': cpp_extension.BuildExtension,
                "clean": CleanCommand,})