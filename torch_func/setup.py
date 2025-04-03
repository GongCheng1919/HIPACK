import os
import subprocess
from setuptools import setup, Command
from torch.utils import cpp_extension
from typing import List, Optional


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        os.system('rm -vrf ./build ./dist ./*.egg-info ./*.so')


class CustomBuildExtension(cpp_extension.BuildExtension):
    """Custom build extension with optimized compilation flags."""
    
    def build_extensions(self):
        # Customize compilation options for all extensions
        for extension in self.extensions:
            # Set optimized compilation flags
            extension.extra_compile_args = [
                '-fopenmp', 
                '-march=native',
                '-fconcepts-ts', 
                '-std=c++17', 
                '-Ofast'
            ]
            
            # Remove unwanted flags
            exclude_flags = [
                '-g', '-fwrapv', '-O2', '-pthread', 
                '-Wno-unused-result', '-Wsign-compare'
            ]
            extension.extra_compile_args = [
                arg for arg in extension.extra_compile_args 
                if arg not in exclude_flags
            ]
        
        # Call the parent build method
        super().build_extensions()


# Hook into cpp_extension's build process for debugging
def debug_build_process():
    # Store original functions
    original_build_extensions = cpp_extension.BuildExtension.build_extensions
    original_write_ninja_file_and_compile_objects = cpp_extension._write_ninja_file_and_compile_objects
    original_write_ninja_file_to_build_library = cpp_extension._write_ninja_file_to_build_library
    
    # Debug build extensions
    def debug_build_extensions(self):
        original_build_extensions(self)
        print("### BuildExtensions: ", self.extensions[0].extra_compile_args)
    
    # Debug ninja file compilation
    def debug_write_ninja_file_and_compile_objects(
            sources: List[str], objects, cflags, post_cflags, 
            cuda_cflags, cuda_post_cflags, cuda_dlink_post_cflags, 
            build_directory: str, verbose: bool, with_cuda: Optional[bool]):
        
        print("#########cflags: ", cflags)
        print("#########post_cflags: ", post_cflags)
        print("#########cuda_cflags: ", cuda_cflags)
        print("#########cuda_post_cflags: ", cuda_post_cflags)
        print("#########cuda_dlink_post_cflags: ", cuda_dlink_post_cflags)
        
        exclude_flags = ['-g', '-fwrapv', "-O2", "-pthread", "-Wno-unused-result", "-Wsign-compare"]
        cflags = [arg for arg in cflags if arg not in exclude_flags]
        
        original_write_ninja_file_and_compile_objects(
            sources, objects, cflags, post_cflags, 
            cuda_cflags, cuda_post_cflags, cuda_dlink_post_cflags, 
            build_directory, verbose, with_cuda
        )
    
    # Debug ninja file library build
    def debug_write_ninja_file_to_build_library(
            name, sources: List[str], extra_cflags, extra_cuda_cflags, 
            extra_ldflags, extra_include_paths, build_directory: str, 
            verbose: bool, with_cuda: Optional[bool], is_standalone: bool = False):
        
        print("#########extra_cflags: ", extra_cflags)
        print("#########extra_ldflags: ", extra_ldflags)
        
        original_write_ninja_file_to_build_library(
            name, sources, extra_cflags, extra_cuda_cflags, 
            extra_ldflags, extra_include_paths, build_directory, 
            verbose, with_cuda, is_standalone
        )
    
    # Replace original functions with debug versions
    cpp_extension.BuildExtension.build_extensions = debug_build_extensions
    cpp_extension._write_ninja_file_and_compile_objects = debug_write_ninja_file_and_compile_objects
    cpp_extension._write_ninja_file_to_build_library = debug_write_ninja_file_to_build_library


def main():
    # Set environment for compilation
    os.environ["CC"] = "g++"
    os.environ["CXX"] = "g++"
    
    # Enable debugging of the build process
    debug_build_process()
    
    # Get system architecture
    machine_name = subprocess.check_output(['uname', '-m']).decode('utf-8').strip()
    
    # Define paths
    direct_conv2d_path = f"/Users/junyi/Desktop/HIPACK/src"
    system_include_path = f'/usr/include/{machine_name}-linux-gnu/'
    
    # Define compilation and linking arguments
    compile_args = [
        '-g',
        '-fopenmp', 
        '-march=native', 
        "-DENABLE_OPENMP",
        '-fconcepts-ts', 
        '-std=c++17', 
        '-Ofast',
        f'-I{system_include_path}',
        f"-I{direct_conv2d_path}"
    ]
    
    link_args = [
        '-lgomp',
        '-lcpuinfo', 
        '-lclog', 
        '-fconcepts-ts',
        '-lopenblas'
    ]
    
    include_dirs = [
        system_include_path,
        direct_conv2d_path
    ]
    
    # Create extension
    extension = cpp_extension.CppExtension(
        'direct_conv2d', 
        ['./directconv2d.cpp'],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        include_dirs=include_dirs,
        library_dirs=[system_include_path]
    )
    
    # Setup package
    setup(
        name='direct_conv2d',
        version='1.0.7',
        description='A direct convolution 2D extension for PyTorch.',
        long_description=open('../README.md').read(),
        long_description_content_type='text/markdown',
        ext_modules=[extension],
        cmdclass={
            'build_ext': cpp_extension.BuildExtension,
            'clean': CleanCommand,
        }
    )


if __name__ == "__main__":
    main()