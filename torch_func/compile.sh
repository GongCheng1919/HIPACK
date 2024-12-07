# g++ your_program.cpp -o your_program -std=c++14 -I/path/to/libtorch/include -I/path/to/libtorch/include/torch/csrc/api/include -L/path/to/libtorch/lib -ltorch -lc10 -Wl,-rpath,/path/to/libtorch/lib

DISTUTILS_DEBUG=1 MAX_JOBS=$(nproc) python setup.py install --user --verbose