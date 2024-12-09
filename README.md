# HIPACK: Efficient Low-Bitwidth Convolution Operator

We present and implement **HIPACK**, an efficient low-bitwidth convolution operator that supports integer computations below 8-bit with high performance. Compared to low-bitwidth quantization libraries like QNNPACK, HIPACK enables **dynamic bitwidth** computations below 8-bit and achieves **over 4x performance improvement**.

---

## Abstract
HiPACK is an efficient acceleration library for sub-byte Neural Network Computation.
HiPACK follows the theoretical approach of adopting multiplication for low-bitwidth convolution and develops a series of novel approaches to fill the efficiency gap of low-bitwidth convolution on wimpy processors with SIMD optimizations. 
It first identifies the inevitable data dependencies of the multiply-to-convolution. Then decoupling the multiplication with unpacking, followed by a series of optimization techniques developed to maximize the data reuse and processing efficiency.  The synergistic combination of the above methods is thoroughly evaluated with various CNN models on ARM processors. Experimental results demonstrate $4\times$ performance improvements compared to existing approaches, enabling efficient execution of low-bitwidth DNNs on resource-constrained ARM devices.

---
## Features

1. **Dynamic Bitwidth Support**: Adapts to quantized computations with bitwidths lower than 8-bit.
2. **High Performance**: Significant performance improvements, achieving up to 4x speedup.
3. **PyTorch Integration**: Provides PyTorch operator interfaces in [torch_func](./torch_func/README.md), making it easy to integrate into existing deep learning workflows.
4. **Support for Various Convolution Shapes**:
   - **DirectConv (nx3)**: Native support for `nx3` convolution shapes.
   - **DirectConv (nxn)**: Extended implementation for arbitrary `nxn` shapes by tiling them into multiple `nx3` convolutions.

---

## Usage in C++ backend
We have prepared our codes and simple experiments in [src](./src) folder, you can use following commend to run the fast expetiments in Raspi4B+.

```shell
$ cd src
$ bash run_bench.sh
```
You can get the following results.
```
config: N1 Ci2 H2 W2 Co2 W3A3 debug1 verbose0
	[W3A3] input[1,2,2,12] * weight[2,2,3,3]: Test pass
	[W3A3] input[1,2,2,12] * weight[2,2,3,3]: Elapsed time: 0.000168 seconds Performance: 0.023943 GFLOPS.
config: N1 Ci2 H2 W2 Co4 W3A3 debug1 verbose0
	[W3A3] input[1,2,2,12] * weight[4,2,3,3]: Test pass
	[W3A3] input[1,2,2,12] * weight[4,2,3,3]: Elapsed time: 0.001268 seconds Performance: 0.006360 GFLOPS.
config: N1 Ci2 H2 W2 Co8 W3A3 debug1 verbose0
	[W3A3] input[1,2,2,12] * weight[8,2,3,3]: Test pass
	[W3A3] input[1,2,2,12] * weight[8,2,3,3]: Elapsed time: 0.000989 seconds Performance: 0.016311 GFLOPS.
config: N1 Ci2 H2 W2 Co16 W3A3 debug1 verbose0
	[W3A3] input[1,2,2,12] * weight[16,2,3,3]: Test pass
	[W3A3] input[1,2,2,12] * weight[16,2,3,3]: Elapsed time: 0.000173 seconds Performance: 0.186667 GFLOPS.
...
...
config: W3A3, save to: logs/test_hipack_perf_W3A3.log
        [W3A3] input[16,3,224,228] * weight[32,3,3,3]: Elapsed time: 0.224631 seconds Performance: 6.397795 GFLOPS.
        [W3A3] input[16,32,112,120] * weight[64,32,3,3]: Elapsed time: 0.248804 seconds Performance: 32.970821 GFLOPS.
        [W3A3] input[16,64,112,120] * weight[64,64,3,3]: Elapsed time: 0.470369 seconds Performance: 34.880142 GFLOPS.
        [W3A3] input[16,64,56,60] * weight[128,64,3,3]: Elapsed time: 0.185535 seconds Performance: 45.727473 GFLOPS.
        [W3A3] input[16,128,56,60] * weight[128,128,3,3]: Elapsed time: 0.343597 seconds Performance: 49.383658 GFLOPS.
        [W3A3] input[16,128,28,36] * weight[256,128,3,3]: Elapsed time: 0.178536 seconds Performance: 60.259073 GFLOPS.
        [W3A3] input[16,256,28,36] * weight[256,256,3,3]: Elapsed time: 0.341874 seconds Performance: 62.937711 GFLOPS.
        [W3A3] input[16,256,14,24] * weight[512,256,3,3]: Elapsed time: 0.234358 seconds Performance: 67.006236 GFLOPS.
        [W3A3] input[16,512,14,24] * weight[512,512,3,3]: Elapsed time: 0.427688 seconds Performance: 73.434252 GFLOPS.
        [W3A3] input[16,512,7,12] * weight[1024,512,3,3]: Elapsed time: 0.221781 seconds Performance: 85.784536 GFLOPS.
        [W3A3] input[16,1024,7,12] * weight[1024,1024,3,3]: Elapsed time: 0.446465 seconds Performance: 85.226671 GFLOPS.
```

## Usage in PyTorch

You can refer to [torch_func](torch_func/README.md) to find detailed infomation.
