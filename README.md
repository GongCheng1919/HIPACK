# HIPACK

This repo contains the detailed implementation of **HIPACK**, an efficient sub-8-bit direct convolution acceleration library to maximize the performance of quantized NN execution on Arm processors. Compared to low-bitwidth quantization libraries like QNNPACK, CMSIS-NN, HIPACK enables **dynamic bitwidth** computations below 8-bit and achieves **over 3.2x** performance improvement.

---

## Design Concepts

HiPACK follows the theoretical approach of adopting multiplication for low-bitwidth convolution and develops a series of novel approaches to fill the efficiency gap of low-bitwidth convolution on wimpy processors with SIMD optimizations and bitwise management. 
HiPack is built upon the following principles:
1. **Multiplication-based Convolution**: Adopts long-bitwidth multiplication for low-bitwidth convolution.
<div align="center">
  <img src="./figures/figure4-crop.pdf" alt="mul-4-conv" width="800"/>
  <p><em>Figure 1: Multiplication for convolution.</em></p>
</div>
2. **Data Dependency Elimination**: Identifies and handles data dependencies in the process of adopting large-bitwidth multiplication for low-bitwidht convolution operations.
3. **SIMD Optimization**: Utilizes SIMD instructions to maximize data reuse with operation decoupling and reordering to improve data parallelism.
4. **Bitwise Management**: Develops optimal segmentation bitwidth identification mechanism and dual interleaved register mechanism to improve the efficiency of low-bitwidth convolution on wimpy processors with bitwise management.

The synergistic combination of the above methods is thoroughly evaluated with various CNN models on ARM processors. Experimental results demonstrate over $3.2\times$ performance improvements compared to existing approaches, enabling efficient execution of low-bitwidth DNNs on resource-constrained ARM devices.

---

## Features

1. **Dynamic Bitwidth Support**: Adapts to quantized computations with bitwidths lower than 8-bit.
2. **High Performance**: Significant performance improvements, achieving a minimum of 3.2x speedup.
3. **PyTorch Integration**: Provides PyTorch operator interfaces in [torch_func](./torch_func/README.md), making it easy to integrate into existing deep learning workflows.
4. **Support for Various Convolution Shapes**:
   - **DirectConv (nx3)**: Native support for `nx3` convolution shapes.
   - **DirectConv (nxn)**: Extended implementation for arbitrary `nxn` shapes by tiling them into multiple `nx3` convolutions.

---

# Implementation Instructions

The native support of `nx3` kernel is implemented with C++ and located in [src](./src) folder. The other convolution kernel sizes are implemented by tiling the convolution into multiple `nx3` convolutions through pytorch function calls in [torch_func](./torch_func/README.md) folder.
The codes are implemented and tested on a Raspberry Pi 4B+ platform.

## Usage in C++ backend

### Customizable parameters
- **N**: Input batch size. (Supported values: 1, 2, 4, 8)
- **Ci**: Number of input channels. (Supported values: 32, 64, 128, 256)
- **H**: Height of input feature map. (Supported values: 8, 16, 32)
- **W**: Width of input feature map. (Currently only support numbers divisible by 12, if not, will be padded with zeros to the nearest number divisible by 12, e.g., 32 will be padded to 36. Recommented values: 12, 24, 36)
- **Co**: Number of output channels. (Supported values: 32, 64, 128, 256)
- **WA_bits**: Bitwidth of weights and activations. (Supported values: 1, 2, 3, 4, 5, 6. Note: values greater than 4 may have the risk of overflow.)
- **verbose**: Whether to print verbose information. (Supported values: 0, 1)
- **debug**: Whether to verify the correctness of the computation. (Supported values: 0, 1)

Based on these parameters, the tensor dimensions for computation are represented as:
- Input shape: [N, Ci, H, W]
- Weight shape: [Co, Ci, 3, 3]

Use the following command to run the fast expetiments on a Raspberry Pi 4B+ platform.
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

Please refer to [torch_func](torch_func/README.md) to find detailed implementations.
