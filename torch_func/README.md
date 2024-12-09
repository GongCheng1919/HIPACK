# PyTorch Integration

This folder provides PyTorch operator interfaces, making it easy to integrate into existing deep learning workflows.

## Usage Guide

### 1. Compile the DirectConv Operator

Run the following command in the current directory to compile the PyTorch DirectConv computation library:

```bash
bash compile.sh
```

Once compiled, the `direct_conv` operator is ready to use for convolution calculations.

> **Note**: You need to install `g++ (10.2.1)` and `PyTorch>=2.2.2` before use `compile.sh`.

### 2. Using the DirectConv Operator

Refer to the file `usage_of_directconv.py` for an example of how to use the `direct_conv` operator for efficient convolution computations.
The following is a simple example.
```python
from direct_conv2d import direct_conv2d
N, Ci, H, W, Co, W_bits,A_bits =16,256,32,36,256,3,3
flops = 2*N*Ci*Co*H*W*3*3
inp = torch.randint(0, 2**A_bits -1, (N, Ci, H, W)).int()
weight = torch.randint(0, 2**W_bits -1, (Co, Ci, 3, 3)).int()
output = direct_conv2d(inp,weight,W_bits, A_bits,1,1,0,0)
```

### 3. Extending to nxn Convolution Shapes

DirectConv natively supports `nx3` convolution shapes. To support arbitrary `nxn` shapes, we extend the implementation by tiling `nxn` convolutions into multiple `nx3` convolutions. For example:
- A **5x5 convolution** can be tiled into **2 5x3 convolutions**.
- A **9x9 convolution** can be tiled into **3 9x3 convolutions**.

Refer to the file `extend_conv2d.py` for details on using the extended convolution operator.

---

We welcome you to explore and use the HIPACK operator on GitHub, and we look forward to your valuable feedback! 😊
