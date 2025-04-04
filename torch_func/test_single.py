import time
import torch
import argparse
import torch.nn.quantized as nnq
from torch.ao.quantization import get_default_qconfig

from direct_conv2d import direct_conv2d

# Constants
RUN_TIMES = 10

def test_both(
    width: int, height: int, batch_size: int, in_channels: int, out_channels: int,
    kernel_size: int, stride: int, padding: int, W_bits: int, A_bits: int
):
    """
    Compare the performance of nn.Conv2d and direct_conv2d.

    Args:
        width (int): Input width.
        height (int): Input height.
        batch_size (int): Batch size.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Kernel size.
        stride (int): Stride.
        padding (int): Padding.
        W_bits (int): Weight quantization bits.
        A_bits (int): Activation quantization bits.

    Returns:
        tuple: Execution times for direct_conv2d and nn.Conv2d.
    """
    # Generate random input and weight tensors
    x = torch.rand((batch_size, in_channels, width, height), requires_grad=False)
    weight = torch.rand((out_channels, in_channels, kernel_size, kernel_size), requires_grad=False)
    print(f"Input  shape:   {tuple(x.shape)}\ttype:", x.dtype)
    print(f"Weight shape:   {tuple(weight.shape)}\ttype:", weight.dtype)

    # nn.Conv2d setup
    conv = torch.nn.Conv2d(
        in_channels=in_channels, 
        out_channels=out_channels, 
        kernel_size=kernel_size, 
        stride=stride, 
        padding=padding, 
        bias=False
    )
    conv.weight = torch.nn.Parameter(weight, requires_grad=False)

    # 先量化输入和权重
    x_q = torch.quantize_per_tensor(x, scale=0.1, zero_point=128, dtype=torch.quint8)
    w_q = torch.quantize_per_tensor(weight, scale=0.05, zero_point=0, dtype=torch.qint8)

    # 创建 nn.quantized.Conv2d
    torch.backends.quantized.engine = "qnnpack"
    qconfig = get_default_qconfig("qnnpack")
    print(qconfig)
    conv_q = nnq.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=False,
        dtype=torch.qint8
    )
    conv_q.set_weight_bias(w_q, None)
    conv_q.scale = 0.1  # 这个是输出 scale
    conv_q.zero_point = 128

    # 预热并计时
    y_q = conv_q(x_q)  # warmup
    start = time.time_ns()
    for _ in range(RUN_TIMES):
        y_q = conv_q(x_q)
    end = time.time_ns()
    time_qconv = (end - start) / 1_000_000
    

    # Measure nn.Conv2d execution time
    yhat = conv(x) # warnup
    time_begin = time.time_ns()
    for _ in range(RUN_TIMES):
        yhat = conv(x)
    time_end = time.time_ns()
    time_conv = (time_end - time_begin) / 1_000_000  # Convert to milliseconds
    print("Output shape (nn.Conv2d):    ", yhat.shape)

    # Measure direct_conv2d execution time
    y = direct_conv2d(
        x.int(), weight.int(), W_bits, A_bits, False, stride, False, False
    ) # warnup
    time_begin = time.time_ns()
    for _ in range(RUN_TIMES):
        y = direct_conv2d(
            x.int(), weight.int(), W_bits, A_bits, False, stride, False, False
        )
    time_end = time.time_ns()
    time_direct = (time_end - time_begin) / 1_000_000  # Convert to milliseconds
    print("Output shape (direct_conv2d):", y.shape)


    
    return time_direct, time_conv, time_qconv

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description='Test direct_conv2d performance')
    parser.add_argument('--width', "-w", type=int, default=224, help='Input width')
    parser.add_argument('--height', type=int, default=224, help='Input height')
    parser.add_argument('--batch_size', "-bs", type=int, default=16, help='Batch size')
    parser.add_argument('--in_channels', "-ic", type=int, default=3, help='Input channels')
    parser.add_argument('--out_channels', "-oc", type=int, default=3, help='Output channels')
    parser.add_argument('--kernel_size', "-ks", type=int, default=3, help='Kernel size')
    parser.add_argument('--stride', "-s", type=int, default=1, help='Stride')
    parser.add_argument('--padding', "-p", type=int, default=0, help='Padding')
    parser.add_argument('--W_bits', "-wb", type=int, default=4, help='Weight quantization bits')
    parser.add_argument('--A_bits', "-ab", type=int, default=4, help='Activation quantization bits')
    args = parser.parse_args()

    # Print arguments
    print("Arguments:", args)

    # Run tests
    time_direct, time_conv, time_qconv = test_both(
        width=args.width,
        height=args.height,
        batch_size=args.batch_size,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        kernel_size=args.kernel_size,
        stride=args.stride,
        padding=args.padding,
        W_bits=args.W_bits,
        A_bits=args.A_bits
    )

    # Print results
    print("\nAverage execution time (ms):")
    print("DirectConv2d:", time_direct / RUN_TIMES)
    print("nn.Conv2d:   ", time_conv / RUN_TIMES)
    print("q.Conv:      ", time_qconv / RUN_TIMES)
    if time_direct < time_conv:
        if time_qconv < time_direct:
            print("🚼 QNNPack is faster")
        else:
            print("✅ DirectConv2d is faster")
    else:
        print("❌ nn.Conv2d is faster")

if __name__ == "__main__":
    main()

