import torch
import torch.nn as nn
import time
from torch.nn.modules.conv import Conv2d
from torch.quantization import QConfig
import direct_conv2d
from collections import defaultdict

result_dict = defaultdict(dict) 
"""
{
    "input_shape=(16, 3, 112, 112), weight_shape=(16, 16, 3, 3)": {
        "qnnpack":  #time
        "hipack":  #time
        "float32":  #time
    }
}

"""

class DCConv(nn.Module):
    """Applies a convolution, batch normalization, and activation function to an input tensor in a neural network."""

    def __init__(self, in_channel, out_channel, W_bits, A_bits, k, s, p=None, g=1, d=1, MT=True, show_time=False, conv_weight=None, conv_bias=None, compare_mode = False):
        """Initializes a standard convolution layer with optional batch normalization and activation."""
        super().__init__()
        # self.conv = nn.Conv2d(in_channel, out_channel, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        # self.bn = nn.BatchNorm2d(out_channel)
        if compare_mode:
            self.nnConv = Conv2d(
                in_channels = in_channel,
                out_channels = out_channel,
                kernel_size = k,
                stride = s,
                padding = p,
                groups = g,
                dilation = d,
            )
            self.nnConv.weight = torch.nn.Parameter(conv_weight, requires_grad=False)
            self.nnConv.bias = torch.nn.Parameter(conv_bias, requires_grad=False) if conv_bias is not None else None
        self.compare_mode = compare_mode
        self.padding = p if type(p) is int else p[0]
        self.stride = s if type(s) is int else s[0]
        self.show_time = show_time
        self.depth_conv = False
        self.MT = MT
        self.W_bits = W_bits
        self.A_bits = A_bits
        self.kernel_size = k
        max_w = 2**(W_bits) - 1
        if conv_weight is None:
            self.weight = torch.randint(0, max_w, (out_channel, in_channel, k, k))
        else:
            self.weight = conv_weight.reshape(out_channel, in_channel, k, k)
        print("Layer DCConv weight shape", self.weight.shape, "stride", self.stride, "padding", self.padding)
        if conv_bias is None:
            self.bias = torch.zeros(out_channel)
        else:
            self.bias = conv_bias.reshape(out_channel)

    def forward(self, x):
        print("")
        print("Input ", tuple(x.shape), "Weight", tuple(self.weight.shape), "stride", self.stride,  "padding", self.padding, "stride", self.stride)
        time_begin = time.time_ns()
        """Applies a convolution followed by batch normalization and an activation function to the input tensor `x`."""
        # print("DCConv before", x.shape, "stride", self.stride, "weight", self.weight.shape, "bias", self.bias.shape)
        y = direct_conv2d.direct_conv2d(
            x.int(), 
            self.weight.int(), 
            self.W_bits, 
            self.A_bits, 
            self.MT, 
            self.stride, 
            self.show_time, 
            self.depth_conv
        )
        time_end = time.time_ns()
        time_direct_conv2d = (time_end - time_begin)/1000/1000

        key = f"input_shape={tuple(x.shape)}, weight_shape={tuple(self.weight.shape)}"
        result_dict.setdefault(key, {}).setdefault("hipack", []).append(time_direct_conv2d)
        print("DCConv output shape", tuple(y.shape), f"time {time_direct_conv2d:.2f}ms")
        
        if self.compare_mode:
            time_start_nn = time.time_ns()
            y_nn = self.nnConv(x)
            time_end_nn = time.time_ns()
            time_nn_conv2d = (time_end_nn - time_start_nn)/1000/1000
            key = f"input_shape={tuple(x.shape)}, weight_shape={tuple(self.nnConv.weight.shape)}"
            result_dict.setdefault(key, {}).setdefault("float32", []).append(time_nn_conv2d)
            print("nnConv output shape", tuple(y_nn.shape), f"time {time_nn_conv2d:.2f}ms")
            
            if time_nn_conv2d < time_direct_conv2d:
                print("❌ nn.Conv2d is faster, using nn.Conv2d")
            else:
                print("✅ DirectConv2d is faster, using DirectConv2d")
            print(f"   Run this \"python ../../test_single.py -bs {x.shape[0]} -ic {x.shape[1]} -oc {y.shape[1]} -k {self.kernel_size} -s {self.stride} -p {self.padding} -wb {self.W_bits} -ab {self.A_bits} --width {x.shape[3]} --height {x.shape[2]}\"")
            return y_nn
        
        if self.kernel_size == 3 and self.padding == 0:
            y = y[:, :, 2:-2, 2:-2] # 去掉两圈
        elif self.kernel_size == 3 and self.padding == 1:
            y = y[:, :, 1:-1, 1:-1] # 去掉一圈
        elif self.padding == 0:
            y = y[:, :, 0:-2, 1:-1] # 去掉多算的边界
             
        print(f"DCConv time {(time_end - time_begin)/1000/1000:.2f}ms, final shape", y.shape)
        
        return y

    # def forward_fuse(self, x):
    #     """Applies a fused convolution and activation function to the input tensor `x`."""
    #     x = x.int()
    #     x = direct_conv2d.direct_conv2d(x, self.weight, self.W_bits, self.A_bits, self.MT, self.stride, self.show_time, self.depth_conv)
    #     return self.act(x)

class PTConv(nn.Module):
    """Applies a PyTorch Conv2d layer for performance comparison with DirectConv2d"""
    
    def __init__(self, in_channel, out_channel, k, s, p=None, g=1, d=1, show_time=False, conv_weight=None, conv_bias=None):
        super().__init__()
        self.padding = p if type(p) is int else p[0]
        self.stride = s if type(s) is int else s[0]
        self.conv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=k,
            stride=s,
            padding=p,
            groups=g,
            dilation=d,
        )
        self.conv.weight = torch.nn.Parameter(conv_weight, requires_grad=False)
        self.conv.bias = torch.nn.Parameter(conv_bias, requires_grad=False) if conv_bias is not None else None
        print("PTConv created with shape", self.conv.weight.shape, "stride", self.stride, "padding", self.padding)
    
    def forward(self, x):
        print("")
        print("PTConv Input", tuple(x.shape), "stride", self.stride, "padding", self.padding)
        time_begin = time.time_ns()
        
        # Apply PyTorch Conv2d
        y = self.conv(x)
        
        time_end = time.time_ns()
        time_pt_conv2d = (time_end - time_begin)/1000/1000
        
        print("PTConv output shape", tuple(y.shape), f"time {time_pt_conv2d:.2f}ms")
        return y

class QNNPackConv(nn.Module):
    """Applies a QNNPACK quantized convolution for performance comparison with DirectConv2d"""
    
    def __init__(self, in_channel, out_channel, k, s, p=None, g=1, d=1, show_time=False, conv_weight=None, conv_bias=None):
        super().__init__()
        self.padding = p if type(p) is int else p[0]
        self.stride = s if type(s) is int else s[0]
        self.show_time = show_time
        self.kernel_size = k
        
        # Create float Conv2d to be quantized
        self.conv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=k,
            stride=s,
            padding=p,
            groups=g,
            dilation=d,
        )
        self.conv_weight = conv_weight
        # Set weights and bias if provided
        if conv_weight is not None:
            self.conv.weight = torch.nn.Parameter(conv_weight, requires_grad=False)
        if conv_bias is not None:
            self.conv.bias = torch.nn.Parameter(conv_bias, requires_grad=False)
            
        # Setup quantization components
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        
        # Configure quantization
        torch.backends.quantized.engine = 'qnnpack'
        qconfig = QConfig(
            weight=torch.quantization.default_observer.with_args(dtype=torch.qint8),
            activation=torch.quantization.default_observer.with_args(dtype=torch.quint8)
        )
        self.conv.qconfig = qconfig
        self.quant.qconfig = qconfig
        self.dequant.qconfig = qconfig
        
        # Prepare and convert the model
        torch.quantization.prepare(self, inplace=True)
        # We'll do the conversion in forward to measure time properly
        self.is_quantized = False
        print("QNNPackConv created with shape", self.conv.weight.shape, "stride", self.stride, "padding", self.padding)
    
    def forward(self, x):
        print("")
        print("QNNPackConv Input", tuple(x.shape), "stride", self.stride, "padding", self.padding)
        
        # Convert the model on first forward pass
        if not self.is_quantized:
            torch.quantization.convert(self, inplace=True)
            self.is_quantized = True
        
        time_begin = time.time_ns()
        
        # Apply quantized convolution
        x = self.quant(x)
        y = self.conv(x)
        y = self.dequant(y)
        
        time_end = time.time_ns()
        time_qnnpack_conv2d = (time_end - time_begin)/1000/1000
        
        key = f"input_shape={tuple(x.shape)}, weight_shape={tuple(self.conv_weight.shape)}"
        result_dict.setdefault(key, {}).setdefault("qnnpack", []).append(time_qnnpack_conv2d)
        print("QNNPackConv output shape", tuple(y.shape), f"time {time_qnnpack_conv2d:.2f}ms")
        return y


def replace_conv2d(model, mode='float', W_bits=4, A_bits=4):
    """
    Replace Conv2d layers in the model with either DirectConv or QNNPack quantized convolution.
    
    Args:
        model: The model to modify.
        mode: One of 'float' (original), 'hipack' (DirectConv), or 'qnnpack' (QNNPack quantization).
        W_bits: Weight bit width for HIPACK.
        A_bits: Activation bit width for HIPACK.
    """
    if mode not in ['float', 'hipack', 'qnnpack']:
        raise ValueError("mode must be one of 'float', 'hipack', or 'qnnpack'")
    
    for name, module in model.named_children():
        if isinstance(module, Conv2d):
            
            print(module.in_channels, module.out_channels, module.kernel_size, module.stride, module.padding, end=" ")
            
            if module.kernel_size[0] == 1:  # Skip kernel size 1
                print("Skip, kernel size is 1, cannot speedup")
                continue

            if module.stride[0] == 2:
                print("Skip, stride is 2")
                continue
            
            print("")
            
            if mode == 'float':
                new_conv = PTConv(
                    in_channel=module.in_channels,
                    out_channel=module.out_channels,
                    k=module.kernel_size[0],
                    s=module.stride,
                    p=module.padding,
                    g=module.groups,
                    d=module.dilation,
                    conv_weight=module.state_dict()["weight"],
                    conv_bias=module.state_dict()["bias"] if "bias" in module.state_dict() else None,
                )
                setattr(model, name, new_conv)
            elif mode == 'hipack':
                # Replace with DirectConv
                new_conv = DCConv(
                    in_channel=module.in_channels,
                    out_channel=module.out_channels,
                    W_bits=W_bits,
                    A_bits=A_bits,
                    k=module.kernel_size[0],
                    s=module.stride,
                    p=module.padding,
                    g=module.groups,
                    d=module.dilation,
                    conv_weight=module.state_dict()["weight"],
                    conv_bias=module.state_dict()["bias"] if "bias" in module.state_dict() else None,
                    compare_mode=True,
                )
                setattr(model, name, new_conv)
            elif mode == 'qnnpack':
                # Replace with QNNPack quantized convolution
                new_conv = QNNPackConv(
                    in_channel=module.in_channels,
                    out_channel=module.out_channels,
                    k=module.kernel_size[0],
                    s=module.stride,
                    p=module.padding,
                    g=module.groups,
                    d=module.dilation,
                    conv_weight=module.state_dict()["weight"],
                    conv_bias=module.state_dict()["bias"] if "bias" in module.state_dict() else None,
                )
                setattr(model, name, new_conv)
        else:
            replace_conv2d(module, mode, W_bits, A_bits)  # Recursively replace


 