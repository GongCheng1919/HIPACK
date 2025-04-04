import os
import torch
from torch import nn
import time
import direct_conv2d 
from torch.nn.modules.conv import Conv2d
os.environ["PATH"] += os.path.dirname(__file__) + ":" + os.environ["PATH"]
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

# disable FutureWarning
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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
        print("DCConv weight shape", self.weight.shape, "stride", self.stride, "padding", self.padding)
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
        print("DCConv output shape", tuple(y.shape), f"time {time_direct_conv2d:.2f}ms")
        
        if self.compare_mode:
            time_start_nn = time.time_ns()
            y_nn = self.nnConv(x)
            time_end_nn = time.time_ns()
            time_nn_conv2d = (time_end_nn - time_start_nn)/1000/1000
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
    


def replace_conv2d(model, enable_hipack:bool, W_bits:int, A_bits:int):
    for name, module in model.named_children():
        if isinstance(module, Conv2d):
            
            print(module.in_channels, module.out_channels, module.kernel_size, module.stride, module.padding, end=" ")
            
            if module.kernel_size[0] == 1: # 等于一的没办法加速
                print("Skip, kernel size is 1")
                continue
            print("")
            
            # pdb.set_trace()
            if not enable_hipack:
                continue

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
                # show_time=True
            )
            setattr(model, name, new_conv)  # 替换层
        else:
            replace_conv2d(module, enable_hipack, W_bits, A_bits)  # 递归替换
            
        continue
        if isinstance(module, Conv):
            
            if module.conv.kernel_size[0] == 1: # 等于一的没办法加速
                print("Skip, kernel size is 1")
                continue
            if module.conv.kernel_size[0] != 3: # 不等于 3 的，大的圈数不一样
                print("Skip, kernel size is not 3")
                continue
            # if module.conv.kernel_size[0] == 3 and module.conv.padding[0] == 1:
            #     continue # segfault
            if module.conv.kernel_size[0] == 3 \
                and module.conv.padding[0] == 1 \
                and module.conv.stride[0] == 2 \
                and module.conv.in_channels == 32 \
                and module.conv.out_channels == 64:
                print("Skip, segfault")
                continue # segfault
            
            if module.conv.kernel_size[0] == 3 \
                and module.conv.stride[0] == 2 \
                and module.conv.in_channels == 64 \
                and module.conv.out_channels == 128:
                print("Skip, segfault")
                continue # segfault
                
            if module.conv.kernel_size[0] == 3 \
                and module.conv.stride[0] == 2 \
                and module.conv.in_channels == 128 \
                and module.conv.out_channels == 256:
                print("Skip, segfault")
                continue # segfault
                
            if module.conv.kernel_size[0] == 3 \
                and module.conv.stride[0] == 2 \
                and module.conv.in_channels == 256 \
                and module.conv.out_channels == 512:
                print("Skip, segfault")
                continue # segfault

            if module.conv.kernel_size[0] == 3 \
                and module.conv.stride[0] == 2 \
                and module.conv.in_channels == 128 \
                and module.conv.out_channels == 128:
                print("Skip, segfault")
                continue # segfault
            
            if module.conv.kernel_size[0] == 3 \
                and module.conv.stride[0] == 2 \
                and module.conv.in_channels == 256 \
                and module.conv.out_channels == 256:
                print("Skip, segfault")
                continue # segfault
            

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)

replace_conv2d(model, True, 4, 4) # 替换模型中的 Conv 层
diffusion = GaussianDiffusion(
    model,
    image_size = 112,
    timesteps = 1    # number of steps
)

sampled_images = diffusion.sample(batch_size = 8, return_all_timesteps = False)
sampled_images.shape # (16, 3, 128, 128)
print(sampled_images.shape)
