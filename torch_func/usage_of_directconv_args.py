import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

from direct_conv2d import direct_conv2d
from utils import MeasureExecutionTime, Qint8Conv2D

class DirectConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                         W_bits=3,A_bits=3):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.weight=nn.Parameter(torch.randint(0, 2**W_bits -1, (out_channels, in_channels, kernel_size[0], kernel_size[1])).float())
        self.W_bits = W_bits
        self.A_bits = A_bits
    def forward(self, x):
        x = torch.clip(x.int(),0,2**self.A_bits-1)
        return direct_conv2d(x, self.weight.int(), self.W_bits,self.A_bits, 1, 1, 0, 0)
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Direct Conv2D')
    parser.add_argument('--N', type=int, default=32, help='batch size')
    parser.add_argument('--Ci', type=int, default=256, help='input channels')
    parser.add_argument('--H', type=int, default=24, help='input height')
    parser.add_argument('--W', type=int, default=24, help='input width')
    parser.add_argument('--Co', type=int, default=256, help='output channels')
    parser.add_argument('--W_bits', type=int, default=3, help='weight bits')
    parser.add_argument('--A_bits', type=int, default=3, help='activation bits')
    args = parser.parse_args()
    N, Ci, H, W, Co, W_bits, A_bits = args.N, args.Ci, args.H, args.W, args.Co, args.W_bits, args.A_bits
    
    flops = 2*N*Ci*Co*H*W*3*3
    inp = torch.randint(0, 2**A_bits -1, (N, Ci, H, W)).float()
    weight = torch.randint(0, 2**W_bits -1, (Co, Ci, 3, 3)).float()
    fconv = nn.Conv2d(Ci, Co, 3, padding=2)
    qconv = Qint8Conv2D(Ci, Co, 3,padding=2)
    dconv = DirectConv2D(Ci, Co, 3, 1,0,W_bits,A_bits)
    fconv.weight.data.copy_(weight)
    dconv.weight.data.copy_(weight.int())

    print(f"input shape: {inp.shape}, weight shape: {weight.shape}")
    with MeasureExecutionTime(measure_name="Float Conv2d",flops=flops):
        output_f = fconv(inp)

    with MeasureExecutionTime(measure_name="Qint8 Conv2d",flops=flops):
        output_q8 = qconv(inp)

    with MeasureExecutionTime(measure_name="Direct Conv2d",flops=flops):
        output_h3 = dconv(inp)
