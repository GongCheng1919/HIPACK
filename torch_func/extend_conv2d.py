import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

from direct_conv2d import direct_conv2d,direct_conv2d_khkw

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
        return direct_conv2d_khkw(x, self.weight.int(), self.W_bits,self.A_bits, 1, 1, 0, 0)
    
class Conv5by2Conv3Padding2(nn.Module):
    def __init__(self,conv5_module,reserve_wide_conv=False):
        super().__init__()
        if reserve_wide_conv:
            self.wide_conv = conv5_module
        self.in_channels = conv5_module.in_channels
        self.out_channels = conv5_module.out_channels
        self.kernel_size = conv5_module.kernel_size
        self.narrow_kernel_size = (self.kernel_size[0],3)
        self.padding = conv5_module.padding
        self.conv3_1 = nn.Conv2d(self.in_channels, self.out_channels, 
                                 kernel_size=self.narrow_kernel_size, 
                                 bias=False,padding=self.padding)
        self.conv3_2 = nn.Conv2d(self.in_channels, self.out_channels, 
                                 kernel_size=self.narrow_kernel_size, 
                                 bias=False,padding=self.padding)
        self.conv3_1.weight.data.copy_(conv5_module.weight.data[:,:,:,:3])
        self.conv3_2.weight.data.copy_(conv5_module.weight.data[:,:,:,2:])
        self.conv3_2.weight.data[:,:,:,0] = 0 # 多余的直接设置成0
        
    def forward(self, input):
        out1 = self.conv3_1(input)
        out2 = self.conv3_2(input)
        out = out1[:,:,:,:-2]+out2[:,:,:,2:]
        return out

class Conv7by3Conv3Padding2(nn.Module):
    def __init__(self,conv7_module,reserve_wide_conv=False):
        super().__init__()
        if reserve_wide_conv:
            self.wide_conv = conv7_module
        self.in_channels = conv7_module.in_channels
        self.out_channels = conv7_module.out_channels
        self.kernel_size = conv7_module.kernel_size
        self.narrow_kernel_size = (self.kernel_size[0],3)
        self.padding = conv7_module.padding
        self.conv3_1 = nn.Conv2d(self.in_channels, self.out_channels, 
                                 kernel_size=self.narrow_kernel_size, 
                                 bias=False,padding=self.padding)
        self.conv3_2 = nn.Conv2d(self.in_channels, self.out_channels, 
                                 kernel_size=self.narrow_kernel_size, 
                                 bias=False,padding=self.padding)
        self.conv3_3 = nn.Conv2d(self.in_channels, self.out_channels, 
                                 kernel_size=self.narrow_kernel_size, 
                                 bias=False,padding=self.padding)
        self.conv3_1.weight.data.copy_(conv7_module.weight.data[:,:,:,:3])
        self.conv3_2.weight.data.copy_(conv7_module.weight.data[:,:,:,2:5])
        self.conv3_3.weight.data.copy_(conv7_module.weight.data[:,:,:,4:])
        self.conv3_2.weight.data[:,:,:,0] = 0 # 多余的直接设置成0
        self.conv3_2.weight.data[:,:,:,2] = 0 # 多余的直接设置成0
        
    def forward(self, input):
        out1 = self.conv3_1(input[:,:,:,:-2])
        out2 = self.conv3_2(input)
        out3 = self.conv3_3(input[:,:,:,2:])
        out = out1[:,:,:,:-2]+out2[:,:,:,2:-2]+out3[:,:,:,2:]
        return out

class Conv9by3Conv3Padding2(nn.Module):
    def __init__(self,conv9_module,reserve_wide_conv=False):
        super().__init__()
        if reserve_wide_conv:
            self.wide_conv = conv9_module
        self.in_channels = conv9_module.in_channels
        self.out_channels = conv9_module.out_channels
        self.kernel_size = conv9_module.kernel_size
        self.narrow_kernel_size = (self.kernel_size[0],3)
        self.padding = conv9_module.padding
        self.conv3_1 = nn.Conv2d(self.in_channels, self.out_channels, 
                                 kernel_size=self.narrow_kernel_size, 
                                 bias=False,padding=self.padding)
        self.conv3_2 = nn.Conv2d(self.in_channels, self.out_channels, 
                                 kernel_size=self.narrow_kernel_size, 
                                 bias=False,padding=self.padding)
        self.conv3_3 = nn.Conv2d(self.in_channels, self.out_channels, 
                                 kernel_size=self.narrow_kernel_size, 
                                 bias=False,padding=self.padding)
        self.conv3_1.weight.data.copy_(conv9_module.weight.data[:,:,:,:3])
        self.conv3_2.weight.data.copy_(conv9_module.weight.data[:,:,:,3:6])
        self.conv3_3.weight.data.copy_(conv9_module.weight.data[:,:,:,6:9])
        
    def forward(self, input):
        out1 = self.conv3_1(input[:,:,:,:-4])
        out2 = self.conv3_2(input[:,:,:,1:-1])
        out3 = self.conv3_3(input[:,:,:,4:])
        out = out1[:,:,:,:-2]+out2[:,:,:,2:-2]+out3[:,:,:,2:]
        return out

class Conv11by4Conv3Padding2(nn.Module):
    def __init__(self,conv11_module,reserve_wide_conv=False):
        super().__init__()
        if reserve_wide_conv:
            self.wide_conv = conv11_module
        self.in_channels = conv11_module.in_channels
        self.out_channels = conv11_module.out_channels
        self.kernel_size = conv11_module.kernel_size
        self.narrow_kernel_size = (self.kernel_size[0],3)
        self.padding = conv11_module.padding
        self.conv3_1 = nn.Conv2d(self.in_channels, self.out_channels, 
                                 kernel_size=self.narrow_kernel_size, 
                                 bias=False,padding=self.padding)
        self.conv3_2 = nn.Conv2d(self.in_channels, self.out_channels, 
                                 kernel_size=self.narrow_kernel_size, 
                                 bias=False,padding=self.padding)
        self.conv3_3 = nn.Conv2d(self.in_channels, self.out_channels, 
                                 kernel_size=self.narrow_kernel_size, 
                                 bias=False,padding=self.padding)
        self.conv3_4 = nn.Conv2d(self.in_channels, self.out_channels, 
                                 kernel_size=self.narrow_kernel_size, 
                                 bias=False,padding=self.padding)
        self.conv3_1.weight.data.copy_(conv11_module.weight.data[:,:,:,:3])
        self.conv3_2.weight.data.copy_(conv11_module.weight.data[:,:,:,3:6])
        self.conv3_3.weight.data.copy_(conv11_module.weight.data[:,:,:,6:9])
        self.conv3_4.weight.data.copy_(conv11_module.weight.data[:,:,:,8:11])
        self.conv3_4.weight.data[:,:,:,0] = 0 # 多余的直接设置成0
        
    def forward(self, input):
        out1 = self.conv3_1(input[:,:,:,:-6])
        out2 = self.conv3_2(input[:,:,:,1:-3])
        out3 = self.conv3_3(input[:,:,:,4:])
        out4 = self.conv3_4(input[:,:,:,6:])
        # print(out1.shape,out2.shape,out3.shape,out4.shape)
        out = out1[:,:,:,:-2]+out2[:,:,:,2:-2]+out3[:,:,:,2:-2]+out4[:,:,:,2:]
        return out
    

class Conv5by2DirectConv3Padding2(nn.Module):
    def __init__(self,conv5_module,reserve_wide_conv=False,W_bits=3,A_bits=3):
        super().__init__()
        if reserve_wide_conv:
            self.wide_conv = conv5_module
        self.in_channels = conv5_module.in_channels
        self.out_channels = conv5_module.out_channels
        self.kernel_size = conv5_module.kernel_size
        self.narrow_kernel_size = (self.kernel_size[0],3)
        self.padding = conv5_module.padding
        self.conv3_1 = DirectConv2D(self.in_channels, self.out_channels, 
                                 kernel_size=self.narrow_kernel_size, W_bits=W_bits,A_bits=A_bits,
                                 padding=self.padding)
        self.conv3_2 = DirectConv2D(self.in_channels, self.out_channels, 
                                 kernel_size=self.narrow_kernel_size, W_bits=W_bits,A_bits=A_bits, 
                                 padding=self.padding)
        self.conv3_1.weight.data.copy_(conv5_module.weight.data[:,:,:,:3])
        self.conv3_2.weight.data.copy_(conv5_module.weight.data[:,:,:,2:])
        self.conv3_2.weight.data[:,:,:,0] = 0 # 多余的直接设置成0
        
    def forward(self, input):
        out1 = self.conv3_1(input)
        out2 = self.conv3_2(input)
        out = out1[:,:,:,:-2]+out2[:,:,:,2:]
        return out

class Conv7by3DirectConv3Padding2(nn.Module):
    def __init__(self,conv7_module,reserve_wide_conv=False,):
        super().__init__()
        if reserve_wide_conv:
            self.wide_conv = conv7_module
        self.in_channels = conv7_module.in_channels
        self.out_channels = conv7_module.out_channels
        self.kernel_size = conv7_module.kernel_size
        self.narrow_kernel_size = (self.kernel_size[0],3)
        self.padding = conv7_module.padding
        self.conv3_1 = DirectConv2D(self.in_channels, self.out_channels, 
                                 kernel_size=self.narrow_kernel_size, W_bits=W_bits,A_bits=A_bits, 
                                 padding=self.padding)
        self.conv3_2 = DirectConv2D(self.in_channels, self.out_channels, 
                                 kernel_size=self.narrow_kernel_size, W_bits=W_bits,A_bits=A_bits, 
                                 padding=self.padding)
        self.conv3_3 = DirectConv2D(self.in_channels, self.out_channels, 
                                 kernel_size=self.narrow_kernel_size, W_bits=W_bits,A_bits=A_bits, 
                                 padding=self.padding)
        self.conv3_1.weight.data.copy_(conv7_module.weight.data[:,:,:,:3])
        self.conv3_2.weight.data.copy_(conv7_module.weight.data[:,:,:,2:5])
        self.conv3_3.weight.data.copy_(conv7_module.weight.data[:,:,:,4:])
        self.conv3_2.weight.data[:,:,:,0] = 0 # 多余的直接设置成0
        self.conv3_2.weight.data[:,:,:,2] = 0 # 多余的直接设置成0
        
    def forward(self, input):
        out1 = self.conv3_1(input[:,:,:,:-2])
        out2 = self.conv3_2(input)
        out3 = self.conv3_3(input[:,:,:,2:])
        out = out1[:,:,:,:-2]+out2[:,:,:,2:-2]+out3[:,:,:,2:]
        return out

class Conv9by3DirectConv3Padding2(nn.Module):
    def __init__(self,conv9_module,reserve_wide_conv=False):
        super().__init__()
        if reserve_wide_conv:
            self.wide_conv = conv9_module
        self.in_channels = conv9_module.in_channels
        self.out_channels = conv9_module.out_channels
        self.kernel_size = conv9_module.kernel_size
        self.narrow_kernel_size = (self.kernel_size[0],3)
        self.padding = conv9_module.padding
        self.conv3_1 = DirectConv2D(self.in_channels, self.out_channels, 
                                 kernel_size=self.narrow_kernel_size, W_bits=W_bits,A_bits=A_bits, 
                                 padding=self.padding)
        self.conv3_2 = DirectConv2D(self.in_channels, self.out_channels, 
                                 kernel_size=self.narrow_kernel_size, W_bits=W_bits,A_bits=A_bits, 
                                 padding=self.padding)
        self.conv3_3 = DirectConv2D(self.in_channels, self.out_channels, 
                                 kernel_size=self.narrow_kernel_size, W_bits=W_bits,A_bits=A_bits, 
                                 padding=self.padding)
        self.conv3_1.weight.data.copy_(conv9_module.weight.data[:,:,:,:3])
        self.conv3_2.weight.data.copy_(conv9_module.weight.data[:,:,:,3:6])
        self.conv3_3.weight.data.copy_(conv9_module.weight.data[:,:,:,6:9])
        
    def forward(self, input):
        out1 = self.conv3_1(input[:,:,:,:-4])
        out2 = self.conv3_2(input[:,:,:,1:-1])
        out3 = self.conv3_3(input[:,:,:,4:])
        out = out1[:,:,:,:-2]+out2[:,:,:,2:-2]+out3[:,:,:,2:]
        return out

class Conv11by4DirectConv3Padding2(nn.Module):
    def __init__(self,conv11_module,reserve_wide_conv=False):
        super().__init__()
        if reserve_wide_conv:
            self.wide_conv = conv11_module
        self.in_channels = conv11_module.in_channels
        self.out_channels = conv11_module.out_channels
        self.kernel_size = conv11_module.kernel_size
        self.narrow_kernel_size = (self.kernel_size[0],3)
        self.padding = conv11_module.padding
        self.conv3_1 = DirectConv2D(self.in_channels, self.out_channels, 
                                 kernel_size=self.narrow_kernel_size, W_bits=W_bits,A_bits=A_bits, 
                                 padding=self.padding)
        self.conv3_2 = DirectConv2D(self.in_channels, self.out_channels, 
                                 kernel_size=self.narrow_kernel_size, W_bits=W_bits,A_bits=A_bits, 
                                 padding=self.padding)
        self.conv3_3 = DirectConv2D(self.in_channels, self.out_channels, 
                                 kernel_size=self.narrow_kernel_size, W_bits=W_bits,A_bits=A_bits, 
                                 padding=self.padding)
        self.conv3_4 = DirectConv2D(self.in_channels, self.out_channels, 
                                 kernel_size=self.narrow_kernel_size, W_bits=W_bits,A_bits=A_bits, 
                                 padding=self.padding)
        self.conv3_1.weight.data.copy_(conv11_module.weight.data[:,:,:,:3])
        self.conv3_2.weight.data.copy_(conv11_module.weight.data[:,:,:,3:6])
        self.conv3_3.weight.data.copy_(conv11_module.weight.data[:,:,:,6:9])
        self.conv3_4.weight.data.copy_(conv11_module.weight.data[:,:,:,8:11])
        self.conv3_4.weight.data[:,:,:,0] = 0 # 多余的直接设置成0
        
    def forward(self, input):
        out1 = self.conv3_1(input[:,:,:,:-6])
        out2 = self.conv3_2(input[:,:,:,1:-3])
        out3 = self.conv3_3(input[:,:,:,4:])
        out4 = self.conv3_4(input[:,:,:,6:])
        # print(out1.shape,out2.shape,out3.shape,out4.shape)
        out = out1[:,:,:,:-2]+out2[:,:,:,2:-2]+out3[:,:,:,2:-2]+out4[:,:,:,2:]
        return out
    

if __name__ == '__main__':
    # 测试代码
    N,Ci,H,W,Co,W_bits,A_bits = 1,1,11,11,1,3,3
    # s = torch.randn(N,Ci,H,W)
    s = torch.randint(0, 2**A_bits -1, (N, Ci, H, W)).float()
    kernel_5x5 = torch.randint(0, 2**W_bits -1, (Co, Ci, 5, 5)).float()
    conv2d_5x5 = nn.Conv2d(Ci, Co, kernel_size=(5,5), padding=(4,2))
    conv2d_5x5.weight.data.copy_(kernel_5x5)

    conv2d_7x7 = nn.Conv2d(Ci, Co, kernel_size=(7,7), padding=(6,2))
    kernel_7x7 = torch.randint(0, 2**W_bits -1, (Co, Ci, 7, 7)).float()
    conv2d_7x7.weight.data.copy_(kernel_7x7)

    conv2d_9x9 = nn.Conv2d(Ci, Co, kernel_size=(9,9), padding=(8,2))
    kernel_9x9 = torch.randint(0, 2**W_bits -1, (Co, Ci, 9, 9)).float()
    conv2d_9x9.weight.data.copy_(kernel_9x9)

    conv2d_11x11 = nn.Conv2d(Ci, Co, kernel_size=(11,11), padding=(10,2))
    kernel_11x11 = torch.randint(0, 2**W_bits -1, (Co, Ci, 11, 11)).float()
    conv2d_11x11.weight.data.copy_(kernel_11x11)

    print(s.shape)
    # Test Simulate Narrow Convolution
    output_5x5 = conv2d_5x5(s)
    conv2d_2x5x3 = Conv5by2Conv3Padding2(conv2d_5x5)
    output_2x5x3 = conv2d_2x5x3(s)
    print("Error between 1x5x5 and 2x5x3:", (output_5x5-output_2x5x3).abs().mean().item()/output_5x5.abs().mean().item())

    output_7x7 = conv2d_7x7(s)
    conv2d_3x7x3 = Conv7by3Conv3Padding2(conv2d_7x7)
    output_3x7x3 = conv2d_3x7x3(s)
    print("Error between 1x7x7 and 3x7x3:", (output_7x7-output_3x7x3).abs().mean().item()/output_7x7.abs().mean().item())

    output_9x9 = conv2d_9x9(s)
    conv2d_3x9x3 = Conv9by3Conv3Padding2(conv2d_9x9)
    output_3x9x3 = conv2d_3x9x3(s)
    print("Error between 1x9x9 and 3x9x3:", (output_9x9-output_3x9x3).abs().mean().item()/output_9x9.abs().mean().item())

    output_11x11 = conv2d_11x11(s)
    conv2d_4x11x3 = Conv11by4Conv3Padding2(conv2d_11x11)
    output_4x11x3 = conv2d_4x11x3(s)
    print("Error between 1x11x11 and 4x11x3:", (output_11x11-output_4x11x3).abs().mean().item()/output_11x11.abs().mean().item())

    # Test Direct Convolution
    print(s.shape)
    direct_conv2d_2x5x3 = Conv5by2DirectConv3Padding2(conv2d_5x5)
    direct_output_2x5x3 = direct_conv2d_2x5x3(s)
    print("Error between 1x5x5 and 2x5x3:", (output_5x5-direct_output_2x5x3).abs().mean().item()/output_5x5.abs().mean().item())

    direct_conv2d_3x7x3 = Conv7by3DirectConv3Padding2(conv2d_7x7)
    direct_output_3x7x3 = direct_conv2d_3x7x3(s)
    print("Error between 1x7x7 and 3x7x3:", (output_7x7-direct_output_3x7x3).abs().mean().item()/output_7x7.abs().mean().item())

    direct_conv2d_3x9x3 = Conv9by3DirectConv3Padding2(conv2d_9x9)
    direct_output_3x9x3 = direct_conv2d_3x9x3(s)
    print("Error between 1x9x9 and 3x9x3:", (output_9x9-direct_output_3x9x3).abs().mean().item()/output_9x9.abs().mean().item())

    direct_conv2d_4x11x3 = Conv11by4DirectConv3Padding2(conv2d_11x11)
    direct_output_4x11x3 = direct_conv2d_4x11x3(s)
    print("Error between 1x11x11 and 4x11x3:", (output_11x11-direct_output_4x11x3).abs().mean().item()/output_11x11.abs().mean().item())
    print(output_5x5)
    print(direct_output_2x5x3)
