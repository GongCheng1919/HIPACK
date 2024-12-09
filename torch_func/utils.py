import torch
import torch.nn as nn
import time
import logging
import numpy as np
import torch.nn.quantized


class MeasureExecutionTime:
    def __init__(self, measure_name = "Execution", flops =None, type='auto', log_to_file=False, log_file='./execution_time.log'):
        self.log_to_file = log_to_file
        self.log_file = log_file
        self.start_time = None
        self.measure_name = measure_name
        self.flops = flops
        self.type=type
        self.type_convert = {"auto":0,"s":1,"ms":1000,"us":1000000,"ns":1000000000}
        if self.type not in self.type_convert.keys():
            raise ValueError(f"Invalid type {self.type}, only support {self.type_convert.keys()}")

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        execution_time = end_time - self.start_time
        type = self.type
        if type == "auto":
            # 自动选择时间单位
            if execution_time > 1:
                type = "s"
            elif execution_time > 0.001:
                type = "ms"
            elif execution_time > 0.000001:
                type = "us"
            else:
                type = "ns"
        display_time = execution_time*self.type_convert[type]
        print(f"{self.measure_name} time: {display_time:.4f} {type}")
        perf,unit = 0,""
        if self.flops is not None:
            perf = self.flops/execution_time
            # auto set 单位
            if perf>1e12:
                perf = perf/1e12
                unit = "TFLOPS"
            elif perf>1e9:
                perf = perf/1e9
                unit = "GFLOPS"
            elif perf>1e6:
                perf = perf/1e6
                unit = "MFLOPS"
            elif perf>1e3:
                perf = perf/1e3
                unit = "KFLOPS"
            print(f"Performance: {perf:.4f}{unit}")
        if self.log_to_file:
            logging.basicConfig(filename=self.log_file, level=logging.INFO)
            logging.info(f"Execution time: {display_time:.4f} {type}")
            if self.flops is not None:
                logging.info(f"Performance: {perf:.4f}{unit}")

class Qint8Conv2D(nn.Module):
    def __init__(self, 
                 in_channels, out_channels, 
                 kernel_size, stride=1, 
                 padding=0, dilation=1, 
                 groups=1, bias=None,return_float=False,engine="qnnpack"):
        super(Qint8Conv2D, self).__init__()
        torch.backends.quantized.engine = engine
        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv = torch.nn.quantized.functional.conv2d
        # self.conv.weight.data = self.conv.weight.data.int()
        self.quant = lambda x: torch.quantize_per_tensor(x, 1, 0, torch.quint8)
        weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        self.weight = torch.quantize_per_tensor(weight, 1, 0, torch.qint8)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.return_float = return_float
    
    def forward(self, inp):
        # 检查，如果不是quint8的类型，则转换一下
        if not inp.dtype == torch.quint8:
            inp = self.quant(inp)
        output = self.conv(inp,self.weight,self.bias,
                           padding=self.padding, stride=self.stride,
                           groups=self.groups,dilation=self.dilation,
                           scale=1.,zero_point=0)
        if self.return_float:
            output = torch.dequantize(output)
        return output
    
    @staticmethod
    def from_float_conv2d(conv2d,copy_weight=True):
        # 从一个标准的Conv2d层构造一个Qint8Conv2D层
        qconv = Qint8Conv2D(conv2d.in_channels, conv2d.out_channels, 
                        conv2d.kernel_size[0], conv2d.stride[0], conv2d.padding[0], 
                        conv2d.dilation[0], conv2d.groups, conv2d.bias)
        if copy_weight:
            qconv.weight = torch.quantize_per_tensor(conv2d.weight.data, 1, 0, torch.qint8)
        return qconv