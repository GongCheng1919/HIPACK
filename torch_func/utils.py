import torch
import torch.nn as nn
import time
import logging
import copy
import numpy as np
import torch.nn.quantized
from torch.quantization import QConfig
import torch.nn.quantized as nnq
from torch.ao.quantization import (
  get_default_qconfig_mapping,
  get_default_qat_qconfig_mapping,
  QConfigMapping,
)
import torch.ao.quantization.quantize_fx as quantize_fx


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
        print(f"{self.measure_name} time: {display_time:.4f} {type}" , end = "  ")
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
        else:
            print()
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
    
def eager_quantize_model(model,shape,
                         prepare = True,
                         engine_name = 'qnnpack',
                         inplace=False):
    # engine_name = 'qnnpack' # 'fbgemm' # 'qnnpack'
    # Set the quantization engine to QNNPACK
    torch.backends.quantized.engine = engine_name
    # qconfig = torch.quantization.get_default_qconfig(engine_name)
    qconfig = QConfig(
        weight=torch.quantization.default_observer.with_args(dtype=torch.qint8),  # Use quint8 for weights
        activation=torch.quantization.default_observer.with_args(dtype=torch.quint8)  # Use quint8 for activations
    )
    q_model = model
    if not inplace:
        q_model = copy.deepcopy(model)

    q_model.eval()
    # 指定要融合的模块序列
    # modules_to_fuse = [['features.0', 'features.1', 'features.2'],
    # 				['features.3', 'features.4', 'features.5'],
    # 				['features.7', 'features.8', 'features.9'],
    # 				['features.10', 'features.11', 'features.12'],
    # 				['features.14', 'features.15', 'features.16'],
    # 				['features.17', 'features.18', 'features.19'],
    # 				["classifier.0","classifier.1"]
    # 				]
    # 融合模块
    # q_model = torch.quantization.fuse_modules(q_model, modules_to_fuse)
    # Prepare the model for quantization
    q_model.qconfig = qconfig 
    # 只量化卷积层
    # for n,m in q_model.named_modules():
    # 	if isinstance(m,(nn.Conv2d,torch.quantization.QuantStub,torch.quantization.DeQuantStub)):
    # 		m.qconfig = qconfig
    torch.quantization.prepare(q_model, inplace=True)
    # Quantize the model
    # run_model_on_data(model, data)
    if prepare:
        calibra_data = torch.rand(shape)
        q_model(calibra_data)
    torch.quantization.convert(q_model, inplace=True)
    return q_model

def fx_quantize_model(model,shape,engine_name = 'qnnpack',inplace=False):
    # engine_name = 'qnnpack' # 'fbgemm' # 'qnnpack'
    # if not inplace:
    #     model_fp = copy.deepcopy(model)

    # post training static quantization
    model_to_quantize = model
    if not inplace:
        model_to_quantize = copy.deepcopy(model)
    # fusion
    model_to_quantize.eval()
    model_to_quantize = quantize_fx.fuse_fx(model_to_quantize)
    qconfig_mapping = get_default_qconfig_mapping(engine_name)
    example_inputs = torch.rand(shape)
    # prepare
    model_prepared = quantize_fx.prepare_fx(model_to_quantize, qconfig_mapping, example_inputs)
    # calibrate (not shown)
    model_prepared(example_inputs)
    # quantize
    model_quantized = quantize_fx.convert_fx(model_prepared)
    return model_quantized