import time
import logging
import torch
from torch import nn
import copy
from tqdm import tqdm
import copy
from torch.quantization import QConfig
import torch.nn.quantized as nnq
from torch.ao.quantization import (
  get_default_qconfig_mapping,
  get_default_qat_qconfig_mapping,
  QConfigMapping,
)
import torch.ao.quantization.quantize_fx as quantize_fx

class MeasureExecutionTime:
    def __init__(self, measure_name = "Execution", type='auto', verbose=True, log_to_file=False, log_file='./execution_time.log'):
        self.log_to_file = log_to_file
        self.log_file = log_file
        self.start_time = None
        self.measure_name = measure_name
        self.execution_time = 0
        self.type=type
        self.verbose = verbose
        self.type_convert = {"auto":0,"s":1,"ms":1000,"us":1000000,"ns":1000000000}
        if self.type not in self.type_convert.keys():
            raise ValueError(f"Invalid type {self.type}, only support {self.type_convert.keys()}")

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def elapsed_time(self):
        if self.start_time is None:
            return 0  # 如果还没开始，返回0
        return time.perf_counter() - self.start_time

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        execution_time = end_time - self.start_time
        self.execution_time = execution_time
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
        if self.verbose:
            print(f"{self.measure_name} time: {display_time:.4f} {type}")
        if self.log_to_file:
            logging.basicConfig(filename=self.log_file, level=logging.INFO)
            logging.info(f"{self.measure_name} time: {display_time:.4f} {type}")


# set_model_quant_bits
def set_model_quant_bits(model, W_bits, A_bits):
    for name, module in model.named_children():
        if hasattr(module, "set_quant_bits"):
            try:
                module.set_quant_bits(W_bits, A_bits)
            except Exception as e:
                raise ValueError(f"Error setting quant bits for module {name}: {e}")
        else:
            set_model_quant_bits(module, W_bits, A_bits)
    return model


# 定义一些工具函数：
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

def test_inference_speed(model,
                         input_shape,
                         qunantize_input=False,
                        test_iters=100,
                        verbose = True,
                        title = ""):
    quant = lambda x: torch.quantize_per_tensor(x, scale=1.0, zero_point=0, dtype=torch.quint8) # torch.quantization.QuantStub()
    dequant = torch.dequantize
    ecalps_time = 0
    batch_size = input_shape[0]
    model.eval()
    with torch.no_grad():
        # for images, labels in testloader:
        for i in tqdm(range(test_iters)):
            images = torch.rand(*input_shape)
            if qunantize_input:
                images = quant(images)
            start_time = time.time()
            outputs = model(images)
            m = outputs.mean().item()
            ecalps_time += time.time()-start_time
    if verbose:
          print(f'{title} Inference time: {ecalps_time} seconds, FPS: {batch_size*test_iters/ecalps_time}')
    return ecalps_time