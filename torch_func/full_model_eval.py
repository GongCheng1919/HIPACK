import torch
import sys, os
notebook_path = os.getcwd()  # Get the current working directory
parent_directory = os.path.dirname(notebook_path)  # Get the parent directory
sys.path.append(parent_directory)
from models.resnet import (
    resnet18, resnet18_qint8, resnet18_dc,
    resnet34, resnet34_qint8, resnet34_dc,
)
from models.vggnet import (
    vgg16_float, vgg16_qint8, vgg16_dc,
)

from models.utils import eager_quantize_model,fx_quantize_model,MeasureExecutionTime
from models.conv import fuse_module
# Set engine QNNPACK
torch.backends.quantized.engine = 'qnnpack'

W_bits, A_bits = 3,3
batchsize = 16
x = torch.rand(batchsize,3,224,224)
# Evaluate latency on VGG16
model =vgg16_float()
q_model = vgg16_qint8()
dc_model = vgg16_dc(W_bits, A_bits)
print(f"Evaluate latency on VGG16 with batchsize of {batchsize}:")
with MeasureExecutionTime(measure_name="Float"):
    out= model(x)

with MeasureExecutionTime(measure_name="Qint8"):
    out= q_model(x)

with MeasureExecutionTime(measure_name="HIPACK"):
    out= dc_model(x)

# Evaluate latency on ResNet18
model =resnet18()
q_model = resnet18_qint8()
dc_model = resnet18_dc(W_bits, A_bits)
print(f"Evaluate latency on ResNet18 with batchsize of {batchsize}:")
with MeasureExecutionTime(measure_name="Float"):
    out= model(x)

with MeasureExecutionTime(measure_name="Qint8"):
    out= q_model(x)

with MeasureExecutionTime(measure_name="HIPACK"):
    out= dc_model(x)

# Evaluate latency on ResNet34
model =resnet34()
q_model = resnet34_qint8()
dc_model = resnet34_dc(W_bits, A_bits)
print(f"Evaluate latency on ResNet34 with batchsize of {batchsize}:")
with MeasureExecutionTime(measure_name="Float"):
    out= model(x)

with MeasureExecutionTime(measure_name="Qint8"):
    out= q_model(x)

with MeasureExecutionTime(measure_name="HIPACK"):
    out= dc_model(x)
