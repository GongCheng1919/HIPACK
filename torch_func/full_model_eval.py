import torch
import sys, os
notebook_path = os.getcwd()  # Get the current working directory
parent_directory = os.path.dirname(notebook_path)  # Get the parent directory
sys.path.append(parent_directory)
from model.resnet import (
    resnet18, resnet18_qint8, resnet18_dc,
    resnet34, resnet34_qint8, resnet34_dc,
)
from model.vggnet import (
    vgg16_float, vgg16_qint8, vgg16_dc,
)
from model.utils import eager_quantize_model,fx_quantize_model,MeasureExecutionTime

sys.path.insert(0, "model/yolov5")

from detect_dc import run as yolo
from detect_qint8 import run as yolo_qint8

from model.conv import fuse_module
# Set engine QNNPACK
torch.backends.quantized.engine = 'qnnpack'

# W_bits, A_bits = 4, 4
# batchsize = 16
# x = torch.rand(batchsize,3,224,224)
# # Evaluate latency on VGG16
# model =vgg16_float()
# q_model = vgg16_qint8()
# dc_model = vgg16_dc(W_bits, A_bits)
# print(f"Evaluate latency on VGG16 with batchsize of {batchsize}:")
# with MeasureExecutionTime(measure_name="Float"):
#     out= model(x)

# with MeasureExecutionTime(measure_name="Qint8"):
#     out= q_model(x)

# with MeasureExecutionTime(measure_name="HIPACK"):
#     out= dc_model(x)

# # Evaluate latency on ResNet18
# model =resnet18()
# q_model = resnet18_qint8()
# dc_model = resnet18_dc(W_bits, A_bits)
# print(f"Evaluate latency on ResNet18 with batchsize of {batchsize}:")
# with MeasureExecutionTime(measure_name="Float"):
#     out= model(x)

# with MeasureExecutionTime(measure_name="Qint8"):
#     out= q_model(x)

# with MeasureExecutionTime(measure_name="HIPACK"):
#     out= dc_model(x)

# # Evaluate latency on ResNet34
# model =resnet34()
# q_model = resnet34_qint8()
# dc_model = resnet34_dc(W_bits, A_bits)
# print(f"Evaluate latency on ResNet34 with batchsize of {batchsize}:")
# with MeasureExecutionTime(measure_name="Float"):
#     out= model(x)

# with MeasureExecutionTime(measure_name="Qint8"):
#     out= q_model(x)

# with MeasureExecutionTime(measure_name="HIPACK"):
#     out= dc_model(x)


W_bits, A_bits = 5,5
batchsize = 32
x = torch.rand(batchsize, 3, 640, 640)
input(f"Press Enter to continue... {os.getpid()}")

with MeasureExecutionTime(measure_name="Float"):
    os.environ["ENABLE_HIPACK"] = "0"
    yolo("yolov5s.pt", x)

with MeasureExecutionTime(measure_name="HIPACK"):
    os.environ["ENABLE_HIPACK"] = "1"
    os.environ["HIPACK_W_BITS"] = str(W_bits)
    os.environ["HIPACK_A_BITS"] = str(A_bits)
    yolo("yolov5s.pt", x)
