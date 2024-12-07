import torch
import sys, os
notebook_path = os.getcwd()  # Get the current working directory
parent_directory = os.path.dirname(notebook_path)  # Get the parent directory
sys.path.append(parent_directory)
# from models.resnet import ResNet18
# from models.vggnet import ExpVGG16_BN
from models.utils import eager_quantize_model,fx_quantize_model,MeasureExecutionTime
from models.conv import fuse_module
from models.mobilenetv2 import mobilenet_v2,mobilenet_v2_qint8,mobilenet_v2_dc
# 设置量化引擎为QNNPACK
torch.backends.quantized.engine = 'qnnpack'

x = torch.rand(16,3,224,224)
model =mobilenet_v2()
q_model = mobilenet_v2_qint8()
dc_model = mobilenet_v2_dc(1,1)
# out= model(x)
# with MeasureExecutionTime(measure_name="Float"):
#     out= model(x)

out= q_model(x)
with MeasureExecutionTime(measure_name="Qint8"):
    out= q_model(x)

out= dc_model(x)
with MeasureExecutionTime(measure_name="DC"):
    out= dc_model(x)


