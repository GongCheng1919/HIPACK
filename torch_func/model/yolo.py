import torch
from replace import replace_conv2d, generate_report
import pandas as pd
import numpy as np

MODEL_NAME = "yolov5s"
def run(mode: str):
    assert mode in ["float", "hipack", "qnnpack"], "mode must be one of 'float', 'hipack', or 'qnnpack'"
    
    model = torch.hub.load("ultralytics/yolov5", MODEL_NAME)
    replace_conv2d(model, mode=mode, W_bits=3, A_bits=3)

    y = model(torch.randn(32, 3, 640, 640))
    print(y.shape)
    
if __name__ == "__main__":
    run("hipack")
    run("qnnpack")
    generate_report(MODEL_NAME)
   
