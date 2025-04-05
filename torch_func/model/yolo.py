import torch
from diffusion.replace import replace_conv2d, result_dict
import pandas as pd
import numpy as np

MODEL_NAME = "yolov5s"
def run(mode: str):
    assert mode in ["float", "hipack", "qnnpack"], "mode must be one of 'float', 'hipack', or 'qnnpack'"
    
    model = torch.hub.load("ultralytics/yolov5", MODEL_NAME)
    replace_conv2d(model, mode=mode, W_bits=4, A_bits=4)

    y = model(torch.randn(32, 3, 640, 640))
    print(y.shape)
    
if __name__ == "__main__":
    run("hipack")
    run("qnnpack")

    df = pd.DataFrame(result_dict)
    df = df.transpose()

    df["hipack_avg"] = df["hipack"].apply(lambda x: np.mean(x))
    df["hipack_std"] = df["hipack"].apply(lambda x: np.std(x))

    df["float32_avg"] = df["float32"].apply(lambda x: np.mean(x))
    df["float32_std"] = df["float32"].apply(lambda x: np.std(x))

    df["qnnpack_avg"] = df["qnnpack"].apply(lambda x: np.mean(x))
    df["qnnpack_std"] = df["qnnpack"].apply(lambda x: np.std(x))

    df.drop(columns=["hipack", "float32", "qnnpack"], inplace=True)
    df.drop(columns=["hipack_std", "float32_std", "qnnpack_std"], inplace=True)

    # compare hipack, float32, qnnpack speed, add a column "winner" to show which one is faster
    df["winner"] = df.apply(lambda row: "hipack_avg" if row["hipack_avg"] < row["float32_avg"] and row["hipack_avg"] < row["qnnpack_avg"] else ("qnnpack_avg" if row["qnnpack_avg"] < row["float32_avg"] else "float32_avg"), axis=1)
    df.to_csv(f"{MODEL_NAME}_result.csv", index=True)
