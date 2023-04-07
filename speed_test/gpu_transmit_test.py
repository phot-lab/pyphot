import torch
import numpy as np
import time


def transmit_gpu(data: np.ndarray):
    start = time.time()

    data = torch.from_numpy(data)
    data = data.to("cuda")

    duration = time.time() - start
    size = data.shape
    print(f"Size: {size}, dduration: {duration} s")


if __name__ == "__main__":
    data1 = np.random.randn(10000)
    data2 = np.random.randn(100000)
    data3 = np.random.randn(1000000)
    transmit_gpu(data1)
    transmit_gpu(data2)
    transmit_gpu(data3)
