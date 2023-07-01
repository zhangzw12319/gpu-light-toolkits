# from https://www.autodl.com/docs/gpu/
# testing whether the hard device (GPU) has problems.

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import torch
m = k = n = 32768 # 8192, 16384, 32768, etc...
a = torch.zeros(m, k, dtype=torch.float32).cuda("cuda:0")
b = torch.zeros(k, n, dtype=torch.float32).cuda("cuda:0")
for _ in range(10000):
    y = torch.matmul(a, b)
torch.cuda.synchronize("cuda:0")