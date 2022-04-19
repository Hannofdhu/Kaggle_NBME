import torch
from torch import nn
#sigmoid是一个类
a = nn.Sigmoid()
print(a(torch.randn(1)))

