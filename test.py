
import torch
from pytorch3d.transforms import *
a = torch.randn(2,6)
b = (a[0] + a[1]).unsqueeze(0)
b = se3_exp_map(b)
a = se3_exp_map(a)
trans1 = a[0].transpose(1,0)
trans2 = a[1].transpose(1,0)
trans = b[0].transpose(1,0)
print(trans1)
print(trans2)
print(trans)