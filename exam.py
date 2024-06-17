import torch
import torch.nn as nn
import torch.nn.functional as F
# Q = torch.FloatTensor([[2,1,4],[7,6,8]])
# K = torch.FloatTensor([[1,2,3],[3,1,4],[3,2,3]])
# V = torch.FloatTensor([[3,3,3],[2,2,2],[4,2,3]])
# soft = nn.Softmax()
# mat = torch.matmul(Q,K.transpose(0,1))
# print(soft(mat))
# a = soft(mat)
# print(torch.matmul(a,V))

# mat = torch.FloatTensor([[12,16,12],[24,32,24]])
# soft = nn.Softmax()
# a = soft(mat)
# print(a)
# b = torch.matmul(a,torch.FloatTensor([3,2,3]).view(3,1))
# print(b)
import numpy as np
import trimesh
import vedo
mesh = vedo.Mesh('/data/lcs/dataset/teeth_full/centered_before/1.vtp')
mesh = vedo.vedo2trimesh(mesh)
m = np.max(np.sqrt(np.sum(mesh.vertices ** 2, axis=1)))
print(m)