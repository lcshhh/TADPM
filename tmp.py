import torch
from utils.pointcloud import write_pointcloud
clouds = torch.load('/data3/leics/dataset/full.pt')
write_pointcloud(clouds[2].cpu().numpy(),'/data3/leics/dataset/gt.ply')