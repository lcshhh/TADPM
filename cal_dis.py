import torch
from utils.pointcloud import write_pointcloud
from pytorch3d.loss import chamfer_distance
before_pointclouds = torch.load('/data3/leics/dataset/gen_res5.pt').cuda() * 40
write_pointcloud(before_pointclouds[0].cpu().numpy(),'/data3/leics/dataset/test.ply')
exit()
number_of_shapes = before_pointclouds.shape[0]
chamfer_distances = torch.zeros(number_of_shapes,number_of_shapes).cuda()
r = 10
num = 0
for i in range(number_of_shapes):
    batch_size = 160
    # print('num:',num)
    for j in range(i+1,number_of_shapes,batch_size):
        start = j
        end = min(j+batch_size,number_of_shapes)
        real_batch_size = end - start
        cd = chamfer_distance(before_pointclouds[i].unsqueeze(0).repeat(real_batch_size,1,1),before_pointclouds[start:end],batch_reduction=None)[0]
        cd = (cd > r).sum()
        if cd > 0:
            num+=1
            break
print(num/number_of_shapes)
