import torch
import os
import sys
from torch.autograd import Variable
import argparse
# from tensorboardX import SummaryWriter
import copy
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
import numpy as np
import torch.nn as nn
import torch.utils.data as data
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule, get_cosine_schedule_with_warmup
import time
from sklearn.manifold import TSNE
import matplotlib
import vedo
from pytorch3d.transforms import Transform3d
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import trimesh
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance
from scipy.spatial.transform import Rotation
from einops import rearrange
from pytorch3d.transforms import se3_exp_map,se3_log_map
from pytorch3d.transforms import euler_angles_to_matrix
from dataset.dataset import FullTeethDataset
from models.tadpm import TADPM
from util import progress_bar
from models.utils import compute_rotation_matrix_from_ortho6d

def seed_torch(seed=12):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# def transform_vertices(vertices,centroids,dofs):
#     '''
#     vertices: [bs, 32, pt_num, 3]
#     centroids: [bs, 32, 3]
#     dofs: [bs, 32, 6]
#     '''
#     angles = rearrange(dofs[:,:,3:]*torch.pi/6,'b n c -> (b n) c')
#     move = rearrange(dofs[:,:,:3],'b n c -> (b n) c').unsqueeze(1) #[b*n,1,3]
#     centroids = rearrange(centroids, 'b n c -> (b n) c').unsqueeze(1)
#     R = euler_angles_to_matrix(angles,'XYZ')
#     vertices = rearrange(vertices,'b n pn c -> (b n) pn c')
#     vertices = torch.bmm(vertices - centroids,R) + centroids + move
#     return vertices

# def transform_mesh(mesh,centroid,dof):
#     '''
#     vertices: [pt_num, 3]
#     centroids: [3]
#     dofs: [bs, 32, 6]
#     '''
#     angles = dof[3:].unsqueeze(0)
#     move = (dof[:3]).unsqueeze(0) #[b*n,1,3]
#     centroids = centroid.unsqueeze(0)
#     # R = euler_angles_to_matrix(angles,'XYZ') #[3,3]
#     R = compute_rotation_matrix_from_ortho6d(angles)[0]
#     vertices = mesh.vertices
#     vertices = torch.matmul(torch.FloatTensor(vertices).to(centroids.device) - centroids,R) + centroids + move
#     mesh.vertices = vertices.cpu().numpy()
#     return mesh
def transform_mesh(mesh,dof):
    '''
    centroid [bs,16,3]
    '''
    bs = dof.shape[0]
    trans_matrix = se3_exp_map(dof.unsqueeze(0)).transpose(2,1) # [16,4,4]
    vertices = torch.from_numpy(mesh.vertices).to(dof.device).float()
    predicted_vertices = Transform3d(matrix=trans_matrix.transpose(2,1)[0]).transform_points(vertices)
    mesh.vertices = predicted_vertices.cpu().numpy()
    return mesh

def move_mesh(mesh,centroid,dof):
    move = (dof[:3]).unsqueeze(0) #[b*n,1,3]
    vertices = torch.FloatTensor(mesh.vertices).cuda()
    vertices = vertices + move
    mesh.vertices = vertices.cpu().numpy()
    return mesh

def transform_teeth(index,centroid,output):
    before_meshes = []
    meshes = []
    gt_meshes = []
    for i in range(32):
        path = os.path.join('/data3/leics/dataset/mesh/single_before',f'{index}_{i}.obj')
        gt_path = os.path.join('/data3/leics/dataset/mesh/single_after',f'{index}_{i}.obj')
        if os.path.exists(path) and os.path.exists(gt_path):
            mesh = trimesh.load_mesh(path)
            gt_mesh = vedo.Mesh(gt_path)
            before_mesh = trimesh.load_mesh(path)
            before_meshes.append(vedo.trimesh2vedo(before_mesh))
            # after_mesh = move_mesh(mesh,centroid[i],output[i])
            after_mesh = transform_mesh(mesh,output[i])
            meshes.append(vedo.trimesh2vedo(after_mesh))
            gt_meshes.append(gt_mesh)
    mesh = vedo.merge(meshes)
    before_mesh = vedo.merge(before_meshes)
    gt_mesh = vedo.merge(gt_meshes)
    os.makedirs('/data3/leics/outputs',exist_ok=True)
    vedo.write(mesh,f'/data3/leics/outputs/after{index}.obj')
    vedo.write(before_mesh,f'/data3/leics/outputs/before{index}.obj')
    vedo.write(gt_mesh,f'/data3/leics/outputs/gt{index}.obj')


# def chamfer_loss(before_points,after_points,centroids,outputs):
#     '''
#     outputs: [bs,teeth_num,6]
#     before_points: [bs,teeth_num,point_num,3]
#     '''
#     bs = before_points.shape[0]
#     predicted_points = transform_vertices(before_points, centroids, outputs)
#     after_points = rearrange(after_points,'b n pn c -> (b n) pn c')
#     loss, _ = chamfer_distance(after_points, predicted_points, point_reduction="sum", norm=1)
#     return loss/bs

def test(net, names, optimizer, scheduler, test_dataset, epoch, args, autoencoder=None):

    net.eval()

    running_loss = 0
    n_samples = 0

    for it, (feats_patch, center_patch, coordinate_patch, face_patch, np_Fs, index, before_points, after_points, centroid,after_centroid, dofs, masks) in enumerate(
            test_dataset):
        faces = face_patch.cuda()
        feats = feats_patch.to(torch.float32).cuda()
        centers = center_patch.to(torch.float32).cuda()
        Fs = np_Fs.cuda()
        cordinates = coordinate_patch.to(torch.float32).cuda()
        before_points = before_points.to(torch.float32).cuda()
        after_points = after_points.to(torch.float32).cuda()
        centroid = centroid.to(torch.float32).cuda()
        after_centroid = after_centroid.to(torch.float32).cuda()
        masks = masks.cuda()
        n_samples += faces.shape[0]
        with torch.no_grad():
            outputs = net(faces, feats, centers, Fs, cordinates, centroid, before_points).to(torch.float32)
            # print(outputs)
            # print(dofs)
            # exit()
            # for j in range(32):
            #     predicted_centroid = centroid[0,j] + outputs[0,j,:3]
            #     print('move:',outputs[0,j,:3])
            #     print(predicted_centroid)
            #     print(after_centroid[0,j])
            for i in range(index.shape[0]):
                transform_teeth(index[i],centroid[i],outputs[i])

if __name__ == '__main__':
    # seed_torch(seed=43)
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--lr_milestones', type=str, default=None)
    parser.add_argument('--num_warmup_steps', type=str, default=None)
    parser.add_argument('--depth', type=int, required=True)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--encoder_depth', type=int, default=6)
    parser.add_argument('--decoder_dim', type=int, default=512)
    parser.add_argument('--decoder_depth', type=int, default=6)
    parser.add_argument('--decoder_num_heads', type=int, default=6)
    parser.add_argument('--dim', type=int, default=384)
    parser.add_argument('--heads', type=int, required=True)
    parser.add_argument('--patch_size', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epoch', type=int, required=True, default=500)
    parser.add_argument('--dataroot', type=str, required=True)
    parser.add_argument('--n_classes', type=int)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--n_worker', type=int, default=8)
    parser.add_argument('--paramroot',type=str, required=True)
    parser.add_argument('--encoder_checkpoint',type=str,default='')
    parser.add_argument('--checkpoint',type=str,default='')
    parser.add_argument('--use_mlp', action='store_true')
    parser.add_argument('--use_pointnet', action='store_true')
    parser.add_argument('--use_ae', action='store_true')
    parser.add_argument('--pure_test', action='store_true')
    parser.add_argument('--lower', action='store_true')
    parser.add_argument('--mask_ratio', type=float, default=0.25)
    parser.add_argument('--channels', type=int, default=10)
    parser.add_argument('--train_ratio', type=float, default=0.9)
    args = parser.parse_args()
    # mode = args.mode
    args.name = args.name
    dataroot = args.dataroot
    paramroot = args.paramroot

    # ========== Dataset ==========
    augments = []
    # dataManager = OriginalFullTeethDataManager(dataroot,paramroot,args.train_ratio,)
    # dataManager = OriginalFullTeethDataManager(dataroot,paramroot,args.train_ratio,)
    # train_dataset = dataManager.train_dataset()
    # test_dataset = dataManager.test_dataset()
    train_dataset = FullTeethDataset(dataroot,paramroot,'train.txt',True)
    test_dataset = FullTeethDataset(dataroot,paramroot,'val.txt',False)
    print(len(train_dataset))
    print(len(test_dataset))
    train_data_loader = data.DataLoader(train_dataset, num_workers=args.n_worker, batch_size=args.batch_size,
                                        shuffle=True, pin_memory=True)
    test_data_loader = data.DataLoader(test_dataset, num_workers=args.n_worker, batch_size=args.batch_size,
                                       shuffle=False, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = TADPM(args).to(device)
    net = nn.DataParallel(net)
    if args.checkpoint != '':
        checkpoint = torch.load(args.checkpoint)
        print('loading model...')
        net.load_state_dict(checkpoint['model'],strict=True)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # if args.mode != 'train':
    

    # ========== Optimizer ==========
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epoch)
    checkpoint_names = []
    checkpoint_path = os.path.join('/data/lcs/new_checkpoints', args.name)

    os.makedirs(checkpoint_path, exist_ok=True)

    # ========== Start Training ==========
    test(net, args.name, optimizer, scheduler, test_data_loader, 0, args)

