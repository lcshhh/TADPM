# baseline, main
import torch
import os
import sys
from torch.autograd import Variable
import argparse
from tensorboardX import SummaryWriter
import copy
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
import numpy as np
import torch.nn as nn
import torch.utils.data as data
from model.meshmae import teethArranger
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule, get_cosine_schedule_with_warmup
import time
from sklearn.manifold import TSNE
import vedo
import math
from pytorch3d.transforms import Transform3d
from model.dataset import TeethRegressorDataManager
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import trimesh
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance
from pytorch3d.transforms import se3_exp_map
def seed_torch(seed=12):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def transform_teeth(index,trans_6dof,args):
    dataroot = args.dataroot
    gtroot = args.gtroot
    trans_matrix = se3_exp_map(trans_6dof).transpose(2,1)
    meshes_upper = []
    before_meshes = []
    gt_meshes = []
    ran = range(32)
    for i in ran:
        mesh_path = os.path.join(dataroot,f'{index}_{i}.obj')
        if os.path.exists(mesh_path):
            mesh = vedo.Mesh(mesh_path)
            mesh_before = mesh.clone(True)
            before_meshes.append(mesh_before)
            mesh.apply_transform(trans_matrix[i])
            meshes_upper.append(mesh)
            gt_mesh = vedo.Mesh(os.path.join(gtroot,f'{index}_{i}.obj'))
            gt_meshes.append(gt_mesh)
    mesh_upper = vedo.merge(meshes_upper)
    before = vedo.merge(before_meshes)
    gt = vedo.merge(gt_meshes)
    outputroot = args.outputroot
    os.makedirs(outputroot,exist_ok=True)
    vedo.write(mesh_upper,os.path.join(outputroot,f'{index}_after.obj'))
    vedo.write(before,os.path.join(outputroot,f'{index}_before.obj'))
    vedo.write(gt,os.path.join(outputroot,f'{index}_gt.obj'))

def test_and_save(net, test_dataset, args):
    net.eval()

    n_samples = 0

    for i, (feats_patch, center_patch, coordinate_patch, face_patch, np_Fs, trans_6dof, trans_matrix,index,before_points,after_points,centroid,after_centroid,before_points_centered) in enumerate(
            test_dataset):
        faces = face_patch.cuda()
        feats = feats_patch.to(torch.float32).cuda()
        centers = center_patch.to(torch.float32).cuda()
        Fs = np_Fs.cuda()
        cordinates = coordinate_patch.to(torch.float32).cuda()
        centroid = centroid.to(torch.float32).cuda()
        before_points = before_points.to(torch.float32).cuda()
        after_points = after_points.to(torch.float32).cuda()
        n_samples += faces.shape[0]
        batch_size = faces.shape[0]
        with torch.no_grad():
            outputs = net(faces, feats, centers, Fs, cordinates, centroid, before_points).detach()
            for i in range(index.shape[0]):
                transform_teeth(index[i],outputs[i],args)

if __name__ == '__main__':
    seed_torch(seed=43)
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'test'])
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default='none')
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
    parser.add_argument('--gtroot', type=str, required=True)
    parser.add_argument('--outputroot', type=str, required=True)
    parser.add_argument('--n_classes', type=int)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--n_worker', type=int, default=8)
    parser.add_argument('--paramroot',type=str, required=True)
    parser.add_argument('--total_checkpoint',type=str,default='none')
    parser.add_argument('--mask_ratio', type=float, default=0.25)
    parser.add_argument('--channels', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='./experiments')
    args = parser.parse_args()
    mode = args.mode
    args.name = args.name
    dataroot = args.dataroot
    paramroot = args.paramroot

    # ========== Dataset ==========
    augments = []
    dataManager = TeethRegressorDataManager(dataroot,paramroot,args.train_ratio,)
    train_dataset = dataManager.train_dataset()
    test_dataset = dataManager.test_dataset()
    print(len(train_dataset))
    print(len(test_dataset))
    train_data_loader = data.DataLoader(train_dataset, num_workers=args.n_worker, batch_size=args.batch_size,
                                        shuffle=True, pin_memory=True)
    test_data_loader = data.DataLoader(test_dataset, num_workers=args.n_worker, batch_size=args.batch_size,
                                       shuffle=False, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = teethArranger(args)
    net = nn.DataParallel(net).to(device)
    checkpoint = torch.load(args.checkpoint)
    net.load_state_dict(checkpoint,strict=True)
    

    # ========== Start Testing ==========
    test_and_save(net, test_data_loader, args)

