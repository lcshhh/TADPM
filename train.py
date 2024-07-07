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
from einops import rearrange, repeat
from pytorch3d.transforms import se3_exp_map,se3_log_map
from pytorch3d.transforms import euler_angles_to_matrix
from dataset.dataset import FullTeethDataset
from models.tadpm import TADPM
from models.utils import compute_rotation_matrix_from_ortho6d
from util import progress_bar

def seed_torch(seed=12):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def transform_vertices(vertices,centroids,dofs):
    '''
    vertices: [bs, 32, pt_num, 3]
    centroids: [bs, 32, 3]
    dofs: [bs, 32, 6]
    '''
    angles = rearrange(dofs[:,:,3:],'b n c -> (b n) c')
    move = rearrange(dofs[:,:,:3],'b n c -> (b n) c').unsqueeze(1) #[b*n,1,3]
    centroids = rearrange(centroids, 'b n c -> (b n) c').unsqueeze(1)
    R = compute_rotation_matrix_from_ortho6d(angles)
    vertices = rearrange(vertices,'b n pn c -> (b n) pn c')
    vertices = torch.bmm(vertices - centroids,R) + centroids + move
    return vertices

def chamfer_loss(before_points,after_points,outputs,masks=None):
    '''
    outputs: [bs,16,6]
    before_points: [bs,16,2048,3]
    '''
    bs = before_points.shape[0]
    loss = torch.FloatTensor([0]).cuda()
    outputs = rearrange(outputs,'b n c -> (b n) c')
    trans_matrix = se3_exp_map(outputs).transpose(1,2)
    before_points = rearrange(before_points,'b n p c -> (b n) p c')
    after_points = rearrange(after_points,'b n p c -> (b n) p c')
    riged_tar = Transform3d(matrix=trans_matrix.transpose(1,2)).transform_points(before_points)
    if masks is not None:
        loss,_ = chamfer_distance(after_points, riged_tar, point_reduction="sum", batch_reduction=None, norm=1)
        loss = (loss * masks.view(loss.shape)).sum()
    else:
        loss,_ = chamfer_distance(after_points, riged_tar, point_reduction="sum", norm=1)
    return loss/bs

def rotation_matrix(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    # a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)


    bs = vec1.shape[0]
    a = nn.functional.normalize(vec1,dim=-1)
    b = nn.functional.normalize(vec2,dim=-1)
    n_vector = torch.cross(a, b, dim=-1)
    c = torch.sum(a*b,dim=-1)
    s = torch.norm(n_vector,dim=-1)
    # s.masked_fill_(masks.flatten(),1)
    n_matrix = torch.zeros(bs,3,3).cuda()
    for i in range(bs):
        n_matrix[i] = torch.tensor([[0, -n_vector[i,2], n_vector[i,1]],
                             [n_vector[i,2], 0, -n_vector[i,0]],
                             [-n_vector[i,1], n_vector[i,0], 0]], dtype=torch.float32)
    I = torch.stack([torch.eye(3) for _ in range(bs)],dim=0).cuda()
    rotation_matrix = I + n_matrix + ((torch.bmm(n_matrix,n_matrix) * ((1 - c.view(-1,1,1)) / (s.view(-1,1,1) ** 2))))
    return rotation_matrix

def rotation_matrix_with_axis(theta, v):
    """
    创建绕任意轴旋转的旋转矩阵
    :param theta: 旋转角度（弧度）
    :param v: 旋转轴的单位向量
    :return: 旋转矩阵
    """
    bs = theta.shape[0]
    K = torch.zeros(bs,3,3).cuda()
    for i in range(bs):
        K[i] = torch.tensor([[0, -v[i,2], v[i,1]],
                        [v[i,2], 0, -v[i,0]],
                        [-v[i,1], v[i,0], 0]]).cuda()
    I = torch.stack([torch.eye(3) for _ in range(bs)],dim=0).cuda()
    R = I + torch.sin(theta).view(-1,1,1) * K + (1 - torch.cos(theta).view(-1,1,1)) * torch.bmm(K, K)
    return R

def align_axis(normal1,normal2,gt_normal1,gt_normal2):
    rot_matrix = rotation_matrix(gt_normal1.view(-1,3),normal1.view(-1,3)).transpose(2,1) #将z轴对齐

    after_axis = torch.bmm(gt_normal2.view(-1,3).unsqueeze(1),rot_matrix)
    rho = torch.cross(after_axis.squeeze(1), normal2.view(-1,3),dim=-1)
    eps = 1e-7
    theta = torch.acos(torch.clamp(torch.sum(normal2.view(-1,3)*after_axis.squeeze(1),dim=-1),min=-1+eps,max=1-eps)).cuda()
    theta = -torch.sign(torch.sum(normal1.view(-1,3)*rho,dim=-1)) * theta
    R = rotation_matrix_with_axis(theta,normal1.view(-1,3))
    RR = torch.bmm(rot_matrix,R)
    return RR

def get_center_and_axis(x):
    '''
    x:[bs,32,8]
    '''
    centers = x[:,:,:3]
    normal1 = x[:,:,3:6]
    normal2 = torch.zeros_like(normal1).to(normal1.device)
    normal2[:,:,:2] = x[:,:,6:]
    normal2[:,:,2] = -(normal1[:,:,0]*x[:,:,6]+normal1[:,:,1]*x[:,:,7])/normal1[:,:,2]
    return centers,normal1,normal2

def get_axis(x):
    '''
    x:[bs,32,8]
    '''
    normal1 = x[:,:,:3]
    normal2 = torch.zeros_like(normal1).to(normal1.device)
    normal2[:,:,:2] = x[:,:,3:]
    normal2[:,2] = -(normal1[:,0]*x[:,3]+normal1[:,1]*x[:,4])/normal1[:,2]
    return normal1,normal2


def centroid_loss(centroid,after_centroid,outputs):
    '''
    centroid [bs,16,3]
    '''
    bs = centroid.shape[0]
    loss = torch.FloatTensor([0]).cuda()
    for i in range(bs):
        trans_matrix = se3_exp_map(outputs[i]).transpose(2,1) # [16,4,4]
        predicted_centroid = Transform3d(matrix=trans_matrix.transpose(2,1)).transform_points(centroid[i].unsqueeze(1)).squeeze(1) # [16,3]
        pre_dis = torch.sqrt(torch.sum(torch.square(predicted_centroid[:-1] - predicted_centroid[1:]),dim=-1))
        after_dis = torch.sqrt(torch.sum(torch.square(after_centroid[i][:-1] - after_centroid[i][1:]),dim=-1))
        loss += torch.abs(pre_dis - after_dis).sum()
    return loss/bs

def cal_add_loss(before_points,after_points,RR,predicted_centroid,centroid, masks):
    points = rearrange(before_points - centroid.unsqueeze(2),'b n p c -> (b n) p c')
    predicted_points = torch.bmm(points,RR)
    predicted_points = predicted_points + rearrange(predicted_centroid,'b n c -> (b n) c').unsqueeze(1)
    after_points = rearrange(after_points,'b n p c -> (b n) p c')
    criterion = nn.MSELoss(reduction='none')
    add_loss = criterion(predicted_points,after_points).sum(dim=(-1,-2))
    add_loss = (add_loss * masks.flatten()).sum()
    return add_loss


def train(net, optim, names, scheduler, train_dataset, epoch, args):
    net.train()
    running_loss = 0
    n_samples = 0

    for it, (feats_patch, center_patch, coordinate_patch, face_patch, np_Fs, index, before_points, after_points, centroid,after_centroid,after_axis,before_axis,masks) in enumerate(
            train_dataset):
        optim.zero_grad()
        faces = face_patch.to(torch.float32).cuda()
        feats = feats_patch.to(torch.float32).cuda()
        centers = center_patch.to(torch.float32).cuda()
        Fs = np_Fs.cuda()
        cordinates = coordinate_patch.cuda()
        centroid = centroid.to(torch.float32).cuda()
        after_centroid = after_centroid.to(torch.float32).cuda()
        after_axis = after_axis.to(torch.float32).cuda()
        axis = after_axis[:,:,:8]
        before_axis = before_axis.to(torch.float32).cuda()
        masks = masks.cuda()
        n_samples += faces.shape[0]
        before_points = before_points.to(torch.float32).cuda()
        after_points = after_points.to(torch.float32).cuda()
        if args.use_mlp:
            outputs = net(faces, feats, centers, Fs, cordinates, centroid, before_points).to(torch.float32).cuda()
        else:
            outputs = net(faces, feats, centers, Fs, cordinates, centroid, before_points, axis).to(torch.float32).cuda()
        criterion = nn.MSELoss(reduction='none')
        loss2 = criterion(axis,outputs).sum(dim=-1)
        center_loss = (loss2 * masks).sum()

        # criterion2 = nn.CosineEmbeddingLoss(reduction='none')
        # bs = axis.shape[0]
        # target = torch.ones(bs).cuda()
        # # axis_loss = torch.FloatTensor([0.]).to(device)
        # # for i in range(bs):
        # normal1 = rearrange(outputs[:,:,3:6],'b n c -> (b n) c')
        # gt_normal1 = rearrange(axis[:,:,3:6],'b n c -> (b n) c')
        # normal2 = rearrange(outputs[:,:,6:],'b n c -> (b n) c')
        # gt_normal2 = rearrange(axis[:,:,6:],'b n c -> (b n) c')
        # target = torch.ones(32*bs).to(device)
        # axis_loss = (criterion2(normal1,gt_normal1,target) + criterion2(normal2,gt_normal2,target))
        # axis_loss = (axis_loss * masks.flatten()).sum()
        # loss2 = centroid_loss(centroid, after_centroid, outputs)
        # loss = chamfer_loss(before_points,after_points,outputs).mean()
        predicted_centroid, gt_normal1, gt_normal2 = get_center_and_axis(outputs)
        gt_normal1 = torch.nn.functional.normalize(gt_normal1,dim=-1)
        gt_normal2 = torch.nn.functional.normalize(gt_normal2,dim=-1)
        normal1 = before_axis[:,:,3:6]
        normal2 = before_axis[:,:,6:]
        RR = align_axis(normal1,normal2,gt_normal1,gt_normal2)
        add_loss = 0.1*cal_add_loss(before_points,after_points,RR,predicted_centroid,centroid,masks)
        print('center_loss:',center_loss)
        print('add_loss',add_loss)
        loss = center_loss + add_loss
        # print('centroid loss:',loss2)
        # loss = loss2
        loss.backward()
        optim.step()
        running_loss += loss.item() * faces.size(0)
        progress_bar(it, len(train_dataset), 'Train Loss: %.3f'% (running_loss/n_samples))

    scheduler.step()
    epoch_loss = running_loss / n_samples
    message = 'epoch ({:}): {:} Train Loss: {:.4f}'.format(names, epoch, epoch_loss)
    with open(os.path.join(args.saveroot, names, 'log.txt'), 'a') as f:
        f.write(message+'\n')
    print()
    print(message)
    if (epoch+1)%50 == 0:
        best_model_wts = copy.deepcopy(net.state_dict())
        torch.save({'model':best_model_wts,'optim':optim.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch}, os.path.join(args.saveroot, names, f'acc-{epoch_loss:.4f}-{epoch}.pkl'))

def test(net, names, optimizer, scheduler, test_dataset, epoch, args, autoencoder=None):

    net.eval()
    if autoencoder!=None:
        autoencoder.eval()

    running_loss = 0
    n_samples = 0

    for it, (feats_patch, center_patch, coordinate_patch, face_patch, np_Fs, index, before_points, after_points, centroid,after_centroid, after_axis,before_axis,masks) in enumerate(
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
        # masks = (masks>0).cuda()
        after_axis = after_axis.to(torch.float32).cuda()
        axis = after_axis[:,:,:8]
        masks = masks.cuda()
        # trans_matrix = trans_matrix.to(torch.float32).cuda()
        n_samples += faces.shape[0]
        # trans_6dof = trans_6dof.to(torch.float32).cuda()
        with torch.no_grad():
            if args.use_mlp:
                outputs = net(faces, feats, centers, Fs, cordinates, centroid, before_points).to(torch.float32).cuda()
            else:
                outputs = net(faces, feats, centers, Fs, cordinates, centroid, before_points, axis).to(torch.float32).cuda()
            # loss = 512*chamfer_loss(before_points,after_points,centroid, outputs, masks)
            # loss = chamfer_loss(before_points,after_points,outputs/10, masks)
            criterion = nn.MSELoss(reduction='none')
            loss2 = criterion(axis[:,:,:3],outputs[:,:,:3]).sum(dim=-1)
            center_loss = (loss2 * masks).sum()

            criterion2 = nn.CosineEmbeddingLoss(reduction='none')
            bs = axis.shape[0]
            target = torch.ones(bs).cuda()
            # axis_loss = torch.FloatTensor([0.]).to(device)
            # for i in range(bs):
            normal1 = rearrange(outputs[:,:,3:6],'b n c -> (b n) c')
            gt_normal1 = rearrange(axis[:,:,3:6],'b n c -> (b n) c')
            normal2 = rearrange(outputs[:,:,6:],'b n c -> (b n) c')
            gt_normal2 = rearrange(axis[:,:,6:],'b n c -> (b n) c')
            target = torch.ones(32*bs).to(device)
            axis_loss = (criterion2(normal1,gt_normal1,target) + criterion2(normal2,gt_normal2,target))
            axis_loss = (axis_loss * masks.flatten()).sum()
            # loss2 = centroid_loss(centroid, after_centroid, outputs)
            # loss = chamfer_loss(before_points,after_points,outputs).mean()
            loss = center_loss + axis_loss
            running_loss += loss.item() * faces.size(0)
            progress_bar(it, len(test_dataset), 'Test Loss: %.3f'% (running_loss/n_samples))

    epoch_loss = running_loss / n_samples
    if test.best_loss > epoch_loss:
        test.best_loss = epoch_loss
        best_model_wts = copy.deepcopy(net.state_dict())
        torch.save({'model':best_model_wts,'optim':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch}, os.path.join(args.saveroot, names, 'best_acc.pkl'))
        # torch.save(best_model_wts, os.path.join('/data/lcs/created_checkpoints', names, 'best_acc.pkl'))

    message = 'epoch ({:}): {:} test Loss: {:.4f}'.format(names, epoch, epoch_loss)
    with open(os.path.join(args.saveroot, names, 'log.txt'), 'a') as f:
        f.write(message+'\n')
    print()
    print(message)

if __name__ == '__main__':
    seed_torch(seed=43)
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'test'])
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
    parser.add_argument('--max_epoch', type=int, default=300)
    parser.add_argument('--dataroot', type=str, required=True)
    parser.add_argument('--n_classes', type=int)
    parser.add_argument('--saveroot', type=str, required=True)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--n_worker', type=int, default=8)
    parser.add_argument('--paramroot',type=str, required=True)
    parser.add_argument('--before_path',type=str, required=True)
    parser.add_argument('--after_path',type=str, required=True)
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
    mode = args.mode
    args.name = args.name
    dataroot = args.dataroot
    paramroot = args.paramroot

    # ========== Dataset ==========
    augments = []
    # dataManager = FullTeethDataManager(dataroot,paramroot,args.train_ratio,)
    # train_dataset = dataManager.train_dataset()
    # test_dataset = dataManager.test_dataset()
    train_dataset = FullTeethDataset(dataroot,paramroot,'train.txt',True,args,2048)
    test_dataset = FullTeethDataset(dataroot,paramroot,'val.txt',False,args,2048)
    print(len(train_dataset))
    print(len(test_dataset))
    train_data_loader = data.DataLoader(train_dataset, num_workers=args.n_worker, batch_size=args.batch_size,
                                        shuffle=True, pin_memory=True, drop_last=True)
    test_data_loader = data.DataLoader(test_dataset, num_workers=args.n_worker, batch_size=args.batch_size,
                                       shuffle=False, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = TADPM(args).to(device)
    net = nn.DataParallel(net)
    if args.checkpoint != '':
        print('...loading...')
        checkpoint = torch.load(args.checkpoint)
        net.load_state_dict(checkpoint['model'],strict=True)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    
    # ========== Optimizer ==========
    # optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epoch)
    if args.optim.lower() == 'adamw':
        optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # if args.lr_milestones.lower() != 'none':
    #     ms = args.lr_milestones
    #     ms = ms.split()
    #     ms = [int(j) for j in ms]
    #     scheduler = MultiStepLR(optimizer, milestones=ms, gamma=0.1)
    # else:
    #     scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(args.num_warmup_steps),
    #                                                 num_training_steps=args.n_epoch + 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epoch)
    checkpoint_names = []
    checkpoint_path = os.path.join(args.saveroot, args.name)

    os.makedirs(checkpoint_path, exist_ok=True)

    train.step = 0
    test.best_loss = 99999

    # ========== Start Training ==========
    for epoch in range(args.n_epoch):
        print('epoch', epoch)
        train(net, optimizer, args.name, scheduler, train_data_loader, epoch, args)
        print('train finished')
        test(net, args.name, optimizer, scheduler, test_data_loader, epoch, args)
        print('test finished')
        