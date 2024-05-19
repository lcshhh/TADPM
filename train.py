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

def matrix_similiar(output, target):
    '''
    output: [bs,6]
    target: [bs,4,4]
    '''
    if output.ndim==3:
        output = output.squeeze(1)
    matrix = se3_exp_map(output).transpose(1,2)
    flattend_matrix = matrix.reshape(matrix.shape[0],-1)
    flattend_target = target.reshape(matrix.shape[0],-1)
    cos_sim = (F.cosine_similarity(flattend_matrix, flattend_target)+1)/2
    cos_sim = 1-torch.mean(cos_sim)
    return cos_sim

def cos_similiar(outputs,targets):
    '''
    outputs: [bs,32,6]
    targets: [bs,32,4,4]
    '''
    bs = outputs.shape[0]
    cos_sim = torch.zeros(bs)
    for i in range(bs):
        output = outputs[i]
        target = targets[i]
        cos_sim[i] = matrix_similiar(output,target)
    return cos_sim.mean()

def chamfer_loss(before_points,after_points,outputs):
    '''
    outputs: [bs,16,6]
    before_points: [bs,16,2048,3]
    '''
    bs = before_points.shape[0]
    loss = torch.FloatTensor([0]).cuda()
    for i in range(bs):
        trans_matrix = se3_exp_map(outputs[i]).transpose(1,2)      # [16,4,4]    
        riged_tar = Transform3d(matrix=trans_matrix.transpose(1,2)).transform_points(before_points[i])
        tmp,_ = chamfer_distance(after_points[i], riged_tar, point_reduction="sum", norm=1)
        loss += tmp
    return loss/bs

def dis_matrix(points):
    """
    points:[32,2048,3]
    """
    n = points.shape[0]
    distance_matrix = torch.zeros(n,n)
    for i in range(n):
        for j in range(n):
            tmp,_ = chamfer_distance(points[i].unsqueeze(0), points[j].unsqueeze(0), point_reduction="sum", norm=1)
            distance_matrix[i][j] = tmp
    return distance_matrix

def cal_distance(centroids):
    '''
    centroids: [16,3]
    '''
    # pdist = nn.PairwiseDistance(p=2)
    dis = torch.sqrt(torch.sum(torch.square(centroids[:-1] - centroids[1:]),dim=-1))
    return dis

def cal_centroid(centroid,after_centroid,outputs):
    '''
    centroid [bs,16,3]
    '''
    bs = centroid.shape[0]
    loss = torch.FloatTensor([0]).cuda()
    for i in range(bs):
        trans_matrix = se3_exp_map(outputs[i]).transpose(2,1) # [16,4,4]
        predicted_centroid = centroid[i] + outputs[i,:,:3]
        pre_dis = torch.sqrt(torch.sum(torch.square(predicted_centroid[:-1] - predicted_centroid[1:]),dim=-1))
        after_dis = torch.sqrt(torch.sum(torch.square(after_centroid[i][:-1] - after_centroid[i][1:]),dim=-1))
        loss += torch.abs(pre_dis - after_dis).sum()
    return loss/bs

def cal_centroid_2(centroid,after_centroid,outputs,index):
    '''
    centroid [bs,16,3]
    '''
    bs = centroid.shape[0]
    loss = torch.FloatTensor([0]).cuda()
    for i in range(bs):
        trans_matrix = se3_exp_map(outputs[i]).transpose(2,1) # [16,4,4]
        predicted_centroid = centroid[i] + outputs[i,:,:3]
        if index == 0:
            pre_dis = torch.sqrt(torch.sum(torch.square(predicted_centroid[:15] - predicted_centroid[1:16]),dim=-1))
            after_dis = torch.sqrt(torch.sum(torch.square(after_centroid[i][:15] - after_centroid[i][1:16]),dim=-1))
        elif index == 1:
            pre_dis = torch.sqrt(torch.sum(torch.square(predicted_centroid[16:-1] - predicted_centroid[17:]),dim=-1))
            after_dis = torch.sqrt(torch.sum(torch.square(after_centroid[i][16:-1] - after_centroid[i][17:]),dim=-1))
        else:
            pre_dis = torch.sqrt(torch.sum(torch.square(predicted_centroid[:16] - predicted_centroid[16:]),dim=-1))
            after_dis = torch.sqrt(torch.sum(torch.square(after_centroid[i][:16] - after_centroid[i][16:]),dim=-1))
        loss += torch.abs(pre_dis - after_dis).sum()
    return loss/bs

def get_centroid_matrix(centroid):
    '''
    centroid [32,3]
    '''
    n = centroid.shape[0]
    distance_matrix = torch.zeros(n,n)
    for i in range(n):
        for j in range(n):
            distance_matrix[i][j] = torch.sum(torch.abs(centroid[i]-centroid[j]))
    return distance_matrix


def cal_centroid_3(centroid,after_centroid,outputs):
    '''
    centroid [bs,32,3]
    '''
    bs = centroid.shape[0]
    loss = torch.FloatTensor([0]).cuda()
    for i in range(bs):
        predicted_centroid = centroid[i] + outputs[i,:,:3]
        matrix1 = get_centroid_matrix(predicted_centroid)
        matrix2 = get_centroid_matrix(after_centroid[i])
        loss += torch.abs(matrix1 - matrix2).sum()
    return loss/bs


def train(net, optim, scheduler, names, train_dataset, epoch, args):
    net.train()
    running_loss = 0
    running_corrects = 0
    n_samples = 0

    for it, (feats_patch, center_patch, coordinate_patch, face_patch, np_Fs, trans_6dof, trans_matrix, index, before_points, after_points, centroid,after_centroid,before_points_centered) in enumerate(
            train_dataset):
        optim.zero_grad()
        faces = face_patch.to(torch.float32).cuda()
        feats = feats_patch.to(torch.float32).cuda()
        centers = center_patch.to(torch.float32).cuda()
        Fs = np_Fs.cuda()
        cordinates = coordinate_patch.cuda()
        centroid = centroid.to(torch.float32).cuda()
        after_centroid = after_centroid.to(torch.float32).cuda()
        before_points_centered = before_points_centered.to(torch.float32).cuda()
        n_samples += faces.shape[0]
        trans_6dof = trans_6dof.to(torch.float32).cuda()
        trans_matrix = trans_matrix.to(torch.float32).cuda()
        before_points = before_points.to(torch.float32).cuda()
        after_points = after_points.to(torch.float32).cuda()
        outputs = net(faces, feats, centers, Fs, cordinates, centroid, before_points, trans_6dof).to(torch.float32)
        loss = 0.1*chamfer_loss(centroid,before_points,after_points,outputs)
        loss2 = cal_centroid_2(centroid,after_centroid,outputs,0) + cal_centroid_2(centroid,after_centroid,outputs,1) + cal_centroid_2(centroid,after_centroid,outputs,2)
        loss3 = cal_centroid_3(centroid,after_centroid,outputs)
        loss += loss2 * 0.5 + loss3 * 0.5
        loss.backward()
        optim.step()
        running_loss += loss.item() * faces.size(0)

    scheduler.step()
    epoch_loss = running_loss / n_samples
    epoch_acc = running_corrects / n_samples
    print('epoch ({:}): {:} Train Loss: {:.4f} Acc: {:.4f}'.format(names, epoch, epoch_loss, epoch_acc))
    message = 'epoch ({:}): {:} Train Loss: {:.4f} Acc: {:.4f}\n'.format(names, epoch, epoch_loss, epoch_acc)
    with open(os.path.join(args.save_dir , names, 'log.txt'), 'a') as f:
        f.write(message)

def test(net, names, test_dataset, epoch, args):
    net.eval()

    running_loss = 0
    running_loss2 = 0
    running_corrects = 0
    n_samples = 0

    for i, (feats_patch, center_patch, coordinate_patch, face_patch, np_Fs, trans_6dof, trans_matrix,index,before_points,after_points,centroid,after_centroid,before_points_centered) in enumerate(
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
        trans_matrix = trans_matrix.to(torch.float32).cuda()
        n_samples += faces.shape[0]
        trans_6dof = trans_6dof.to(torch.float32).cuda()
        with torch.no_grad():
            outputs = net(faces, feats, centers, Fs, cordinates, centroid, before_points, trans_6dof).to(torch.float32)
            loss = chamfer_loss(centroid,before_points,after_points,outputs)
            loss2 = cal_centroid(centroid,after_centroid,outputs)
            running_loss += loss.item() * faces.size(0)
            running_loss2 += loss2.item() * faces.size(0)

    epoch_loss = running_loss / n_samples
    centroid_loss = running_loss2 / n_samples
    if (epoch+1)%50 == 0:
        best_model_wts = copy.deepcopy(net.state_dict())
        torch.save(best_model_wts, os.path.join(args.save_dir, names, f'acc-{epoch_loss:.4f}-{epoch}.pkl'))

    message = 'epoch ({:}): {:} test Loss: {:.4f} centroid dis: {:.4f} '.format(names, epoch, epoch_loss, centroid_loss
                                                                                       )
    with open(os.path.join(args.save_dir, names, 'log.txt'), 'a') as f:
        f.write(message)
    print(message)


if __name__ == '__main__':
    seed_torch(seed=43)
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'test'])
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--encoder_checkpoint', type=str, default='none')
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
    parser.add_argument('--checkpoint',type=str,default='none')
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

    # ========== Load Model ==========
    if args.checkpoint != 'none':
        checkpoint = torch.load(args.checkpoint)
        net.load_state_dict(checkpoint,strict=True)
    

    # ========== Optimizer ==========
    optim = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_milestones.lower() != 'none':
        ms = args.lr_milestones
        ms = ms.split()
        ms = [int(j) for j in ms]
        scheduler = MultiStepLR(optim, milestones=ms, gamma=0.1)
    else:
        scheduler = get_cosine_schedule_with_warmup(optim, num_warmup_steps=int(args.num_warmup_steps),
                                                    num_training_steps=args.n_epoch)

    print(scheduler)
    checkpoint_names = []
    checkpoint_path = os.path.join(args.save_dir, args.name)

    os.makedirs(checkpoint_path, exist_ok=True)
    train.step = 0

    # ========== Start Training ==========
    for epoch in range(args.n_epoch):
        print('epoch', epoch)
        train(net, optim, scheduler, args.name, train_data_loader, epoch)
        print('train finished')
        test(net, args.name, test_data_loader, epoch)
        print('test finished')

