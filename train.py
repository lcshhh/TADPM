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
    

# def chamfer_loss(before_points,after_points,centroids,outputs,mask=None):
#     '''
#     outputs: [bs,teeth_num,6]
#     before_points: [bs,teeth_num,point_num,3]
#     mask: [bs,32]
#     '''
#     bs = before_points.shape[0]
#     np = before_points.shape[2]
#     predicted_points = transform_vertices(before_points, centroids, outputs)
#     after_points = rearrange(after_points,'b n pn c -> (b n) pn c')
#     mask = repeat(mask,'b n -> (b n) np c', np=np, c=3)
#     bool_mask = (mask < 0.5)
#     predicted_points = predicted_points.masked_fill(bool_mask,value=0)
#     after_points = after_points.masked_fill(bool_mask,value=0)
#     loss, _ = chamfer_distance(after_points, predicted_points, point_reduction="mean", norm=1)
#     return loss

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

def chamfer_loss2(index,outputs):
    bs = index.shape[0]
    loss = torch.FloatTensor([0.]).cuda()
    for j in range(bs):
        trans_matrix = se3_exp_map(outputs[j]).transpose(1,2)
        for i in range(32):
            before_path = os.path.join('/data3/leics/dataset/mesh/single_before',f'{index[j]}_{i}.obj')
            after_path = os.path.join('/data3/leics/dataset/mesh/single_after',f'{index[j]}_{i}.obj')
            if os.path.exists(before_path) and os.path.exists(after_path):
                before_mesh = trimesh.load_mesh(before_path)
                after_mesh = trimesh.load_mesh(after_path)
                before_points = torch.FloatTensor(before_mesh.vertices).cuda()
                after_points = torch.FloatTensor(after_mesh.vertices).cuda()
                predicted_points = Transform3d(matrix=trans_matrix.transpose(1,2)[i]).transform_points(before_points)
                cd_loss,_ = chamfer_distance(after_points.unsqueeze(0), predicted_points.unsqueeze(0), point_reduction="sum", norm=1)
                loss += cd_loss
    return loss/bs
    
            

def add_loss(before_points,after_points,outputs,masks=None):
    bs = before_points.shape[0]
    loss = torch.FloatTensor([0]).cuda()
    outputs = rearrange(outputs,'b n c -> (b n) c')
    trans_matrix = se3_exp_map(outputs).transpose(1,2)
    before_points = rearrange(before_points,'b n p c -> (b n) p c')
    after_points = rearrange(after_points,'b n p c -> (b n) p c')
    riged_tar = Transform3d(matrix=trans_matrix.transpose(1,2)).transform_points(before_points)
    criterion = nn.MSELoss(reduction='none')
    # if masks is not None:
    #     loss,_ = chamfer_distance(after_points, riged_tar, point_reduction="sum", batch_reduction=None, norm=1)
    #     loss = (loss * masks.view(loss.shape)).mean()
    # else:
    #     loss,_ = chamfer_distance(after_points, riged_tar, point_reduction="sum", norm=1)
    loss = criterion(after_points,riged_tar).sum(dim=(1,2))
    loss = (loss * masks.view(loss.shape)).sum()
    return loss/bs

# def centroid_loss(before_centroids,after_centroids,outputs,mask=None):
#     '''
#     before_centroids:[bs, 32, 3]
#     after_centroids:[bs, 32, 3]
#     outputs:[bs, 32, 6]
#     '''
#     bs = before_centroids.shape[0]
#     n = before_centroids.shape[1]
#     predicted_centrodis = before_centroids + outputs[:,:,:3]
#     criterion = nn.MSELoss(reduction='none')
#     loss = criterion(after_centroids,predicted_centrodis).sum(dim=-1)
#     loss = (loss * mask.float()).sum()
#     non_zero_elements = mask.sum()
#     mse_loss_val = loss / non_zero_elements
#     # predicted_points = transform_vertices(before_points, centroids, outputs)
#     # after_points = rearrange(after_points,'b n pn c -> (b n) pn c')
#     # mask = repeat(mask,'b n -> (b n) np c', np=np, c=3)
#     # predicted_points = predicted_points.masked_fill(mask,value=0)
#     # after_points = after_points.masked_fill(mask,value=0)
#     # loss, _ = chamfer_distance(after_points, predicted_points, point_reduction="sum", norm=2)
#     return mse_loss_val

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

def train(net, optim, names, scheduler, train_dataset, epoch, args):
    net.train()
    running_loss = 0
    n_samples = 0

    for it, (feats_patch, center_patch, coordinate_patch, face_patch, np_Fs, index, before_points, after_points, centroid,after_centroid,dofs,masks) in enumerate(
            train_dataset):
        optim.zero_grad()
        faces = face_patch.to(torch.float32).cuda()
        feats = feats_patch.to(torch.float32).cuda()
        centers = center_patch.to(torch.float32).cuda()
        Fs = np_Fs.cuda()
        cordinates = coordinate_patch.cuda()
        centroid = centroid.to(torch.float32).cuda()
        after_centroid = after_centroid.to(torch.float32).cuda()
        dofs = dofs.to(torch.float32).cuda()
        masks = masks.cuda()
        n_samples += faces.shape[0]
        before_points = before_points.to(torch.float32).cuda()
        after_points = after_points.to(torch.float32).cuda()
        if args.use_mlp:
            outputs = net(faces, feats, centers, Fs, cordinates, centroid, before_points).to(torch.float32).cuda()
        else:
            outputs = net(faces, feats, centers, Fs, cordinates, centroid, before_points, dofs).to(torch.float32).cuda()
        # loss1 = chamfer_loss(before_points,after_points,outputs/10, masks)
        loss1 = chamfer_loss2(index,outputs/10)
        criterion = nn.MSELoss(reduction='none')
        loss2 = criterion(dofs,outputs/10).sum(dim=-1)
        loss2 = 30*(loss2 * masks).sum()
        # loss2 = centroid_loss(centroid, after_centroid, outputs)
        # loss = chamfer_loss(before_points,after_points,outputs).mean()
        print('loss1:',loss1)
        print('loss2:',loss2)
        loss = loss2 + loss1
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

    for it, (feats_patch, center_patch, coordinate_patch, face_patch, np_Fs, index, before_points, after_points, centroid,after_centroid, dofs,masks) in enumerate(
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
        dofs = dofs.to(torch.float32).cuda()
        masks = masks.cuda()
        # trans_matrix = trans_matrix.to(torch.float32).cuda()
        n_samples += faces.shape[0]
        # trans_6dof = trans_6dof.to(torch.float32).cuda()
        with torch.no_grad():
            if args.use_mlp:
                outputs = net(faces, feats, centers, Fs, cordinates, centroid, before_points).to(torch.float32).cuda()
            else:
                outputs = net(faces, feats, centers, Fs, cordinates, centroid, before_points, dofs).to(torch.float32).cuda()
            # loss = 512*chamfer_loss(before_points,after_points,centroid, outputs, masks)
            loss = chamfer_loss(before_points,after_points,outputs/10, masks)
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
    train_dataset = FullTeethDataset(dataroot,paramroot,'train.txt',True)
    test_dataset = FullTeethDataset(dataroot,paramroot,'val.txt',False)
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
        checkpoint = torch.load(args.checkpoint)
        net.load_state_dict(checkpoint['model'],strict=True)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    
    # ========== Optimizer ==========
    # optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epoch)
    if args.optim.lower() == 'adamw':
        optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_milestones.lower() != 'none':
        ms = args.lr_milestones
        ms = ms.split()
        ms = [int(j) for j in ms]
        scheduler = MultiStepLR(optimizer, milestones=ms, gamma=0.1)
    else:
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(args.num_warmup_steps),
                                                    num_training_steps=args.n_epoch + 1)
    checkpoint_names = []
    checkpoint_path = os.path.join(args.saveroot, args.name)

    os.makedirs(checkpoint_path, exist_ok=True)

    train.step = 0
    test.best_loss = 1000

    # ========== Start Training ==========
    for epoch in range(args.n_epoch):
        print('epoch', epoch)
        train(net, optimizer, args.name, scheduler, train_data_loader, epoch, args)
        print('train finished')
        test(net, args.name, optimizer, scheduler, test_data_loader, epoch, args)
        print('test finished')