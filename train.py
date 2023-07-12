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
import matplotlib
import vedo
from pytorch3d.transforms import Transform3d
from model.dataset import TeethRegressorDataManager
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import trimesh
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance
from pytorch3d.transforms import se3_exp_map

colors = [[0, 114, 189],
          [217, 83, 26],
          [238, 177, 32],
          [126, 47, 142],
          [117, 142, 48],
          [76, 190, 238],
          [162, 19, 48],
          [240, 166, 202],
          [50, 114, 189],
          [217, 23, 26],
          [158, 177, 32],
          [126, 47, 92],
          [117, 92, 48],
          [76, 140, 238],
          [112, 19, 48],
          [190, 166, 202],
          [20, 104, 179],
          [237, 73, 6],
          [258, 167, 12],
          [146, 37, 122],
          [137, 132, 28],
          [96, 180, 218],
          [182, 9, 28],
          [250, 156, 192],
          [70, 104, 169],
          [137, 13, 6],
          [178, 167, 12],
          [146, 17, 72],
          [137, 82, 28],
          [96, 130, 218],
          [132, 9, 28],
          [210, 156, 182],
          [222, 156, 192],
          [40, 104, 179],
          [197, 23, 23],
          [158, 197, 2],
          [106, 47, 52],
          [117, 52, 98],
          [76, 110, 238],
          [112, 99, 8],
          [110, 196, 12]]


def seed_torch(seed=12):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def plot_embedding(data, label, title):

    print(len(label))
    cmap = cm.rainbow(np.linspace(0, 1, len(label)))

    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()

    for i in range(data.shape[0]):
        c = cm.rainbow(int(255 / 40 * label[i]))
        # plt.text(data[i, 0], data[i, 1], str(label[i]), color=plt.cm.Set1(label[i] / 40),  fontdict={'weight': 'bold', 'size': 7})
        plt.scatter(data[i, 0], data[i, 1], color=c, alpha=0.5)
    plt.xticks()
    plt.yticks()
    plt.title(title, fontsize=9)

    return fig

def cos_sim(output,trans):
    # n = output.shape[0]
    # predicted_matrix= se3_exp_map(predicted_6dof.reshape(-1,6)).reshape(-1,16)
    # gt_matrix = se3_exp_map(gt_6dof.reshape(-1,6)).reshape(-1,16)
    # cos_sim = (F.cosine_similarity(predicted_matrix, gt_matrix)+1)/2
    # cos_sim = 1-torch.mean(cos_sim)
    # return cos_sim
    # output = 
    pass

def matrix_similiar(output, target):
    # target_points = target_points.float()
    # output = output.squeeze(1).squeeze(1)
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
        # N,P,C = before_points.shape
        # before_points_pre = torch.cat([before_points[i],torch.ones(N,P,1).cuda()],dim=-1).permute(0,2,1)
        # riged_tar = torch.bmm(trans_matrix,before_points_pre).permute(0,2,1)
        # riged_tar = riged_tar[:,:,:3]
        tmp,_ = chamfer_distance(after_points[i], riged_tar, point_reduction="sum", norm=1)
        loss += tmp
    return loss/bs

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
    # pdist = nn.PairwiseDistance(p=2)
    bs = centroid.shape[0]
    loss = torch.FloatTensor([0]).cuda()
    # criterion = nn.L1Loss(reduction='mean')
    for i in range(bs):
        trans_matrix = se3_exp_map(outputs[i]).transpose(2,1) # [16,4,4]
        predicted_centroid = Transform3d(matrix=trans_matrix.transpose(2,1)).transform_points(centroid[i].unsqueeze(1)).squeeze(1) # [16,3]
        pre_dis = torch.sqrt(torch.sum(torch.square(predicted_centroid[:-1] - predicted_centroid[1:]),dim=-1))
        after_dis = torch.sqrt(torch.sum(torch.square(after_centroid[i][:-1] - after_centroid[i][1:]),dim=-1))
        loss += torch.abs(pre_dis - after_dis).sum()
    # for i in range(bs):
    #     trans_matrix = se3_exp_map(outputs[i]).transpose(2,1) # [16,4,4]
    #     predicted_centroid = Transform3d(matrix=trans_matrix.transpose(2,1)).transform_points(centroid[i].unsqueeze(1)).squeeze(1) # [16,3]
    #     # pre_dis = torch.sqrt(torch.sum(torch.square(predicted_centroid[:-1] - predicted_centroid[1:]),dim=-1))
    #     # after_dis = torch.sqrt(torch.sum(torch.square(after_centroid[i][:-1] - after_centroid[i][1:]),dim=-1))
    #     loss += torch.abs(after_centroid - predicted_centroid).sum()
    return loss/bs

def cal_centroid_2(centroid,after_centroid,outputs,index):
    '''
    centroid [bs,16,3]
    '''
    # pdist = nn.PairwiseDistance(p=2)
    bs = centroid.shape[0]
    loss = torch.FloatTensor([0]).cuda()
    # criterion = nn.L1Loss(reduction='mean')
    for i in range(bs):
        trans_matrix = se3_exp_map(outputs[i]).transpose(2,1) # [16,4,4]
        predicted_centroid = Transform3d(matrix=trans_matrix.transpose(2,1)).transform_points(centroid[i].unsqueeze(1)).squeeze(1) # [16,3]
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
    # for i in range(bs):
    #     trans_matrix = se3_exp_map(outputs[i]).transpose(2,1) # [16,4,4]
    #     predicted_centroid = Transform3d(matrix=trans_matrix.transpose(2,1)).transform_points(centroid[i].unsqueeze(1)).squeeze(1) # [16,3]
    #     # pre_dis = torch.sqrt(torch.sum(torch.square(predicted_centroid[:-1] - predicted_centroid[1:]),dim=-1))
    #     # after_dis = torch.sqrt(torch.sum(torch.square(after_centroid[i][:-1] - after_centroid[i][1:]),dim=-1))
    #     loss += torch.abs(after_centroid - predicted_centroid).sum()
    return loss/bs
        
        





def train(net, optim, scheduler, names, criterion, train_dataset, epoch, args, autoencoder=None):
    net.train()
    if autoencoder!=None:
        autoencoder.eval()
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
        # mask = mask.cuda()
        before_points_centered = before_points_centered.to(torch.float32).cuda()
        # labels = labels.cuda()
        n_samples += faces.shape[0]
        trans_6dof = trans_6dof.to(torch.float32).cuda()
        trans_matrix = trans_matrix.to(torch.float32).cuda()
        before_points = before_points.to(torch.float32).cuda()
        after_points = after_points.to(torch.float32).cuda()
        if args.use_ae:
            embedding = net.get_feature(faces, feats, centers, Fs, cordinates, centroid, before_points, trans_6dof)
            rec_embedding = autoencoder.get_encode(embedding)
            outputs = net.process_feature(rec_embedding,trans_6dof).to(torch.float32)
        else:
            outputs = net(faces, feats, centers, Fs, cordinates, centroid, before_points, trans_6dof).to(torch.float32)
        loss = 0.03*chamfer_loss(before_points,after_points,outputs)
        loss2 = cal_centroid_2(centroid,after_centroid,outputs,0) + cal_centroid_2(centroid,after_centroid,outputs,1) + cal_centroid_2(centroid,after_centroid,outputs,2)
        # loss += loss2
        # loss3 = 100*criterion(outputs, trans_6dof)
        # print(loss3)
        # loss += loss3
        # loss = loss3
        # _, preds = torch.max(outputs, 1)
        # running_corrects += torch.sum(preds == labels.data)
        loss.backward()
        optim.step()
        running_loss += loss.item() * faces.size(0)

    scheduler.step()
    epoch_loss = running_loss / n_samples
    epoch_acc = running_corrects / n_samples
    print('epoch ({:}): {:} Train Loss: {:.4f} Acc: {:.4f}'.format(names, epoch, epoch_loss, epoch_acc))
    message = 'epoch ({:}): {:} Train Loss: {:.4f} Acc: {:.4f}\n'.format(names, epoch, epoch_loss, epoch_acc)
    with open(os.path.join('/data/lcs/new_checkpoints', names, 'log.txt'), 'a') as f:
        f.write(message)

def test(net, names, criterion, test_dataset, epoch, args, autoencoder=None):

    # for net_ in net:
    #     net_.eval()
    #     voted.append(ClassificationMajorityVoting(args.n_classes))
    net.eval()
    if autoencoder!=None:
        autoencoder.eval()

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
        # labels = labels.cuda()
        centroid = centroid.to(torch.float32).cuda()
        after_centroid = after_centroid.to(torch.float32).cuda()
        # mask = mask.cuda()
        trans_matrix = trans_matrix.to(torch.float32).cuda()
        n_samples += faces.shape[0]
        trans_6dof = trans_6dof.to(torch.float32).cuda()
        with torch.no_grad():
            if args.use_ae:
                embedding = net.get_feature(faces, feats, centers, Fs, cordinates, centroid, before_points, trans_6dof)
                rec_embedding = autoencoder.get_encode(embedding)
                outputs = net.process_feature(rec_embedding,trans_6dof).to(torch.float32)
            else:
                outputs = net(faces, feats, centers, Fs, cordinates, centroid, before_points, trans_6dof).to(torch.float32)
            loss = chamfer_loss(before_points,after_points,outputs)
            loss2 = cal_centroid(centroid,after_centroid,outputs)
            # loss = cos_similiar(outputs,trans_matrix)
            running_loss += loss.item() * faces.size(0)
            running_loss2 += loss2.item() * faces.size(0)

    epoch_loss = running_loss / n_samples
    centroid_loss = running_loss2 / n_samples
    if test.best_loss > epoch_loss:
        test.best_loss = epoch_loss
        best_model_wts = copy.deepcopy(net.state_dict())
        #torch.save(best_model_wts, os.path.join('checkpoints', names, f'acc-{epoch_acc:.4f}-{epoch:.4f}.pkl'))
        torch.save(best_model_wts, os.path.join('/data/lcs/new_checkpoints', names, 'best_acc.pkl'))
    if (epoch+1)%50 == 0:
        best_model_wts = copy.deepcopy(net.state_dict())
        torch.save(best_model_wts, os.path.join('/data/lcs/new_checkpoints', names, f'acc-{epoch_loss:.4f}-{epoch}.pkl'))
        # torch.save(best_model_wts, os.path.join('/data/lcs/checkpoints', names, 'best_acc.pkl'))

    message = 'epoch ({:}): {:} test Loss: {:.4f} centroid dis: {:.4f} '.format(names, epoch, epoch_loss, centroid_loss
                                                                                       )
    with open(os.path.join('/data/lcs/new_checkpoints', names, 'log.txt'), 'a') as f:
        f.write(message)
    print(message)

def transform_teeth(index,trans_6dof,lower=False):
    # if lower:
    #     dataroot = '/data/lcs/first_lower/single_centered'
    # else:
    #     dataroot = '/data/lcs/first_upper/single_centered_normed'
    dataroot = '/data/lcs/batch2_merged_final/transform_single'
    trans_matrix = se3_exp_map(trans_6dof).transpose(2,1)
    meshes_upper = []
    # meshes_lower = []
    before_meshes = []
    ran = range(1,33)
    for i in ran:
        mesh_path = os.path.join(dataroot,f'{index}_{i}.obj')
        if os.path.exists(mesh_path):
            mesh = vedo.Mesh(mesh_path)
            mesh_before = mesh.clone(True)
            before_meshes.append(mesh_before)
            mesh.apply_transform(trans_matrix[i-1])
            meshes_upper.append(mesh)
            # else:
            #     meshes_lower.append(mesh)
    mesh_upper = vedo.merge(meshes_upper)
    before = vedo.merge(before_meshes)
    # mesh_lower = vedo.merge(meshes_lower)
    outputroot = '/data/lcs/batch2_merged_final/test_output'
    os.makedirs(outputroot,exist_ok=True)
    vedo.write(mesh_upper,os.path.join(outputroot,f'{index}_after.obj'))
    vedo.write(before,os.path.join(outputroot,f'{index}_before.obj'))
    # if lower:
    #     vedo.write(mesh_upper,os.path.join(outputroot,f'{index}_lower.obj'))
    # else:
    #     vedo.write(mesh_upper,os.path.join(outputroot,f'{index}_upper.obj'))
    # vedo.write(mesh_lower,os.path.join(outputroot,f'{index}_lower.obj'))

def pure_transform_teeth(index,trans_6dof,lower=False):
    dataroot = '/data/lcs/batch2_merged_final/transform_single'
    trans_matrix = se3_exp_map(trans_6dof).transpose(2,1)
    meshes_upper = []
    before_meshes = []
    # meshes_lower = []
    # outputroot2 = '/data/lcs/upper_jaw/before_single_5'
    # os.makedirs(outputroot2,exist_ok=True)
    ran = range(1,33)
    for i in ran:
        mesh_path = os.path.join(dataroot,f'{index}_{i}.obj')
        if os.path.exists(mesh_path):
            mesh = vedo.Mesh(mesh_path)
            mesh_before = mesh.clone(True)
            mesh.apply_transform(trans_matrix[i-1])
            meshes_upper.append(mesh)
            # vedo.write(mesh,os.path.join(outputroot2,f'{index}-{i}-before.obj'))
            before_meshes.append(mesh_before)
            # else:
            #     meshes_lower.append(mesh)
    mesh_upper = vedo.merge(meshes_upper)
    mesh_before = vedo.merge(before_meshes)
    # mesh_lower = vedo.merge(meshes_lower)
    outputroot = '/data/lcs/batch2_merged_final/test_output'
    os.makedirs(outputroot,exist_ok=True)
    vedo.write(mesh_upper,os.path.join(outputroot,f'{index}_after.obj'))
    vedo.write(mesh_before,os.path.join(outputroot,f'{index}_before.obj'))

def test_and_save(net, names, criterion, test_dataset, epoch, args):

    # for net_ in net:
    #     net_.eval()
    #     voted.append(ClassificationMajorityVoting(args.n_classes))
    net.eval()

    running_loss = 0
    running_corrects = 0
    n_samples = 0

    for i, (feats_patch, center_patch, coordinate_patch, face_patch, np_Fs, trans_6dof, trans_matrix,index,before_points,after_points,centroid,after_centroid,before_points_centered) in enumerate(
            test_dataset):
        faces = face_patch.cuda()
        feats = feats_patch.to(torch.float32).cuda()
        centers = center_patch.to(torch.float32).cuda()
        Fs = np_Fs.cuda()
        cordinates = coordinate_patch.to(torch.float32).cuda()
        centroid = centroid.to(torch.float32).cuda()
        # labels = labels.cuda()
        before_points = before_points.to(torch.float32).cuda()
        after_points = after_points.to(torch.float32).cuda()
        trans_matrix = trans_matrix.to(torch.float32).cuda()
        n_samples += faces.shape[0]
        batch_size = faces.shape[0]
        trans_6dof = trans_6dof.to(torch.float32).cuda()
        with torch.no_grad():
            outputs = net(faces, feats, centers, Fs, cordinates, centroid, before_points, trans_6dof).detach()
            # loss = chamfer_loss(before_points,after_points,outputs)
            # running_loss += loss.item() * faces.size(0)
            for i in range(index.shape[0]):
                transform_teeth(index[i],outputs[i])
    # epoch_loss = running_loss / n_samples
    # print(epoch_loss)

def pure_test_and_save(net, criterion, test_dataset, epoch, args):

    net.eval()

    for i, (feats_patch, center_patch, coordinate_patch, face_patch, np_Fs,index,centroid,before_points) in enumerate(
            test_dataset):
        faces = face_patch.cuda()
        feats = feats_patch.to(torch.float32).cuda()
        centers = center_patch.to(torch.float32).cuda()
        Fs = np_Fs.cuda()
        cordinates = coordinate_patch.to(torch.float32).cuda()
        centroid = centroid.to(torch.float32).cuda()
        before_points = before_points.to(torch.float32).cuda()
        # labels = labels.cuda()
        with torch.no_grad():
            outputs = net(faces, feats, centers, Fs, cordinates, centroid, before_points).detach()
            for i in range(len(index)):
                pure_transform_teeth(index[i],outputs[i])
            


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
    parser.add_argument('--n_classes', type=int)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--n_worker', type=int, default=8)
    parser.add_argument('--paramroot',type=str, required=True)
    parser.add_argument('--global_checkpoint',type=str,default='none')
    parser.add_argument('--total_checkpoint',type=str,default='none')
    parser.add_argument('--use_mlp', action='store_true')
    parser.add_argument('--use_pointnet', action='store_true')
    parser.add_argument('--use_ae', action='store_true')
    parser.add_argument('--pure_test', action='store_true')
    parser.add_argument('--lower', action='store_true')
    parser.add_argument('--ae_checkpoint',type=str,default='none')
    # parser.add_argument('--augment_scale', action='store_true')
    # parser.add_argument('--augment_orient', action='store_true')
    # parser.add_argument('--augment_deformation', action='store_true')
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
    # if args.augment_scale:
    #     augments.append('scale')
    # if args.augment_orient:
    #     augments.append('orient')
    # if args.augment_deformation:
    #     augments.append('deformation')
    dataManager = TeethRegressorDataManager(dataroot,paramroot,args.train_ratio,)
    train_dataset = dataManager.train_dataset()
    # test_dataset = ClassificationDataset(dataroot, train=False)
    test_dataset = dataManager.test_dataset()
    print(len(train_dataset))
    print(len(test_dataset))
    train_data_loader = data.DataLoader(train_dataset, num_workers=args.n_worker, batch_size=args.batch_size,
                                        shuffle=True, pin_memory=True)
    test_data_loader = data.DataLoader(test_dataset, num_workers=args.n_worker, batch_size=args.batch_size,
                                       shuffle=False, pin_memory=True)
    if args.pure_test:
        pure_test_dataset = teethTestDataset('/data/lcs/batch2_merged_final/before_remesh')
        print(len(pure_test_dataset))
        pure_test_data_loader = data.DataLoader(pure_test_dataset, num_workers=args.n_worker, batch_size=args.batch_size,
                                       shuffle=False, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = teethArranger(args).to(device)
    auto_encoder = None
    if args.use_ae:     
        auto_encoder = AutoEncoder(1840).to(device)
        checkpoint1 = torch.load(args.ae_checkpoint)
        auto_encoder.load_state_dict(checkpoint1)
    # if args.mode != 'train':
    if args.total_checkpoint != 'none':
        checkpoint = torch.load(args.total_checkpoint)
        net.load_state_dict(checkpoint,strict=True)

    # ========== Optimizer ==========
    if args.optim.lower() == 'adamw':
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
    criterion = nn.MSELoss()
    checkpoint_names = []
    checkpoint_path = os.path.join('/data/lcs/new_checkpoints', args.name)

    os.makedirs(checkpoint_path, exist_ok=True)

    # if args.checkpoint.lower() != 'none':
    #     net.load_state_dict(torch.load(args.checkpoint), strict=False)

    train.step = 0
    test.best_loss = 1000

    # ========== Start Training ==========

    if args.mode == 'train':
        for epoch in range(args.n_epoch):
            # train_data_loader.dataset.set_epoch(epoch)
            print('epoch', epoch)
            train(net, optim, scheduler, args.name, criterion, train_data_loader, epoch, args, autoencoder=auto_encoder)
            print('train finished')
            test(net, args.name, criterion, test_data_loader, epoch, args, autoencoder=auto_encoder)
            print('test finished')
    
    elif args.pure_test:
        pure_test_and_save(net, criterion, pure_test_data_loader, 0, args)



    else:

        test_and_save(net, args.name, criterion, test_data_loader, 0, args)
