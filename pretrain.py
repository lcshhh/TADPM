import torch
import os
import sys
from torch.autograd import Variable
import argparse
import copy
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
import numpy as np
import torch.nn as nn
import torch.utils.data as data
from dataset.dataset import teethDataset,TeethDataManager
from models.mesh_mae import Mesh_mae
import torch.backends.cudnn as cudnn
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule, get_cosine_schedule_with_warmup
import time


def  train(net, optim, scheduler, names, train_dataset, epoch, args):
    net.train()
    running_loss = 0

    n_samples = 0

    for it, (feats_patch, center_patch,coordinate_patch, face_patch,  np_Fs, label, mesh_paths) in enumerate(
            train_dataset):
        optim.zero_grad()
        faces = face_patch.to(torch.float32).cuda()
        feats = feats_patch.to(torch.float32).cuda()
        centers = center_patch.to(torch.float32).cuda()
        Fs = np_Fs.cuda()
        cordinates = coordinate_patch.to(torch.float32).cuda()
        n_samples += faces.shape[0]
        loss = net(faces, feats, centers, Fs, cordinates).mean()
        loss.backward()
        optim.step()
        running_loss += loss.item() * faces.size(0)

    epoch_loss = running_loss / n_samples


    scheduler.step()
    # if (epoch+1) % 5 == 0 and train.best_loss > epoch_loss:
    #     train.best_loss = epoch_loss
    #     train.best_epoch = epoch
    #     best_model_wts = copy.deepcopy(net.state_dict())
    #     torch.save(best_model_wts, os.path.join(args.saveroot, names, f'loss-{epoch_loss:.4f}-{epoch}.pkl'))
    print('epoch ({:}): {:} Train Loss: {:.4f}'.format(names, epoch, epoch_loss))

def test(net, optim, scheduler, names, test_dataset, epoch, args):
    net.eval()
    running_loss = 0

    n_samples = 0

    for it, (feats_patch, center_patch,coordinate_patch, face_patch,  np_Fs, label, mesh_paths) in enumerate(
            test_dataset):
        faces = face_patch.to(torch.float32).cuda()
        feats = feats_patch.to(torch.float32).cuda()
        centers = center_patch.to(torch.float32).cuda()
        Fs = np_Fs.cuda()
        cordinates = coordinate_patch.to(torch.float32).cuda()
        n_samples += faces.shape[0]
        loss = net(faces, feats, centers, Fs, cordinates).mean()
        running_loss += loss.item() * faces.size(0)

    epoch_loss = running_loss / n_samples


    if (epoch+1) % 5 == 0 and test.best_loss > epoch_loss:
        test.best_loss = epoch_loss
        test.best_epoch = epoch
        best_model_wts = copy.deepcopy(net.state_dict())
        torch.save({'model':best_model_wts, 'optim':optim.state_dict()}, os.path.join(args.saveroot, names, f'loss-{epoch_loss:.4f}-{epoch}.pkl'))
        # torch.save(optim.state_dict(),os.path.join(args.saveroot, names, f'loss-{epoch_loss:.4f}-{epoch}-optim.pkl'))
    print('epoch ({:}): {:} Test Loss: {:.4f}'.format(names, epoch, epoch_loss))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'test'])
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--lr_milestones', type=str, default=None)
    parser.add_argument('--num_warmup_steps', type=str, default=None)
    parser.add_argument('--depth', type=int, required=True)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--n_dropout', type=int, default=1)
    parser.add_argument('--encoder_depth', type=int, default=6)
    parser.add_argument('--decoder_depth', type=int, default=6)
    parser.add_argument('--decoder_dim', type=int, default=512)
    parser.add_argument('--decoder_num_heads', type=int, default=6)
    parser.add_argument('--dim', type=int, default=384)
    parser.add_argument('--weight', type=float, default=0.2)
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--heads', type=int, required=True)
    parser.add_argument('--patch_size', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epoch', type=int, default=500)
    parser.add_argument('--max_epoch', type=int, default=300)
    parser.add_argument('--dataroot', type=str, required=True)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--n_worker', type=int, default=8)
    parser.add_argument('--augment_scale', action='store_true')
    parser.add_argument('--augment_orient', action='store_true')
    parser.add_argument('--augment_deformation', action='store_true')
    parser.add_argument('--channels', type=int, default=10)
    parser.add_argument('--mask_ratio', type=float, default=0.25)
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--saveroot', type=str, required=True)

    args = parser.parse_args()
    mode = args.mode
    dataroot = args.dataroot

    # ========== Dataset ==========
    augments = []
    if args.augment_scale:
        augments.append('scale')
    if args.augment_orient:
        augments.append('orient')

    dataManager = TeethDataManager(dataroot,augment=augments)
    train_dataset = dataManager.train_dataset()
    test_dataset = dataManager.test_dataset()
    print(len(train_dataset))
    print(len(test_dataset))
    train_data_loader = data.DataLoader(train_dataset, num_workers=args.n_worker, batch_size=args.batch_size,
                                        shuffle=True, pin_memory=True)
    test_data_loader = data.DataLoader(test_dataset, num_workers=args.n_worker, batch_size=args.batch_size,
                                           shuffle=True, pin_memory=True)

    # ========== Network ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    net = Mesh_mae(masking_ratio=args.mask_ratio,
                   channels=args.channels,
                   num_heads=args.heads,
                   encoder_depth=args.encoder_depth,
                   embed_dim=args.dim,
                   decoder_num_heads=args.decoder_num_heads,
                   decoder_depth=args.decoder_depth,
                   decoder_embed_dim=args.decoder_dim,
                   patch_size=args.patch_size,
                   weight=args.weight
                   ).to(device)
    print("using data parallel")
    net = torch.nn.DataParallel(net)
    if args.checkpoint != '':
        print(args.checkpoint)
        net.load_state_dict(torch.load(args.checkpoint)['model'], strict=True)
        cudnn.benchmark = True

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
                                                    num_training_steps=args.max_epoch + 1)

    print(scheduler)

    # ========== MISC ==========

    checkpoint_names = []
    checkpoint_path = os.path.join(args.saveroot, args.name)
    os.makedirs(checkpoint_path, exist_ok=True)


    test.best_loss = 999
    test.best_epoch = 0
    # ========== Start Training ==========

    if args.mode == 'train':
        for epoch in range(args.n_epoch):
            print('epoch', epoch)
            train(net, optim, scheduler, args.name, train_data_loader, epoch, args)
            print('train finished')
            test(net, optim, scheduler, args.name, train_data_loader, epoch, args)
            print('test finished')

