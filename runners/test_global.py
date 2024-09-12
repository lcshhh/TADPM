import torch
import torch.nn as nn
from utils import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from pytorch3d.loss import chamfer_distance
from utils.pointcloud import write_pointcloud
import numpy as np
from torchvision import transforms

def test_global(args, config, logger):
    # build dataset
    config.model.args = args
    (_, test_dataloader)= builder.dataset_builder(args, config.dataset.test)
    # build model
    base_model = builder.model_builder(config.model)
    

    # resume ckpts
    # base_model.load_model_from_ckpt(args.ckpts)
    builder.load_model_from_ckpt(base_model,args.ckpts)
 
    logger.info('Using Data parallel ...' , logger = logger)
    device = torch.device('cuda') 
    base_model = nn.DataParallel(base_model).to(device)
    # optimizer & scheduler
    logger.info(f"[TEST] Start test")
    base_model.eval()  # set model to eval mode
    losses = AverageMeter(['loss','rec','kl'])
    with torch.no_grad():
        for idx, (index,point,centers,axis,masks) in enumerate(test_dataloader):
            point = point.cuda().float()
            masks = masks.cuda()
            mu, log_var, reconstructed = base_model(point,axis)
            kl_loss = 0.001*(-0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp()))
            rec_loss = (torch.stack([chamfer_distance(point[:,i],reconstructed[:,i],batch_reduction=None)[0] for i in range(32)],dim=1) * masks).sum()
            clouds = []
            for i in range(32):
                if masks[0][i] > 0:
                    clouds.append(reconstructed[0,i])
            clouds = torch.cat(clouds,dim=0)
            write_pointcloud(clouds.cpu().numpy(),f'/data3/leics/dataset/test.ply')
            loss = kl_loss + rec_loss
            losses.update([loss.item(),kl_loss.item(),rec_loss.item()])

        logger.info('[TEST] Loss rec_loss kl_loss = %s' % (['%.4f' % l for l in losses.avg()]))


    # Add testing results to TensorBoard