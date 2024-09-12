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

def test_single(args, config, logger):
    # build dataset
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
    losses = AverageMeter(['loss'])
    with torch.no_grad():
        for idx, (point, label) in enumerate(test_dataloader):
            point = point.cuda().to(torch.float32)
            fine = base_model(point)
            loss = 1000*chamfer_distance(point,fine,point_reduction='mean')[0]
            losses.update([loss.item()])

        logger.info('[Test] loss = %s' % (['%.4f' % l for l in losses.avg()]))


    # Add testing results to TensorBoard