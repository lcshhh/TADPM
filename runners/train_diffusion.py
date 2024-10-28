import torch
import torch.nn as nn
from utils import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from pytorch3d.loss import chamfer_distance
from einops import rearrange

import numpy as np
from torchvision import transforms
def create_attn_mask(masks):
    masks = masks.float()
    attn_mask = torch.bmm(masks.unsqueeze(2),masks.unsqueeze(1))
    return attn_mask < 0.5

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

def rotation_matrix(vec1, vec2, masks=None):
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

def align_axis(normal1,normal2,gt_normal1,gt_normal2, masks=None):
    rot_matrix = rotation_matrix(gt_normal1.view(-1,3),normal1.view(-1,3),masks).transpose(2,1) #将z轴对齐

    after_axis = torch.bmm(gt_normal2.view(-1,3).unsqueeze(1),rot_matrix)
    rho = torch.cross(after_axis.squeeze(1), normal2.view(-1,3),dim=-1)
    eps = 1e-7
    theta = torch.acos(torch.clamp(torch.sum(normal2.view(-1,3)*after_axis.squeeze(1),dim=-1),min=-1+eps,max=1-eps)).cuda()
    theta = -torch.sign(torch.sum(normal1.view(-1,3)*rho,dim=-1)) * theta
    R = rotation_matrix_with_axis(theta,normal1.view(-1,3))
    RR = torch.bmm(rot_matrix,R)
    return RR

def train_diffusion(args, config, train_writer, val_writer, logger):
    # build dataset
    config.model.args = args
    (train_sampler, train_dataloader), (_, test_dataloader),= builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
    # build model
    base_model = builder.model_builder(config.model)
    
    # parameter setting
    start_epoch = 0
    # resume ckpts
    if args.resume:
        start_epoch, best_metrics = builder.resume_model(base_model, args, logger = logger)
    else:
        if args.ckpts != '':
            # base_model.load_model_from_ckpt(args.ckpts)
            checkpoints = torch.load(args.ckpts)
            base_model.load_state_dict(checkpoints['base_model'])
        else:
            logger.info('Training from scratch')
    best_metrics = 99999
 
    logger.info('Using Data parallel ...' , logger = logger)
    device = torch.device('cuda') 
    base_model = nn.DataParallel(base_model).to(device)
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    
    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    # trainval
    # training
    for epoch in range(start_epoch, config.max_epoch + 1):
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['loss'])
        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)

        for idx, (index,point,centers,axis,masks) in enumerate(train_dataloader):      #TODO
            optimizer.zero_grad()
            data_time.update(time.time() - batch_start_time)

            #####TODO
            point = point.cuda().float()
            centers = centers.cuda().float()
            axis = axis.cuda().float()
            masks = masks.cuda()
            latents, predicted_latents = base_model(point)

            # diffusion loss
            criterion = nn.MSELoss()
            loss = 10*criterion(latents,predicted_latents)

            #######

            loss.backward()
            optimizer.step()
            losses.update([loss.item()])

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()
            
            if idx % 5 == 0:
                logger.info('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Loss= %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']))
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss', losses.avg(0), epoch)

        # print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
        #     (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],optimizer.param_groups[0]['lr']), logger = logger)
        logger.info('[Training] EPOCH: %d EpochTime = %.3f (s) Loss = %s lr = %.6f' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],optimizer.param_groups[0]['lr']))

        if epoch % args.val_freq == 0 and epoch != 0:
            # Validate the current model
            metrics = validate(base_model, test_dataloader, epoch, val_writer, args, config, logger=logger)

            # better = metrics.better_than(best_metrics)
            # Save ckeckpoints
            if metrics < best_metrics:
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, best_metrics, 'ckpt-best', args, logger = logger)
                logger.info("--------------------------------------------------------------------------------------------")

        builder.save_checkpoint(base_model, optimizer, epoch, best_metrics, 'ckpt-last', args, logger = logger)      
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()

def validate(base_model, test_dataloader, epoch, val_writer, args, config, logger = None):
    logger.info(f"[VALIDATION] Start validating epoch {epoch}")
    base_model.eval()  # set model to eval mode
    losses = AverageMeter(['loss'])
    with torch.no_grad():
        for idx, (index,point,centers,axis,masks) in enumerate(test_dataloader):
            point = point.cuda().float()
            centers = centers.cuda().float()
            axis = axis.cuda().float()
            masks = masks.cuda()
            latents, predicted_latents = base_model(point)

            # diffusion loss
            criterion = nn.MSELoss()
            loss = criterion(latents,predicted_latents)
            losses.update([loss.item()])

        logger.info('[Validation] EPOCH: %d  Loss = %s' % (epoch,['%.4f' % l for l in losses.avg()]))


    # Add testing results to TensorBoard
    # if val_writer is not None:
    #     val_writer.add_scalar('Metric/loss', losses.avg(), epoch)

    return losses.avg()[0]