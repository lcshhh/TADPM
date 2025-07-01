import torch
import torch.nn as nn
from utils import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from pytorch3d.loss import chamfer_distance
from einops import rearrange
from torch import autograd

import numpy as np
from torchvision import transforms
from pytorch3d.transforms import rotation_6d_to_matrix

def robust_compute_rotation_matrix_from_ortho6d(poses):
    """
    Instead of making 2nd vector orthogonal to first
    create a base that takes into account the two predicted
    directions equally
    """
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    x = torch.nn.functional.normalize(x_raw,dim=-1)  # batch*3
    y = torch.nn.functional.normalize(y_raw)  # batch*3
    middle = torch.nn.functional.normalize(x + y)
    orthmid = torch.nn.functional.normalize(x - y)
    x = torch.nn.functional.normalize(middle + orthmid)
    y = torch.nn.functional.normalize(middle - orthmid)
    z = torch.nn.functional.normalize(torch.cross(x, y, dim=-1))

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix

def train_tadpm(args, config, train_writer, val_writer, logger):
    # build dataset
    config.model.args = args
    (train_sampler, train_dataloader), (_, test_dataloader),= builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
    # build model
    base_model = builder.model_builder(config.model)
    
    # parameter setting
    start_epoch = 0
    best_metrics = 99999
    # resume ckpts
    if args.resume:
        start_epoch, best_metrics = builder.resume_model(base_model, args, logger = logger)
        # best_metrics = Acc_Metric(best_metrics)
    else:
        if args.ckpts != '':
            # base_model.load_model_from_ckpt(args.ckpts)
            checkpoints = torch.load(args.ckpts)
            base_model.load_state_dict(checkpoints['base_model'])
        else:
            logger.info('Training from scratch')
 
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
        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        losses = AverageMeter(['loss'])
        # validate(base_model, test_dataloader, epoch, val_writer, arg s, config, logger=logger)
        for idx, (index,feats,center,cordinates,faces,Fs,before_points,after_points,centroid,after_centroid,gt_params,masks) in enumerate(train_dataloader): 
            optimizer.zero_grad()
            data_time.update(time.time() - batch_start_time)

            #####TODO
            faces = faces.to(torch.float32).cuda()
            feats = feats.to(torch.float32).cuda()
            center = center.to(torch.float32).cuda()
            gt_params = gt_params.to(torch.float32).cuda()
            Fs = Fs.cuda()
            cordinates = cordinates.cuda()
            centroid = centroid.to(torch.float32).cuda()
            after_centroid = after_centroid.to(torch.float32).cuda()
            before_points = before_points.to(torch.float32).cuda()
            after_points = after_points.to(torch.float32).cuda()
            masks = masks.cuda()
            #######
            outputs = base_model(faces, feats, center, Fs, cordinates, centroid, before_points, gt_params).to(torch.float32).cuda()
            predicted_translation = outputs[:,:,:3]
            rot6d = rearrange(outputs[:,:,3:],'b n c -> (b n) c')
            criterion = nn.MSELoss(reduction='none')
            rot_matrix = rotation_6d_to_matrix(rot6d)
            # rot_matrix = robust_compute_rotation_matrix_from_ortho6d(rot6d)
            
            predicted_points = rearrange(before_points - centroid.unsqueeze(2),'b n p c -> (b n) p c')
            predicted_points = torch.bmm(predicted_points,rot_matrix)
            predicted_points = predicted_points + rearrange(predicted_translation,'b n c->(b n) c').unsqueeze(1)
            after_points = rearrange(after_points,'b n p c -> (b n) p c')
            loss = criterion(predicted_points,after_points).sum(dim=(1,2))
            loss = 0.001*(loss * masks.flatten()).sum()
            #######
            loss.backward()
            optimizer.step()
            losses.update([loss.item()])

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()
            
            if idx % 5 == 0:
                logger.info('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Loss = %s lr = %.6f' %
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
        for idx, (index,feats,center,cordinates,faces,Fs,before_points,after_points,centroid,after_centroid,gt_params,masks) in enumerate(test_dataloader):

            ###
            faces = faces.to(torch.float32).cuda()
            feats = feats.to(torch.float32).cuda()
            center = center.to(torch.float32).cuda()
            gt_params = gt_params.to(torch.float32).cuda()
            Fs = Fs.cuda()
            cordinates = cordinates.cuda()
            centroid = centroid.to(torch.float32).cuda()
            after_centroid = after_centroid.to(torch.float32).cuda()
            before_points = before_points.to(torch.float32).cuda()
            after_points = after_points.to(torch.float32).cuda()
            masks = masks.cuda()
            outputs = base_model(faces, feats, center, Fs, cordinates, centroid, before_points, gt_params).to(torch.float32).cuda()

            predicted_translation = outputs[:,:,:3]
            rot6d = rearrange(outputs[:,:,3:],'b n c -> (b n) c')
            criterion = nn.MSELoss(reduction='none')
            rot_matrix = rotation_6d_to_matrix(rot6d)
            predicted_points = rearrange(before_points - centroid.unsqueeze(2),'b n p c -> (b n) p c')
            predicted_points = torch.bmm(predicted_points,rot_matrix)
            predicted_points = predicted_points + rearrange(predicted_translation,'b n c->(b n) c').unsqueeze(1)
            after_points = rearrange(after_points,'b n p c -> (b n) p c')
            loss = criterion(predicted_points,after_points).sum(dim=-1).sqrt()
            loss = 40*(loss * masks.flatten().unsqueeze(1)).mean()
            ###
            losses.update([loss.item()])

        logger.info('[Validation] EPOCH: %d  Loss = %s' % (epoch,['%.4f' % l for l in losses.avg()]))


    # Add testing results to TensorBoard
    # if val_writer is not None:
    #     val_writer.add_scalar('Metric/loss', losses.avg(), epoch)

    return losses.avg()[0]