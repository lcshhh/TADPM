import torch
import torch.nn as nn
from utils import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from pytorch3d.loss import chamfer_distance

import numpy as np
from torchvision import transforms
def create_attn_mask(masks):
    attn_mask = torch.bmm(masks.unsqueeze(2),masks.unsqueeze(1))
    return attn_mask == 0

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

def train_global(args, config, train_writer, val_writer, logger):
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
        start_epoch, best_metric = builder.resume_model(base_model, args, logger = logger)
        # best_metrics = Acc_Metric(best_metrics)
    else:
        if args.ckpts != '':
            base_model.load_model_from_ckpt(args.ckpts)
        else:
            logger.info('Training from scratch')
 
    logger.info('Using Data parallel ...' , logger = logger)
    device = torch.device('cuda') 
    base_model = nn.DataParallel(base_model).to(device)
    # optimizer & scheduler
    best_metrics = 99999
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
        losses = AverageMeter(['loss','rec_loss','kl_loss'])
        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)

        for idx, (index,point,centers,axis,masks) in enumerate(train_dataloader):      #TODO
            optimizer.zero_grad()
            data_time.update(time.time() - batch_start_time)

            #####TODO
            point = point.cuda().float()
            masks = masks.cuda()
            attn_mask = create_attn_mask(masks)
            mu, log_var, reconstructed = base_model(point,axis,attn_mask)
            kl_loss = 0.001*(-0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp()))
            rec_loss = (torch.stack([chamfer_distance(point[:,i],reconstructed[:,i],batch_reduction=None)[0] for i in range(32)],dim=1) * masks).sum()
            loss = kl_loss + rec_loss
            #######

            loss.backward()
            optimizer.step()
            losses.update([loss.item(),kl_loss.item(),rec_loss.item()])

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()
            
            if idx % 5 == 0:
                logger.info('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Loss rec_loss kl_loss = %s lr = %.6f' %
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
        logger.info('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
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
        # if (config.max_epoch - epoch) < 10:
        #     builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()

def validate(base_model, test_dataloader, epoch, val_writer, args, config, logger = None):
    logger.info(f"[VALIDATION] Start validating epoch {epoch}")
    base_model.eval()  # set model to eval mode
    losses = AverageMeter(['loss','rec','kl'])
    with torch.no_grad():
        for idx, (index,point,centers,axis,masks) in enumerate(test_dataloader):
            point = point.cuda().float()
            masks = masks.cuda()
            mu, log_var, reconstructed = base_model(point,axis)
            kl_loss = 0.001*(-0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp()))
            rec_loss = (torch.stack([chamfer_distance(point[:,i],reconstructed[:,i],batch_reduction=None)[0] for i in range(32)],dim=1) * masks).sum()
            loss = kl_loss + rec_loss
            losses.update([loss.item(),kl_loss.item(),rec_loss.item()])

        logger.info('[Validation] EPOCH: %d  Loss rec_loss kl_loss = %s' % (epoch,['%.4f' % l for l in losses.avg()]))


    # Add testing results to TensorBoard
    # if val_writer is not None:
    #     val_writer.add_scalar('Metric/loss', losses.avg(), epoch)

    return losses.avg()[0]