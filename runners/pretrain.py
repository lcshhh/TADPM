import torch
import torch.nn as nn
from utils import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from pytorch3d.loss import chamfer_distance
from models.MeshMAE import Mesh_mae

import numpy as np
from torchvision import transforms

def pretrain(args, config, train_writer, val_writer, logger):
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader),= builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
    # build model
    base_model = Mesh_mae()
    device = torch.device('cuda') 
    logger.info('Using Data parallel ...' , logger = logger)
    base_model = nn.DataParallel(base_model).to(device)
    # parameter setting
    start_epoch = 0

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger = logger)
        # best_metrics = Acc_Metric(best_metrics)
    else:
        if args.ckpts != '':
            checkpoints = torch.load(args.ckpts)
            base_model.load_state_dict(checkpoints['model'])
        else:
            logger.info('Training from scratch')
 
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
        # losses = AverageMeter(['loss','rec_loss'])
        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)

        for idx, (feats, center, cordinates, faces, Fs) in enumerate(train_dataloader):
            losses = AverageMeter(['loss'])
            optimizer.zero_grad()
            data_time.update(time.time() - batch_start_time)

            #####TODO
            faces = faces.to(torch.float32).cuda()
            feats = feats.to(torch.float32).cuda()
            centers = center.to(torch.float32).cuda()
            Fs = Fs.cuda()
            cordinates = cordinates.to(torch.float32).cuda()
            loss = base_model(faces, feats, centers, Fs, cordinates).mean()
            #######

            loss.backward()
            optimizer.step()
            losses.update([loss.item()])

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()


            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()
            
            if idx % 10 == 0:
                logger.info('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Loss, rec = %s lr = %.6f' %
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
        logger.info('[Training] EPOCH: %d EpochTime = %.3f (s) Loss, rec = %s lr = %.6f' %
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
    losses = AverageMeter(['loss'])
    with torch.no_grad():
        for idx, (feats, center, cordinates, faces, Fs) in enumerate(test_dataloader):
            faces = faces.to(torch.float32).cuda()
            feats = feats.to(torch.float32).cuda()
            centers = center.to(torch.float32).cuda()
            Fs = Fs.cuda()
            cordinates = cordinates.to(torch.float32).cuda()
            loss = base_model(faces, feats, centers, Fs, cordinates).mean()
            losses.update([loss.item()])

        logger.info('[Validation] EPOCH: %d  Loss= %s' % (epoch,['%.4f' % l for l in losses.avg()]))


    # Add testing results to TensorBoard
    return losses.avg()[0]