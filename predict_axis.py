from utils import dist_utils, misc
from utils.logger import *
from utils.config import *
from loguru import logger
import argparse
import time
import os
import torch
from tensorboardX import SummaryWriter
from datasets import *
from models import *
from runners.train_axis import train_axis
from runners.test_axis import test_axis
from runners.generate_axis import generate_axis
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', 
        type = str, 
        help = 'yaml config file')   
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch'],
        default='none',
        help='job launcher')     
    # seed 
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')      
    # bn
    parser.add_argument(
        '--sync_bn', 
        action='store_true', 
        default=False, 
        help='whether to use sync bn')
    # some args
    parser.add_argument('--save_root', type = str, required=True)
    parser.add_argument('--exp_name', type = str, required=True)
    parser.add_argument('--ckpts', type = str, default='')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--val_freq', type = int, default=1, help = 'test freq')
    parser.add_argument('--PCN_checkpoint', type = str, default='')
    parser.add_argument(
        '--test', 
        action='store_true', 
        default=False, 
        help = 'test mode for certain ckpt')
    parser.add_argument(
        '--generate', 
        action='store_true', 
        default=False, 
        help = 'test mode for certain ckpt')
    parser.add_argument(
        '--finetune_model', 
        action='store_true', 
        default=False, 
        help = 'finetune modelnet with pretrained weight')
    parser.add_argument(
        '--scratch_model', 
        action='store_true', 
        default=False, 
        help = 'training modelnet from scratch')
    
    args = parser.parse_args()

    if args.test:
        args.exp_name = 'test_' + args.exp_name
    args.experiment_path = os.path.join(args.save_root, args.exp_name)
    args.tfboard_path = os.path.join(args.save_root,args.exp_name,'TFBoard')
    args.log_name = Path(args.config).stem
    create_experiment_dir(args)
    return args

def create_experiment_dir(args):
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
        print('Create experiment path successfully at %s' % args.experiment_path)
    if not os.path.exists(args.tfboard_path):
        os.makedirs(args.tfboard_path)
        print('Create TFBoard path successfully at %s' % args.tfboard_path)

def main():
    # args
    args = get_args()
    # CUDA
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True
    # init distributed env first, since logger depends on the dist info.
    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    logger.add(os.path.join(args.experiment_path, f'{timestamp}.log'))
    # log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    # logger = get_root_logger(log_file=log_file, name=args.log_name)
    # define the tensorboard writer
    if not args.test:
        train_writer = SummaryWriter(os.path.join(args.tfboard_path, 'train'))
        val_writer = SummaryWriter(os.path.join(args.tfboard_path, 'test'))
    # config
    config = get_config(args, logger = logger)
    # batch size
    config.dataset.train.bs = config.total_bs
    # if config.dataset.get('extra_train'):
    #     config.dataset.extra_train.others.bs = config.total_bs * 2
    config.dataset.val.bs = config.total_bs
    if config.dataset.get('test'):
        config.dataset.test.bs = config.total_bs 
    # log
    log_args_to_file(args, logger, os.path.join(args.experiment_path, f'args.txt'))
    # log_config_to_file(config, logger, os.path.join(args.experiment_path, f'args.txt'))
    # exit()
    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        misc.set_random_seed(args.seed + args.local_rank, deterministic=args.deterministic) # seed + rank, for augmentation
        
    # run
    if args.test:
        test_axis(args, config, logger)
    elif args.generate:
        generate_axis(args, config, logger)
    else:
        train_axis(args, config, train_writer, val_writer, logger)


if __name__ == '__main__':
    main()
