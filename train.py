import time
import torch
import random
import datetime
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from logs.configuration import *
from dataloader import prepare_data

def get_args_parser():

    parser = argparse.ArgumentParser()
    
    # 
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # ViT loss function
    parser.add_argument('--smoothing', type=float, default=0.0, help='Label smoothing (default: 0.1)')

    # training stage
    parser.add_argument('--is_linear', default=False, action='store_true', help="the type of the classifier")
    parser.add_argument('--augment', default=False, action='store_true', help='whether augment images')
    parser.add_argument('--sample_pos', default=False, action="store_true")
    parser.add_argument('--is_mix', default=False, action="store_true")
    parser.add_argument('--recon_mission', default=False, action='store_true')
    parser.add_argument('--global_pool', default=False, action='store_true')

    # Hyperparameters of loss function 
    parser.add_argument('--lambda1', type=float, default=1e-3, help='hyperparameter of adaptive contrastive loss')
    parser.add_argument('--lambda2', type=float, default=1.0, help='hyperparameter of classification loss')
    parser.add_argument('--tao', type=float, default=4e-1 , help='temperature coefficient of infoNCE loss')
    parser.add_argument('--margin', type=float, default=8e-3, help='margin of reconstruction L2 loss')
    parser.add_argument('--ada_iter', type=int, default=0, help='iteration for adaptive contrastive loss')
    parser.add_argument('--decoder_drop', type=float, default=0.0, help="Dropout rate of the decoder") 
    parser.add_argument('--inter_ada', type=int, default=25, help="training interval of the domain classifier")
    parser.add_argument('--max_epoch_ada', type=int, default=100, help="maximum adaptive training epoch")

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,help='images input size')
    parser.add_argument('--mask_ratio', default=0.75, type=float, help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--norm_pix_loss', action='store_true',help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=0, metavar='N', help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--trial_seed', type=int, default=0, help='Trial number (used for seeding split_dataset and random_hparams).')
    parser.add_argument('--holdout_fraction', type=float, default=0.01)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0)
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--dataset', default='DomainNet', type=str, help='dataset')
    parser.add_argument('--data_path', default='../dataset', type=str, help='dataset path')
    parser.add_argument('--output_dir', default='../output_dir', help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='../output_dir', help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default=None, help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    return parser

def main(args):
    
    configure_experiment(args.output_dir, rank=misc.get_rank())
    logger = get_logger()
    logger.info(args)
    
    train_loaders, eval_loaders, eval_weights, eval_loader_names, total_len = prepare_data(args)
    train_iterator = zip(*train_loaders)

    import models_dmae
    if args.recon_mission:
        args.model = "mae_vit_recon"
    else:
        args.model = "mae_vit_cls"

    model = models_dmae.__dict__[args.model](args=args, norm_pix_loss=args.norm_pix_loss)
    from engine_dmae import train_one_epoch
    device = torch.device(args.device)
    model.to(device)

    # fix the seed for reproducibility
    # seed = args.seed + misc.get_rank()

    # seed = args.seed
    # np.random.seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # if True:  # args.distributed:
    #     num_tasks = misc.get_world_size()
    #     global_rank = misc.get_rank()
    #     sampler_train = torch.utils.data.DistributedSampler(
    #         dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    #     )
    #     print("Sampler_train = %s" % str(sampler_train))
    # else:
    #     sampler_train = torch.utils.data.RandomSampler(dataset_train)

    # if global_rank == 0 and args.log_dir is not None:
    #     os.makedirs(args.log_dir, exist_ok=True)
    #     log_writer = SummaryWriter(log_dir=args.log_dir)
    # else:
    #     log_writer = None
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    optimizer_domain = torch.optim.SGD(model.reweight_classifier.parameters(), 0.0005, momentum=0.99, weight_decay=args.weight_decay)
    
    misc.load_model(args=args, model=model, optimizer=optimizer, logger=logger)

    start_time = time.time()
    logger.info("Start training for {} epochs".format(args.epochs))
    logger.info("Parameters {:.2f}M".format(sum([x.numel() for x in model.parameters() if x.requires_grad])/1e6))
    
    for epoch in range(args.start_epoch, args.epochs):
        
        logger.info("Epoch:[{}/{}], Lr:{:.5f}".format(epoch, args.epochs, args.lr))
        train_one_epoch(model, train_iterator, optimizer, optimizer_domain, epoch, args, total_len, logger)
      
    if epoch!=0 and args.dataset=="DomainNet":
        misc.save_model(args, epoch, model, optimizer)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(exist_ok=True)
    
    # torch.distributed.init_process_group('nccl', init_method='env://', timeout=datetime.timedelta(seconds=1800))
    # torch.cuda.set_device(args.local_rank)
    main(args)
