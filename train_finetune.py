import time
import torch
import random
import datetime
import argparse
import numpy as np
from pathlib import Path
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from logs.configuration import *
from dataloader import prepare_data, prepare_finetune_data

import models_finetune
from engine_finetune import train_one_epoch, eval_model

def get_args_parser():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--experiment', type=str, default='ft')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus: [32*3, 64*3, 12*3, 12*3]')
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
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr): [0.025, 0.05, 5e-5, 5e-5]')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=0, metavar='N', help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--trial_seed', type=int, default=0, help='Trial number (used for seeding split_dataset and random_hparams).')
    parser.add_argument('--val_fraction', default=0.01, type=float, help="fraction of labeled training data: [1%, 5%, 10%, 100%]")
    parser.add_argument('--holdout_fraction', type=float, default=0.01, help='fraction of validation set/(validation set+training set): [0.991, 0.955, 0.91, 0.10]')
    parser.add_argument('--uda_holdout_fraction', type=float, default=0)
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--dataset', default='DomainNet', type=str, help='dataset')
    parser.add_argument('--data_path', default='../dataset', type=str, help='dataset path')
    parser.add_argument('--output_dir', default='../output_dir', help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='../output_dir', help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='/path/to/the/checkpoints', help='resume from checkpoint')

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
    train_loaders, eval_loaders, eval_weights, eval_loader_names, total_len, env_samples = prepare_finetune_data(args, logger)
    train_iterator = zip(*train_loaders)

    args.model = "Finetune"
    model = models_finetune.__dict__[args.model](args=args, norm_pix_loss=args.norm_pix_loss)
    
    # freeze all the blocks except the head if args.val_fraction<0.10
    if args.val_fraction<0.10:
        torch.nn.init.trunc_normal_(model.head.weight, std=0.01)
        model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
        for _, p in model.named_parameters():
            p.requires_grad = False
        for _, p in model.head.named_parameters():
            p.requires_grad = True
    
    device = torch.device(args.device)
    model.to(device)

    # fix the seed for reproducibility
    # seed = args.seed + misc.get_rank()
    seed = args.seed
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    best_perf = 0
    no_impr_counter = 0
    record_epoch = None

    # if args.val_fraction>=0.10:
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    # optimizer = torch.optim.SGD(model.head.parameters(), args.lr, momentum=0.99, weight_decay=args.weight_decay)
    
    misc.load_model(args=args, model=model, optimizer=optimizer, logger=logger)

    start_time = time.time()
    logger.info("Start training for {} epochs".format(args.epochs))
    logger.info("Parameters {:.2f}M".format(sum([x.numel() for x in model.parameters() if x.requires_grad])/1e6))
    
    for epoch in range(args.start_epoch, args.epochs):
        logger.info("Epoch:[{}/{}], Lr:{:.5f}".format(epoch, args.epochs, args.lr))
        train_one_epoch(model, train_iterator, optimizer, epoch, args, total_len, logger)
        
        if epoch%5==0 and epoch!=0 and not args.recon_mission:
            eval_iterator = zip(eval_loader_names, eval_loaders, eval_weights)
            current_perf = eval_model(model, eval_iterator, device, epoch, args, total_len, logger, env_samples)
            if current_perf>best_perf:
                record_epoch = epoch
                no_impr_counter = 0

            best_perf = max(best_perf, current_perf)
            no_impr_counter += 1
            if no_impr_counter>=30:
                break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {} | Best perf in validation set: {:.2f}% in epoch {}'.format(total_time_str, best_perf, record_epoch))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(exist_ok=True)
    
    # torch.distributed.init_process_group('nccl', init_method='env://', timeout=datetime.timedelta(seconds=1800))
    # torch.cuda.set_device(args.local_rank)

    main(args)
