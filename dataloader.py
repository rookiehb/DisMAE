import datasets
import util.misc as misc
from util.fast_data_loader import *

def prepare_finetune_data(args, logger):
    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args)
    else:
        raise NotImplementedError
    
    in_splits = []; out_splits = []; total_len = 0
    env_samplesInfo = ""
    env_samples = []
    for env_i, env in enumerate(dataset):
        
        env_samplesInfo += "env{}: {}, ".format(env_i, len(env))
        env_samples.append(len(env))
        total_len += len(env)
        if env_i in args.test_envs:
            out, in_ = misc.split_dataset(
                            env, int(len(env) * 0.8),
                            misc.seed_hash(args.trial_seed, env_i))
        else:
            out, in_ = misc.split_dataset(
                            env, int(len(env) * args.holdout_fraction),
                            misc.seed_hash(args.trial_seed, env_i))

        in_weights, out_weights = None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
    
    
    logger.info(env_samplesInfo)
    train_loaders = [InfiniteDataLoader(dataset=env, weights=in_weights, 
                            batch_size=args.batch_size, num_workers=args.num_workers)
                        for i, (env, in_weights) in enumerate(in_splits) if i not in args.test_envs]         
    
    eval_loaders = [FastDataLoader(
                    dataset=env, batch_size=args.batch_size, num_workers=args.num_workers)
                    for env, _ in (in_splits + out_splits)]
   
    eval_weights = [None for _, weights in (in_splits + out_splits)]
    eval_loader_names = ['env{}_in'.format(i) for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i) for i in range(len(out_splits))]
    
    if args.experiment=='ft':
        if args.val_fraction>=0.1:
            total_len = int(total_len*(1-args.holdout_fraction)/env_i/args.batch_size)
        else:
            # in case of infinite sampling loop
            total_len = int(total_len*(1-args.holdout_fraction)/env_i/4)
    else:
        total_len = int(total_len*(1-args.holdout_fraction)/args.batch_size//env_i)+1
        
    return train_loaders, eval_loaders, eval_weights, eval_loader_names, total_len, env_samples



def prepare_data(args):
        
    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args)
    else:
        raise NotImplementedError

    in_splits = []; out_splits = []; total_len = 0
    for env_i, env in enumerate(dataset):

        total_len += len(env)
        out, in_ = misc.split_dataset(
                        env, int(len(env) * args.holdout_fraction),
                        misc.seed_hash(args.trial_seed, env_i))
        if False:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            # if uda is not None:
            #     uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights = None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
    

    train_loaders = [InfiniteDataLoader(dataset=env, weights=in_weights, 
                            batch_size=args.batch_size, num_workers=args.num_workers)
                        for i, (env, in_weights) in enumerate(in_splits) if i not in args.test_envs]         

    eval_loaders = [FastDataLoader(
                    dataset=env, batch_size=args.batch_size, num_workers=args.num_workers)
                    for env, _ in (in_splits + out_splits)]
    
    eval_weights = [None for _, weights in (in_splits + out_splits)]
    eval_loader_names = ['env{}_in'.format(i) for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i) for i in range(len(out_splits))]
    
    total_len = int(total_len*(1-args.holdout_fraction)/args.batch_size//env_i)
    return train_loaders, eval_loaders, eval_weights, eval_loader_names, total_len

