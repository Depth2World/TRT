import os 
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import argparse
from models.dy_model import dy_model
import random
import logging
def get_args_parser():
    # build the arg_parser
    parser = argparse.ArgumentParser()
    # set path params
    parser.add_argument("--param_store", type=bool, default=True)
    parser.add_argument("--model_name", type=str, default="debug") #  TST_Large_DP_final_model_upgrade
    parser.add_argument("--dataset", type=str, default='xxx')
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--train_bacth_size", type=int, default=1)
    parser.add_argument("--test_bacth_size", type=int, default=1)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr_rate", type=float, default=1e-4)
    parser.add_argument("--weit_decay", type=int, default=1e-4)
    parser.add_argument("--opter", type=str, default="adamw")
    parser.add_argument("--seed", type=int, default=3407, help="seed for data loading")
    parser.add_argument("--model_dir", type=str, default="")
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--resmod_dir", type=str, default=None)
    parser.add_argument("--num_save", type=int, default=300) #800
    
    parser.add_argument("--data_size", type=int, default=64)
    
    parser.add_argument("--down_scale", type=int, default=1)
    parser.add_argument("--mask", action='store_true', help='default False')
    parser.add_argument("--sp_ds_scale", type=int, default=8)
    parser.add_argument("--kl_scale", type=float, default=0.1)
    parser.add_argument("--nlm_scale", type=float, default=0.1)
    parser.add_argument("--po_scale", type=float, default=0)
    parser.add_argument("--tv3d_scale", type=float, default=1e-6)
    parser.add_argument("--num_coders", type=int, default=1)
    
    parser.add_argument("--loss_weit", type=float, default=0.)
    parser.add_argument("--grad_clip", type=int, default=0.1, help="gradient clipping max norm")
    parser.add_argument("--noise_idx", type=int, default=1)
    parser.add_argument("--num_head", type=int, default=8)
    parser.add_argument("--drop_attn", type=float, default=0.)
    parser.add_argument("--drop_proj", type=float, default=0.)
    parser.add_argument("--drop_path", type=float, default=0.)
    
    ### dataset txt path
    parser.add_argument("--root_path", type=str, default='') 
    parser.add_argument("--train_total_path", type=str, default='') 
    parser.add_argument("--test_total_path", type=str, default='') 
    
    
    # set optimization params
    parser.add_argument("--epo_warm", type=int, default=5)
    parser.add_argument("--epo_cool", type=int, default=5)
    
    # set distributed training params
    parser.add_argument("--local_rank", type=int, default=-1)
    
    parser.add_argument("--dp_gpus", type=str, default="0,1,2,3")# 
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--world_size", type=int, default=1, help="number of total machines")
    parser.add_argument("--dist_url", type=str, default="env://", help="url used to setup the distributed training")
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    return parser.parse_args()


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_seeds(seed=0, cuda_deterministic=True):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

    if cuda_deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.deterministic = False
        cudnn.benchmark = False
        

def main(args):
    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.distributed = True
    
    torch.cuda.set_device(args.local_rank)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    
    init_seeds(args.seed + args.local_rank)
    print("Now process is {}".format(args.local_rank))

    # setup_for_distributed(args.rank == 0)
    model = dy_model(args)
    model.do(args.local_rank, args.nprocs)
    logging.info('complete!')
    
if __name__=="__main__":
    args= get_args_parser()
    args.nprocs = torch.cuda.device_count()
    main(args)
