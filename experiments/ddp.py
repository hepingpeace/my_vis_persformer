import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import subprocess
import numpy as np
import random

def ddp_init(args):
    args.proc_id, args.gpu, args.world_size = 0, 0, 1
    
    if args.use_slurm == True:
        setup_slurm(args)
    else:
        setup_dist_launch(args)

    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) >= 1

    if args.distributed:
        setup_distributed(args)

    # deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.proc_id)
    np.random.seed(args.proc_id)
    random.seed(args.proc_id)
