import torch
import torch.optim
import torch.nn as nn
import numpy as np
import glob
import time
import shutil
import os
from os import mkdir, write
from tqdm import tqdm
from os.path import ops
from models.networks import Loss_crit
from utils import eval_3D_lane
from utils.utils import *
from utils.Visualizer import *

from tensorboardX import SummaryWriter

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from .ddp import *

class Runner:
    def __init__(self, args):
        self.args = args
        # Check GPU availability
        if args.proc_id == 0:
            if not args.no_cuda and not torch.cuda.is_available():
                raise Exception("No gpu available for usage")
            if int(os.getenv('WORLD_SIZE', 1)) >= 1:
                print("Let's use", os.environ['WORLD_SIZE'], "GPUs!")
                torch.cuda.empty_cache()

        save_id = args.mod
        args.save_json_path = args.save_path
        args.save_path = os.path.join(args.save_path, save_id)
        if args.proc_id == 0:
            mkdir_if_missing(args.save_path)
            mkdir_if_missing(os.path.join(args.save_path, 'example/'))
            mkdir_if_missing(os.path.join(args.save_path, 'example/train'))
            mkdir_if_missing(os.path.join(args.save_path, 'example/valid'))

        # Get Dataset
        if args.proc_id == 0:
            print("Loading Dataset ...")
        self.val_gt_file = ops.join(args.save_path, 'test.json')
        self.train_dataset, self.train_loader, self.train_sampler = self._get_train_dataset()
        self.valid_dataset, self.valid_loader, self.valid_sampler = self._get_valid_dataset()

        # self.crit_string = 'loss_gflat'
        self.crit_string = args.crit_string
        # Define loss criteria
        if self.crit_string == 'loss_gflat_3D':
            self.criterion = Loss_crit.Laneline_loss_gflat_3D(args.batch_size, self.train_dataset.num_types,
                                                              self.train_dataset.anchor_x_steps, self.train_dataset.anchor_y_steps,
                                                              self.train_dataset._x_off_std, self.train_dataset._y_off_std,
                                                              self.train_dataset._z_std, args.pred_cam, args.no_cuda)
        elif self.crit_string == 'loss_gflat_novis':
            self.criterion = Loss_crit.Laneline_loss_gflat_novis_withdict(self.train_dataset.num_types, args.num_y_steps, args.pred_cam)
        else:
            self.criterion = Loss_crit.Laneline_loss_gflat_multiclass(self.train_dataset.num_types, args.num_y_steps,
                                                                      args.pred_cam, args.num_category, args.no_3d, args.loss_dist)
        if 'openlane' in args.dataset_name:
            self.evaluator = eval_3D_lane.LaneEval(args)
        else:
            self.evaluator = eval_3D_lane.LaneEval(args)
        
        # Tensorboard writer
        if not args.no_tb and args.proc_id == 0:
            tensorboard_path = os.path.join(args.save_path, 'Tensorboard/')
            mkdir_if_missing(tensorboard_path)
            self.writer = SummaryWriter(tensorboard_path)
        # initialize visual saver
        self.vs_saver = Visualizer(args)
        if args.proc_id == 0:
            print("Init Done!")
    
    def train(self):
        args = self.args

        # Get Dataset
        train_dataset = self.train_dataset
        train_loader = self.train_loader
        train_sampler = self.train_sampler

        # Define model or resume
        if args.model_name == "PersFormer":
            model, optimizer, scheduler, best_epoch, lowest_loss, best_f1_epoch, best_val_f1 = self._get_model_ddp()
        
        criterion = self.criterion
        if not args.no_cuda:
            device = torch.device("cuda", args.local_rank)
            criterion = criterion.to(device)
        bceloss = nn.BCEWithLogitsLoss()

        # Print model basic info
        if args.proc_id == 0:
            if args.model_name == "PersFormer":
                print(40*"="+"\nArgs:{}\n".format(args)+40*"=")
                print("Init model: '{}'".format(args.mod))
                print("Number of parameters in model {} is {:.3f}M".format(args.mod, sum(tensor.numel() for tensor in model.parameters())/1e6))
        
        # image matrix
        _S_im_inv = torch.from_numpy(np.array([[1/np.float(args.resize_w),                         0, 0],
                                                    [                        0, 1/np.float(args.resize_h), 0],
                                                    [                        0,                         0, 1]], dtype=np.float32)).cuda()
        _S_im = torch.from_numpy(np.array([[args.resize_w,              0, 0],
                                                [            0,  args.resize_h, 0],
                                                [            0,              0, 1]], dtype=np.float32)).cuda()
        # 设置TensorBoard日志记录和保存/恢复模型变量。
        if not args.no_tb and args.proc_id == 0:
            writer = self.writer
        vs_saver = self.vs_saver

        # Start training and validation for nepochs
        for epoch in range(args.start_epoch, args.nepochs):
            if args.proc_id == 0:
                print("\n => Start train set for EPOCH {}".format(epoch + 1))
            if (args.proc_id == 0) and (args.model_name == "PersFormer"):
                lr = optimizer.param_groups[0]['lr']
                print('lr is set to {}'.format(lr))