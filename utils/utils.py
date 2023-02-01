import argparse
import errno
import os
import sys

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.init as init
import torch.optim
from torch.optim import lr_scheduler

def define_args():
    parser = argparse.ArgumentParser(description='PersFormer_3DLane_Detection')
    # Paths settings
    parser.add_argument('--dataset_name', type=str, help='the dataset name')
    parser.add_argument('--data_dir', type=str, help='The path of dataset json files (annotations)')
    parser.add_argument('--dataset_dir', type=str, help='The path of dataset image files (images)')
    parser.add_argument('--save_path', type=str, default='data_splits/', help='directory to save output')
    parser.add_argument('--use_memcache', type=str2bool, nargs='?', const=True, default=True, help='if use memcache')
    # Dataset settings
    parser.add_argument('--org_h', type=int, default=1080, help='height of the original image')
    parser.add_argument('--org_w', type=int, default=1920, help='width of the original image')
    parser.add_argument('--crop_y', type=int, default=0, help='crop from image')
    parser.add_argument('--cam_height', type=float, default=1.55, help='height of camera in meters')
    parser.add_argument('--pitch', type=float, default=3, help='pitch angle of camera to ground in centi degree')
    parser.add_argument('--fix_cam', type=str2bool, nargs='?', const=True, default=False, help='if to use fix camera')
    parser.add_argument('--no_3d', action='store_true', help='if a dataset include laneline 3D attributes')
    parser.add_argument('--no_centerline', action='store_true', help='if a dataset include centerline')
    parser.add_argument('--num_category', type=int, default=2, help='number of lane category, including background')
    # PersFormer settings
    parser.add_argument('--mod', type=str, default='PersFormer', help='model to train')
    parser.add_argument("--pretrained", type=str2bool, nargs='?', const=True, default=True, help="use pretrained model to start training")
    parser.add_argument("--batch_norm", type=str2bool, nargs='?', const=True, default=True, help="apply batch norm")
    parser.add_argument("--pred_cam", type=str2bool, nargs='?', const=True, default=False, help="use network to predict camera online?")
    parser.add_argument('--ipm_h', type=int, default=208, help='height of inverse projective map (IPM)')
    parser.add_argument('--ipm_w', type=int, default=128, help='width of inverse projective map (IPM)')
    parser.add_argument('--resize_h', type=int, default=360, help='height of the resized image (input of net)')
    parser.add_argument('--resize_w', type=int, default=480, help='width of the resized image (input of net)')
    parser.add_argument('--y_ref', type=float, default=20.0, help='the reference Y distance in meters from where lane association is determined')
    parser.add_argument('--prob_th', type=float, default=0.5, help='probability threshold for selecting output lanes')
    parser.add_argument('--encoder', type=str, default='ResNext101', help='feature extractor:'
                                                                          'ResNext101/VGG19/DenseNet161/InceptionV3/MobileNetV2/ResNet101/EfficientNet-Bx')
    parser.add_argument('--feature_channels', type=int, default=128, help='number of channels after encoder')
    parser.add_argument('--num_proj', type=int, default=4, help='number of projection layers')
    parser.add_argument('--num_att', type=int, default=3, help='number of attention encoding layers')
    parser.add_argument('--use_proj', type=str2bool, nargs='?', const=True, default=True, help='proj features in 2D pathway')
    parser.add_argument('--use_fpn', type=str2bool, nargs='?', const=True, default=False, help='use FPN features')
    parser.add_argument('--use_default_anchor', type=str2bool, nargs='?', const=True, default=False, help='use default anchors in 2D and 3D')
    parser.add_argument('--nms_thres_3d', type=float, default=1.0, help='nms threshold to filter detections in BEV, unit: meter')
    parser.add_argument('--new_match', type=str2bool, nargs='?', const=True, default=False, help='Allow multiple anchors to match the same GT during 3D data loading')
    parser.add_argument('--match_dist_thre_3d', type=float, default=2.0, help='Threshold to match an anchor to GT when using new_match, unit: meter')
    # LaneATT settings
    parser.add_argument('--max_lanes', type=int, default=6, help='max lane number detection in LaneATT')
    parser.add_argument('--S', type=int, default=72, help='max sample number in img height')
    parser.add_argument('--anchor_feat_channels', type=int, default=64, help='number of anchor feature channels')
    parser.add_argument('--cls_loss_weight', type=float, default=1.0, help='cls loss weight w.r.t. reg loss in 2D prediction')
    parser.add_argument('--reg_vis_loss_weight', type=float, default=1.0, help='reg vis loss weight w.r.t. reg loss in 2D prediction')
    parser.add_argument('--nms_thres', type=float, default=45.0, help='nms threshold')
    parser.add_argument('--conf_th', type=float, default=0.1, help='confidence threshold for selecting output 2D lanes')
    parser.add_argument('--vis_th', type=float, default=0.1, help='visibility threshold for output 2D lanes')
    parser.add_argument('--loss_att_weight', type=float, default=100.0, help='2D lane losses weight w.r.t. 3D lane losses')
    # General model settings
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--nepochs', type=int, default=100, help='total numbers of epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--no_cuda', action='store_true', help='if gpu available')
    parser.add_argument('--nworkers', type=int, default=0, help='num of threads')
    parser.add_argument('--seg_start_epoch', type=int, default=1, help='Number of epochs to perform segmentation pretraining')
    parser.add_argument('--channels_in', type=int, default=3, help='num channels of input image')
    parser.add_argument('--test_mode', action='store_true', help='prevents loading latest saved model')
    parser.add_argument('--start_epoch', type=int, default=0, help='prevents loading latest saved model')
    parser.add_argument('--evaluate', action='store_true', help='only perform evaluation')
    parser.add_argument('--resume', type=str, default='', help='resume latest saved run')
    parser.add_argument('--vgg_mean', type=float, default=[0.485, 0.456, 0.406], help='Mean of rgb used in pretrained model on ImageNet')
    parser.add_argument('--vgg_std', type=float, default=[0.229, 0.224, 0.225], help='Std of rgb used in pretrained model on ImageNet')
    # Optimizer settings
    parser.add_argument('--optimizer', type=str, default='adamw', help='adam/adamw/sgd/rmsprop')
    parser.add_argument('--weight_init', type=str, default='normal', help='normal, xavier, kaiming, orhtogonal weights initialisation')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='L2 weight decay/regularisation on?')
    parser.add_argument('--lr_decay', action='store_true', help='decay learning rate with rule')
    parser.add_argument('--niter', type=int, default=900, help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', type=int, default=400, help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--lr_policy', default=None, help='learning rate policy: lambda|step|cosine|cosine_warm')
    parser.add_argument('--gamma', type=float, default=0.1, help='multiplicative factor of learning rate decay')
    parser.add_argument('--lr_decay_iters', type=int, default=10, help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--T_max', type=int, default=10, help='maximum number of iterations')
    parser.add_argument('--T_0', type=int, default=500, help='number of iterations for the first restart')
    parser.add_argument('--T_mult', type=int, default=1, help='a factor increases T_i after a restart')
    parser.add_argument('--eta_min', type=float, default=1e-3, help='minimum learning rate')
    parser.add_argument('--clip_grad_norm', type=int, default=0, help='performs gradient clipping')
    # CUDNN usage
    parser.add_argument("--cudnn", type=str2bool, nargs='?', const=True, default=True, help="cudnn optimization active")
    # Tensorboard settings
    parser.add_argument("--no_tb", type=str2bool, nargs='?', const=True, default=False, help="Use tensorboard logging by tensorflow")
    # Print and Save settings
    parser.add_argument('--print_freq', type=int, default=500, help='padding')
    parser.add_argument('--save_freq', type=int, default=500, help='padding')
    # DDP setting
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument('--gpu', type=int, default = 0)
    parser.add_argument('--world_size', type=int, default = 1)
    parser.add_argument('--nodes', type=int, default = 1)
    return parser


# trick from stackoverflow
def str2bool(argument):
    if argument.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif argument.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Wrong argument in argparse, should be a boolean')

#如果没有文件的时候，就创建文件
def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def homographic_transformation(Matrix, x, y):
    """
    Helper function to transform coordinates defined by transformation matrix

    Args:
            Matrix (multi dim - array): 3x3 homography matrix
            x (array): original x coordinates
            y (array): original y coordinates
    """
    ones = np.ones((1, len(y)))
    coordinates = np.vstack((x, y, ones))
    trans = np.matmul(Matrix, coordinates)

    x_vals = trans[0, :]/trans[2, :]
    y_vals = trans[1, :]/trans[2, :]
    return x_vals, y_vals

def projective_transformation(Matrix, x, y, z):
    """
    Helper function to transform coordinates defined by transformation matrix

    Args:
            Matrix (multi dim - array): 3x4 projection matrix
            x (array): original x coordinates
            y (array): original y coordinates
            z (array): original z coordinates
    """
    ones = np.ones((1, len(z)))
    coordinates = np.vstack((x, y, z, ones))
    trans = np.matmul(Matrix, coordinates)

    x_vals = trans[0, :]/trans[2, :]
    y_vals = trans[1, :]/trans[2, :]
    return x_vals, y_vals