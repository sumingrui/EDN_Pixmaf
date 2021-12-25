# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
import json
from yacs.config import CfgNode as CN

# Configuration variables
cfg = CN(new_allowed=True)

cfg.OUTPUT_DIR = 'results'
cfg.DEVICE = 'cuda'
cfg.DEBUG = False
cfg.LOGDIR = ''
cfg.VAL_VIS_BATCH_FREQ = 200
cfg.TRAIN_VIS_ITER_FERQ = 1000
cfg.SEED_VALUE = -1

cfg.TRAIN = CN(new_allowed=True)


cfg.MODEL = CN(new_allowed=True)

cfg.MODEL.PyMAF = CN(new_allowed=True)

## switch
cfg.TRAIN.VAL_LOOP = True

cfg.TEST = CN(new_allowed=True)




cfg.PixMAF = CN(new_allowed=True)
cfg.PixMAF.USE_PIXMAF = True
cfg.PixMAF.N_ITER = 4
# cfg.PixMAF.MLP_DIM = [[512, 256, 128, 5],[256, 128, 64, 10],[128, 64, 32, 10],[64, 32, 16, 10]]
cfg.PixMAF.MLP_DIM = [[512, 256, 128, 5],[256, 128, 64, 5],[128, 64, 32, 5],[64, 32, 16, 5]]
cfg.PixMAF.BACKBONE = 'Pix2pixHD'
cfg.PixMAF.USE_SILHOUETTE = True

cfg.LOSS = CN(new_allowed=True)
cfg.LOSS.KP_2D_W = 100.0
cfg.LOSS.SILHOUETTES_W = 0.00001



def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    # return cfg.clone()
    return cfg

def update_cfg(cfg_file):
    cfg.merge_from_file(cfg_file)
    return cfg


def parse_args(args):
    cfg_file = args.cfg_file
    if cfg_file is not None:
        cfg = update_cfg(cfg_file)
    else:
        cfg = get_cfg_defaults()
    return cfg

def parse_args_extend(args):
        parse_args(args)
