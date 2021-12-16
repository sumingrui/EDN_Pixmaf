# This script is borrowed and extended from https://github.com/shunsukesaito/PIFu/blob/master/lib/model/SurfaceClassifier.py
# Modify by Armin: msu073@uottawa.ca

from packaging import version
import torch
import scipy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from ..core.cfgs import cfg
from ..utils.geometry import projection

import logging
logger = logging.getLogger(__name__)

class MAF_Extractor(nn.Module):
    ''' Mesh-aligned Feature Extrator

    As discussed in the paper, we extract mesh-aligned features based on 2D projection of the mesh vertices.
    The features extrated from spatial feature maps will go through a MLP for dimension reduction.
    '''

    def __init__(self, deconv_layer_list, device=torch.device('cuda')):
        super().__init__()

        self.device = device
        self.filters = []
        self.num_views = 1
        filter_channels = cfg.MODEL.PixMAF.MLP_DIM[deconv_layer_list]
        self.last_op = nn.ReLU(True) 

        for l in range(0, len(filter_channels) - 1): # 3
            if 0 != l:
                self.filters.append(
                    nn.Conv1d(
                        filter_channels[l] + filter_channels[0],
                        filter_channels[l + 1],
                        1))
            else:
                self.filters.append(nn.Conv1d(
                    filter_channels[l],
                    filter_channels[l + 1],
                    1))

            self.add_module("conv%d" % l, self.filters[l])
        
        self.im_feat = None
        self.cam = None

        # downsample SMPL mesh and assign part labels
        # from https://github.com/nkolot/GraphCMR/blob/master/data/mesh_downsampling.npz
        smpl_mesh_graph = np.load('pixmaf_data/mesh_downsampling.npz', allow_pickle=True, encoding='latin1')

        A = smpl_mesh_graph['A']
        U = smpl_mesh_graph['U']
        D = smpl_mesh_graph['D'] # shape: (2,)

        # downsampling
        ptD = []
        for i in range(len(D)):
            d = scipy.sparse.coo_matrix(D[i])
            i = torch.LongTensor(np.array([d.row, d.col]))
            v = torch.FloatTensor(d.data)
            ptD.append(torch.sparse.FloatTensor(i, v, d.shape))
        
        # downsampling mapping from 6890 points to 431 points
        # ptD[0].to_dense() - Size: [1723, 6890]
        # ptD[1].to_dense() - Size: [431. 1723]
        Dmap = torch.matmul(ptD[1].to_dense(), ptD[0].to_dense()) # 6890 -> 431   torch.Size([431, 6890])
        self.register_buffer('Dmap', Dmap)

    def reduce_dim(self, feature):
        '''
        Dimension reduction by multi-layer perceptrons
        :param feature: list of [B, C_s, N] point-wise features before dimension reduction
        :return: [B, C_p x N] concatantion of point-wise features after dimension reduction
        '''
        y = feature
        tmpy = feature
        for i, f in enumerate(self.filters):
            y = self._modules['conv' + str(i)](
                y if i == 0
                else torch.cat([y, tmpy], 1)
            )
            if i != len(self.filters) - 1:
                y = F.leaky_relu(y)
            if self.num_views > 1 and i == len(self.filters) // 2:
                y = y.view(
                    -1, self.num_views, y.shape[1], y.shape[2]
                ).mean(dim=1)
                tmpy = feature.view(
                    -1, self.num_views, feature.shape[1], feature.shape[2]
                ).mean(dim=1)

        y = self.last_op(y)

        y = y.view(y.shape[0], -1)
        return y

    def sampling(self, points, im_feat=None, z_feat=None):
        '''
        Given 2D points, sample the point-wise features for each point, 
        the dimension of point-wise features will be reduced from C_s to C_p by MLP.
        Image features should be pre-computed before this call.
        :param points: [B, N, 2] image coordinates of points
        :im_feat: [B, C_s, H_s, W_s] spatial feature maps 
        :return: [B, C_p x N] concatantion of point-wise features after dimension reduction
        '''
        if im_feat is None:
            im_feat = self.im_feat

        batch_size = im_feat.shape[0]

        if version.parse(torch.__version__) >= version.parse('1.3.0'):
            # Default grid_sample behavior has changed to align_corners=False since 1.3.0.
            point_feat = torch.nn.functional.grid_sample(im_feat, points.unsqueeze(2), align_corners=True)[..., 0]
        else:
            # print('im_feat:',im_feat.shape) # 0: torch.Size([1, 512, 32, 64])
            # print('points.unsqueeze(2):',points.unsqueeze(2).shape) # 0: torch.Size([1, 441, 1, 2])
            point_feat = torch.nn.functional.grid_sample(im_feat, points.unsqueeze(2))[..., 0]
            # print('point_feat:',point_feat.shape) # 0: torch.Size([1, 512, 441])

        mesh_align_feat = self.reduce_dim(point_feat)
        # print('mesh_align_feat:',mesh_align_feat.shape) # 0:torch.Size([1, 2205])
        # print('')
        return mesh_align_feat

    def forward(self, p, res, s_feat=None, cam=None, **kwargs):
        ''' Returns mesh-aligned features for the 3D mesh points.

        Args:
            p (tensor): [B, N_m, 3] mesh vertices
            s_feat (tensor): [B, C_s, H_s, W_s] spatial feature maps
            cam (tensor): [B, 3] camera
        Return:
            mesh_align_feat (tensor): [B, C_p x N_m] mesh-aligned features
        '''
        if cam is None:
            cam = self.cam
        
        p_proj_2d = projection(p, cam, res, retain_z=False)
        # print('p_proj_2d:',p_proj_2d.shape)
        mesh_align_feat = self.sampling(p_proj_2d, s_feat)
        return mesh_align_feat
