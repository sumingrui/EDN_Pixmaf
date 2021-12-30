### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### From PIX2PIXHD

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import functools
import neural_renderer as nr
from .smpl import SMPL, SMPL_MODEL_DIR, get_smpl_faces
from ..utils.geometry import perspective_projection
from ..core.cfgs import cfg
from ..core import path_config
from ..utils.renderer import PyRenderer

###############################################################################
# Functions
###############################################################################

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

###############################################################################
# Downsampling and Upsampling
###############################################################################

class Down_Sampling(nn.Module):
    def __init__(self, input_nc=6, ngf=64, n_downsampling=4, n_blocks=9, 
                                    norm_layer=get_norm_layer(norm_type='instance'), padding_type='reflect'):
        assert(n_blocks >= 0)
        super(Down_Sampling, self).__init__()        

        #self.avgpool = nn.AvgPool2d((8,16), stride=(8,16))
        activation = nn.ReLU(True)      
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        self.model = nn.Sequential(*model)
            
    def forward(self, input):
        s_feat = self.model(input)    
        #g_feat = self.avgpool(s_feat)

        return s_feat      

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


def Up_Sampling(output_nc =3, ngf=64, n_downsampling=4, norm_layer=get_norm_layer(norm_type='instance'), padding_type='reflect'):
    activation = nn.ReLU(True)             
    model = []
    ### upsample         
    for i in range(n_downsampling):
        mult = 2**(n_downsampling - i)
        model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                    norm_layer(int(ngf * mult / 2)), activation] 
    model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
    model = nn.Sequential(*model)
    return model       


# class Up_Sampling(nn.Module):
#     def __init__(self, output_nc =3, ngf=64, n_downsampling=4, norm_layer=get_norm_layer(norm_type='instance'), padding_type='reflect'):
#         super(Up_Sampling, self).__init__()

#         activation = nn.ReLU(True)             
#         model = []
#         ### upsample         
#         for i in range(n_downsampling):
#             mult = 2**(n_downsampling - i)
#             model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
#                         norm_layer(int(ngf * mult / 2)), activation]   
#         model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
#         self.model = nn.Sequential(*model)
        
#     def forward(self, input):
#         return  self.model(input)



###############################################################################
# Losses
###############################################################################

class Pixmaf_Loss():
    def __init__(self, device = 'cuda'):
        super(Pixmaf_Loss, self).__init__()
        self.device = device
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        self.kp2d_loss_dict = {}
        self.cam_loss_dict = {}

    # for openpose
    def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d, openpose_weight=1.0, gt_weight=0.0):
        """ Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        The available keypoints are different for each dataset.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        conf[:, :25] *= openpose_weight
        # conf[:, 25:] *= gt_weight
        pred_keypoints_2d = pred_keypoints_2d[:,0:25,:]
        loss = (conf * self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
        return loss

    def get_kp2d_loss(self, smpl_outs, gt_keypoints_2d, batch_size, img_res, focal_length = 5000. ,device = 'cuda'):
        smpl = SMPL(SMPL_MODEL_DIR, batch_size=batch_size, create_transl=False).to(device)
        len_loop = len(smpl_outs) # 5
        for l_i in range(len_loop):
            # Ignore first intial value
            if l_i == 0:
                continue
            pred_rotmat = smpl_outs[l_i]['rotmat']
            pred_betas = smpl_outs[l_i]['theta'][:, 3:13]
            pred_camera =  smpl_outs[l_i]['theta'][:, :3]

            pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:],
                                global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
            pred_vertices = pred_output.vertices
            pred_joints = pred_output.joints

            # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz] in 3D given the bounding box size
            # This camera translation can be used in a full perspective projection
            pred_cam_t = torch.stack([pred_camera[:,1],
                                    pred_camera[:,2],
                                    2*focal_length/(img_res * pred_camera[:,0] +1e-9)],dim=-1)

            camera_center = torch.zeros(batch_size, 2, device=device)
            pred_keypoints_2d = perspective_projection(pred_joints,
                                                    rotation=torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1),
                                                    translation=pred_cam_t,
                                                    focal_length=focal_length,
                                                    camera_center=camera_center)
            # Normalize keypoints to [-1,1]
            pred_keypoints_2d = pred_keypoints_2d / (img_res / 2.)   

            loss_keypoints = self.keypoint_loss(pred_keypoints_2d, gt_keypoints_2d) * cfg.LOSS.KP_2D_W
            self.kp2d_loss_dict['loss_keypoints_{}'.format(l_i)] = loss_keypoints

            # Camera
            # force the network to predict positive depth values
            loss_cam = ((torch.exp(-pred_camera[:,0]*10)) ** 2 ).mean()
            self.cam_loss_dict['loss_cam_{}'.format(l_i)] = loss_cam

        kp2d_loss = torch.stack(list(self.kp2d_loss_dict.values())).sum()
        cam_loss = torch.stack(list(self.cam_loss_dict.values())).sum()

        return kp2d_loss, cam_loss

    # 考虑crop_res = 384
    def get_silhouette_loss(self, crop_img, smpl_outs, crop_res=384, batch_size=1, device='cuda'):
        smpl = SMPL(SMPL_MODEL_DIR, batch_size=batch_size, create_transl=False).to(device)
        smpl_out = smpl_outs[-1]
        pred_rotmat = smpl_out['rotmat']
        pred_betas = smpl_out['theta'][:, 3:13]
        pred_camera =  smpl_out['theta'][:, :3]

        pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:],
                            global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices

        # for silhouette loss
        silhouette_model = Silhouette_model(crop_res).to(self.device)
        silhouettes_img = silhouette_model(pred_vertices,pred_camera)

        # crop_img: 1
        # silhouettes_img: 2
        crop_img_index = (crop_img>0).nonzero()
        silhouettes_img_index = (silhouettes_img>0).nonzero()
        diff_index = (((crop_img>0) == (silhouettes_img>0))==0).nonzero()

        l1 = 0
        l2_squared = 0
        
        for i in range(diff_index.shape[0]):
            a = crop_img[diff_index[i][0],diff_index[i][1]]
            b = silhouettes_img[diff_index[i][0],diff_index[i][1]]

            # 点在1上
            if a!=0 and b==0:
                l1 += self._caldist_l1(diff_index[i].float(),silhouettes_img_index.float())
        
            # 点在2上
            elif a==0 and b!= 0:
                l2_squared += self._caldist_squared_l2(diff_index[i].float(),crop_img_index.float())

            else:
                print('Error')

        silhouette_loss = (l1 + l2_squared) * cfg.LOSS.SILHOUETTES_W
        return silhouette_loss

    def _caldist_squared_l2(self,pt,pts):
        d = torch.min(torch.sum((pts - pt)*(pts - pt), dim=1))
        return d

    def _caldist_l1(self,pt,pts):
        d = torch.min(torch.sum(torch.abs(pts - pt), dim=1))
        return d


class Silhouette_model(nn.Module):
    def __init__(self, crop_res=384):
        super(Silhouette_model, self).__init__()

        # faces
        faces = get_smpl_faces()
        faces = faces.astype(np.int32)
        faces = torch.from_numpy(faces)[None, :, :]
        self.register_buffer('faces', faces)

        textures = np.load(path_config.VERTEX_TEXTURE_FILE)
        self.textures = torch.from_numpy(textures).cuda().float()

        # setup renderer
        self.focal_length = 5000
        self.render_res = crop_res
        renderer = nr.Renderer(dist_coeffs=None, orig_size=self.render_res,
                                           image_size=self.render_res,
                                           camera_mode='projection',
                                           light_intensity_ambient=1,
                                           light_intensity_directional=0,
                                           anti_aliasing=True)
        self.renderer = renderer

    def forward(self, vertices, camera):
        cam_t = torch.stack([camera[:,1], camera[:,2], 2*self.focal_length/(self.render_res * camera[:,0] +1e-9)],dim=-1)
        batch_size = vertices.shape[0] 
        K = torch.eye(3, device=vertices.device)
        K[0,0] = self.focal_length 
        K[1,1] = self.focal_length 
        K[2,2] = 1
        K[0,2] = self.render_res / 2.
        K[1,2] = self.render_res / 2.
        K = K[None, :, :].expand(batch_size, -1, -1)
        R = torch.eye(3, device=vertices.device)[None, :, :].expand(batch_size, -1, -1)
        t = cam_t.unsqueeze(1)
        self.faces = self.faces.expand(batch_size, -1, -1)

        # render_silhouettes
        silhouettes_img =  self.renderer(vertices, self.faces, textures=self.textures, mode='silhouettes',K=K, R=R, t=t)  
        return silhouettes_img
            
###############################################################################
# Render
###############################################################################

def render_smpl(smpl_output, bboxes, img, orig_width, orig_height):
    bboxes = bboxes.numpy()
    # img = img.numpy().transpose(0,2,3,1)
    pred_camera = smpl_output['theta'][:, :3].cpu().detach().numpy()
    pred_vertices = smpl_output['verts'].cpu().detach().numpy()

    # print(bboxes.shape)
    # # print(img.shape)
    # print(pred_camera.shape)
    # print(pred_vertices.shape)

    # orig_cam = convert_crop_cam_to_orig_img(
    #             cam=pred_camera,
    #             bbox=bboxes,
    #             img_width=orig_width,
    #             img_height=orig_height
    #         )

    # render
    img = np.zeros((500,500,3))
    color_type = 'purple'
    mesh_filename = None
    renderer = PyRenderer(resolution=(orig_width, orig_height))
    img = renderer(
        pred_vertices.squeeze(0),
        img=img,
        cam=pred_camera.squeeze(0),
        color_type=color_type,
        mesh_filename=mesh_filename
    )

    side_img = np.zeros_like(img)

    side_img = renderer(
        pred_vertices.squeeze(0),
        img=side_img,
        cam=pred_camera.squeeze(0),
        color_type=color_type,
        angle=270,
        axis=[0,1,0],
    )

    total_img = np.concatenate([img, side_img], axis=1)
    
    return total_img



def convert_crop_cam_to_orig_img(cam, bbox, img_width, img_height):
    '''
    Convert predicted camera from cropped image coordinates
    to original image coordinates
    :param cam (ndarray, shape=(3,)): weak perspective camera in cropped img coordinates
    :param bbox (ndarray, shape=(4,)): bbox coordinates (c_x, c_y, h)
    :param img_width (int): original image width
    :param img_height (int): original image height
    :return:
    '''
    cx, cy, h = bbox[:,0], bbox[:,1], bbox[:,2]
    hw, hh = img_width / 2., img_height / 2.
    sx = cam[:,0] * (1. / (img_width / h))
    sy = cam[:,0] * (1. / (img_height / h))
    tx = ((cx - hw) / hw / sx) + cam[:,1]
    ty = ((cy - hh) / hh / sy) + cam[:,2]
    orig_cam = np.stack([sx, sy, tx, ty]).T
    return orig_cam


def prepare_rendering_results(vibe_results, nframes):
    frame_results = [{} for _ in range(nframes)]
    for person_id, person_data in vibe_results.items():
        for idx, frame_id in enumerate(person_data['frame_ids']):
            frame_results[frame_id][person_id] = {
                'verts': person_data['verts'][idx],
                'cam': person_data['orig_cam'][idx],
                # 'cam': person_data['pred_cam'][idx],
            }

    # naive depth ordering based on the scale of the weak perspective camera
    for frame_id, frame_data in enumerate(frame_results):
        # sort based on y-scale of the cam in original image coords
        sort_idx = np.argsort([v['cam'][1] for k,v in frame_data.items()])
        frame_results[frame_id] = OrderedDict(
            {list(frame_data.keys())[i]:frame_data[list(frame_data.keys())[i]] for i in sort_idx}
        )

    return frame_results