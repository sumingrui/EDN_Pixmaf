### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### From PIX2PIXHD

from tokenize import Exponent
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
from .pred_cam_to_orig_cam import convert_crop_cam_to_orig_img, Renderer

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

def move_dict_to_device(dict, device, tensor2float=False):
    for k,v in dict.items():
        if isinstance(v, torch.Tensor):
            if tensor2float:
                dict[k] = v.float().to(device)
            else:
                dict[k] = v.to(device)

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
        self.enc_loss = batch_encoder_disc_l2_loss
        self.dec_loss = batch_adv_disc_l2_loss

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

    # for motion_discriminator
    def get_motion_disc_loss(self, pred_motion, motion_discriminator, data_motion_mosh):
        # pred_motion [1,2,85] [batchsize,sequence_len,theta]
        end_idx = 75
        start_idx = 6
        g_motion_disc_loss = self.enc_loss(motion_discriminator(pred_motion[:, :, start_idx:end_idx]))
        g_motion_disc_loss = g_motion_disc_loss * cfg.LOSS.D_MOTION_LOSS_W

        fake_motion = pred_motion.detach()
        real_motion = data_motion_mosh['theta']
        fake_disc_value = motion_discriminator(fake_motion[:, :, start_idx:end_idx])
        real_disc_value = motion_discriminator(real_motion[:, :, start_idx:end_idx])
        _, _, d_motion_disc_loss = self.dec_loss(real_disc_value, fake_disc_value)
        d_motion_disc_loss = d_motion_disc_loss * cfg.LOSS.D_MOTION_LOSS_W

        return g_motion_disc_loss, d_motion_disc_loss


    # for silhouette 考虑crop_res = 384
    def get_silhouette_loss(self, crop_img, smpl_out, crop_res=384, batch_size=1, device='cuda'):
        smpl = SMPL(SMPL_MODEL_DIR, batch_size=batch_size, create_transl=False).to(device)
        pred_rotmat = smpl_out['rotmat']
        pred_betas = smpl_out['theta'][:, 3:13]
        pred_camera =  smpl_out['theta'][:, :3]

        pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:],
                            global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices

        # for silhouette loss
        silhouette_model = Silhouette_model(crop_res).to(self.device)
        silhouettes_img = silhouette_model(pred_vertices,pred_camera).squeeze(0)

        # crop_img: 1
        # silhouettes_img: 2
        crop_img_index = (crop_img>0).nonzero(as_tuple=False)
        silhouettes_img_index = (silhouettes_img>0).nonzero(as_tuple=False)
        diff_index = (((crop_img>0) == (silhouettes_img>0))==0).nonzero(as_tuple=False)

        l1 = 0
        l2_squared = 0
        
        for i in range(diff_index.shape[0]):
            a = crop_img[diff_index[i][0],diff_index[i][1]]
            b = silhouettes_img[diff_index[i][0],diff_index[i][1]]

            # 点在1上
            if a!=0 and b==0:
                try:    
                    l1 += self._caldist_l1(diff_index[i].float(),silhouettes_img_index.float())
                except:
                    print('------------------SIL L1 ERROR------------------')
                    print('a: ',a)
                    print('b: ',b)
                    print('diff_index[i]: ',diff_index[i].float())
                    print('silhouettes_img_index: ',silhouettes_img_index.shape)
                    print('-----------------------END-----------------------')
        
            # 点在2上
            elif a==0 and b!= 0:
                try:
                    l2_squared += self._caldist_squared_l2(diff_index[i].float(),crop_img_index.float())
                except:
                    print('------------------SIL L2 ERROR------------------')
                    print('a: ',a)
                    print('b: ',b)
                    print('diff_index[i]: ',diff_index[i].float())
                    print('crop_img_index: ',crop_img_index.shape)
                    print('-----------------------END-----------------------')

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


def batch_encoder_disc_l2_loss(disc_value):
    '''
        Inputs:
            disc_value: N x 25
    '''
    k = disc_value.shape[0]
    return torch.sum((disc_value - 1.0) ** 2) * 1.0 / k

def batch_adv_disc_l2_loss(real_disc_value, fake_disc_value):
    '''
        Inputs:
            disc_value: N x 25
    '''
    ka = real_disc_value.shape[0]
    kb = fake_disc_value.shape[0]
    lb, la = torch.sum(fake_disc_value ** 2) / kb, torch.sum((real_disc_value - 1) ** 2) / ka
    return la, lb, la + lb  
            
###############################################################################
# Render
###############################################################################

def render_smpl(smpl_output, bboxes, imgs, orig_width=512, orig_height=256):
    '''
    bboxes (2,4)
    pred_camera (2,3)
    pred_vertices (2,6890,3)
    '''
    bboxes = bboxes.numpy()
    # img = img.numpy().transpose(0,2,3,1)
    pred_camera = torch.cat((smpl_output[0]['theta'][:, :3],smpl_output[1]['theta'][:, :3]),dim=0).cpu().detach().numpy()
    pred_vertices = torch.cat((smpl_output[0]['verts'],smpl_output[1]['verts']),dim=0).cpu().detach().numpy()

    orig_cam = convert_crop_cam_to_orig_img(
            cam=pred_camera,
            bbox=bboxes,
            img_width=orig_width,
            img_height=orig_height
        )

    # render
    renderer = Renderer(resolution=(orig_width, orig_height), orig_img=True, wireframe=True)
    mesh_filename = None
    img_render_0 = renderer.render(
                    imgs[0],
                    pred_vertices[0],
                    cam=orig_cam[0],
                    color=(0,255,0),
                    mesh_filename=mesh_filename,
                )

    img_render_1 = renderer.render(
                    imgs[1],
                    pred_vertices[1],
                    cam=orig_cam[1],
                    color=(0,255,0),
                    mesh_filename=mesh_filename,
                )
   
    return img_render_0, img_render_1