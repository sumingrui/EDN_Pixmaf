# neural renderer

import torch
from torch import nn
import numpy as np
import neural_renderer as nr
from .smpl import get_smpl_faces
from core import path_config

class Render_layer(nn.Module):
    def __init__(self):
        super(Render_layer, self).__init__()

        # faces
        faces = get_smpl_faces()
        faces = faces.astype(np.int32)
        faces = torch.from_numpy(faces)[None, :, :]
        self.register_buffer('faces', faces)

        textures = np.load(path_config.VERTEX_TEXTURE_FILE)
        self.textures = torch.from_numpy(textures).cuda().float()

        # setup renderer
        self.focal_length = 5000
        self.render_res = 224
        renderer = nr.Renderer(dist_coeffs=None, orig_size=self.render_res,
                                           image_size=self.render_res,
                                           camera_mode='projection',
                                           light_intensity_ambient=1,
                                           light_intensity_directional=0,
                                           anti_aliasing=True)
        self.renderer = renderer

    def forward(self, vertices, camera):

        # print(camera.dtype)
        # print(camera.shape) # torch.Size([1, 3])
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
        # render
        rgb_img,depth_img,alpha_img =  self.renderer(vertices, self.faces, textures=self.textures, mode=None, K=K, R=R, t=t)  
        return rgb_img,depth_img,alpha_img

        # render_silhouettes
        # silhouettes_img =  self.renderer(vertices, self.faces, textures=self.textures, mode='silhouettes',K=K, R=R, t=t)  
        # return silhouettes_img