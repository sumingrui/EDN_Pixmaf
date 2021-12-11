### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import joblib

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        ### label maps    
        self.dir_label = os.path.join(opt.dataroot, opt.phase + '_label')              
        self.label_paths = sorted(make_dataset(self.dir_label))

        ### real images
        if opt.isTrain:
            self.dir_image = os.path.join(opt.dataroot, opt.phase + '_img')  
            self.image_paths = sorted(make_dataset(self.dir_image))

            self.dir_vibe = os.path.join(opt.dataroot, 'train_vibe.pkl')  
            self.vibe_results = joblib.load(self.dir_vibe)

        ### load face bounding box coordinates size 128x128
        if opt.face_discrim or opt.face_generator:
            self.dir_facetext = os.path.join(opt.dataroot, opt.phase + '_facetexts128')
            print('----------- loading face bounding boxes from %s ----------' % self.dir_facetext)
            self.facetext_paths = sorted(make_dataset(self.dir_facetext))


        self.dataset_size = len(self.label_paths) 
      
    def __getitem__(self, index):        
        ### label maps
        paths = self.label_paths
        label_path = paths[index]              
        label = Image.open(label_path).convert('RGB')        
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label)
        original_label_path = label_path

        image_tensor = next_label = next_image = face_tensor = 0
        other_params = {}
        next_other_params = {}
        ### real images 
        if self.opt.isTrain:
            image_path = self.image_paths[index]   
            image = Image.open(image_path).convert('RGB')    
            transform_image = get_transform(self.opt, params)     
            image_tensor = transform_image(image).float()

            # 添加bbox
            other_params['bboxes'] = self.vibe_results[1]['bboxes'][index] #(4,)

            other_params['pred_cam'] = self.vibe_results[1]['pred_cam'][index] #(3,)
            other_params['orig_cam'] = self.vibe_results[1]['orig_cam'][index]
            other_params['pose'] = self.vibe_results[1]['pose'][index] #(72,)
            other_params['betas'] = self.vibe_results[1]['betas'][index] #(10,)
            # unused
            other_params['verts'] = self.vibe_results[1]['verts'][index] #(6890, 3)
            other_params['joints3d'] = self.vibe_results[1]['joints3d'][index] #(49, 3)
            other_params['frame_ids'] = self.vibe_results[1]['frame_ids'][index]
            other_params['kp_2d'] = self.vibe_results[1]['kp_2d'][index]

        is_next = index < len(self) - 1
        if self.opt.gestures:
            is_next = is_next and (index % 64 != 63)

        """ Load the next label, image pair """
        if is_next:

            paths = self.label_paths
            label_path = paths[index+1]              
            label = Image.open(label_path).convert('RGB')        
            params = get_params(self.opt, label.size)          
            transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            next_label = transform_label(label).float()
            
            if self.opt.isTrain:
                image_path = self.image_paths[index+1]   
                image = Image.open(image_path).convert('RGB')
                transform_image = get_transform(self.opt, params)      
                next_image = transform_image(image).float()

                # 添加bbox
                next_other_params['bboxes'] = self.vibe_results[1]['bboxes'][index+1] #(4,)

                next_other_params['pred_cam'] = self.vibe_results[1]['pred_cam'][index] #(3,)
                next_other_params['orig_cam'] = self.vibe_results[1]['orig_cam'][index]
                next_other_params['pose'] = self.vibe_results[1]['pose'][index] #(72,)
                next_other_params['betas'] = self.vibe_results[1]['betas'][index] #(10,)
                # unused
                next_other_params['verts'] = self.vibe_results[1]['verts'][index] #(6890, 3)
                next_other_params['joints3d'] = self.vibe_results[1]['joints3d'][index] #(49, 3)
                next_other_params['frame_ids'] = self.vibe_results[1]['frame_ids'][index]
                next_other_params['kp_2d'] = self.vibe_results[1]['kp_2d'][index]

        """ If using the face generator and/or face discriminator """
        if self.opt.face_discrim or self.opt.face_generator:
            facetxt_path = self.facetext_paths[index]
            facetxt = open(facetxt_path, "r")
            face_tensor = torch.IntTensor(list([int(coord_str) for coord_str in facetxt.read().split()]))

        input_dict = {'label': label_tensor.float(), 'image': image_tensor, 'other_params':other_params,
                      'path': original_label_path, 'face_coords': face_tensor,
                      'next_label': next_label, 'next_image': next_image, 'next_other_params':next_other_params }
        return input_dict

    def __len__(self):
        return len(self.label_paths)

    def name(self):
        return 'AlignedDataset'