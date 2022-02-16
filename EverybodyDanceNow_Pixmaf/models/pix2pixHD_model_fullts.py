### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import torch
import os
from collections import OrderedDict
import pickle
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

# Pixmaf
from .pixmaf_net.models.networks import Pixmaf_Loss, Silhouette_model
from .pixmaf_net.models.motion_discriminator import MotionDiscriminator
from .pixmaf_net.core.cfgs import cfg
from .pixmaf_net.utils.geometry import estimate_translation
from .pixmaf_net.smplify import SMPLify
from .pixmaf_net.core.constants import FOCAL_LENGTH

class Pix2PixHDModel(BaseModel):
    def name(self):
        return 'Pix2PixHDModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none': # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        # 记录使用SPIN方法是否有更新opt数据
        self.countOPT = 0

        ##### define networks        
        # Generator network
        netG_input_nc = opt.label_nc
        if not opt.no_instance:
            netG_input_nc += 1          
        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG, 
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers, 
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)        

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = 4*opt.output_nc
            if not opt.no_instance:
                netD_input_nc += 1
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, 
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

        # Face discriminator network
        if self.isTrain and opt.face_discrim:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = 2*opt.output_nc
            if not opt.no_instance:
                netD_input_nc += 1
            self.netDface = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, 
                                          1, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids, netD='face')

        #Face residual network
        if opt.face_generator:
            if opt.faceGtype == 'unet':
                self.faceGen = networks.define_G(opt.output_nc*2, opt.output_nc, 32, 'unet', 
                                          n_downsample_global=2, n_blocks_global=5, n_local_enhancers=0, 
                                          n_blocks_local=0, norm=opt.norm, gpu_ids=self.gpu_ids)
            elif opt.faceGtype == 'global':
                self.faceGen = networks.define_G(opt.output_nc*2, opt.output_nc, 64, 'global', 
                                      n_downsample_global=3, n_blocks_global=5, n_local_enhancers=0, 
                                      n_blocks_local=0, norm=opt.norm, gpu_ids=self.gpu_ids)
            else:
                raise('face generator not implemented!')

        # Body SMPL AMASS discriminator network
        if opt.use_pixmaf: 
            self.netDmotion = MotionDiscriminator(
                rnn_size=cfg.TRAIN.MOT_DISCR.HIDDEN_SIZE,
                input_size=69,
                num_layers=cfg.TRAIN.MOT_DISCR.NUM_LAYERS,
                output_size=1,
                feature_pool=cfg.TRAIN.MOT_DISCR.FEATURE_POOL,
                attention_size=None if cfg.TRAIN.MOT_DISCR.FEATURE_POOL !='attention' else cfg.TRAIN.MOT_DISCR.ATT.SIZE,
                attention_layers=None if cfg.TRAIN.MOT_DISCR.FEATURE_POOL !='attention' else cfg.TRAIN.MOT_DISCR.ATT.LAYERS,
                attention_dropout=None if cfg.TRAIN.MOT_DISCR.FEATURE_POOL !='attention' else cfg.TRAIN.MOT_DISCR.ATT.DROPOUT
            ).to(cfg.DEVICE)
            print(self.netDmotion)
        
            # Initialize SMPLify fitting module
            self.smplify = SMPLify(step_size=1e-2, batch_size=cfg.SMPLIFY.BATCH_SIZE, num_iters=cfg.SMPLIFY.NUM_ITER, \
                                    focal_length=FOCAL_LENGTH, device=cfg.DEVICE)
            
            # Initialize best fits
            file_best_fits = os.path.join(opt.dataroot, 'best_fits.pkl')
            with open(file_best_fits,'rb') as f:
                self.best_fits = pickle.load(f)

        print('---------- Networks initialized -------------')

        # load networks
        if (not self.isTrain or opt.continue_train or opt.load_pretrain):
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)            
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)
                self.load_network(self.netDmotion, 'Dmotion', opt.which_epoch, pretrained_path)
                if opt.face_discrim:
                    self.load_network(self.netDface, 'Dface', opt.which_epoch, pretrained_path)
            if opt.face_generator:
                self.load_network(self.faceGen, 'Gface', opt.which_epoch, pretrained_path)

        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)   
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:             
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)
            if opt.use_l1:
                self.criterionL1 = torch.nn.L1Loss()

            # loss for pixmaf
            self.criterionPixmaf = Pixmaf_Loss()
        
            # Loss names
            self.loss_names = ['G_GAN', 'G_GAN_Feat', 'G_VGG', \
                                'G_2DKP', 'G_CAM', 'G_SMPL', 'G_VERTS', \
                                'G_SIL', 'G_MOTION', 'G_SHAPECOH', \
                                'D_real', 'D_fake', 'D_MOTION',\
                                'G_GANface', 'D_realface', 'D_fakeface']

            # initialize optimizers
            # optimizer G
            if opt.niter_fix_global > 0:
                print('------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
                params_dict = dict(self.netG.named_parameters())
                G_params = []
                for key, value in params_dict.items():       
                    if key.startswith('model' + str(opt.n_local_enhancers)):
                        G_params += [{'params':[value],'lr':opt.lr}]
                    else:
                        G_params += [{'params':[value],'lr':0.0}]                            
            else:
                G_params = list(self.netG.parameters())

            if opt.face_generator:
                G_params = list(self.faceGen.parameters())
            else:
                if opt.niter_fix_main == 0:
                    # TODO 这里有没有问题
                    pass
                    #G_params += list(self.netG.parameters())
            # 选择更新的参数
            # self.optimizer_G = torch.optim.Adam(G_params, lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_G = torch.optim.Adam([
                {'params': [param for name, param in self.netG.named_parameters() if (name.startswith('feature_extractor') or name.startswith('deconv_layers'))]},
                {'params': [param for name, param in self.netG.named_parameters() if (name.startswith('maf_extractor') or name.startswith('regressor'))],'betas':(0.9, 0.999)}],
                lr=opt.lr, 
                betas=(opt.beta1, 0.999),
                weight_decay=0,
                )

            # optimizer D
            if opt.niter_fix_main > 0:
                print('------------- Only training the face discriminator network (for %d epochs) ------------' % opt.niter_fix_main)
                D_params = list(self.netDface.parameters())                         
            else:
                if opt.face_discrim:
                    D_params = list(self.netD.parameters()) + list(self.netDface.parameters())   
                else:
                    D_params = list(self.netD.parameters())   

            if self.opt.use_pixmaf: 
                D_params_motion = list(self.netDmotion.parameters())                 

            # optimizer add D motion
            self.optimizer_D = torch.optim.Adam([
                {'params':D_params},
                {'params':D_params_motion,'lr':opt.lr_Dmotion,'betas':(0.9, 0.999),'weight_decay':0.0001}],
                lr=opt.lr, 
                betas=(opt.beta1, 0.999),
                weight_decay=0,
                )

            # optimizer without D motion
            self.optimizer_D_wo_motion = torch.optim.Adam(D_params, lr=opt.lr, betas=(opt.beta1, 0.999))

    def encode_input(self, label_map, real_image=None, next_label=None, next_image=None, zeroshere=None, \
                        other_params=None, next_other_params=None, infer=False):

        input_label = label_map.data.float().cuda()
        input_label = Variable(input_label, volatile=infer)

        # next label for training
        if next_label is not None:
            next_label = next_label.data.float().cuda()
            next_label = Variable(next_label, volatile=infer)

        # real images for training
        if real_image is not None:
            real_image = Variable(real_image.data.float().cuda())

        # real images for training
        if next_image is not None:
            next_image = Variable(next_image.data.float().cuda())

        if zeroshere is not None:
            zeroshere = zeroshere.data.float().cuda()
            zeroshere = Variable(zeroshere, volatile=infer)
        
        if other_params is not None:
            for k in other_params.keys():
                other_params[k] = Variable(other_params[k].data.float().cuda())

        if next_other_params is not None:
            for k in next_other_params.keys():
                next_other_params[k] = Variable(next_other_params[k].data.float().cuda())

        return input_label, real_image, next_label, next_image, zeroshere, other_params, next_other_params

    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def discriminate_4(self, s0, s1, i0, i1, use_pool=False):
        input_concat = torch.cat((s0, s1, i0.detach(), i1.detach()), dim=1)
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def discriminateface(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.netDface.forward(fake_query)
        else:
            return self.netDface.forward(input_concat)
    

    def forward(self, label, next_label, image, next_image, face_coords, zeroshere, \
                other_params, next_other_params, data_motion_mosh, infer=False):
        # Encode Inputs
        input_label, real_image, next_label, next_image, zeroshere, other_params, next_other_params = self.encode_input(label, image, \
                     next_label=next_label, next_image=next_image, zeroshere=zeroshere, \
                     other_params = other_params, next_other_params = next_other_params)
        if self.opt.face_discrim:
            miny = face_coords.data[0][0]
            maxy = face_coords.data[0][1]
            minx = face_coords.data[0][2]
            maxx = face_coords.data[0][3]

        initial_I_0 = 0

        # Fake Generation I_0
        input_concat = torch.cat((input_label, zeroshere), dim=1) 

        #face residual for I_0
        face_residual_0 = 0
        if self.opt.face_generator:
            initial_I_0 = self.netG.forward(input_concat)
            face_label_0 = input_label[:, :, miny:maxy, minx:maxx]
            face_residual_0 = self.faceGen.forward(torch.cat((face_label_0, initial_I_0[:, :, miny:maxy, minx:maxx]), dim=1))
            I_0 = initial_I_0.clone()
            I_0[:, :, miny:maxy, minx:maxx] = initial_I_0[:, :, miny:maxy, minx:maxx] + face_residual_0
        else:
            Outlist_0 = self.netG.forward(input_concat,other_params)
            I_0 = Outlist_0['Generator']
            S_0 = Outlist_0['smpl_out']

        input_concat1 = torch.cat((next_label, I_0), dim=1)

        #face residual for I_1
        face_residual_1 = 0
        if self.opt.face_generator:
            initial_I_1 = self.netG.forward(input_concat1)
            face_label_1 = next_label[:, :, miny:maxy, minx:maxx]
            face_residual_1 = self.faceGen.forward(torch.cat((face_label_1, initial_I_1[:, :, miny:maxy, minx:maxx]), dim=1))
            I_1 = initial_I_1.clone()
            I_1[:, :, miny:maxy, minx:maxx] = initial_I_1[:, :, miny:maxy, minx:maxx] + face_residual_1
        else:
            Outlist_1 = self.netG.forward(input_concat1,next_other_params)
            I_1 = Outlist_1['Generator']
            S_1 = Outlist_1['smpl_out']

        loss_D_fake_face = loss_D_real_face = loss_G_GAN_face = 0
        fake_face_0 = fake_face_1 = real_face_0 = real_face_1 = 0
        fake_face = real_face = face_residual = 0
        if self.opt.face_discrim:

            fake_face_0 = I_0[:, :, miny:maxy, minx:maxx]
            fake_face_1 = I_1[:, :, miny:maxy, minx:maxx]
            real_face_0 = real_image[:, :, miny:maxy, minx:maxx]
            real_face_1 = next_image[:, :, miny:maxy, minx:maxx]

            # Fake Detection and Loss
            pred_fake_pool_face = self.discriminateface(face_label_0, fake_face_0, use_pool=True)
            loss_D_fake_face += 0.5 * self.criterionGAN(pred_fake_pool_face, False)

            # Face Real Detection and Loss        
            pred_real_face = self.discriminateface(face_label_0, real_face_0)
            loss_D_real_face += 0.5 * self.criterionGAN(pred_real_face, True)

            # Face GAN loss (Fake Passability Loss)        
            pred_fake_face = self.netDface.forward(torch.cat((face_label_0, fake_face_0), dim=1))        
            loss_G_GAN_face += 0.5 * self.criterionGAN(pred_fake_face, True)

            pred_fake_pool_face = self.discriminateface(face_label_1, fake_face_1, use_pool=True)
            loss_D_fake_face += 0.5 * self.criterionGAN(pred_fake_pool_face, False)

            # Face Real Detection and Loss        
            pred_real_face = self.discriminateface(face_label_1, real_face_1)
            loss_D_real_face += 0.5 * self.criterionGAN(pred_real_face, True)

            # Face GAN loss (Fake Passability Loss)        
            pred_fake_face = self.netDface.forward(torch.cat((face_label_1, fake_face_1), dim=1))        
            loss_G_GAN_face += 0.5 * self.criterionGAN(pred_fake_face, True)

            fake_face = torch.cat((fake_face_0, fake_face_1), dim=3)
            real_face = torch.cat((real_face_0, real_face_1), dim=3)

            if self.opt.face_generator:
                face_residual = torch.cat((face_residual_0, face_residual_1), dim=3)

        # Fake Detection and Loss
        # I_0, I_1 detach 不管G，只看D
        pred_fake_pool = self.discriminate_4(input_label, next_label, I_0, I_1, use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)        

        # Real Detection and Loss       
        # real_image, next_image detach
        pred_real = self.discriminate_4(input_label, next_label, real_image, next_image)
        loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)        
        pred_fake = self.netD.forward(torch.cat((input_label, next_label, I_0, I_1), dim=1))        
        loss_G_GAN = self.criterionGAN(pred_fake, True)
        
        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
                   
        # VGG feature matching loss
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG0 = self.criterionVGG(I_0, real_image) * self.opt.lambda_feat
            loss_G_VGG1 = self.criterionVGG(I_1, next_image) * self.opt.lambda_feat
            loss_G_VGG = loss_G_VGG0 + loss_G_VGG1 
            if self.opt.netG == 'global': #need 2x VGG for artifacts when training local
                loss_G_VGG *= 0.5
            if self.opt.face_discrim:
                loss_G_VGG += 0.5 * self.criterionVGG(fake_face_0, real_face_0) * self.opt.lambda_feat
                loss_G_VGG += 0.5 * self.criterionVGG(fake_face_1, real_face_1) * self.opt.lambda_feat

        if self.opt.use_l1:
            loss_G_VGG += (self.criterionL1(I_1, next_image)) * self.opt.lambda_A

        # 开始pixMAF相关的loss的训练

        # 从best_fits.pkl获得opt数据用于训练
        
        blank_keypoints_2d = torch.zeros((self.opt.batchSize,24,3)).to(cfg.DEVICE)
        gt_keypoints_2d_0 = torch.cat((other_params['openpose_kp_2d'],blank_keypoints_2d),dim=1) # torch.Size([1, 25, 3]) + torch.Size([1, 24, 3])
        gt_keypoints_2d_1 = torch.cat((next_other_params['openpose_kp_2d'],blank_keypoints_2d),dim=1)
        # gt_keypoints_2d_0 = other_params['openpose_kp_2d'] 
        # gt_keypoints_2d_1 = next_other_params['openpose_kp_2d']

        img_res_0 = int(other_params['bboxes'][0][2])
        img_res_1 = int(next_other_params['bboxes'][0][2])

        frame_id_0 = int(other_params['frame_ids'].cpu().detach().numpy())
        frame_id_1 = int(next_other_params['frame_ids'].cpu().detach().numpy())

        opt_pose_0 = torch.from_numpy(self.best_fits['pose'][frame_id_0]).to(cfg.DEVICE).unsqueeze(0) # torch.Size([1, 72])
        opt_pose_1 = torch.from_numpy(self.best_fits['pose'][frame_id_1]).to(cfg.DEVICE).unsqueeze(0)

        opt_beta_0 = torch.from_numpy(self.best_fits['betas'][frame_id_0]).to(cfg.DEVICE).unsqueeze(0) # torch.Size([1, 10])
        opt_beta_1 = torch.from_numpy(self.best_fits['betas'][frame_id_1]).to(cfg.DEVICE).unsqueeze(0)

        opt_joints_0 = torch.from_numpy(self.best_fits['joints3d'][frame_id_0]).to(cfg.DEVICE).unsqueeze(0) # torch.Size([1, 49, 3])
        opt_joints_1 = torch.from_numpy(self.best_fits['joints3d'][frame_id_1]).to(cfg.DEVICE).unsqueeze(0)

        opt_vertices_0 = torch.from_numpy(self.best_fits['verts'][frame_id_0]).to(cfg.DEVICE).unsqueeze(0) # torch.Size([1, 6890, 3])
        opt_vertices_1 = torch.from_numpy(self.best_fits['verts'][frame_id_1]).to(cfg.DEVICE).unsqueeze(0)
   
        # 这里因为img_res不一样 因此只能一个一个算
        opt_cam_t_0 = estimate_translation(opt_joints_0, gt_keypoints_2d_0, focal_length=FOCAL_LENGTH, img_size=img_res_0)
        opt_cam_t_1 = estimate_translation(opt_joints_1, gt_keypoints_2d_1, focal_length=FOCAL_LENGTH, img_size=img_res_1)

        # 得到reprojection_loss
        opt_joint_loss_0 = self.smplify.get_fitting_loss(opt_pose_0, opt_beta_0, opt_cam_t_0,
                                                       0.5 * img_res_0 * torch.ones(cfg.SMPLIFY.BATCH_SIZE, 2, device=cfg.DEVICE),
                                                       gt_keypoints_2d_0).mean(dim=-1)
        
        opt_joint_loss_1 = self.smplify.get_fitting_loss(opt_pose_1, opt_beta_1, opt_cam_t_1,
                                                       0.5 * img_res_1 * torch.ones(cfg.SMPLIFY.BATCH_SIZE, 2, device=cfg.DEVICE),
                                                       gt_keypoints_2d_1).mean(dim=-1)
        
        if self.opt.run_smplify:
            pred_pose_0 = S_0[-1]['theta'][:, 13:]
            pred_pose_1 = S_1[-1]['theta'][:, 13:]
            pred_beta_0 = S_0[-1]['pred_shape']
            pred_beta_1 = S_1[-1]['pred_shape']
            pred_cam_0 = S_0[-1]['theta'][:, :3]
            pred_cam_1 = S_1[-1]['theta'][:, :3]
            pred_cam_t_0 = torch.stack([pred_cam_0[:,1],
                                    pred_cam_0[:,2],
                                    2*FOCAL_LENGTH/(img_res_0 * pred_cam_0[:,0] +1e-9)],dim=-1)
            pred_cam_t_1 = torch.stack([pred_cam_1[:,1],
                                    pred_cam_1[:,2],
                                    2*FOCAL_LENGTH/(img_res_1 * pred_cam_1[:,0] +1e-9)],dim=-1)
            # De-normalize 2D keypoints from [-1,1] to pixel space
            gt_keypoints_2d_orig_0 = gt_keypoints_2d_0.clone()
            gt_keypoints_2d_orig_0[:, :, :-1] = 0.5 * img_res_0 * (gt_keypoints_2d_orig_0[:, :, :-1] + 1)
            gt_keypoints_2d_orig_1 = gt_keypoints_2d_1.clone()
            gt_keypoints_2d_orig_1[:, :, :-1] = 0.5 * img_res_1 * (gt_keypoints_2d_orig_1[:, :, :-1] + 1)

            # ================= SMPL OUTPUT 0 =================
            # Run SMPLify optimization starting from the network prediction
            new_opt_vertices_0, new_opt_joints_0,\
            new_opt_pose_0, new_opt_betas_0,\
            new_opt_cam_t_0, new_opt_joint_loss_0 = self.smplify(
                                        pred_pose_0.detach(), pred_beta_0.detach(),
                                        pred_cam_t_0.detach(),
                                        0.5 * img_res_0 * torch.ones(cfg.SMPLIFY.BATCH_SIZE, 2, device=cfg.DEVICE),
                                        gt_keypoints_2d_orig_0)
            new_opt_joint_loss_0 = new_opt_joint_loss_0.mean(dim=-1)

            # Will update the dictionary for the examples where the new loss is less than the current one
            update_0 = (new_opt_joint_loss_0 < opt_joint_loss_0) # tensor([False], device='cuda:0')

            if update_0.detach().item():
                opt_joint_loss_0[update_0] = new_opt_joint_loss_0[update_0]
                opt_vertices_0[update_0, :] = new_opt_vertices_0[update_0, :]
                opt_joints_0[update_0, :] = new_opt_joints_0[update_0, :]
                opt_pose_0[update_0, :] = new_opt_pose_0[update_0, :]
                opt_beta_0[update_0, :] = new_opt_betas_0[update_0, :]
                opt_cam_t_0[update_0, :] = new_opt_cam_t_0[update_0, :]

                self.best_fits['pose'][frame_id_0] = opt_pose_0.squeeze(0).cpu().detach().numpy()
                self.best_fits['betas'][frame_id_0] = opt_beta_0.squeeze(0).cpu().detach().numpy()
                self.best_fits['joints3d'][frame_id_0] = opt_joints_0.squeeze(0).cpu().detach().numpy()
                self.best_fits['verts'][frame_id_0] = opt_vertices_0.squeeze(0).cpu().detach().numpy()

                self.countOPT += 1

            # ================= SMPL OUTPUT 1 =================
            # Run SMPLify optimization starting from the network prediction
            new_opt_vertices_1, new_opt_joints_1,\
            new_opt_pose_1, new_opt_betas_1,\
            new_opt_cam_t_1, new_opt_joint_loss_1 = self.smplify(
                                        pred_pose_1.detach(), pred_beta_1.detach(),
                                        pred_cam_t_1.detach(),
                                        0.5 * img_res_1 * torch.ones(cfg.SMPLIFY.BATCH_SIZE, 2, device=cfg.DEVICE),
                                        gt_keypoints_2d_orig_1)
            new_opt_joint_loss_1 = new_opt_joint_loss_1.mean(dim=-1)

            # Will update the dictionary for the examples where the new loss is less than the current one
            update_1 = (new_opt_joint_loss_1 < opt_joint_loss_1) # tensor([False], device='cuda:0')

            if update_1.detach().item():
                opt_joint_loss_1[update_1] = new_opt_joint_loss_1[update_1]
                opt_vertices_1[update_1, :] = new_opt_vertices_1[update_1, :]
                opt_joints_1[update_1, :] = new_opt_joints_1[update_1, :]
                opt_pose_1[update_1, :] = new_opt_pose_1[update_1, :]
                opt_beta_1[update_1, :] = new_opt_betas_1[update_1, :]
                opt_cam_t_1[update_1, :] = new_opt_cam_t_1[update_1, :]

                self.best_fits['pose'][frame_id_1] = opt_pose_1.squeeze(0).cpu().detach().numpy()
                self.best_fits['betas'][frame_id_1] = opt_beta_1.squeeze(0).cpu().detach().numpy()
                self.best_fits['joints3d'][frame_id_1] = opt_joints_1.squeeze(0).cpu().detach().numpy()
                self.best_fits['verts'][frame_id_1] = opt_vertices_1.squeeze(0).cpu().detach().numpy()

                self.countOPT += 1
        
        # Replace extreme betas with zero betas
        opt_beta_0[(opt_beta_0.abs() > 3).any(dim=-1)] = 0.
        opt_beta_1[(opt_beta_1.abs() > 3).any(dim=-1)] = 0.

        # Assert whether a fit is valid by comparing the joint loss with the threshold
        valid_fit_0 = (opt_joint_loss_0 < cfg.SMPLIFY.THRESHOLD).to(cfg.DEVICE)
        valid_fit_1 = (opt_joint_loss_1 < cfg.SMPLIFY.THRESHOLD).to(cfg.DEVICE)

        # add:
        # keypoints 2d loss, camera loss, smpl loss, vertex loss
        loss_G_kp2d = 0
        loss_G_cam = 0
        loss_G_smpl = 0
        loss_G_verts = 0
        if self.opt.use_pixmaf:    
            loss_G_kp2d_0, loss_G_cam_0, loss_G_smpl_0, loss_G_verts_0 = \
                self.criterionPixmaf.get_losses(S_0, gt_keypoints_2d_0, self.opt.batchSize, img_res_0, opt_pose_0, opt_beta_0, opt_vertices_0, valid_fit_0)
            loss_G_kp2d_1, loss_G_cam_1, loss_G_smpl_1, loss_G_verts_1 = \
                self.criterionPixmaf.get_losses(S_1, gt_keypoints_2d_1, self.opt.batchSize, img_res_1, opt_pose_1, opt_beta_1, opt_vertices_1, valid_fit_1)
            loss_G_kp2d = (loss_G_kp2d_0 + loss_G_kp2d_1)*0.5
            loss_G_cam = (loss_G_cam_0 + loss_G_cam_1)*0.5
            loss_G_smpl = (loss_G_smpl_0 + loss_G_smpl_1)*0.5
            loss_G_verts = (loss_G_verts_0 + loss_G_verts_1)*0.5

        # 加入silhouette loss
        loss_G_silhouette = 0
        if self.opt.use_pixmaf and  self.opt.use_silhouette:
            # 计算剪影loss
            silhouette_loss0 = 0
            silhouette_loss1 = 0
            # for smpl_out_0 in S_0:
            #     silhouette_loss0 += self.criterionPixmaf.get_silhouette_loss(other_params['silhouette'].squeeze(0),smpl_out_0)
            # for smpl_out_1 in S_1:
            #     silhouette_loss1 += self.criterionPixmaf.get_silhouette_loss(next_other_params['silhouette'].squeeze(0),smpl_out_1)
            
            smpl_out_0 = S_0[-1]
            silhouette_loss0 = self.criterionPixmaf.get_silhouette_loss(other_params['silhouette'].squeeze(0),smpl_out_0)
            smpl_out_1 = S_1[-1]
            silhouette_loss1 = self.criterionPixmaf.get_silhouette_loss(next_other_params['silhouette'].squeeze(0),smpl_out_1)
            loss_G_silhouette = (silhouette_loss0+silhouette_loss1)*0.5     
        
        # 加入beta一致loss
        loss_G_shapeCoherence = 0
        if self.opt.use_pixmaf and self.opt.use_shapeCoherence:  
            loss_G_shapeCoherence = self.criterionPixmaf.get_betas_coherence_loss(S_0[-1]['pred_shape'],S_1[-1]['pred_shape'])
        
        # 加入motion_discriminator
        loss_G_motion = 0
        loss_D_motion = 0
        if self.opt.use_pixmaf:  
            pred_motion = torch.cat((S_0[-1]['theta'],S_1[-1]['theta']),dim=0).unsqueeze(0)
            loss_G_motion, loss_D_motion = self.criterionPixmaf.get_motion_disc_loss(pred_motion, self.netDmotion, data_motion_mosh)

        # Only return the fake_B image if necessary to save BW
        return [ [ loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_G_kp2d, loss_G_cam, loss_G_smpl, loss_G_verts, \
                    loss_G_silhouette, loss_G_motion, loss_G_shapeCoherence,\
                    loss_D_real, loss_D_fake, loss_D_motion,\
                    loss_G_GAN_face, loss_D_real_face,  loss_D_fake_face], \
                        None if not infer else [torch.cat((I_0, I_1), dim=3), fake_face, face_residual, initial_I_0, \
                                                [S_0[-1], S_1[-1]] ] ]

    def inference(self, label, prevouts, face_coords):

        # Encode Inputs        
        input_label, _, _, _, prevouts = self.encode_input(Variable(label), zeroshere=Variable(prevouts), infer=True)

        if self.opt.face_generator:
            miny = face_coords[0][0]
            maxy = face_coords[0][1]
            minx = face_coords[0][2]
            maxx = face_coords[0][3]

        """ new face """
        I_0 = 0
        # Fake Generation

        input_concat = torch.cat((input_label, prevouts), dim=1) 
        initial_I_0 = self.netG.forward(input_concat)

        if self.opt.face_generator:
            face_label_0 = input_label[:, :, miny:maxy, minx:maxx]
            face_residual_0 = self.faceGen.forward(torch.cat((face_label_0, initial_I_0[:, :, miny:maxy, minx:maxx]), dim=1))
            I_0 = initial_I_0.clone()
            I_0[:, :, miny:maxy, minx:maxx] = initial_I_0[:, :, miny:maxy, minx:maxx] + face_residual_0
            fake_face_0 = I_0[:, :, miny:maxy, minx:maxx]
            return I_0
        return initial_I_0

    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        return edge.float()

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        self.save_network(self.netDmotion, 'Dmotion', which_epoch, self.gpu_ids)
        if self.opt.face_discrim:
             self.save_network(self.netDface, 'Dface', which_epoch, self.gpu_ids)
        if self.opt.face_generator:
            self.save_network(self.faceGen, 'Gface', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.opt.face_generator:
            params += list(self.faceGen.parameters())           
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999)) 
        print('------------ Now also finetuning global generator -----------')

    def update_fixed_params_netD(self):
        params = list(self.netD.parameters()) + list(self.netDface.parameters())         
        self.optimizer_D = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999)) 
        print('------------ Now also finetuning multiscale discriminator -----------')

    def update_learning_rate(self):
        '''
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd        
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr  
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
        '''
        # 修改为变成0.1倍
        for param_group in self.optimizer_G.param_groups:
            old_lr = param_group['lr']
            param_group['lr'] = param_group['lr'] * self.opt.lr_factor
            print('update optimizer_G learning rate: %f -> %f' % (old_lr, param_group['lr']))
        for param_group in self.optimizer_D.param_groups:
            old_lr = param_group['lr']
            param_group['lr'] = param_group['lr'] * self.opt.lr_factor
            print('update optimizer_D learning rate: %f -> %f' % (old_lr, param_group['lr']))
        for param_group in self.optimizer_D_wo_motion.param_groups:
            old_lr = param_group['lr']
            param_group['lr'] = param_group['lr'] * self.opt.lr_factor 
            print('update optimizer_D_wo_motion learning rate: %f -> %f' % (old_lr, param_group['lr']))