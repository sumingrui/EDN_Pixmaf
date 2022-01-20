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

import torch
import joblib
import numpy as np
import os.path as osp
from torch.utils.data import Dataset
from skimage.util.shape import view_as_windows
from models.pixmaf_net.core.cfgs import VIBE_DB_DIR


def split_into_chunks(vid_names, seqlen, stride):
    video_start_end_indices = []

    video_names, group = np.unique(vid_names, return_index=True)
    perm = np.argsort(group)
    video_names, group = video_names[perm], group[perm]

    indices = np.split(np.arange(0, vid_names.shape[0]), group[1:])

    for idx in range(len(video_names)):
        indexes = indices[idx]
        if indexes.shape[0] < seqlen:
            continue
        chunks = view_as_windows(indexes, (seqlen,), step=stride)
        start_finish = chunks[:, (0, -1)].tolist()
        video_start_end_indices += start_finish

    return video_start_end_indices

class AMASS(Dataset):
    def __init__(self, seqlen, debug):
        self.seqlen = seqlen
        self.stride = seqlen
        self.debug = debug
        self.db = self.load_db()
        self.vid_indices = split_into_chunks(self.db['vid_name'], self.seqlen, self.stride)
        del self.db['vid_name']
        print(f'AMASS dataset number of videos: {len(self.vid_indices)}')

    def __len__(self):
        return len(self.vid_indices)

    def __getitem__(self, index):
        return self.get_single_item(index)

    def load_db(self):
        if not self.debug:
            db_file = osp.join(VIBE_DB_DIR, 'amass_db_ori.pt')
        else:
            db_file = osp.join(VIBE_DB_DIR, 'amass_db_debug.pt')
        db = joblib.load(db_file)
        return db

    def get_single_item(self, index):
        start_index, end_index = self.vid_indices[index]
        thetas = self.db['theta'][start_index:end_index+1]

        cam = np.array([1., 0., 0.])[None, ...]
        cam = np.repeat(cam, thetas.shape[0], axis=0)
        theta = np.concatenate([cam, thetas], axis=-1)

        target = {
            'theta': torch.from_numpy(theta).float(),  # cam, pose and shape
        }
        return target



