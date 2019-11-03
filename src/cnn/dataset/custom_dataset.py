import os
import pickle
import random

import pandas as pd
import numpy as np
import torch
import cv2
import pydicom

from .. import factory
from ..utils.logger import log
from ...utils import mappings, misc

## nanashi
def bsb_window(img):
    brain_img = window_image(img, 40, 80)
    subdural_img = window_image(img, 80, 200)
    bone_img = window_image(img, 600, 2800)
    greywhite_img = window_image(img, 32, 8)
    soft_img = window_image(img, 20, 350)
    
    bsb_img = np.zeros((brain_img.shape[0], brain_img.shape[1], 3))
    bsb_img[:, :, 0] = brain_img
    bsb_img[:, :, 1] = subdural_img
    bsb_img[:, :, 2] = bone_img
    bsb_img[:, :, 3] = grey-white_img
    bsb_img[:, :, 4] = soft_img
    return bsb_img

def apply_window_policy(image, row, policy):
    if policy == 1:
        image1 = misc.apply_window(image, 40, 80) # brain
        image2 = misc.apply_window(image, 80, 200) # subdural
        image3 = misc.apply_window(image, row.WindowCenter, row.WindowWidth)
        image1 = (image1 - 0) / 80
        image2 = (image2 - (-20)) / 200
        image3 = (image3 - image3.min()) / (image3.max()-image3.min())
        image = np.array([
            image1 - image1.mean(),
            image2 - image2.mean(),
            image3 - image3.mean(),
        ]).transpose(1,2,0)
    elif policy == 2:
        image1 = misc.apply_window(image, 40, 80) # brain
        image2 = misc.apply_window(image, 80, 200) # subdural
        image3 = misc.apply_window(image, 40, 380) # bone
        image1 = (image1 - 0) / 80
        image2 = (image2 - (-20)) / 200
        image3 = (image3 - (-150)) / 380
        image = np.array([
            image1,# - image1.mean(),
            image2,# - image2.mean(),
            image3, # - image3.mean(),
        ]).transpose(1,2,0)
    elif policy == 3: # 5 windows
        image1 = misc.apply_window(image, 40, 80) # brain
        image2 = misc.apply_window(image, 80, 200) # subdural
        image3 = misc.apply_window(image, 40, 380) # soft
        image4 = misc.apply_window(image, 600, 2800) # bone
        image5 = misc.apply_window(image, 8, 32+32) # gw, doubled the range to make it more reasonable
        image1 = (image1 - 0) / 80
        image2 = (image2 - (-20)) / 200
        image3 = (image3 - (-150)) / 380
        image4 = (image4 - (-800)) / 2800
        image5 = (image5 - (-24)) / 64
        image = np.array([
            image1,# - image1.mean(),
            image2,# - image2.mean(),
            image3, # - image3.mean(),
            image4,
            image5,
        ]).transpose(1,2,0)
    elif policy == 4:  #  -50–150, 100–300 and 250–450.
        image1 = misc.apply_window(image, 50, 200) # brain
        image2 = misc.apply_window(image, 200, 200) # subdural
        image3 = misc.apply_window(image, 350, 200) # bone
        image1 = (image1 - (-50)) / 200
        image2 = (image2 - (100)) / 200
        image3 = (image3 - (250)) / 200
        image = np.array([
            image1,# - image1.mean(),
            image2,# - image2.mean(),
            image3, # - image3.mean(),
        ]).transpose(1,2,0)
        
    else:
        raise NotImplementedError

    return image


def apply_dataset_policy(df, policy):
    if policy == 'all':
        pass
    elif policy == 'pos==neg':
        df_positive = df[df.labels != '']
        df_negative = df[df.labels == '']
        df_sampled = df_negative.sample(len(df_positive))
        df = pd.concat([df_positive, df_sampled], sort=False)
    else:
        raise
    log('applied dataset_policy %s (%d records)' % (policy, len(df)))

    return df


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, folds):
        self.cfg = cfg

        log(f'dataset_policy: {self.cfg.dataset_policy}')
        log(f'window_policy: {self.cfg.window_policy}')

        self.transforms = factory.get_transforms(self.cfg)
        with open(cfg.annotations, 'rb') as f:
            self.df = pickle.load(f)

        if folds:
            self.df = self.df[self.df.fold.isin(folds)]
            log('read dataset (%d records)' % len(self.df))

        self.df = apply_dataset_policy(self.df, self.cfg.dataset_policy)
        # self.df = self.df.sample(560)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        path = '%s/%s.dcm' % (self.cfg.imgdir, row.ID)

        dicom = pydicom.dcmread(path)
        image = dicom.pixel_array
        image = misc.rescale_image(image, row.RescaleSlope, row.RescaleIntercept)
        image = apply_window_policy(image, row, self.cfg.window_policy)

        image = self.transforms(image=image)['image']

        target = np.array([0.0] * len(mappings.label_to_num))
        for label in row.labels.split():
            cls = mappings.label_to_num[label]
            target[cls] = 1.0

        if hasattr(self.cfg, 'spread_diagnosis'):
            for label in row.LeftLabel.split() + row.RightLabel.split():
                cls = mappings.label_to_num[label]
                target[cls] += self.cfg.propagate_diagnosis
        target = np.clip(target, 0.0, 1.0)

        return image, torch.FloatTensor(target), row.ID
