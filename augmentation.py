# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 11:11:04 2017

@author: asiu
"""

import numpy as np
import keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

def augmentation(imgs, masks):
    print("Augmentation Model...")
    """ Generate batches of image data augmentation. The data will be looped over in batches."""

    gen = ImageDataGenerator(rotation_range=10, # Degree range for random rotation
                             width_shift_range=0.1, # Fraction of total width for random horizontal shift
                             height_shift_range=0.1, # Fraction of total height for random vertical shift
                             shear_range=0.1, # Shear angle in counter-clockwise direction as radins
                             zoom_range=0.1, # Zoom in or out within 10% range
                             horizontal_flip=True, # Randomly flip inputs horizontally
                             fill_mode='nearest', # Points outside boundaries are filled
                             cval=0.)
    
    n = imgs.shape[0]
    aug_imgs, aug_masks = [], []
        
    # Generate augmented image and mask in the same way for each observation
    i = 0
    for batch in gen.flow(imgs, batch_size=1, shuffle=False, seed=123):
        aug_imgs.append(imgs[i])
        aug_imgs.append(batch[0])
        i = i+1
        if i>=n:
            break
    i = 0
    for batch in gen.flow(masks, batch_size=1, shuffle=False, seed=123):
        aug_masks.append(masks[i])
        aug_masks.append(batch[0])
        i = i+1
        if i>=n:
            break
    
    aug_imgs = np.array(aug_imgs)
    aug_masks = np.array(aug_masks)
    print("Shape of Augmented images: {}".format(aug_imgs.shape))
    
    return aug_imgs, aug_masks