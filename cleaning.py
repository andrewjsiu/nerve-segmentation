# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 10:21:45 2017

@author: asiu
"""
import numpy as np
from data import load_train_data
from skimage.util import view_as_blocks # Blocks are non-overlapping views of the input array.
import scipy.spatial.distance as spdist

imgs, masks, fnames = load_train_data()
subjects = np.load('np_data/subjects_train.npy')

def load_patient(pid):
    im = imgs[np.where(subjects==pid)]
    mk = masks[np.where(subjects==pid)]
    fn = fnames[np.where(subjects==pid)]
    return im, mk, fn

def compute_img_hist(img):
    """ Divide image into 21x29 blocks of size 20x20
        and compute a histogram for each row of 29 blocks """
    blocks = view_as_blocks(img.astype(np.float32)/255.0, block_shape=(20,20))
    hists = [np.histogram(block, bins=np.linspace(0,1,10))[0] for block in blocks]
    return np.concatenate(hists)

smooth = 1
def dice(mask1, mask2):
    mask1 = (mask1.astype(np.float32)/255.0).flatten()
    mask2 = (mask2.astype(np.float32)/255.0).flatten()
    intersection = np.sum(np.multiply(mask1, mask2))
    return (2 * intersection + smooth) / (np.sum(mask1) + np.sum(mask2) + smooth)

def keep_valid(pid):

    im, mk, fn = load_patient(pid)
    im_hists = [compute_img_hist(i) for i in im]
    im_hists = np.array(im_hists)
    
    """ Compute pairwise distances between image histograms 
        Convert distance vector to a square-form distance matrix
        Find similar pairs while excluding self-matching pairs.
        Need to determine a threshold for similar pairs """
        
    dist = spdist.squareform(spdist.pdist(im_hists, metric='cosine'))
    close_pairs = ((dist + np.eye(dist.shape[0])) < 0.008)
    close_ij = np.transpose(np.nonzero(close_pairs))

    inconsistent = [(i, j) for i, j in close_ij if dice(mk[i], mk[j])<0.2]
    inconsistent = np.array(inconsistent)
    
    valids = np.ones(len(im), dtype=np.bool)
    for i, j in inconsistent:
        if np.sum(mk[i]) == 0:
            valids[i] = False
        if np.sum(mk[j]) == 0:
            valids[j] = False
    
    print("For patient {}, {} images are kept and {} removed."
          .format(pid, np.sum(valids), np.sum(~valids)))
    
    return im[valids], mk[valids], fn[valids]

def create_clean_data():
    for pid in np.unique(subjects):
        if pid == 1:
            imv, mkv, fnv = keep_valid(pid)
        else:
            i, m, f = keep_valid(pid)
            imv = np.concatenate((imv, i), axis=0)
            mkv = np.concatenate((mkv, m), axis=0)
            fnv = np.concatenate((fnv, f), axis=0)
    
    print("Total number of images kept: {}".format(imv.shape[0]))
    print("Number of images removed due to inconsistent labels: {}".format(imgs.shape[0]-imv.shape[0]))
    
    np.save('np_data/clean_imgs.npy', imv)
    np.save('np_data/clean_masks.npy', mkv)
    np.save('np_data/clean_fnames.npy', fnv)
    

def load_clean_data():
    imgs_clean = np.load('np_data/clean_imgs.npy')
    masks_clean = np.load('np_data/clean_masks.npy')
    fnames_clean = np.load('np_data/clean_fnames.npy')
    return imgs_clean, masks_clean, fnames_clean

# Execute only if it is run, not if it is imported
if __name__ == '__main__':
    create_clean_data()
