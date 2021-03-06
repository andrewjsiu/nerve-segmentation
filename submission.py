from __future__ import print_function

import numpy as np
from skimage.transform import resize
from data import image_cols, image_rows

def prep(img):
    img = img.astype('float32')
    img = resize(img, (image_rows, image_cols), preserve_range=True, mode='constant')
    img = (img > 0.5).astype(np.uint8)  # threshold
    return img


def run_length_enc(label):
    """ Use run-length encoding on pixel values to reduce file size. 
    Submit pairs of values that contain a start position and a run length.
    Use space delimited list of pairs. The pixels are numbered from top to 
    bottom and then left to right. Headers are [img, pixels]. """
    
    from itertools import chain
    x = label.transpose().flatten()
    y = np.where(x > 0)[0]
    if len(y) < 10:  # consider as empty
        return ''
    z = np.where(np.diff(y) > 1)[0]
    # The start positions are y[z+1] plus the initial y[0] 
    start = np.insert(y[z+1], 0, y[0])
    # The end positions are y[z] plus the last y[-1] 
    end = np.append(y[z], y[-1])
    length = end - start
    res = [[s+1, l+1] for s, l in zip(list(start), list(length))]
    res = list(chain.from_iterable(res))
    return ' '.join([str(r) for r in res])


def submission():
    from data import load_test_data
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = np.load('imgs_mask_test.npy')

    argsort = np.argsort(imgs_id_test)
    imgs_id_test = imgs_id_test[argsort]
    imgs_test = imgs_test[argsort]
    
    total = imgs_test.shape[0]
    ids = []
    rles = []
    count = 0
    for i in range(total):
        img = imgs_test[i]
        img = prep(img)
        if (np.sum(img) < 3500): # threshold for making the empty mask predictions
            img = np.zeros((image_rows, image_cols))
            count += 1
        rle = run_length_enc(img)
        rles.append(rle)
        ids.append(imgs_id_test[i])

        if i % 1000 == 0:
            print('{}/{}'.format(i, total))

    print("Empty Masks: {} as {:.1f}%".format(count, count/total*100))
    first_row = 'img,pixels'
    file_name = 'submission.csv'

    with open(file_name, 'w+') as f:
        f.write(first_row + '\n')
        for i in range(total):
            s = str(ids[i]) + ',' + rles[i]
            f.write(s + '\n')

if __name__ == '__main__':
    submission()
