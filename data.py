import os
import numpy as np

import cv2

data_path = 'raw_images/'

# image resolution is 420x580
image_rows = 420
image_cols = 580


def create_train_data():
    train_data_path = os.path.join(data_path, 'train')
    # There are 5635 ultrasound images and their corresponding binary mask images 
    # showing the Brachial Plexus nerve segments
    images = list(filter((lambda image: 'mask' not in image), os.listdir(train_data_path)))
    total = len(images)
    
    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    subjects = np.ndarray((total,), dtype=np.uint8)

    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    
    for i, image_name in enumerate(images):
        
        # Load images from files
        image_mask_name = image_name.split('.')[0] + '_mask.tif'
        # There are 47 patients and each patient has about 120 scanned images
        subject_id = image_name.split('_')[0]
        img = cv2.imread(os.path.join(train_data_path, image_name), cv2.IMREAD_GRAYSCALE)
        img_mask = cv2.imread(os.path.join(train_data_path, image_mask_name), cv2.IMREAD_GRAYSCALE)

        img = np.array([img])
        img_mask = np.array([img_mask])
        
        # About 47% of the images don't have a mask of BP nerves
        imgs[i] = img
        imgs_mask[i] = img_mask
        subjects[i] = subject_id
        
        if i % 1000 == 0:
            print('Done: {0}/{1} images'.format(i, total))
    print('Loading done.')

    np.save('np_data/subjects_train.npy', subjects)
    np.save('np_data/imgs_train.npy', imgs)
    np.save('np_data/imgs_mask_train.npy', imgs_mask)
    print('Saving to .npy files done.')


def load_train_data():
    imgs_train = np.load('np_data/imgs_train.npy')
    imgs_mask_train = np.load('np_data/imgs_mask_train.npy')
    subjects = np.load('np_data/subjects_train.npy')
    return imgs_train, imgs_mask_train, subjects


def create_test_data():
    test_data_path = os.path.join(data_path, 'test')
    # There are 5508 ultrasound images
    images = os.listdir(test_data_path)
    total = len(images)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=np.int32)

    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    # Load images from files
    for i, image_name in enumerate(images):
        img_id = int(image_name.split('.')[0])
        img = cv2.imread(os.path.join(test_data_path, image_name), cv2.IMREAD_GRAYSCALE)

        img = np.array([img])

        imgs[i] = img
        imgs_id[i] = img_id

        if i % 1000 == 0:
            print('Done: {0}/{1} images'.format(i, total))
    print('Loading done.')
    
    # save images as binary format files for faster loading later
    np.save('np_data/imgs_test.npy', imgs)
    np.save('np_data/imgs_id_test.npy', imgs_id)
    print('Saving to .npy files done.')


def load_test_data():
    imgs_test = np.load('np_data/imgs_test.npy')
    test_id = np.load('np_data/imgs_id_test.npy')
    return imgs_test, test_id

# Execute only if it is run, not if it is imported
if __name__ == '__main__':
    create_train_data()
    create_test_data()
