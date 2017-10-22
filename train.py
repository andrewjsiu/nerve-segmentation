from __future__ import print_function

import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
# Keras has pre-built layers for building deep learning models
from keras.layers import Input, concatenate, Convolution2D, MaxPooling2D, Conv2DTranspose
from keras.layers import Dropout, BatchNormalization, Dense, Flatten
from keras.optimizers import Adam
# Callbacks give a view on internal states and statistics of model during training
from keras.callbacks import ModelCheckpoint 
from keras import backend as K # handles low-level operations such as tensor products

from data import load_train_data, load_test_data
from cleaning import load_clean_data
from augmentation import augmentation

K.set_image_data_format('channels_last')  # The default is TensorFlow

img_rows, img_cols = 96, 96

smooth = 1


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet():
    """ U-Net Architecture: Convolutional Networks for Biomedial Image Segmentation 
    The main idea is to first contract the network by pooling operators and then
    increase network resolution by upsampling operators. Upsampled output is combined
    with high resolution features from the contracting path.
    """
        
    # Define the shape of the input data to the model
    inputs = Input(shape=(img_rows, img_cols, 1))
    
    # Convolution layer: number of filters, (width, height) of filters, default stride is 1, 
    # activation is Rectified Linear function,
    # padding='same' adds zero padding to produce output of the same size if stride is 1, 
    # (if padding='valid', no padding is added and excess columns/rows are dropped)
    # Apply a 3x3 convolution with 32 filters on a 96x96 image 
    conv1 = Convolution2D(32, (3, 3), activation='elu', padding='same', 
                          kernel_initializer='he_normal', bias_initializer='zeros')(inputs)
    conv1 = BatchNormalization(axis=1)(conv1)
    conv1 = Convolution2D(32, (3, 3), activation='elu', padding='same',
                          kernel_initializer='he_normal', bias_initializer='zeros')(conv1)
    conv1 = BatchNormalization(axis=1)(conv1)

    # Pool layer: downsamples the spatial dimensions of the input
    # Apply max pooling operation with 2x2 receptive fields to discard 75% of activations
    # The maxpooling turns 96x96 image to 48x48 feature maps
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(0.5)(pool1)
    
    conv2 = Convolution2D(64, (3, 3), activation='elu', padding='same',
                          kernel_initializer='he_normal', bias_initializer='zeros')(pool1)
    conv2 = BatchNormalization(axis=1)(conv2)
    conv2 = Convolution2D(64, (3, 3), activation='elu', padding='same',
                          kernel_initializer='he_normal', bias_initializer='zeros')(conv2)
    conv2 = BatchNormalization(axis=1)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(0.5)(pool2)

    conv3 = Convolution2D(128, (3, 3), activation='elu', padding='same',
                          kernel_initializer='he_normal', bias_initializer='zeros')(pool2)
    conv3 = BatchNormalization(axis=1)(conv3)
    conv3 = Convolution2D(128, (3, 3), activation='elu', padding='same',
                          kernel_initializer='he_normal', bias_initializer='zeros')(conv3)
    conv3 = BatchNormalization(axis=1)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(0.5)(pool3)

    conv4 = Convolution2D(256, (3, 3), activation='elu', padding='same',
                          kernel_initializer='he_normal', bias_initializer='zeros')(pool3)
    conv4 = BatchNormalization(axis=1)(conv4)
    conv4 = Convolution2D(256, (3, 3), activation='elu', padding='same',
                          kernel_initializer='he_normal', bias_initializer='zeros')(conv4)
    conv4 = BatchNormalization(axis=1)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)
    
    conv5 = Convolution2D(512, (3, 3), activation='elu', padding='same',
                          kernel_initializer='he_normal', bias_initializer='zeros')(pool4)
    conv5 = BatchNormalization(axis=1)(conv5)
    conv5 = Convolution2D(512, (3, 3), activation='elu', padding='same',
                          kernel_initializer='he_normal', bias_initializer='zeros')(conv5)
    conv5 = BatchNormalization(axis=1)(conv5)
    conv5 = Dropout(0.5)(conv5)
    
    # Generate output to predict nerve presence as an auxiliary branch
    pre = Convolution2D(1, (1, 1), activation='sigmoid', kernel_initializer='he_normal')(conv5)
    pre = Flatten()(pre)
    aux_out = Dense(1, activation='sigmoid', name='aux_output')(pre)
    
    # Transposed Convolution layer (Deconvolution): # of filters, (width, height) of filter,
    # Concatenate a list of inputs that have the same shape
    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Convolution2D(256, (3, 3), activation='elu', padding='same',
                          kernel_initializer='he_normal', bias_initializer='zeros')(up6)
    conv6 = BatchNormalization(axis=1)(conv6)
    conv6 = Convolution2D(256, (3, 3), activation='elu', padding='same',
                          kernel_initializer='he_normal', bias_initializer='zeros')(conv6)
    conv6 = BatchNormalization(axis=1)(conv6)
    conv6 = Dropout(0.5)(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Convolution2D(128, (3, 3), activation='elu', padding='same',
                          kernel_initializer='he_normal', bias_initializer='zeros')(up7)
    conv7 = BatchNormalization(axis=1)(conv7)
    conv7 = Convolution2D(128, (3, 3), activation='elu', padding='same',
                          kernel_initializer='he_normal', bias_initializer='zeros')(conv7)
    conv7 = BatchNormalization(axis=1)(conv7)
    conv7 = Dropout(0.5)(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Convolution2D(64, (3, 3), activation='elu', padding='same',
                          kernel_initializer='he_normal', bias_initializer='zeros')(up8)
    conv8 = BatchNormalization(axis=1)(conv8)
    conv8 = Convolution2D(64, (3, 3), activation='elu', padding='same',
                          kernel_initializer='he_normal', bias_initializer='zeros')(conv8)
    conv8 = BatchNormalization(axis=1)(conv8)
    conv8 = Dropout(0.5)(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Convolution2D(32, (3, 3), activation='elu', padding='same',
                          kernel_initializer='he_normal', bias_initializer='zeros')(up9)
    conv9 = BatchNormalization(axis=1)(conv9)
    conv9 = Convolution2D(32, (3, 3), activation='elu', padding='same',
                          kernel_initializer='he_normal', bias_initializer='zeros')(conv9)
    conv9 = BatchNormalization(axis=1)(conv9)
    conv9 = Dropout(0.5)(conv9)

    conv10 = Convolution2D(1, (1, 1), activation='sigmoid',
                           kernel_initializer='he_normal',
                           name='main_output')(conv9)

    # Create a model with input and two output layers
    # Use Adam optimzer evaluated by Dice Coefficient on the main output and
    # accuracy on the auxiliary output of predicting nerve presence
    model = Model(inputs=[inputs], outputs=[conv10, aux_out])
    model.compile(optimizer=Adam(lr=1e-5), 
                  loss={'main_output': dice_coef_loss, 'aux_output': 'binary_crossentropy'},
                  metrics={'main_output': dice_coef, 'aux_output': 'acc'},
                  loss_weights={'main_output': 1., 'aux_output': 0.5})

    return model


def preprocess(imgs):
    """ Resize to image resolution to (img_rows, img_cols)"""
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_rows, img_cols), preserve_range=True, mode='constant')

    # Increase the dimension of array 
    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p


def train_and_predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_mask_train, _ = load_clean_data()

    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization
    imgs_train = (imgs_train - mean) / std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train = imgs_mask_train / 255.  # scale masks to [0, 1]

    # Augmenting images and masks
    imgs_train, imgs_mask_train = augmentation(imgs_train, imgs_mask_train)
    
    # Get nerve presence indicators
    presence = np.array([int(np.sum(imgs_mask_train[i]) > 0) for i in range(imgs_mask_train.shape[0])])

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    # Save the latest best model as HDF5 file format (Hierarchical Data Format)
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    model.fit(imgs_train, [imgs_mask_train, presence], 
              batch_size=128, 
              epochs=20, 
              verbose=1, 
              shuffle=True,
              validation_split=0.2,
              callbacks=[model_checkpoint])

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test = (imgs_test - mean) / std

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('weights.h5')

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    np.save('imgs_mask_test.npy', imgs_mask_test[0])
    np.save('exist_mask_test.npy', imgs_mask_test[1])

    """
    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    pred_dir = 'preds'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for image, image_id in zip(imgs_mask_test, imgs_id_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        imsave(os.path.join(pred_dir, str(image_id) + '_pred.png'), image)
    """
    return model
    
# Execute only if it is run, not if it is imported
if __name__ == '__main__':
    train_and_predict()
