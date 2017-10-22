# Identifying Nerve Structures on Ultrasound Images

## Introduction

Automating the work of segmenting objects on image data can create much practical value. One recent Kaggle competition provides a dataset of ultrasound images of the human neck. An accurate identification of the nerve structures can help surgeons better place catheters to block or mitigate pain during a surgical procedure. 

On these ultrasound images, our task is to build a model to predict the exact pixels of an image that contain a collection of nerves called the Brachial Plexus (BP). The training set includes more than five thousand ultrasound images along with labeled masks that experts have manually annotated the BP.

The models are evaluated on the mean Dice coefficient, which is a similarity measure of the predicted masks and the ground truth. The formula is given by twice the elements in the intersection of the predicted set of pixels and the true set of pixels divided by the sum of all elements in the two sets. 

The advantage of Dice coefficient is that it can be applied to data where the BP is present or absent. For images where the BP is present, a Dice coefficient closer to 1 indicates greater pixel-wise agreement between the truth and the positive predictions. 
When the BP is absent in the ground truth, the Dice coefficient is defined to be 1 if the predicted mask is also empty. But note that even a single false positive prediction can still hurt the Dice coefficient. To minimize the false positives in these cases, we can add an auxiliary output to predict the presence of BP nerves for each image.

## Baseline Model

It is useful to first start with a simple baseline model with an end-to-end working code. Then we can analyze the prediction errors and identify areas where we can make the most improvements. 

My baseline model is a convolutional neural network with a U-Net architecture, that was recently developed by Ronneberger, Fischer and Brox (2015) for segmenting biological images. The main idea is to first contract the network by pooling operators and then increase network resolution by upsampling operators. Upsampled output at each level is combined with high resolution features from the contracting path of the corresponding level. The authors showed that such a network can be trained from very few images and outperforms a sliding-window convolutional network.

A submission of all empty masks gets a score of 0.53 on the private leaderboard, suggesting that there are about 53% of the images that do not have the BP nerves. My baseline model that incorporates dropout layers and batch normalization for regularization achieved a score of 0.59 on the private leaderboard. 

On the technical side, I train my deep learning models using Tensorflow and GeForce GTX 1050. Compared to using CPU only, the training time is reduced to about 3-5 hours (with about 6-9 minutes per epoch for 20-30 epochs). 
Data Exploration

Using the best baseline model, we check the predictions against the ground truth to see where the model does well or poorly. The examination reveals that there are some inconsistent labeling for the same patient. For instance, the figures below show that patient 29 has an image numbered 32 that contains a BP mask in the ground truth, but a very similar image numbered 76 contains no mask in the ground truth. Since the two images are nearly identical, the model generates the same predicted masks and is bound to get low mean accuracy. 

Moreover, there are dozens of cases like this for almost each of the 47 patients in the training set (each patient has about 120 ultrasound images for each patient). Such large instances of inconsistent labeling creates a rather low upper bound that any predictive models can achieve.

**Image File Name: 29_32.tif**

<img src="https://github.com/andrewjsiu/nerve-segmentation/blob/master/images/mask29_32.png" height="200">
 
**Image File Name: 29_76.tif** 

<img src="https://github.com/andrewjsiu/nerve-segmentation/blob/master/images/mask29_76.png" height="200">

## Correcting Inconsistent Labels

Here we assume that when two images are very similar but their label masks are inconsistent such that one indicates the presence of BP nerves while the other absence, the empty mask was a mistake and thus remove this image with its mask. 

We measure the similarity between two images by comparing their pixel value histograms as follows: Each image is divided into 21 x 29 blocks and the size of each block is 20 x 20 pixels. We then compute a histogram for each row of 29 blocks and concatenate all 21 histograms into one vector. For any pair of two images, we calculate the cosine distance between their two vectors of histograms. 
For instance, patient 29 has 120 images with 7140 pairs (=120\*119/2). The pairwise distance is roughly normally distributed and slightly skewed to the right. We use the threshold of 0.008 to define the set of most similar pairs of images. For patient 29, the most similar pairs occupy about 1.4% of all pairs. 

<img src="https://github.com/andrewjsiu/nerve-segmentation/blob/master/images/distances.png" height="360">
 
Within the set of similar pairs, the most different pair (29_110.tif and 29_80.tif) have a distance of 0.0079 and still appear to be quite similar in the images shown below. Both of their masks identify the presence of the BP nerves with a Dice coefficient of 0.49. In this case, we will keep both images since we do not know which mask is more accurate and the truth is perhaps somewhere in between. 

<img src="https://github.com/andrewjsiu/nerve-segmentation/blob/master/images/similar.png" height="425">
 
To correct for all inconsitent images, we loop through all 47 patients in the training set and removed a total of 860 images that have empty masks and their similar images indicate the presence of the BP nerves. 

## Data Augmentation 

Removing inconsistent images makes the training set much cleaner but also reduces the number of inputs we feed into the model. One way to get more training data is to use data augmentation. For instance, we can flip a training image horizontally and add it as a new training example. We can also randomly rotate training images, shift them horizontally or vertically, and zoom in or out of them. For each augmented example, we apply the same transformation to the training image and its corresponding mask. Below is an example of augmented image and mask. Adding such augmented training images often help prevent overfitting and regularize the model. 

<img src="https://github.com/andrewjsiu/nerve-segmentation/blob/master/images/augment.png" height="400">

## Predicting the Presence of Target Nerves

After cleaning inconsistent data and applying data augmentation, the newly trained model did not seem to improve the overall score on the leaderboard at first. We then plot the sum of positive predictions for an image against the sum of its actual true pixel values for all images. There is a significantly positive correlation, but there are still many cases where the true mask is empty but the model predicts at least some positive pixel values. If we turn all images with a predicted sum that is less 3500 into an empty mask prediction, we can obtain a private leaderboard score of 0.66, which is a big improvement to our baseline model. 

<img src="https://github.com/andrewjsiu/nerve-segmentation/blob/master/images/errors.png" height="400">

## Future Works

To further improve the model’s ability to accurately predict the presence of the BP nerves, we can add an auxiliary output at the middle of the neural network after the contraction path in addition to the main output of final segmentation.


