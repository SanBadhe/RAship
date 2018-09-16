## Methods
74 images (64.9%) were used for training, 10 images (8.8%) for validation, and 30 images (26.3%) were used for test.
A U-Net convolutional neural network was used as the segmentation architecture using the Keras library and Tensorflow
backend. The loss function was a sum of Dice Coefficient and binary cross-entropy. The Adam optimizer was used with a
learning rate of 0.0001. All weights were randomly initialized. Augmentation consisted of horizontal flipping, shearing,
rotation, scaling, and translation. Training was stopped after a plateau in the validation looss. Intersection over union (IOU), jaccard_distance_loss, and Dice Coefficient were used to evaluate the accuracy of the segmentations

## Result
On the hold-out test dataset, obtain an average Dice Coefficient value of 98.97 and an average Intersection over union (IoU) value of 98.10.

## How to use
1) data:- we used for training data. it has subfolder images and masks for orignal images and ground truth images respectively. Images name should be consucatuve and same. 
2) data1:- we used for testing and saving the augmentation images created by keras ImageDataGenerator.
3) main.py:- Run this file to create predicted mask. predicted mask will be created in test folder with filename_prediction name.
4) idx.csv - File should contain X-ray filenames as first column, mask filenames as second column. 
5) test.py- Is use to created overlay image of orignal image with grouth truth mask and predicted mask.
6) results1:- In this foldar overlay of ground truth and predicted mask with orignal image will be generated. 

## Output
overlay of ground truth and predicted mask with orignal image. blue filled part is predicted region and red boundary part is ground truth.

![alt text](https://github.com/SanBadhe/RAship/blob/master/Segmentation/results1/19.png)




