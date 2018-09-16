###Methods
This was an IRB-exempt study using 114 de-identified HIPAA compliant lateral chest radiographs on unique patients. The
images were pre-processed using contrast-adaptive histogram equalization (CLAHE). All visible vertebrae were manually
segmented on the lateral radiograph using ImageJ (NIH) by a medical student and verified by a board-certified radiologist.
74 images (64.9%) were used for training, 10 images (8.8%) for validation, and 30 images (26.3%) were used for test.
A U-Net convolutional neural network was used as the segmentation architecture using the Keras library and Tensorflow
backend. The loss function was a sum of Dice Coefficient and binary cross-entropy. The Adam optimizer was used with a
learning rate of 0.0001. All weights were randomly initialized. Augmentation consisted of horizontal flipping, shearing,
rotation, scaling, and translation. Training was stopped after a plateau in the validation lo
