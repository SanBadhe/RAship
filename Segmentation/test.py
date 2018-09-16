from skimage import morphology, color
import pandas as pd
from model import *
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import skimage.io as io
import skimage.transform as trans

def loadTestData(df, path, target_size=(512, 512), flag_multi_class=False, as_gray=True):
    """This function loads data preprocessed with `preprocess_JSRT.py`"""
    X, y = [], []
    for i, item in df.iterrows():
        img = io.imread(path + item[0],as_gray = as_gray)
        img = img / 255
        img = trans.resize(img, target_size)
        img = np.reshape(img, img.shape + (1,)) if (not flag_multi_class) else img
        mask = io.imread(path + item[1],as_gray = as_gray)
        mask = mask / 255
        mask = np.reshape(mask, mask.shape + (1,)) if (not flag_multi_class) else mask
        X.append(img)
        y.append(mask)
    X = np.array(X)
    y = np.array(y)

    print ('### Data loaded')
    print ('\t{}'.format(path))
    print ('\t{}\t{}'.format(X.shape, y.shape))
    return X, y

def IoU(y_true, y_pred):
    """Returns Intersection over Union score for ground truth and predicted masks."""
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    union = np.logical_or(y_true_f, y_pred_f).sum()
    return ((intersection + 1) * 1. / (union + 1))

def Dice(y_true, y_pred):
    """Returns Dice Similarity Coefficient for ground truth and predicted masks."""
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    return ((2. * intersection + 1.) / (y_true.sum() + y_pred.sum() + 1.))

def masked(img, gt, mask, alpha=1):
    """Returns image with GT lung field outlined with red, predicted lung field
    filled with blue."""
    rows, cols = img.shape
    color_mask = np.zeros((rows, cols, 3))
    boundary = np.subtract(morphology.dilation(gt, morphology.disk(3)), gt , dtype=np.float32)
    color_mask[mask == 0] = [0, 0, 1]
    color_mask[boundary == 1] = [1, 0, 0]
    img_color = np.dstack((img, img, img))

    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)

    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)
    return img_masked

def remove_small_regions(img, size):
    """Morphologically removes small (less than size) connected regions of 0s or 1s."""
    img = morphology.remove_small_objects(img, size)
    img = morphology.remove_small_holes(img, size)
    return img

# Path to csv-file. File should contain X-ray filenames as first column,
# mask filenames as second column.
csv_path = 'idx.csv'
#test folder path
path = 'data1/test/'

df = pd.read_csv(csv_path)



# Load test data
im_shape = (512, 512)
X, y = loadTestData(df, path, im_shape)

n_test = X.shape[0]
inp_shape = X[0].shape

# Load model
model = unet()
model.load_weights("unet_gold.hdf5")

# For inference standard keras ImageGenerator is used.
test_gen = ImageDataGenerator(rescale=1.)

ious = np.zeros(n_test)
dices = np.zeros(n_test)

i = 0
for xx, yy in test_gen.flow(X, y, batch_size=1):
    img = np.squeeze(xx)
    pred = model.predict(xx)[..., 0].reshape(inp_shape[:2])
    mask = yy[..., 0].reshape(inp_shape[:2])

    # Binarize masks
    gt = mask > 0.5
    pr = pred > 0.7

    # Remove regions smaller than 2% of the image
    #pr = remove_small_regions(pr, 0.02 * np.prod(im_shape))

    io.imsave('results1/{}'.format(df.iloc[i][0]), masked(img, gt, pr, 1))

    ious[i] = IoU(gt, pr)
    dices[i] = Dice(gt, pr)
    print(df.iloc[i][0], ious[i], dices[i])

    i += 1
    if i == n_test:
        break

print('Mean IoU:', ious.mean())
print('Mean Dice:', dices.mean())

