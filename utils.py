import numpy as np
import os
import imageio
from scipy.ndimage import uniform_filter

import logging
import colorlog


def log_color():

    color_logger = logging.getLogger('ROOT')
    color_logger.setLevel(logging.DEBUG)

    log_handler = logging.StreamHandler()
    log_handler.setLevel(logging.DEBUG)

    fmt_str = '%(log_color)s[%(levelname)s]: %(message)s (%(asctime)s)'

    log_colors = {
        'DEBUG': 'black, bg_white',
        'INFO': 'bg_green',
        'WARNING': 'bg_yellow',
        'ERROR': 'bg_red',
        'CRITICAL': 'bg_purple'
    }

    the_format = colorlog.ColoredFormatter(fmt_str, log_colors = log_colors)
    log_handler.setFormatter(the_format)
    color_logger.addHandler(log_handler)

    return color_logger

LOGGER = log_color()


def load_dataset(path, class_names, data_type, dtype=np.float32):

  print('********** loading data **********')

  # Map class names to integer labels
  class_to_label = {class_id: i for i, class_id in enumerate(class_names)}

  X = []
  Y = []

  for i, class_id in enumerate(class_names):
    img_files = os.listdir(os.path.join(path, data_type, class_id))
    num_images = len(img_files)
    
    x_block = np.zeros((num_images, 1, 48, 48), dtype=dtype)
    y_block = class_to_label[class_id] * np.ones(num_images, dtype=np.int64)
    for j, img_file in enumerate(img_files):
      img_file = os.path.join(path, data_type, class_id, img_file)
      img = imageio.imread(img_file)
      if img.ndim == 2:
        ## grayscale file
        img.shape = (48, 48, 1)
      x_block[j] = img.transpose(2, 0, 1)
    X.append(x_block)
    Y.append(y_block)
      
  # We need to concatenate all data
  X = np.concatenate(X, axis=0)
  Y = np.concatenate(Y, axis=0)
  
  return X, Y


def extract_features(imgs, feature_fns, verbose=False):
  """
  Given pixel data for images and several feature functions that can operate on
  single images, apply all feature functions to all images, concatenating the
  feature vectors for each image and storing the features for all images in
  a single matrix.

  Inputs:
  - imgs: N x H X W X C array of pixel data for N images.
  - feature_fns: List of k feature functions. The ith feature function should
    take as input an H x W x D array and return a (one-dimensional) array of
    length F_i.
  - verbose: Boolean; if true, print progress.

  Returns:
  An array of shape (N, F_1 + ... + F_k) where each column is the concatenation
  of all features for a single image.
  """
  num_images = imgs.shape[0]
  if num_images == 0:
    return np.array([])

  # Use the first image to determine feature dimensions
  feature_dims = []
  first_image_features = []
  for feature_fn in feature_fns:
    feats = feature_fn(imgs[0].squeeze())
    assert len(feats.shape) == 1, 'Feature functions must be one-dimensional'
    feature_dims.append(feats.size)
    first_image_features.append(feats)

  # Now that we know the dimensions of the features, we can allocate a single
  # big array to store all features as columns.
  total_feature_dim = sum(feature_dims)
  imgs_features = np.zeros((num_images, total_feature_dim))
  imgs_features[0] = np.hstack(first_image_features).T

  # Extract features for the rest of the images.
  for i in range(1, num_images):
    idx = 0
    for feature_fn, feature_dim in zip(feature_fns, feature_dims):
      next_idx = idx + feature_dim
      imgs_features[i, idx:next_idx] = feature_fn(imgs[i].squeeze())
      idx = next_idx

  return imgs_features


def hog_feature(im):
  """Compute Histogram of Gradient (HOG) feature for an image
  
       Modified from skimage.feature.hog
       http://pydoc.net/Python/scikits-image/0.4.2/skimage.feature.hog
     
     Reference:
       Histograms of Oriented Gradients for Human Detection
       Navneet Dalal and Bill Triggs, CVPR 2005
     
    Parameters:
      im : an input grayscale or rgb image
      
    Returns:
      feat: Histogram of Gradient (HOG) feature
    
  """
  
  # convert rgb to grayscale if needed
  image = im

  sx, sy = image.shape # image size
  orientations = 9 # number of gradient bins
  cx, cy = (8, 8) # pixels per cell

  gx = np.zeros(image.shape)
  gy = np.zeros(image.shape)
  gx[:, :-1] = np.diff(image, n=1, axis=1) # compute gradient on x-direction
  gy[:-1, :] = np.diff(image, n=1, axis=0) # compute gradient on y-direction
  grad_mag = np.sqrt(gx ** 2 + gy ** 2) # gradient magnitude
  grad_ori = np.arctan2(gy, (gx + 1e-15)) * (180 / np.pi) + 90 # gradient orientation

  n_cellsx = int(np.floor(sx / cx))  # number of cells in x
  n_cellsy = int(np.floor(sy / cy))  # number of cells in y
  # compute orientations integral images
  orientation_histogram = np.zeros((n_cellsx, n_cellsy, orientations))
  for i in range(orientations):
    # create new integral image for this orientation
    # isolate orientations in this range
    temp_ori = np.where(grad_ori < 180 / orientations * (i + 1),
                        grad_ori, 0)
    temp_ori = np.where(grad_ori >= 180 / orientations * i,
                        temp_ori, 0)
    # select magnitudes for those orientations
    cond2 = temp_ori > 0
    temp_mag = np.where(cond2, grad_mag, 0)
    orientation_histogram[:,:,i] = uniform_filter(temp_mag, size=(cx, cy))[cx//2::cx, cy//2::cy].T
  
  return orientation_histogram.ravel()


def acc_classes(y_pred, label):

  acc_list = []
  cls_num = np.max(label) + 1

  for cls in range(cls_num):
    
    cls_pred = y_pred[label == cls]
    cls_corr = np.ones(cls_pred.shape) * [cls_pred == cls]
    cls_acc = np.sum(cls_corr) / len(cls_pred)
    acc_list.append(cls_acc)
  
  return acc_list


# if __name__ == '__main__':
#     y_pred =  np.array([0,0,0,0,0,1,1,1,1,1])
#     y_label = np.array([0,0,0,0,1,0,1,1,1,1])
#     print(acc_classes(y_pred, y_label))