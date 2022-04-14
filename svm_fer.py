import numpy as np
from utils import load_dataset, extract_features, hog_feature, acc_classes, LOGGER
import scipy.io as io
from svm_classifier import LinearSVM
import matplotlib.pyplot as plt


# load train val test data
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
FER_path = './dataset/fer'
X_train, Y_train = load_dataset(FER_path, class_names, 'train')
X_val, Y_val = load_dataset(FER_path, class_names, 'val')
X_test, Y_test = load_dataset(FER_path, class_names, 'test')
print(X_train.shape)
print(X_val.shape)
print(X_test.shape)


# change this when you work on different questions
# question = 'pixel' # choose from pixel, hog, deep
# question = 'hog' # choose from pixel, hog, deep
question = 'deep' # choose from pixel, hog, deep


if question == 'pixel':
    # reshape 2D images to 1D
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))

elif question == 'hog':
    feature_fns = [hog_feature]
    X_train = extract_features(X_train, feature_fns)
    X_val = extract_features(X_val, feature_fns)
    X_test = extract_features(X_test, feature_fns)

elif question == 'deep':
    X_train = io.loadmat('./dataset/train_features.mat')['features']
    X_val = io.loadmat('./dataset/val_features.mat')['features']
    X_test = io.loadmat('./dataset/test_features.mat')['features']


# Preprocessing: Subtract the mean feature
mean_feat = np.mean(X_train, axis=0, keepdims=True)
X_train -= mean_feat
X_val -= mean_feat
X_test -= mean_feat
# Preprocessing: Divide by standard deviation
std_feat = np.std(X_train, axis=0, keepdims=True)
X_train /= std_feat
X_val /= std_feat
X_test /= std_feat

# augmented vector
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])


svm = LinearSVM()
loss_svm_1 = svm.train(X_train, Y_train, learning_rate=1.e-4, reg=1.e0, num_iters=8000)
loss_svm_2 = svm.train(X_train, Y_train, learning_rate=1.e-5, reg=1.e0, num_iters=1000)
loss_svm_3 = svm.train(X_train, Y_train, learning_rate=1.e-4, reg=1.e0, num_iters=1000)
loss_svm_4 = svm.train(X_train, Y_train, learning_rate=1.e-6, reg=1.e0, num_iters=1000)

# evaluate the performance on both the training and validation set
y_train_pred = svm.predict(X_train)
LOGGER.info('training accuracy: %f' % (np.mean(Y_train == y_train_pred), ))

y_val_pred = svm.predict(X_val)
LOGGER.info('validation accuracy: %f' % (np.mean(Y_val == y_val_pred), ))

# evaluate the performance on test set
final_test = True
if final_test:
    y_test_pred = svm.predict(X_test)
    LOGGER.info('test accuracy: %f' % (np.mean(Y_test == y_test_pred), ))
    LOGGER.info(acc_classes(y_test_pred, Y_test))

loss_svm = loss_svm_1 + loss_svm_2 + loss_svm_3 + loss_svm_4

plot_x = np.linspace(0, len(loss_svm)-1, len(loss_svm))
plt.title('Loss')
plt.plot(plot_x, loss_svm)
plt.savefig('loss.png')


# test

# def train(lr, reg, iters):
    
#     svm = LinearSVM()
#     loss_svm = svm.train(X_train, Y_train, learning_rate=lr, reg=reg, num_iters=iters)

#     # evaluate the performance on both the training and validation set
#     y_train_pred = svm.predict(X_train)
#     LOGGER.info('training accuracy: %f' % (np.mean(Y_train == y_train_pred), ))

#     y_val_pred = svm.predict(X_val)
#     LOGGER.info('validation accuracy: %f' % (np.mean(Y_val == y_val_pred), ))

#     # evaluate the performance on test set
#     final_test = True
#     if final_test:
#         y_test_pred = svm.predict(X_test)
#         LOGGER.info('test accuracy: %f' % (np.mean(Y_test == y_test_pred), ))
#         LOGGER.info(acc_classes(y_test_pred, Y_test))

#     plot_x = np.linspace(0, len(loss_svm)-1, len(loss_svm))
#     plt.title('Loss')
#     plt.plot(plot_x, loss_svm)
#     plt.savefig('loss.png')


# LOGGER.warning('---')
# train(1.e-4, 1.e0, 8000)