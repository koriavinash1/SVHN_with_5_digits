from __future__ import print_function
from six.moves import cPickle as pickle
from six.moves import range
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model
from keras import optimizers
from keras.layers import Input, Flatten, Dense, GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.callbacks import CSVLogger
import keras.backend as K
import numpy as np
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


pickle_file = '../dataset/SVHN_multi.pickle'
image_size = 32
num_labels = 11 # 0-9, + blank 
num_channels = 3 # grayscale with x,y pos

batch_size = 64
patch_size = 5
depth1 = 16
depth2 = 32
depth3 = 64
num_hidden1 = 64
shape = [batch_size, image_size, image_size, num_channels]

nx, ny = (image_size, image_size)
xt = np.linspace(0, 1, nx)
yt = np.linspace(0, 1, ny)
xpos, ypos = np.meshgrid(xt, yt)

def pre_processing(images):
    processed_images = []
    for image in images:
        # print(image.shape)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.reshape(image, (image_size,image_size))
        asd = np.swapaxes(np.swapaxes(np.array(np.concatenate([[image], [xpos], [ypos]])), 0, 1), 1, 2)
        # resized = cv2.resize(asd, (32, 32, 3), interpolation = cv2.INTER_CUBIC)
        processed_images.append(asd)
    return np.array(processed_images)

def label_processing(labels):
    processed_labels = []
    for label in labels:
        new_label = []
        for i in range(1, 6):
            test = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            test[label[i]] = 1
            new_label.append(test)
        processed_labels.append(np.array(new_label, ndmin=2))
    return np.array(processed_labels, ndmin = 3)

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = pre_processing(save['train_dataset'])
    train_labels = label_processing(save['train_labels'])
    valid_dataset = pre_processing(save['valid_dataset'])
    valid_labels = label_processing(save['valid_labels'])
    test_dataset = pre_processing(save['test_dataset'])
    test_labels = label_processing(save['test_labels'])
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
             print('Test set', test_dataset.shape, test_labels.shape)

def main_RL(image, patch_size):
    for x