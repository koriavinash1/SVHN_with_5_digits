from __future__ import print_function
from six.moves import cPickle as pickle
from six.moves import range
import tensorflow as tf
from keras.preprocessing import image
from keras.models import Model
from keras import optimizers
from keras.fr_cnn import VGG19, losses
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

# keras.layers.convolutional.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)


pickle_file = '../dataset/SVHN_multi_BBox.pickle'
image_size = 32
num_labels = 11 # 0-9, + blank 
num_channels = 5 # HSV with x,y pos

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

with open(pickle_file, 'rb') as f:
	save = pickle.load(f)
	train_dataset = pre_processing(save['train_dataset'])
	train_labels = save['train_labels']
	valid_dataset = pre_processing(save['valid_dataset'])
	valid_labels = save['valid_labels']
	test_dataset = pre_processing(save['test_dataset'])
	test_labels = save['test_labels']
	del save  # hint to help gc free up memory
	print('Training set', train_dataset.shape, train_labels.shape)
	print('Validation set', valid_dataset.shape, valid_labels.shape)
	print('Test set', test_dataset.shape, test_labels.shape)

model_vgg19_conv = VGG19(weights='imagenet', include_top=False)
model_vgg19_conv.summary()

img_input = Input(shape=(image_size, image_size, num_channels), name = 'image_input')
shared_layers = VGG19.nn_base(img_input, trainable=True)

output_vgg16_conv = model_vgg19_conv(input)

# Faster RCNN implementation
# RPN implementation

anchor_box_scales = [30, 60, 90, 120]
anchor_box_ratios = [[1, 1], [1, 2], [2, 1]]
# number of ROIs at once

num_anchors = len(anchor_box_scales) * len(anchor_box_ratios)
rpn = nn.rpn(shared_layers, num_anchors)

rpn = VGG19.rpn(shared_inputs, )
