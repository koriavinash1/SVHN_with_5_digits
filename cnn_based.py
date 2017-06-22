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

# keras.layers.convolutional.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)


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

model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
model_vgg16_conv.summary()

input = Input(shape=(image_size, image_size, num_channels), name = 'image_input')
output_vgg16_conv = model_vgg16_conv(input)
# output_vgg16_conv = GlobalAveragePooling2D()(output_vgg16_conv)

x = Flatten(name='flatten')(output_vgg16_conv)
d1 = Dense(4096, activation='relu', name='d1fc1')(x)
d1 = Dense(1024, activation='relu', name='d1fc2')(d1)
d1 = Dense(11, activation='softmax', name='d1pred')(d1)

d2 = Dense(4096, activation='relu', name='d2fc1')(x)
d2 = Dense(1024, activation='relu', name='d2fc2')(d2)
d2 = Dense(11, activation='softmax', name='d2pred')(d2)

d3 = Dense(4096, activation='relu', name='d3fc1')(x)
d3 = Dense(1024, activation='relu', name='d3fc2')(d3)
d3 = Dense(11, activation='softmax', name='d3pred')(d3)

d4 = Dense(4096, activation='relu', name='d4fc1')(x)
d4 = Dense(1024, activation='relu', name='d4fc2')(d4)
d4 = Dense(11, activation='softmax', name='d4pred')(d4)

d5 = Dense(4096, activation='relu', name='d5fc1')(x)
d5 = Dense(1024, activation='relu', name='d5fc2')(d5)
d5 = Dense(11, activation='softmax', name='d5pred')(d5)

my_model = Model(input=input, output=[d1, d2, d3, d4, d5])


def mean_accuracy(true, pred):
    return K.mean(pred)

# + K.mean(K.variable(value=td2), pd2) + K.mean(K.variable(value=td3), pd3) + K.mean(K.variable(value=td4), pd4) + K.mean(K.variable(value=td5), pd5)

# for layer in model_vgg16_conv.layers:
#    layer.trainable = False
my_model.summary()

# sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
my_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath="../results/best_model/fn_model.{epoch:02d}-{val_acc:.6f}.hdf5", verbose=1, monitor='val_d1pred_acc', save_best_only=True, save_weights_only=False, mode='max', period=1)
tf_board = TensorBoard(log_dir='../results/logs', histogram_freq=0, write_graph=True, write_images=True)
csv_logger = CSVLogger('../results/training.log')

early_stopping = EarlyStopping(monitor='val_d5pred_loss', patience=12)

my_model.fit(train_dataset, [train_labels[:,0], train_labels[:,1], train_labels[:,2], train_labels[:,3], train_labels[:,4]], batch_size=batch_size, nb_epoch=100, validation_data = (valid_dataset, [valid_labels[:,0], valid_labels[:,1], valid_labels[:,2], valid_labels[:,3], valid_labels[:,4]]), callbacks=[tf_board, csv_logger])

my_model.save("../results/final_svhn.hdf5")
score, acc = my_model.evaluate(test_dataset, [test_labels[:,0], test_labels[:,1], test_labels[:,2], test_labels[:,3], test_labels[:,4]], batch_size=batch_size)
resultsfile = open("../results/results.txt", 'w')
resultsfile.write("test_acc : "+str(acc))
resultsfile.close()
