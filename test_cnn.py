import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import load_model
from keras import optimizers
from keras.layers.convolutional import Conv2D
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.callbacks import CSVLogger
import numpy as np
import time
import math
import cv2
import os
import sys
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

cam = cv2.VideoCapture(camport)
time.sleep(0.1)

# some input arrays
nx, ny = (32, 32)
xt = np.linspace(0, 1, nx)
yt = np.linspace(0, 1, ny)
xpos, ypos = np.meshgrid(xt, yt)

def cap_img():
    ret, im=cam.read()
    cv2.imshow("orig_image",im)
    cv2.waitKey(10)
    return im

def pre_processing(image, flag=0):
    # print "Pre-processing the image...."
    channel_1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if flag == 1:
        ret, channel_1 = cv2.threshold(channel_1, 90, 255, cv2.THRESH_BINARY_INV)
        cv2.imshow(str(IS_POSITION_BASED) + "test_image", channel_1)
        cv2.waitKey(50)
    resized = cv2.resize(channel_1, (32, 32), interpolation = cv2.INTER_CUBIC)
    cv2.imwrite("grey_input_image.png", resized)
    # unrolled = np.array(resized, dtype="float32").flatten()
    asd = np.swapaxes(np.swapaxes(np.array(np.concatenate([[resized], [xpos], [ypos]])), 0, 1), 1, 2)
    cv2.imwrite("position_input_image.png", asd)
    return np.array(asd, ndmin=3)

model = load_model("../results/final_svhn.hdf5")

subimages = []
load_int = input("Press 1 load an image or 2 to take new image or 3 for real time:  ")
load_str = "abcd123321"

if load_int == 1:
    while not os.path.exists(load_str):
    	load_str = input("Enter image path...  ")
    test_image = cv2.imread(load_str)
    batch_x = pre_processing(test_image, 0)
    result_out = model.predict(batch_x)
    print "IMAGE UPLOADED IS:  {}".format(result_out)
elif load_int == 2:
    t = time.time()
    print "IMAGE WILL BE TAKEN AFTER 5sec."
    while time.time() - t < 10:
        cap_image = cap_img()
    test_image = cap_img()
    cam.release()
    batch_x = pre_processing(test_image, 1)
    result_out = model.predict(batch_x)
    # find_contours(test_image)
    print "IMAGE UPLOADED IS:  {}".format(result_out)
elif load_int == 3:
    while True:
        image = cap_img()
        batch_x = pre_processing(image, 1)
        result_out = model.predict(batch_x)
        cv2.putText(image, str(result_out[0]), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.imshow("original image", image)
        # cv2.waitKey(1)



# """
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# input_1 (InputLayer)         (None, None, None, 3)     0         
# _________________________________________________________________
# block1_conv1 (Conv2D)        (None, None, None, 64)    1792      
# _________________________________________________________________
# block1_conv2 (Conv2D)        (None, None, None, 64)    36928     
# _________________________________________________________________
# block1_pool (MaxPooling2D)   (None, None, None, 64)    0         
# _________________________________________________________________
# block2_conv1 (Conv2D)        (None, None, None, 128)   73856     
# _________________________________________________________________
# block2_conv2 (Conv2D)        (None, None, None, 128)   147584    
# _________________________________________________________________
# block2_pool (MaxPooling2D)   (None, None, None, 128)   0         
# _________________________________________________________________
# block3_conv1 (Conv2D)        (None, None, None, 256)   295168    
# _________________________________________________________________
# block3_conv2 (Conv2D)        (None, None, None, 256)   590080    
# _________________________________________________________________
# block3_conv3 (Conv2D)        (None, None, None, 256)   590080    
# _________________________________________________________________
# block3_pool (MaxPooling2D)   (None, None, None, 256)   0         
# _________________________________________________________________
# block4_conv1 (Conv2D)        (None, None, None, 512)   1180160   
# _________________________________________________________________
# block4_conv2 (Conv2D)        (None, None, None, 512)   2359808   
# _________________________________________________________________
# block4_conv3 (Conv2D)        (None, None, None, 512)   2359808   
# _________________________________________________________________
# block4_pool (MaxPooling2D)   (None, None, None, 512)   0         
# _________________________________________________________________
# block5_conv1 (Conv2D)        (None, None, None, 512)   2359808   
# _________________________________________________________________
# block5_conv2 (Conv2D)        (None, None, None, 512)   2359808   
# _________________________________________________________________
# block5_conv3 (Conv2D)        (None, None, None, 512)   2359808   
# _________________________________________________________________
# block5_pool (MaxPooling2D)   (None, None, None, 512)   0         
# =================================================================
# Total params: 14,714,688
# Trainable params: 14,714,688
# Non-trainable params: 0
# _________________________________________________________________
# cnn_based.py:81: UserWarning: Update your `Model` call to the Keras 2 API: `Model(outputs=Tensor("pr..., inputs=Tensor("im...)`
#   my_model = Model(input=input, output=x)
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# image_input (InputLayer)     (None, 32, 32, 3)         0         
# _________________________________________________________________
# vgg16 (Model)                multiple                  14714688  
# _________________________________________________________________
# flatten (Flatten)            (None, 512)               0         
# _________________________________________________________________
# fc1 (Dense)                  (None, 4096)              2101248   
# _________________________________________________________________
# fc2 (Dense)                  (None, 4096)              16781312  
# _________________________________________________________________
# fc3 (Dense)                  (None, 1024)              4195328   
# _________________________________________________________________
# fc4 (Dense)                  (None, 1024)              1049600   
# _________________________________________________________________
# predictions (Dense)          (None, 6)                 6150      
# =================================================================
# Total params: 38,848,326
# Trainable params: 38,848,326
# Non-trainable params: 0
# _________________________________________________________________
# """