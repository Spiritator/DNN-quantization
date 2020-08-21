# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 10:31:37 2018

@author: Yung-Yu Tsai

evaluate accuracy of model weight
"""


#setup

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Input
from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras.datasets import mnist
from simulator.utils_tool.confusion_matrix import show_confusion_matrix
import numpy as np
import time

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

#%%
# model setup

model=load_model('../mnist_lenet5_model.h5')
model.load_weights('../mnist_lenet5_weight.h5')

model.summary()

#%%
# evaluate model

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

t = time.time()

test_result = model.evaluate(x_test, y_test, verbose=1)

t = time.time()-t

prediction = model.predict(x_test, verbose=1)
prediction = np.argmax(prediction, axis=1)
        
print('\nruntime: %f s'%t)
print('\nTest loss:', test_result[0])
print('Test accuracy:', test_result[1])

show_confusion_matrix(np.argmax(y_test, axis=1),prediction,np.arange(0,10),'Confusion Matrix',normalize=False)

