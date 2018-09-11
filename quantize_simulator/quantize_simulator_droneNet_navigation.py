# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 17:17:07 2018

@author: Yung-Yu Tsai

evaluate quantized testing result with custom Keras quantize layer 
"""

# setup

import keras
import numpy as np
from keras.layers import Input
import keras.backend as K
import time
import pandas as pd


from models.model_library import quantized_droneNet, convert_original_weight_layer_name
from utils_tool.dataset_setup import dataset_setup
from metrics.topk_metrics import top2_acc

# dimensions of our images.
img_width, img_height = 250, 140

class_number=4

weight_name='../../navigation_droneNet_v2_250x140_weight.h5'
dataset_dir='../../navigation_dataset/validation'
nb_validation_samples = 1450
batch_size = 50

if K.image_data_format() == 'channels_first':
    input_shape = Input(shape=(3, img_width, img_height))
else:
    input_shape = Input(shape=(img_width, img_height, 3))

#%%
# model setup

model=quantized_droneNet(2, nbits=8, fbits=4, BN_nbits=10, BN_fbits=5, rounding_method='nearest', inputs=input_shape,  include_top=True, classes=class_number)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy',top2_acc])
weight_name=convert_original_weight_layer_name(weight_name)
model.load_weights(weight_name)
print('orginal weight loaded')

x_train, x_test, y_train, y_test, datagen, input_shape = dataset_setup('ImageDataGenerator', img_rows = img_width, img_cols = img_height, data_augmentation = False, data_dir = dataset_dir)

t = time.time()

test_result = model.evaluate_generator(datagen, steps=nb_validation_samples//batch_size)

t = time.time()-t

prediction = model.predict_generator(datagen,nb_validation_samples//batch_size)
prediction = np.argmax(prediction, axis=1)
        
print('\nTest loss:', test_result[0])
print('Test top1 accuracy:', test_result[1])
print('Test top2 accuracy:', test_result[2])

print(datagen.class_indices)
pd.crosstab(datagen.classes,prediction,
            rownames=['label'],colnames=['predict'])


