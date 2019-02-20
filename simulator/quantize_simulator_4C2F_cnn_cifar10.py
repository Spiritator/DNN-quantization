# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 15:29:34 2018

@author: Yung-Yu Tsai

evaluate quantized testing result with custom Keras quantize layer 
"""

# setup

import keras
import numpy as np
import keras.backend as K
import time


from models.model_library import quantized_4C2F
from utils_tool.weight_conversion import convert_original_weight_layer_name
from utils_tool.dataset_setup import dataset_setup
from utils_tool.confusion_matrix import show_confusion_matrix
from metrics.topk_metrics import top2_acc
from approximation.estimate import comp_num_estimate

#%%
# model setup

weight_name='../../cifar10_4C2F_weight.h5'
batch_size=25


# model setup
# all augments use the same quantize precision
#model=quantized_4C2F(nbits=8,fbits=4,rounding_method='nearest')
model=quantized_4C2F(nbits=10,fbits=5,rounding_method='nearest',batch_size=batch_size,intrinsic=False)
# each augment uses different quantize precision. information list [input, weight, output]
#model=quantized_4C2F(nbits=[12,6,12],fbits=[6,3,6],rounding_method='nearest')
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy',top2_acc])
weight_name=convert_original_weight_layer_name(weight_name)
model.load_weights(weight_name)
print('orginal weight loaded')

#%%
#dataset setup

x_train, x_test, y_train, y_test, class_indices, datagen, input_shape = dataset_setup('cifar10')

#%%
# view test result

t = time.time()

test_result = model.evaluate(x_test, y_test, verbose=1, batch_size=batch_size)

t = time.time()-t

print('\nruntime: %f s'%t)
print('\nTest loss:', test_result[0])
print('Test top1 accuracy:', test_result[1])
print('Test top2 accuracy:', test_result[2])

computaion_esti=comp_num_estimate(model)
print('\nTotal # of computations:', computaion_esti['total_MAC'])
print('Total # of MAC bits:', computaion_esti['total_MAC_bits'])

#%%
# draw confusion matrix

print('\n')
prediction = model.predict(x_test, verbose=1, batch_size=batch_size)
prediction = np.argmax(prediction, axis=1)

show_confusion_matrix(np.argmax(y_test, axis=1),prediction,class_indices,'Confusion Matrix',figsize=(8,6),normalize=False)

