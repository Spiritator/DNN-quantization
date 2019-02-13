# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 11:33:23 2018

@author: Yung-Yu Tsai

evaluate quantized testing result with custom Keras quantize layer 
"""

# setup

import keras
import numpy as np
import keras.backend as K
import time


from models.model_library import quantized_lenet5
from utils_tool.weight_conversion import convert_original_weight_layer_name
from utils_tool.dataset_setup import dataset_setup
from utils_tool.confusion_matrix import show_confusion_matrix
from metrics.topk_metrics import top2_acc
from approximation.estimate import comp_num_estimate

#%%
# model setup

weight_name='../../mnist_lenet5_weight.h5'

# model setup
# all augments use the same quantize precision
#model=quantized_lenet5(nbits=8,fbits=4,rounding_method='nearest')
# each augment uses different quantize precision. information list [input, weight, output]
#model=quantized_lenet5(nbits=[10,4,10],fbits=[5,2,5],rounding_method='nearest')
# intrinsic quantization
model=quantized_lenet5(nbits=8,fbits=4,rounding_method='nearest',batch_size=1,intrinsic=True)

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy',top2_acc])
weight_name=convert_original_weight_layer_name(weight_name)
model.load_weights(weight_name)
print('orginal weight loaded')

#%%
#dataset setup

x_train, x_test, y_train, y_test, class_indices, datagen, input_shape = dataset_setup('mnist')

#%%
# view test result
t = time.time()

test_result = model.evaluate(x_test, y_test, verbose=1)

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
prediction = model.predict(x_test, verbose=1)
prediction = np.argmax(prediction, axis=1)

show_confusion_matrix(np.argmax(y_test, axis=1),prediction,class_indices,'Confusion Matrix',normalize=False)


