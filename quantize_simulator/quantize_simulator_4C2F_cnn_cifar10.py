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
import pandas as pd


from models.model_library import quantized_4C2F, convert_original_weight_layer_name
from utils_tool.dataset_setup import dataset_setup
from metrics.topk_metrics import top2_acc

#%%
# model setup

weight_name='../../cifar10_4C2F_weight.h5'

# model setup
model=quantized_4C2F(nbits=10,fbits=5,rounding_method='nearest')
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy',top2_acc])
weight_name=convert_original_weight_layer_name(weight_name)
model.load_weights(weight_name)
print('orginal weight loaded')

x_train, x_test, y_train, y_test, datagen, input_shape = dataset_setup('cifar10')

t = time.time()

test_result = model.evaluate(x_test, y_test, verbose=0)

t = time.time()-t

prediction = model.predict(x_test, verbose=0)
prediction = np.argmax(prediction, axis=1)
        
print('\nruntime: %f s'%t)
print('\nTest loss:', test_result[0])
print('Test top1 accuracy:', test_result[1])
print('Test top2 accuracy:', test_result[2])

pd.crosstab(np.argmax(y_test, axis=1),prediction,
            rownames=['label'],colnames=['predict'])

