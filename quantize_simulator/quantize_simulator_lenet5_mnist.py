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
import pandas as pd


from models.model_library import quantized_lenet5, convert_original_weight_layer_name
from utils_tool.dataset_setup import dataset_setup
from metrics.topk_metrics import top2_acc

#%%
# model setup

weight_name='../../mnist_lenet5_weight.h5'

# model setup
model=quantized_lenet5()
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy',top2_acc])
weight_name=convert_original_weight_layer_name(weight_name)
model.load_weights(weight_name)
print('orginal weight loaded')

x_train, x_test, y_train, y_test, datagen, input_shape = dataset_setup('mnist')

test_result = model.evaluate(x_test, y_test, verbose=0)

prediction = model.predict(x_test, verbose=0)
prediction = np.argmax(prediction, axis=1)
        
print('\nTest loss:', test_result[0])
print('Test top1 accuracy:', test_result[1])
print('Test top2 accuracy:', test_result[2])

pd.crosstab(np.argmax(y_test, axis=1),prediction,
            rownames=['label'],colnames=['predict'])

