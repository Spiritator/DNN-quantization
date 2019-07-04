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
#from keras.utils import multi_gpu_model

from simulator.models.model_library import quantized_lenet5
from simulator.utils_tool.weight_conversion import convert_original_weight_layer_name
from simulator.utils_tool.dataset_setup import dataset_setup
from simulator.utils_tool.confusion_matrix import show_confusion_matrix
from simulator.metrics.topk_metrics import top2_acc
from simulator.metrics.FT_metrics import acc_loss, relative_acc, pred_miss, top2_pred_miss, conf_score_vary_10, conf_score_vary_20
from simulator.approximation.estimate import comp_num_estimate
from simulator.inference.evaluate import evaluate_FT

#%%
# model setup

weight_name='../mnist_lenet5_weight.h5'
batch_size=25

# model setup
# all augments use the same quantize precision
#model=quantized_lenet5(nbits=8,fbits=4,rounding_method='nearest')
# each augment uses different quantize precision. information list [input, weight, output]
#model=quantized_lenet5(nbits=[10,4,10],fbits=[5,2,5],rounding_method='nearest')
# intrinsic quantization
#model=quantized_lenet5(nbits=8,fbits=2,rounding_method='nearest',batch_size=batch_size,quant_mode='hybrid',overflow_mode=[True,False,True])
#model=quantized_lenet5(nbits=8,fbits=3,rounding_method='nearest',batch_size=batch_size,quant_mode='hybrid')
model=quantized_lenet5(nbits=8,fbits=3,rounding_method=['down','nearest','down'],batch_size=batch_size,quant_mode='hybrid')
#model=quantized_lenet5(nbits=8,fbits=[6,6,6],rounding_method=['down','nearest','down'],batch_size=batch_size,quant_mode='intrinsic')

weight_name=convert_original_weight_layer_name(weight_name)
model.load_weights(weight_name)
print('orginal weight loaded')

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy',top2_acc])
#model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy',top2_acc,acc_loss,relative_acc,pred_miss,top2_pred_miss,pred_vary_10,pred_vary_20])

# multi GPU
#parallel_model = multi_gpu_model(model, gpus=2)
#parallel_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy',top2_acc])


#%%
#dataset setup

x_train, x_test, y_train, y_test, class_indices, datagen, input_shape = dataset_setup('mnist')

#%%
# view test result
t = time.time()

#test_result = model.evaluate(x_test, y_test, verbose=1, batch_size=batch_size)
from keras.losses import categorical_crossentropy
prediction = model.predict(x_test, verbose=1,batch_size=batch_size)
test_result = evaluate_FT('lenet',prediction=prediction,test_label=y_test,loss_function=categorical_crossentropy,metrics=['accuracy',top2_acc,acc_loss,relative_acc,pred_miss,top2_pred_miss,conf_score_vary_10,conf_score_vary_20])

t = time.time()-t
print('\nruntime: %f s'%t)
for key in test_result.keys():
    print('Test %s\t:'%key, test_result[key])

computaion_esti=comp_num_estimate(model)
print('\nTotal # of computations:', computaion_esti['total_MAC'])
print('Total # of MAC bits:', computaion_esti['total_MAC_bits'])

#%%
# draw confusion matrix

print('\n')
#prediction = model.predict(x_test, verbose=1,batch_size=batch_size)
prediction = np.argmax(prediction, axis=1)

show_confusion_matrix(np.argmax(y_test, axis=1),prediction,class_indices,'Confusion Matrix',normalize=False)


