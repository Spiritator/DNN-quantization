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
import scipy


from models.model_library import quantized_lenet5
from utils_tool.weight_conversion import convert_original_weight_layer_name
from utils_tool.dataset_setup import dataset_setup
from utils_tool.confusion_matrix import show_confusion_matrix
from metrics.topk_metrics import top2_acc
from testing.fault_list import generate_model_stuck_fault
from testing.fault_core import generate_model_modulator
from metrics.FT_metrics import acc_loss, relative_acc, pred_miss, top2_pred_miss, conf_score_vary_10, conf_score_vary_50
from inference.evaluate import evaluate_FT

#%%
# setting parameter

weight_name='../../mnist_lenet5_weight.h5'
model_word_length=8
model_fractional_bit=3
rounding_method='nearest'
batch_size=20
fault_rate=0.0001

#%%
# fault generation

# model for get configuration
model=quantized_lenet5(nbits=model_word_length,
                       fbits=model_fractional_bit,
                       rounding_method=rounding_method,
                       batch_size=batch_size,
                       quant_mode=None)
t = time.time()

model_ifmap_fault_dict_list, model_ofmap_fault_dict_list, model_weight_fault_dict_list\
=generate_model_stuck_fault(model,fault_rate,
                            batch_size,
                            model_word_length,
                            param_filter=[True,True,True],
                            fast_gen=True,
                            return_modulator=True,
                            bit_loc_distribution='uniform',
                            bit_loc_pois_lam=None,
                            fault_type='flip')

#model_ifmap_fault_dict_list, model_ofmap_fault_dict_list, model_weight_fault_dict_list\
#=generate_model_modulator(model,
#                          model_word_length,
#                          model_fractional_bit,
#                          model_ifmap_fault_dict_list, 
#                          model_ofmap_fault_dict_list, 
#                          model_weight_fault_dict_list,
#                          fast_gen=True)
t = time.time()-t
print('\nfault gen time: %f s'%t)

# FC layer no fault
#model_weight_fault_dict_list[6]=[None,None]
#model_weight_fault_dict_list[7]=[None,None]
#model_ifmap_fault_dict_list[6]=None
#model_ifmap_fault_dict_list[7]=None
#model_ofmap_fault_dict_list[6]=None
#model_ofmap_fault_dict_list[7]=None

#%%
# model setup

t = time.time()
model=quantized_lenet5(nbits=model_word_length,
                       fbits=model_fractional_bit,
                       rounding_method=rounding_method,
                       batch_size=batch_size,
                       quant_mode='hybrid',
                       ifmap_fault_dict_list=model_ifmap_fault_dict_list,
                       ofmap_fault_dict_list=model_ofmap_fault_dict_list,
                       weight_fault_dict_list=model_weight_fault_dict_list)
t = time.time()-t
print('\nModel build time: %f s'%t)

print('Model compiling...')
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy',top2_acc])
print('Model compiled !')
model.load_weights(weight_name)
print('orginal weight loaded')

#%%
#dataset setup

x_train, x_test, y_train, y_test, class_indices, datagen, input_shape = dataset_setup('mnist')

#%%
# view test result

t = time.time()

#test_result = model.evaluate(x_test, y_test, verbose=1, batch_size=batch_size)
from keras.losses import categorical_crossentropy
prediction = model.predict(x_test, verbose=1,batch_size=batch_size)
test_result = evaluate_FT('lenet',prediction=prediction,test_label=y_test,loss_function=categorical_crossentropy,metrics=['accuracy',top2_acc,acc_loss,relative_acc,pred_miss,top2_pred_miss,conf_score_vary_10,conf_score_vary_50])

t = time.time()-t
print('\nruntime: %f s'%t)
for key in test_result.keys():
    print('Test %s\t:'%key, test_result[key])

#%%
# draw confusion matrix

print('\n')
#prediction = model.predict(x_test, verbose=1, batch_size=batch_size)
prediction = np.argmax(prediction, axis=1)

show_confusion_matrix(np.argmax(y_test, axis=1),prediction,class_indices,'Confusion Matrix',normalize=False)


