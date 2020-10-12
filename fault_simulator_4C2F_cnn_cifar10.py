# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 11:33:23 2018

@author: Yung-Yu Tsai

evaluate fault injection testing result of 4C2F CNN
"""

# setup

import numpy as np
import tensorflow.keras.backend as K
import time


from simulator.models.model_library import quantized_4C2F
from simulator.utils_tool.dataset_setup import dataset_setup
from simulator.utils_tool.confusion_matrix import show_confusion_matrix
from simulator.metrics.topk_metrics import top2_acc
from simulator.fault.fault_list import generate_model_stuck_fault
from simulator.fault.fault_core import generate_model_modulator
from tensorflow.keras.losses import categorical_crossentropy
from simulator.metrics.FT_metrics import acc_loss, relative_acc, pred_miss, top2_pred_miss, conf_score_vary_10, conf_score_vary_50
from simulator.inference.evaluate import evaluate_FT

#%% setting parameter

weight_name='../cifar10_4C2FBN_weight_fused_BN.h5'
model_word_length=16
model_fractional_bit=12
rounding_method='nearest'
batch_size=20
fault_rate=0.0001

#%% fault generation

# model for get configuration
model=quantized_4C2F(nbits=model_word_length,
                     fbits=model_fractional_bit,
                     rounding_method=rounding_method,
                     batch_size=batch_size,
                     quant_mode=None)

t = time.time()

model_ifmap_fault_dict_list, model_ofmap_fault_dict_list, model_weight_fault_dict_list\
=generate_model_stuck_fault(model,
                            fault_rate,
                            batch_size,
                            model_word_length,
                            param_filter=[True,True,True],
                            fast_gen=True,
                            return_modulator=True,
                            bit_loc_distribution='uniform',
                            bit_loc_pois_lam=2)


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
#model_weight_fault_dict_list[10]=[None,None]
#model_weight_fault_dict_list[13]=[None,None]
#model_ifmap_fault_dict_list[10]=None
#model_ifmap_fault_dict_list[13]=None
#model_ofmap_fault_dict_list[10]=None
#model_ofmap_fault_dict_list[13]=None


#%% model setup

t = time.time()
model=quantized_4C2F(nbits=model_word_length,
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

#%% dataset setup

x_train, x_test, y_train, y_test, class_indices, datagen, input_shape = dataset_setup('cifar10')

#%% view test result

t = time.time()

#test_result = model.evaluate(x_test, y_test, verbose=1, batch_size=batch_size)
prediction = model.predict(x_test, verbose=1,batch_size=batch_size)
test_result = evaluate_FT('cifar10',prediction=prediction,test_label=y_test,loss_function=categorical_crossentropy,metrics=['accuracy',top2_acc,acc_loss,relative_acc,pred_miss,top2_pred_miss,conf_score_vary_10,conf_score_vary_50])

t = time.time()-t
print('\nruntime: %f s'%t)
for key in test_result.keys():
    print('Test %s\t:'%key, test_result[key])

#%% draw confusion matrix

print('\n')
#prediction = model.predict(x_test, verbose=1, batch_size=batch_size)
prediction = np.argmax(prediction, axis=1)

show_confusion_matrix(np.argmax(y_test, axis=1),prediction,class_indices,'Confusion Matrix',figsize=(8,6),normalize=False)


