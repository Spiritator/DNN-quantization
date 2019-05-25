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


from models.model_library import quantized_4C2F
from utils_tool.weight_conversion import convert_original_weight_layer_name
from utils_tool.dataset_setup import dataset_setup
from utils_tool.confusion_matrix import show_confusion_matrix
from metrics.topk_metrics import top2_acc
from testing.fault_list import generate_model_stuck_fault
from testing.fault_core import generate_model_modulator
from metrics.FT_metrics import acc_loss, relative_acc, pred_miss, top2_pred_miss, pred_vary_10, pred_vary_20
from inference.evaluate import evaluate_FT

#%%
# setting parameter

weight_name='../../cifar10_4C2FBN_weight_fused_BN.h5'
model_word_length=16
model_fractional_bit=12
rounding_method='nearest'
batch_size=20
fault_rate=0.0001

#%%
# fault generation

# model for get configuration
model=quantized_4C2F(nbits=model_word_length,
                     fbits=model_fractional_bit,
                     rounding_method=rounding_method,
                     batch_size=batch_size,
                     quant_mode=None)

model_ifmap_fault_dict_list, model_ofmap_fault_dict_list, model_weight_fault_dict_list\
=generate_model_stuck_fault(model,
                            fault_rate,
                            batch_size,
                            model_word_length,
                            bit_loc_distribution='uniform',
                            bit_loc_pois_lam=2)


model_ifmap_fault_dict_list, model_ofmap_fault_dict_list, model_weight_fault_dict_list\
=generate_model_modulator(model,
                          model_word_length,
                          model_fractional_bit,
                          model_ifmap_fault_dict_list, 
                          model_ofmap_fault_dict_list, 
                          model_weight_fault_dict_list)

# FC layer no fault
#model_weight_fault_dict_list[10]=[None,None]
#model_weight_fault_dict_list[13]=[None,None]
#model_ifmap_fault_dict_list[10]=None
#model_ifmap_fault_dict_list[13]=None
#model_ofmap_fault_dict_list[10]=None
#model_ofmap_fault_dict_list[13]=None


#%%
# model setup

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

#%%
#dataset setup

x_train, x_test, y_train, y_test, class_indices, datagen, input_shape = dataset_setup('cifar10')

#%%
# view test result

t = time.time()

#test_result = model.evaluate(x_test, y_test, verbose=1, batch_size=batch_size)
from keras.losses import categorical_crossentropy
prediction = model.predict(x_test, verbose=1,batch_size=batch_size)
test_result = evaluate_FT('cifar10',prediction=prediction,test_label=y_test,loss_function=categorical_crossentropy,metrics=['accuracy',top2_acc,acc_loss,relative_acc,pred_miss,top2_pred_miss,pred_vary_10,pred_vary_20])

t = time.time()-t
print('\nruntime: %f s'%t)
for key in test_result.keys():
    print('Test %s\t:'%key, test_result[key])

#%%
# draw confusion matrix

print('\n')
#prediction = model.predict(x_test, verbose=1, batch_size=batch_size)
prediction = np.argmax(prediction, axis=1)

show_confusion_matrix(np.argmax(y_test, axis=1),prediction,class_indices,'Confusion Matrix',figsize=(8,6),normalize=False)


