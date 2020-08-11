# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 13:39:35 2020

@author: Yung-Yu Tsai

evaluate computation unit fault injection testing result of LeNet-5
"""

# setup

import keras
import numpy as np
import keras.backend as K
import time


from simulator.models.model_library import quantized_lenet5
from simulator.utils_tool.dataset_setup import dataset_setup
from simulator.utils_tool.confusion_matrix import show_confusion_matrix
from simulator.metrics.topk_metrics import top2_acc
from simulator.comp_unit.PEarray import PEarray, PE_mapping_forward, PE_mapping_backward
from simulator.comp_unit.tile import tile_PE, tile_FC_PE
from simulator.comp_unit.mac import mac_unit
from simulator.testing.fault_core import generate_model_modulator
from simulator.metrics.FT_metrics import acc_loss, relative_acc, pred_miss, top2_pred_miss, conf_score_vary_10, conf_score_vary_50
from simulator.inference.evaluate import evaluate_FT

#%%
# setting parameter

weight_name='../mnist_lenet5_weight.h5'
model_word_length=8
model_fractional_bit=4
rounding_method='nearest'
batch_size=20

# PE array fault simulation parameter
mac_config='../pe_mapping_config/lenet/ws/mac_unit_config.json'
model_wl=model_word_length

memory_column_priority=['Tm','Tc','Tr','Tn']
memory_row_priority=['Tr','Tm','Tc','Tn']

fast_mode=True

#%%
# fault generation

# model for get configuration
model=quantized_lenet5(nbits=model_word_length,
                       fbits=model_fractional_bit,
                       rounding_method=rounding_method,
                       batch_size=batch_size,
                       quant_mode=None)

model_mac_math_fault_dict_list=[None for i in range(8)] 

# Tile mapping
wght_tile=tile_PE((3,3,16,32),is_fmap=False,wl=8)
ifmap_tile=tile_PE((1,28,28,16),is_fmap=True,wl=8)
ofmap_tile=tile_PE((1,28,28,32),is_fmap=True,wl=8)
# PE represent computation unit
PE=mac_unit(mac_config)
# PE array
MXU=PEarray(16,16,mac_config=PE)
# assign fault dictionary
fault_loc,fault_info=MXU.make_single_SA_fault(n_bit=model_wl, fault_type='flip')

# conv1
ofmap_tile_conv1=tile_PE((1,28,28,8),is_fmap=True,wl=model_wl)
ifmap_tile_conv1=tile_PE((1,28,28,1),is_fmap=True,wl=model_wl)
wght_tile_conv1 =tile_PE((5,5,1,8),is_fmap=False,wl=model_wl)
ofmap_config_conv1='../pe_mapping_config/lenet/ws/ofmap_config_conv1.json'
ifmap_config_conv1='../pe_mapping_config/lenet/ws/ifmap_config_conv1.json'
wght_config_conv1 ='../pe_mapping_config/lenet/ws/wght_config_conv1.json'
MXU_config='../pe_mapping_config/lenet/ws/MXU_config_conv1.json'

# conv2
ofmap_tile_conv2=tile_PE((1,14,14,36),is_fmap=True,wl=model_wl)
ifmap_tile_conv2=tile_PE((1,14,14,16),is_fmap=True,wl=model_wl)
wght_tile_conv2 =tile_PE((5,5,16,36),is_fmap=False,wl=model_wl)

# FC1
ofmap_tile_fc1=tile_FC_PE((1,8),is_fmap=True,wl=model_wl)
ifmap_tile_fc1=tile_FC_PE((1,882),is_fmap=True,wl=model_wl)
wght_tile_fc1 =tile_FC_PE((882,8),is_fmap=False,wl=model_wl)

# FC2
ofmap_tile_fc2=tile_FC_PE((1,10),is_fmap=True,wl=model_wl)
ifmap_tile_fc2=tile_FC_PE((1,128),is_fmap=True,wl=model_wl)
wght_tile_fc2 =tile_FC_PE((128,10),is_fmap=False,wl=model_wl)

# generate fault dictionary
model_ifmap_fault_dict_list[1],model_ofmap_fault_dict_list[1],model_weight_fault_dict_list[1]\
=generate_layer_memory_mapping(model.layers[1],
                               GLB_ifmap,GLB_wght,GLB_ofmap,
                               ifmap_tile_conv1,wght_tile_conv1,ofmap_tile_conv1,
                               fast_mode=fast_mode)

model_ifmap_fault_dict_list[3],model_ofmap_fault_dict_list[3],model_weight_fault_dict_list[3]\
=generate_layer_memory_mapping(model.layers[3],
                               GLB_ifmap,GLB_wght,GLB_ofmap,
                               ifmap_tile_conv2,wght_tile_conv2,ofmap_tile_conv2,
                               fast_mode=fast_mode)

model_ifmap_fault_dict_list[6],model_ofmap_fault_dict_list[6],model_weight_fault_dict_list[6]\
=generate_layer_memory_mapping(model.layers[6],
                               GLB_ifmap,GLB_wght,GLB_ofmap,
                               ifmap_tile_fc1,wght_tile_fc1,ofmap_tile_fc1,
                               fast_mode=fast_mode)

model_ifmap_fault_dict_list[7],model_ofmap_fault_dict_list[7],model_weight_fault_dict_list[7]\
=generate_layer_memory_mapping(model.layers[7],
                               GLB_ifmap,GLB_wght,GLB_ofmap,
                               ifmap_tile_fc2,wght_tile_fc2,ofmap_tile_fc2,
                               fast_mode=fast_mode)
#%%
# generate modulator

model_ifmap_fault_dict_list, model_ofmap_fault_dict_list, model_weight_fault_dict_list\
=generate_model_modulator(model,
                          model_word_length,
                          model_fractional_bit,
                          model_ifmap_fault_dict_list, 
                          model_ofmap_fault_dict_list, 
                          model_weight_fault_dict_list,
                          fast_gen=True)

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


