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

from simulator.comp_unit.PEarray import PEarray
from simulator.comp_unit.tile import tile_PE, tile_FC_PE
from simulator.comp_unit.mac import mac_unit
from simulator.comp_unit.mapping_flow import PE_mapping_forward, PE_mapping_backward, mapping_valid_checker

from simulator.metrics.FT_metrics import acc_loss, relative_acc, pred_miss, top2_pred_miss, conf_score_vary_10, conf_score_vary_50
from simulator.inference.evaluate import evaluate_FT

#%%
# setting parameter

weight_name='../mnist_lenet5_weight.h5'
model_word_length=8
model_fractional_bit=3
rounding_method=['down','nearest','down']
batch_size=20

# PE array fault simulation parameter
mac_config='../pe_mapping_config/lenet/ws/mac_unit_config.json'
model_wl=model_word_length

#%%
# dataflow setup

# model for get configuration
model=quantized_lenet5(nbits=model_word_length,
                       fbits=model_fractional_bit,
                       rounding_method=rounding_method,
                       batch_size=batch_size,
                       quant_mode=None)

model_mac_math_fault_dict_list=[None for i in range(8)] 

# PE represent computation unit
PE=mac_unit(mac_config)
# PE array
MXU=PEarray(8,8,mac_config=PE)
# assign fault dictionary
fault_loc,fault_info=MXU.make_single_SA_fault(n_bit=model_wl, fault_type='flip')

# conv1
ofmap_tile_conv1=tile_PE((1,28,28,8),is_fmap=True,wl=model_wl)
ifmap_tile_conv1=tile_PE((1,28,28,1),is_fmap=True,wl=model_wl)
wght_tile_conv1 =tile_PE((5,5,1,8),is_fmap=False,wl=model_wl)
ofmap_config_conv1='../pe_mapping_config/lenet/ws/ofmap_config_conv1.json'
ifmap_config_conv1='../pe_mapping_config/lenet/ws/ifmap_config_conv1.json'
wght_config_conv1 ='../pe_mapping_config/lenet/ws/wght_config_conv1.json'
MXU_config_conv1  ='../pe_mapping_config/lenet/ws/MXU_config_conv1.json'

#check=mapping_valid_checker(ifmap_tile_conv1,wght_tile_conv1,ofmap_tile_conv1,MXU,
#                            ifmap_config_conv1,wght_config_conv1,ofmap_config_conv1,MXU_config_conv1,
#                            print_detail=True)
#MXU.clear_all()

# conv2
ofmap_tile_conv2=tile_PE((1,14,14,16),is_fmap=True,wl=model_wl)
ifmap_tile_conv2=tile_PE((1,14,14,16),is_fmap=True,wl=model_wl)
wght_tile_conv2 =tile_PE((5,5,16,16),is_fmap=False,wl=model_wl)
ofmap_config_conv2='../pe_mapping_config/lenet/ws/ofmap_config_conv2.json'
ifmap_config_conv2='../pe_mapping_config/lenet/ws/ifmap_config_conv2.json'
wght_config_conv2 ='../pe_mapping_config/lenet/ws/wght_config_conv2.json'
MXU_config_conv2  ='../pe_mapping_config/lenet/ws/MXU_config_conv2.json'

#check=mapping_valid_checker(ifmap_tile_conv2,wght_tile_conv2,ofmap_tile_conv2,MXU,
#                            ifmap_config_conv2,wght_config_conv2,ofmap_config_conv2,MXU_config_conv2,
#                            print_detail=True)
#MXU.clear_all()

## FC1
#ofmap_tile_fc1=tile_FC_PE((1,8),is_fmap=True,wl=model_wl)
#ifmap_tile_fc1=tile_FC_PE((1,882),is_fmap=True,wl=model_wl)
#wght_tile_fc1 =tile_FC_PE((882,8),is_fmap=False,wl=model_wl)
#ofmap_config_fc1='../pe_mapping_config/lenet/ws/ofmap_config_fc1.json'
#ifmap_config_fc1='../pe_mapping_config/lenet/ws/ifmap_config_fc1.json'
#wght_config_fc1 ='../pe_mapping_config/lenet/ws/wght_config_fc1.json'
#MXU_config_fc1  ='../pe_mapping_config/lenet/ws/MXU_config_fc1.json'

#check=mapping_valid_checker(ifmap_tile_fc1,wght_tile_fc1,ofmap_tile_fc1,MXU,
#                            ifmap_config_fc1,wght_config_fc1,ofmap_config_fc1,MXU_config_fc1,
#                            print_detail=True)
#MXU.clear_all()

## FC2
#ofmap_tile_fc2=tile_FC_PE((1,10),is_fmap=True,wl=model_wl)
#ifmap_tile_fc2=tile_FC_PE((1,128),is_fmap=True,wl=model_wl)
#wght_tile_fc2 =tile_FC_PE((128,10),is_fmap=False,wl=model_wl)
#ofmap_config_fc2='../pe_mapping_config/lenet/ws/ofmap_config_fc2.json'
#ifmap_config_fc2='../pe_mapping_config/lenet/ws/ifmap_config_fc2.json'
#wght_config_fc2 ='../pe_mapping_config/lenet/ws/wght_config_fc2.json'
#MXU_config_fc2  ='../pe_mapping_config/lenet/ws/MXU_config_fc2.json'

#check=mapping_valid_checker(ifmap_tile_fc2,wght_tile_fc2,ofmap_tile_fc2,MXU,
#                            ifmap_config_fc2,wght_config_fc2,ofmap_config_fc2,MXU_config_fc2,
#                            print_detail=True)
#MXU.clear_all()

#%%
# generate fault dictionary
PE_mapping_forward(ifmap_tile_conv1,wght_tile_conv1,ofmap_tile_conv1,MXU,
                   ifmap_config_conv1,wght_config_conv1,ofmap_config_conv1,MXU_config_conv1,
                   pre_plan=True,print_detail=True)
MXU.gen_PEarray_permanent_fault_dict(fault_loc, fault_info, mac_config=True)
model_mac_math_fault_dict_list[1] = PE_mapping_backward(model.layers[1], MXU, print_detail=True)
MXU.clear_all()

PE_mapping_forward(ifmap_tile_conv2,wght_tile_conv2,ofmap_tile_conv2,MXU,
                   ifmap_config_conv2,wght_config_conv2,ofmap_config_conv2,MXU_config_conv2,
                   pre_plan=True,print_detail=True)
MXU.gen_PEarray_permanent_fault_dict(fault_loc, fault_info, mac_config=True)
model_mac_math_fault_dict_list[3] = PE_mapping_backward(model.layers[3], MXU, print_detail=True)
MXU.clear_all()

#PE_mapping_forward(ifmap_tile_fc1,wght_tile_fc1,ofmap_tile_fc1,MXU,
#                   ifmap_config_fc1,wght_config_fc1,ofmap_config_fc1,MXU_config_fc1,
#                   pre_plan=True,print_detail=True)
#MXU.gen_PEarray_permanent_fault_dict(fault_loc, fault_info, mac_config=True)
#model_mac_math_fault_dict_list[6] = PE_mapping_backward(model.layers[7], MXU, print_detail=True)
#MXU.clear_all()
#
#PE_mapping_forward(ifmap_tile_fc2,wght_tile_fc2,ofmap_tile_fc2,MXU,
#                   ifmap_config_fc2,wght_config_fc2,ofmap_config_fc2,MXU_config_fc2,
#                   pre_plan=True,print_detail=True)
#MXU.gen_PEarray_permanent_fault_dict(fault_loc, fault_info, mac_config=True)
#model_mac_math_fault_dict_list[7] = PE_mapping_backward(model.layers[7], MXU, print_detail=True)
#MXU.clear_all()

K.clear_session()

#%%
# model setup

t = time.time()
model=quantized_lenet5(nbits=model_word_length,
                       fbits=model_fractional_bit,
                       rounding_method=rounding_method,
                       batch_size=batch_size,
                       quant_mode='hybrid',
                       ofmap_fault_dict_list=model_mac_math_fault_dict_list,
                       mac_unit=PE)
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


