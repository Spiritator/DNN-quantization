# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 13:39:35 2020

@author: Yung-Yu Tsai

evaluate computation unit fault injection testing result of LeNet-5
"""

# setup

import numpy as np
import tensorflow.keras.backend as K
import time, os, pickle

from simulator.models.model_library import quantized_lenet5
from simulator.utils_tool.dataset_setup import dataset_setup
from simulator.utils_tool.confusion_matrix import show_confusion_matrix
from simulator.metrics.topk_metrics import top2_acc

from simulator.comp_unit.PEarray import PEarray
from simulator.comp_unit.tile import tile_PE, tile_FC_PE
from simulator.comp_unit.mac import mac_unit, preprocess_model_mac_fault
from simulator.comp_unit.mapping_flow import PE_mapping_forward, PE_mapping_backward, mapping_valid_checker

from simulator.metrics.FT_metrics import acc_loss, relative_acc, pred_miss, top2_pred_miss, conf_score_vary_10, conf_score_vary_50
from simulator.inference.evaluate import evaluate_FT

#%% setting parameter

noise_inject=True

weight_name='../mnist_lenet5_weight.h5'
model_word_length=8
model_fractional_bit=3
rounding_method=['down','nearest','down']
batch_size=20

# PE array fault simulation parameter
config_dir='../pe_mapping_config'
network_dir='lenet'
dataflow_dir='ws'
config_dir=os.path.join(config_dir, network_dir, dataflow_dir)
mac_config=os.path.join(config_dir,'mac_unit_config.json')
model_wl=model_word_length

#%% dataflow setup

# model for get configuration
model=quantized_lenet5(nbits=model_word_length,
                       fbits=model_fractional_bit,
                       rounding_method=rounding_method,
                       batch_size=batch_size,
                       quant_mode=None)


# PE represent computation unit
PE=mac_unit(mac_config, noise_inject=noise_inject)
# PE array
MXU=PEarray(8,8,mac_config=PE)

# conv1
ofmap_tile_conv1=tile_PE((1,28,28,8),is_fmap=True,wl=model_wl)
ifmap_tile_conv1=tile_PE((1,28,28,1),is_fmap=True,wl=model_wl)
wght_tile_conv1 =tile_PE((5,5,1,8),is_fmap=False,wl=model_wl)
ofmap_config_conv1=os.path.join(config_dir,'ofmap_config_conv1.json')
ifmap_config_conv1=os.path.join(config_dir,'ifmap_config_conv1.json')
wght_config_conv1 =os.path.join(config_dir,'wght_config_conv1.json')
MXU_config_conv1  =os.path.join(config_dir,'MXU_config_conv1.json')

#check=mapping_valid_checker(ifmap_tile_conv1,wght_tile_conv1,ofmap_tile_conv1,MXU,
#                            ifmap_config_conv1,wght_config_conv1,ofmap_config_conv1,MXU_config_conv1,
#                            print_detail=True)
#MXU.clear_all()

# conv2
ofmap_tile_conv2=tile_PE((1,14,14,16),is_fmap=True,wl=model_wl)
ifmap_tile_conv2=tile_PE((1,14,14,16),is_fmap=True,wl=model_wl)
wght_tile_conv2 =tile_PE((5,5,16,16),is_fmap=False,wl=model_wl)
ofmap_config_conv2=os.path.join(config_dir,'ofmap_config_conv2.json')
ifmap_config_conv2=os.path.join(config_dir,'ifmap_config_conv2.json')
wght_config_conv2 =os.path.join(config_dir,'wght_config_conv2.json')
MXU_config_conv2  =os.path.join(config_dir,'MXU_config_conv2.json')

#check=mapping_valid_checker(ifmap_tile_conv2,wght_tile_conv2,ofmap_tile_conv2,MXU,
#                            ifmap_config_conv2,wght_config_conv2,ofmap_config_conv2,MXU_config_conv2,
#                            print_detail=True)
#MXU.clear_all()

## FC1
#ofmap_tile_fc1=tile_FC_PE((1,8),is_fmap=True,wl=model_wl)
#ifmap_tile_fc1=tile_FC_PE((1,882),is_fmap=True,wl=model_wl)
#wght_tile_fc1 =tile_FC_PE((882,8),is_fmap=False,wl=model_wl)
#ofmap_config_fc1=os.path.join(config_dir,'ofmap_config_fc1.json')
#ifmap_config_fc1=os.path.join(config_dir,'ifmap_config_fc1.json')
#wght_config_fc1 =os.path.join(config_dir,'wght_config_fc1.json')
#MXU_config_fc1  =os.path.join(config_dir,'MXU_config_fc1.json')

#check=mapping_valid_checker(ifmap_tile_fc1,wght_tile_fc1,ofmap_tile_fc1,MXU,
#                            ifmap_config_fc1,wght_config_fc1,ofmap_config_fc1,MXU_config_fc1,
#                            print_detail=True)
#MXU.clear_all()

## FC2
#ofmap_tile_fc2=tile_FC_PE((1,10),is_fmap=True,wl=model_wl)
#ifmap_tile_fc2=tile_FC_PE((1,128),is_fmap=True,wl=model_wl)
#wght_tile_fc2 =tile_FC_PE((128,10),is_fmap=False,wl=model_wl)
#ofmap_config_fc2=os.path.join(config_dir,'ofmap_config_fc2.json')
#ifmap_config_fc2=os.path.join(config_dir,'ifmap_config_fc2.json')
#wght_config_fc2 =os.path.join(config_dir,'wght_config_fc2.json')
#MXU_config_fc2  =os.path.join(config_dir,'MXU_config_fc2.json')

#check=mapping_valid_checker(ifmap_tile_fc2,wght_tile_fc2,ofmap_tile_fc2,MXU,
#                            ifmap_config_fc2,wght_config_fc2,ofmap_config_fc2,MXU_config_fc2,
#                            print_detail=True)
#MXU.clear_all()

#%% generate fault dictionary

# assign fault dictionary
fault_loc,fault_info=MXU.make_single_SA_fault(n_bit=model_wl, fault_type='flip')

model_mac_math_fault_dict_list=[None for i in range(8)] 

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

# make preprocess data
with open('../test_fault_dictionary_stuff/wght_distribution_info_lenet.pickle', 'rb') as fdfile:
    lenet_wght_distribution_info = pickle.load(fdfile)
with open('../test_fault_dictionary_stuff/ifmap_distribution_info_lenet.pickle', 'rb') as fdfile:
    lenet_ifmap_distribution_info = pickle.load(fdfile)
    
model_preprocess_data_list=preprocess_model_mac_fault(model, PE, model_mac_math_fault_dict_list,
                                                      model_fmap_dist_stat_list=lenet_ifmap_distribution_info,
                                                      model_wght_dist_stat_list=lenet_wght_distribution_info)

K.clear_session()

#%% model setup

t = time.time()
model=quantized_lenet5(nbits=model_word_length,
                       fbits=model_fractional_bit,
                       rounding_method=rounding_method,
                       batch_size=batch_size,
                       quant_mode='hybrid',
                       ofmap_fault_dict_list=model_preprocess_data_list,
                       mac_unit=PE)
t = time.time()-t
print('\nModel build time: %f s'%t)

print('Model compiling...')
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy',top2_acc])
print('Model compiled !')
model.load_weights(weight_name)
print('orginal weight loaded')

#%% dataset setup

x_train, x_test, y_train, y_test, class_indices, datagen, input_shape = dataset_setup('mnist')

#%% view test result

t = time.time()

#test_result = model.evaluate(x_test, y_test, verbose=1, batch_size=batch_size)
from tensorflow.keras.losses import categorical_crossentropy
prediction = model.predict(x_test, verbose=1,batch_size=batch_size)
test_result = evaluate_FT('lenet',prediction=prediction,test_label=y_test,loss_function=categorical_crossentropy,metrics=['accuracy',top2_acc,acc_loss,relative_acc,pred_miss,top2_pred_miss,conf_score_vary_10,conf_score_vary_50])

t = time.time()-t
print('\nruntime: %f s'%t)
for key in test_result.keys():
    print('Test %s\t:'%key, test_result[key])

#%% draw confusion matrix

print('\n')
#prediction = model.predict(x_test, verbose=1, batch_size=batch_size)
prediction = np.argmax(prediction, axis=1)

show_confusion_matrix(np.argmax(y_test, axis=1),prediction,class_indices,'Confusion Matrix',normalize=False)


