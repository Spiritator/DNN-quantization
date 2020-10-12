# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 22:57:31 2020

@author: Yung-Yu Tsai

evaluate computation unit fault injection testing result of 4C2F CNN
"""

import numpy as np
import tensorflow.keras.backend as K
import time, os, pickle

from simulator.models.model_library import quantized_4C2F
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

noise_inject=False

weight_name='../cifar10_4C2FBN_weight_fused_BN.h5'
model_word_length=16
model_fractional_bit=12
rounding_method=['down','nearest','down']
batch_size=20

# PE array fault simulation parameter
config_dir='../pe_mapping_config'
network_dir='4C2F'
dataflow_dir='ws'
#PEarraysize='8x8'
#PEarraysize='16x16'
PEarraysize='32x32'
config_dir=os.path.join(config_dir, network_dir, dataflow_dir, PEarraysize)
mac_config=os.path.join(config_dir,'mac_unit_config.json')
model_wl=model_word_length
mapping_verbose=5

#%% dataflow setup

# model for get configuration
model=quantized_4C2F(nbits=model_word_length,
                     fbits=model_fractional_bit,
                     rounding_method=rounding_method,
                     batch_size=batch_size,
                     quant_mode=None)

#%% 8x8 PE

# # PE represent computation unit
# PE=mac_unit(mac_config, noise_inject=noise_inject)
# # PE array
# MXU=PEarray(8,8,mac_config=PE)

# # conv1
# ofmap_tile_conv1=tile_PE((1,16,16,16),is_fmap=True,wl=model_wl)
# ifmap_tile_conv1=tile_PE((1,16,16,3),is_fmap=True,wl=model_wl)
# wght_tile_conv1 =tile_PE((3,3,3,16),is_fmap=False,wl=model_wl)
# ofmap_config_conv1=os.path.join(config_dir,'ofmap_config_conv1.json')
# ifmap_config_conv1=os.path.join(config_dir,'ifmap_config_conv1.json')
# wght_config_conv1 =os.path.join(config_dir,'wght_config_conv1.json')
# MXU_config_conv1  =os.path.join(config_dir,'MXU_config_conv1.json')

# check=mapping_valid_checker(ifmap_tile_conv1,wght_tile_conv1,ofmap_tile_conv1,MXU,
#                             ifmap_config_conv1,wght_config_conv1,ofmap_config_conv1,MXU_config_conv1,
#                             print_detail=True)
# MXU.clear_all()

# # conv2
# ofmap_tile_conv2=tile_PE((1,15,15,16),is_fmap=True,wl=model_wl)
# ifmap_tile_conv2=tile_PE((1,17,17,16),is_fmap=True,wl=model_wl)
# wght_tile_conv2 =tile_PE((3,3,16,16),is_fmap=False,wl=model_wl)
# ofmap_config_conv2=os.path.join(config_dir,'ofmap_config_conv2.json')
# ifmap_config_conv2=os.path.join(config_dir,'ifmap_config_conv2.json')
# wght_config_conv2 =os.path.join(config_dir,'wght_config_conv2.json')
# MXU_config_conv2  =os.path.join(config_dir,'MXU_config_conv2.json')

# check=mapping_valid_checker(ifmap_tile_conv2,wght_tile_conv2,ofmap_tile_conv2,MXU,
#                             ifmap_config_conv2,wght_config_conv2,ofmap_config_conv2,MXU_config_conv2,
#                             print_detail=True)
# MXU.clear_all()

# # conv3
# ofmap_tile_conv3=tile_PE((1,15,15,16),is_fmap=True,wl=model_wl)
# ifmap_tile_conv3=tile_PE((1,15,15,16),is_fmap=True,wl=model_wl)
# wght_tile_conv3 =tile_PE((3,3,16,16),is_fmap=False,wl=model_wl)
# ofmap_config_conv3=os.path.join(config_dir,'ofmap_config_conv3.json')
# ifmap_config_conv3=os.path.join(config_dir,'ifmap_config_conv3.json')
# wght_config_conv3 =os.path.join(config_dir,'wght_config_conv3.json')
# MXU_config_conv3  =os.path.join(config_dir,'MXU_config_conv3.json')

# check=mapping_valid_checker(ifmap_tile_conv3,wght_tile_conv3,ofmap_tile_conv3,MXU,
#                             ifmap_config_conv3,wght_config_conv3,ofmap_config_conv3,MXU_config_conv3,
#                             print_detail=True)
# MXU.clear_all()

# # conv4
# ofmap_tile_conv4=tile_PE((1,13,13,16),is_fmap=True,wl=model_wl)
# ifmap_tile_conv4=tile_PE((1,15,15,16),is_fmap=True,wl=model_wl)
# wght_tile_conv4 =tile_PE((3,3,16,16),is_fmap=False,wl=model_wl)
# ofmap_config_conv4=os.path.join(config_dir,'ofmap_config_conv4.json')
# ifmap_config_conv4=os.path.join(config_dir,'ifmap_config_conv4.json')
# wght_config_conv4 =os.path.join(config_dir,'wght_config_conv4.json')
# MXU_config_conv4  =os.path.join(config_dir,'MXU_config_conv4.json')

# check=mapping_valid_checker(ifmap_tile_conv4,wght_tile_conv4,ofmap_tile_conv4,MXU,
#                             ifmap_config_conv4,wght_config_conv4,ofmap_config_conv4,MXU_config_conv4,
#                             print_detail=True)
# MXU.clear_all()

# # FC1
# ofmap_tile_fc1=tile_FC_PE((1,8),is_fmap=True,wl=model_wl)
# ifmap_tile_fc1=tile_FC_PE((1,288),is_fmap=True,wl=model_wl)
# wght_tile_fc1 =tile_FC_PE((288,8),is_fmap=False,wl=model_wl)
# ofmap_config_fc1=os.path.join(config_dir,'ofmap_config_fc1.json')
# ifmap_config_fc1=os.path.join(config_dir,'ifmap_config_fc1.json')
# wght_config_fc1 =os.path.join(config_dir,'wght_config_fc1.json')
# MXU_config_fc1  =os.path.join(config_dir,'MXU_config_fc1.json')

# check=mapping_valid_checker(ifmap_tile_fc1,wght_tile_fc1,ofmap_tile_fc1,MXU,
#                             ifmap_config_fc1,wght_config_fc1,ofmap_config_fc1,MXU_config_fc1,
#                             print_detail=True)
# MXU.clear_all()

# # FC2
# ofmap_tile_fc2=tile_FC_PE((1,10),is_fmap=True,wl=model_wl)
# ifmap_tile_fc2=tile_FC_PE((1,176),is_fmap=True,wl=model_wl)
# wght_tile_fc2 =tile_FC_PE((176,10),is_fmap=False,wl=model_wl)
# ofmap_config_fc2=os.path.join(config_dir,'ofmap_config_fc2.json')
# ifmap_config_fc2=os.path.join(config_dir,'ifmap_config_fc2.json')
# wght_config_fc2 =os.path.join(config_dir,'wght_config_fc2.json')
# MXU_config_fc2  =os.path.join(config_dir,'MXU_config_fc2.json')

# check=mapping_valid_checker(ifmap_tile_fc2,wght_tile_fc2,ofmap_tile_fc2,MXU,
#                             ifmap_config_fc2,wght_config_fc2,ofmap_config_fc2,MXU_config_fc2,
#                             print_detail=True)
# MXU.clear_all()

#%% 16x16 PE

# # PE represent computation unit
# PE=mac_unit(mac_config, noise_inject=noise_inject)
# # PE array
# MXU=PEarray(16,16,mac_config=PE)

# # conv1
# ofmap_tile_conv1=tile_PE((1,16,16,32),is_fmap=True,wl=model_wl)
# ifmap_tile_conv1=tile_PE((1,16,16,3),is_fmap=True,wl=model_wl)
# wght_tile_conv1 =tile_PE((3,3,3,32),is_fmap=False,wl=model_wl)
# ofmap_config_conv1=os.path.join(config_dir,'ofmap_config_conv1.json')
# ifmap_config_conv1=os.path.join(config_dir,'ifmap_config_conv1.json')
# wght_config_conv1 =os.path.join(config_dir,'wght_config_conv1.json')
# MXU_config_conv1  =os.path.join(config_dir,'MXU_config_conv1.json')

# check=mapping_valid_checker(ifmap_tile_conv1,wght_tile_conv1,ofmap_tile_conv1,MXU,
#                             ifmap_config_conv1,wght_config_conv1,ofmap_config_conv1,MXU_config_conv1,
#                             print_detail=True)
# MXU.clear_all()

# # conv2
# ofmap_tile_conv2=tile_PE((1,15,15,32),is_fmap=True,wl=model_wl)
# ifmap_tile_conv2=tile_PE((1,17,17,32),is_fmap=True,wl=model_wl)
# wght_tile_conv2 =tile_PE((3,3,32,32),is_fmap=False,wl=model_wl)
# ofmap_config_conv2=os.path.join(config_dir,'ofmap_config_conv2.json')
# ifmap_config_conv2=os.path.join(config_dir,'ifmap_config_conv2.json')
# wght_config_conv2 =os.path.join(config_dir,'wght_config_conv2.json')
# MXU_config_conv2  =os.path.join(config_dir,'MXU_config_conv2.json')

# check=mapping_valid_checker(ifmap_tile_conv2,wght_tile_conv2,ofmap_tile_conv2,MXU,
#                             ifmap_config_conv2,wght_config_conv2,ofmap_config_conv2,MXU_config_conv2,
#                             print_detail=True)
# MXU.clear_all()

# # conv3
# ofmap_tile_conv3=tile_PE((1,15,15,32),is_fmap=True,wl=model_wl)
# ifmap_tile_conv3=tile_PE((1,15,15,32),is_fmap=True,wl=model_wl)
# wght_tile_conv3 =tile_PE((3,3,32,32),is_fmap=False,wl=model_wl)
# ofmap_config_conv3=os.path.join(config_dir,'ofmap_config_conv3.json')
# ifmap_config_conv3=os.path.join(config_dir,'ifmap_config_conv3.json')
# wght_config_conv3 =os.path.join(config_dir,'wght_config_conv3.json')
# MXU_config_conv3  =os.path.join(config_dir,'MXU_config_conv3.json')

# check=mapping_valid_checker(ifmap_tile_conv3,wght_tile_conv3,ofmap_tile_conv3,MXU,
#                             ifmap_config_conv3,wght_config_conv3,ofmap_config_conv3,MXU_config_conv3,
#                             print_detail=True)
# MXU.clear_all()

# # conv4
# ofmap_tile_conv4=tile_PE((1,13,13,32),is_fmap=True,wl=model_wl)
# ifmap_tile_conv4=tile_PE((1,15,15,32),is_fmap=True,wl=model_wl)
# wght_tile_conv4 =tile_PE((3,3,32,32),is_fmap=False,wl=model_wl)
# ofmap_config_conv4=os.path.join(config_dir,'ofmap_config_conv4.json')
# ifmap_config_conv4=os.path.join(config_dir,'ifmap_config_conv4.json')
# wght_config_conv4 =os.path.join(config_dir,'wght_config_conv4.json')
# MXU_config_conv4  =os.path.join(config_dir,'MXU_config_conv4.json')

# check=mapping_valid_checker(ifmap_tile_conv4,wght_tile_conv4,ofmap_tile_conv4,MXU,
#                             ifmap_config_conv4,wght_config_conv4,ofmap_config_conv4,MXU_config_conv4,
#                             print_detail=True)
# MXU.clear_all()

# # FC1
# ofmap_tile_fc1=tile_FC_PE((1,16),is_fmap=True,wl=model_wl)
# ifmap_tile_fc1=tile_FC_PE((1,576),is_fmap=True,wl=model_wl)
# wght_tile_fc1 =tile_FC_PE((576,16),is_fmap=False,wl=model_wl)
# ofmap_config_fc1=os.path.join(config_dir,'ofmap_config_fc1.json')
# ifmap_config_fc1=os.path.join(config_dir,'ifmap_config_fc1.json')
# wght_config_fc1 =os.path.join(config_dir,'wght_config_fc1.json')
# MXU_config_fc1  =os.path.join(config_dir,'MXU_config_fc1.json')

# check=mapping_valid_checker(ifmap_tile_fc1,wght_tile_fc1,ofmap_tile_fc1,MXU,
#                             ifmap_config_fc1,wght_config_fc1,ofmap_config_fc1,MXU_config_fc1,
#                             print_detail=True)
# MXU.clear_all()

# # FC2
# ofmap_tile_fc2=tile_FC_PE((1,10),is_fmap=True,wl=model_wl)
# ifmap_tile_fc2=tile_FC_PE((1,512),is_fmap=True,wl=model_wl)
# wght_tile_fc2 =tile_FC_PE((512,10),is_fmap=False,wl=model_wl)
# ofmap_config_fc2=os.path.join(config_dir,'ofmap_config_fc2.json')
# ifmap_config_fc2=os.path.join(config_dir,'ifmap_config_fc2.json')
# wght_config_fc2 =os.path.join(config_dir,'wght_config_fc2.json')
# MXU_config_fc2  =os.path.join(config_dir,'MXU_config_fc2.json')

# check=mapping_valid_checker(ifmap_tile_fc2,wght_tile_fc2,ofmap_tile_fc2,MXU,
#                             ifmap_config_fc2,wght_config_fc2,ofmap_config_fc2,MXU_config_fc2,
#                             print_detail=True)
# MXU.clear_all()

#%% 32x32 PE

# PE represent computation unit
PE=mac_unit(mac_config, noise_inject=noise_inject)
# PE array
MXU=PEarray(32,32,mac_config=PE)

# conv1
ofmap_tile_conv1=tile_PE((1,32,32,32),is_fmap=True,wl=model_wl)
ifmap_tile_conv1=tile_PE((1,32,32,3),is_fmap=True,wl=model_wl)
wght_tile_conv1 =tile_PE((3,3,3,32),is_fmap=False,wl=model_wl)
ofmap_config_conv1=os.path.join(config_dir,'ofmap_config_conv1.json')
ifmap_config_conv1=os.path.join(config_dir,'ifmap_config_conv1.json')
wght_config_conv1 =os.path.join(config_dir,'wght_config_conv1.json')
MXU_config_conv1  =os.path.join(config_dir,'MXU_config_conv1.json')

check=mapping_valid_checker(ifmap_tile_conv1,wght_tile_conv1,ofmap_tile_conv1,MXU,
                            ifmap_config_conv1,wght_config_conv1,ofmap_config_conv1,MXU_config_conv1,
                            print_detail=True)
MXU.clear_all()

# conv2
ofmap_tile_conv2=tile_PE((1,30,30,32),is_fmap=True,wl=model_wl)
ifmap_tile_conv2=tile_PE((1,32,32,32),is_fmap=True,wl=model_wl)
wght_tile_conv2 =tile_PE((3,3,32,32),is_fmap=False,wl=model_wl)
ofmap_config_conv2=os.path.join(config_dir,'ofmap_config_conv2.json')
ifmap_config_conv2=os.path.join(config_dir,'ifmap_config_conv2.json')
wght_config_conv2 =os.path.join(config_dir,'wght_config_conv2.json')
MXU_config_conv2  =os.path.join(config_dir,'MXU_config_conv2.json')

check=mapping_valid_checker(ifmap_tile_conv2,wght_tile_conv2,ofmap_tile_conv2,MXU,
                            ifmap_config_conv2,wght_config_conv2,ofmap_config_conv2,MXU_config_conv2,
                            print_detail=True)
MXU.clear_all()

# conv3
ofmap_tile_conv3=tile_PE((1,15,15,64),is_fmap=True,wl=model_wl)
ifmap_tile_conv3=tile_PE((1,15,15,32),is_fmap=True,wl=model_wl)
wght_tile_conv3 =tile_PE((3,3,32,64),is_fmap=False,wl=model_wl)
ofmap_config_conv3=os.path.join(config_dir,'ofmap_config_conv3.json')
ifmap_config_conv3=os.path.join(config_dir,'ifmap_config_conv3.json')
wght_config_conv3 =os.path.join(config_dir,'wght_config_conv3.json')
MXU_config_conv3  =os.path.join(config_dir,'MXU_config_conv3.json')

check=mapping_valid_checker(ifmap_tile_conv3,wght_tile_conv3,ofmap_tile_conv3,MXU,
                            ifmap_config_conv3,wght_config_conv3,ofmap_config_conv3,MXU_config_conv3,
                            print_detail=True)
MXU.clear_all()

# conv4
ofmap_tile_conv4=tile_PE((1,13,13,32),is_fmap=True,wl=model_wl)
ifmap_tile_conv4=tile_PE((1,15,15,64),is_fmap=True,wl=model_wl)
wght_tile_conv4 =tile_PE((3,3,64,32),is_fmap=False,wl=model_wl)
ofmap_config_conv4=os.path.join(config_dir,'ofmap_config_conv4.json')
ifmap_config_conv4=os.path.join(config_dir,'ifmap_config_conv4.json')
wght_config_conv4 =os.path.join(config_dir,'wght_config_conv4.json')
MXU_config_conv4  =os.path.join(config_dir,'MXU_config_conv4.json')

check=mapping_valid_checker(ifmap_tile_conv4,wght_tile_conv4,ofmap_tile_conv4,MXU,
                            ifmap_config_conv4,wght_config_conv4,ofmap_config_conv4,MXU_config_conv4,
                            print_detail=True)
MXU.clear_all()

# FC1
ofmap_tile_fc1=tile_FC_PE((1,32),is_fmap=True,wl=model_wl)
ifmap_tile_fc1=tile_FC_PE((1,576),is_fmap=True,wl=model_wl)
wght_tile_fc1 =tile_FC_PE((576,32),is_fmap=False,wl=model_wl)
ofmap_config_fc1=os.path.join(config_dir,'ofmap_config_fc1.json')
ifmap_config_fc1=os.path.join(config_dir,'ifmap_config_fc1.json')
wght_config_fc1 =os.path.join(config_dir,'wght_config_fc1.json')
MXU_config_fc1  =os.path.join(config_dir,'MXU_config_fc1.json')

check=mapping_valid_checker(ifmap_tile_fc1,wght_tile_fc1,ofmap_tile_fc1,MXU,
                            ifmap_config_fc1,wght_config_fc1,ofmap_config_fc1,MXU_config_fc1,
                            print_detail=True)
MXU.clear_all()

# FC2
ofmap_tile_fc2=tile_FC_PE((1,10),is_fmap=True,wl=model_wl)
ifmap_tile_fc2=tile_FC_PE((1,512),is_fmap=True,wl=model_wl)
wght_tile_fc2 =tile_FC_PE((512,10),is_fmap=False,wl=model_wl)
ofmap_config_fc2=os.path.join(config_dir,'ofmap_config_fc2.json')
ifmap_config_fc2=os.path.join(config_dir,'ifmap_config_fc2.json')
wght_config_fc2 =os.path.join(config_dir,'wght_config_fc2.json')
MXU_config_fc2  =os.path.join(config_dir,'MXU_config_fc2.json')

check=mapping_valid_checker(ifmap_tile_fc2,wght_tile_fc2,ofmap_tile_fc2,MXU,
                            ifmap_config_fc2,wght_config_fc2,ofmap_config_fc2,MXU_config_fc2,
                            print_detail=True)
MXU.clear_all()

#%% generate fault dictionary

# assign fault dictionary
fault_loc,fault_info=MXU.make_single_SA_fault(n_bit=model_wl, fault_type='flip')

model_mac_math_fault_dict_list=[None for i in range(14)] 

PE_mapping_forward(ifmap_tile_conv1,wght_tile_conv1,ofmap_tile_conv1,MXU,
                    ifmap_config_conv1,wght_config_conv1,ofmap_config_conv1,MXU_config_conv1,
                    pre_plan=True,verbose=mapping_verbose)
MXU.gen_PEarray_permanent_fault_dict(fault_loc, fault_info, mac_config=True)
model_mac_math_fault_dict_list[1] = PE_mapping_backward(model.layers[1], MXU, verbose=mapping_verbose)
MXU.clear_all()

PE_mapping_forward(ifmap_tile_conv2,wght_tile_conv2,ofmap_tile_conv2,MXU,
                   ifmap_config_conv2,wght_config_conv2,ofmap_config_conv2,MXU_config_conv2,
                   pre_plan=True,verbose=mapping_verbose)
MXU.gen_PEarray_permanent_fault_dict(fault_loc, fault_info, mac_config=True)
model_mac_math_fault_dict_list[2] = PE_mapping_backward(model.layers[2], MXU, verbose=mapping_verbose)
MXU.clear_all()

PE_mapping_forward(ifmap_tile_conv3,wght_tile_conv3,ofmap_tile_conv3,MXU,
                    ifmap_config_conv3,wght_config_conv3,ofmap_config_conv3,MXU_config_conv3,
                    pre_plan=True,verbose=mapping_verbose)
MXU.gen_PEarray_permanent_fault_dict(fault_loc, fault_info, mac_config=True)
model_mac_math_fault_dict_list[5] = PE_mapping_backward(model.layers[5], MXU, verbose=mapping_verbose)
MXU.clear_all()

PE_mapping_forward(ifmap_tile_conv4,wght_tile_conv4,ofmap_tile_conv4,MXU,
                   ifmap_config_conv4,wght_config_conv4,ofmap_config_conv4,MXU_config_conv4,
                   pre_plan=True,verbose=mapping_verbose)
MXU.gen_PEarray_permanent_fault_dict(fault_loc, fault_info, mac_config=True)
model_mac_math_fault_dict_list[6] = PE_mapping_backward(model.layers[6], MXU, verbose=mapping_verbose)
MXU.clear_all()

PE_mapping_forward(ifmap_tile_fc1,wght_tile_fc1,ofmap_tile_fc1,MXU,
                  ifmap_config_fc1,wght_config_fc1,ofmap_config_fc1,MXU_config_fc1,
                  pre_plan=True,verbose=mapping_verbose)
MXU.gen_PEarray_permanent_fault_dict(fault_loc, fault_info, mac_config=True)
model_mac_math_fault_dict_list[10] = PE_mapping_backward(model.layers[10], MXU, verbose=mapping_verbose)
MXU.clear_all()

PE_mapping_forward(ifmap_tile_fc2,wght_tile_fc2,ofmap_tile_fc2,MXU,
                  ifmap_config_fc2,wght_config_fc2,ofmap_config_fc2,MXU_config_fc2,
                  pre_plan=True,verbose=mapping_verbose)
MXU.gen_PEarray_permanent_fault_dict(fault_loc, fault_info, mac_config=True)
model_mac_math_fault_dict_list[12] = PE_mapping_backward(model.layers[12], MXU, verbose=mapping_verbose)
MXU.clear_all()

# make preprocess data
with open('../test_fault_dictionary_stuff/wght_distribution_info_cnn4C2FfusedBN.pickle', 'rb') as fdfile:
    c4f2fusedBN_wght_distribution_info = pickle.load(fdfile)
with open('../test_fault_dictionary_stuff/ifmap_distribution_info_cnn4C2FfusedBN.pickle', 'rb') as fdfile:
    c4f2fusedBN_ifmap_distribution_info = pickle.load(fdfile)
    
model_preprocess_data_list=preprocess_model_mac_fault(model, PE, model_mac_math_fault_dict_list,
                                                      model_fmap_dist_stat_list=c4f2fusedBN_ifmap_distribution_info,
                                                      model_wght_dist_stat_list=c4f2fusedBN_wght_distribution_info)

K.clear_session()

#%% model setup

t = time.time()
model=quantized_4C2F(nbits=model_word_length,
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

x_train, x_test, y_train, y_test, class_indices, datagen, input_shape = dataset_setup('cifar10')

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
prediction = np.argmax(prediction, axis=1)
show_confusion_matrix(np.argmax(y_test, axis=1),prediction,class_indices,'Confusion Matrix',normalize=False)

