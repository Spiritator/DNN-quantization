# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 10:05:46 2020

@author: Yung-Yu Tsai

An example of using inference scheme to arange analysis and save result.
evaluate PE array fault injection testing result of LeNet-5
"""

import os
import tensorflow.keras.backend as K

from simulator.inference.scheme import inference_scheme
from simulator.models.model_library import quantized_lenet5
from simulator.metrics.topk_metrics import top2_acc
from simulator.metrics.FT_metrics import acc_loss,relative_acc,pred_miss,top2_pred_miss,conf_score_vary_10,conf_score_vary_50
from tensorflow.keras.losses import categorical_crossentropy
from simulator.comp_unit.tile import tile_PE, tile_FC_PE
from simulator.comp_unit.PEarray import PEarray
from simulator.comp_unit.mac import mac_unit
from simulator.comp_unit.mapping_flow import PE_mapping_forward,PE_mapping_backward


#%% setting parameter

result_save_folder='../test_result/mnist_lenet5_PE_fault'
dataflow_type='ws'
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

test_rounds=20

#%% model & fault information setup

# model for get configuration
def call_model():
    return quantized_lenet5(nbits=model_word_length,
                            fbits=model_fractional_bit,
                            batch_size=batch_size,
                            quant_mode=None)

# PE represent computation unit
PE=mac_unit(mac_config)
# PE array
MXU=PEarray(8,8,mac_config=PE)
# assign fault dictionary
fault_locs=list()
fault_infos=list()
for i in range(test_rounds):
    loc_tmp,info_tmp=MXU.make_single_SA_fault(n_bit=model_wl, fault_type='flip')
    fault_locs.append(loc_tmp)
    fault_infos.append(info_tmp)

#%% PE mapping setup

# conv1
ofmap_tile_conv1=tile_PE((1,28,28,8),is_fmap=True,wl=model_wl)
ifmap_tile_conv1=tile_PE((1,28,28,1),is_fmap=True,wl=model_wl)
wght_tile_conv1 =tile_PE((5,5,1,8),is_fmap=False,wl=model_wl)
ofmap_config_conv1=os.path.join(config_dir,'ofmap_config_conv1.json')
ifmap_config_conv1=os.path.join(config_dir,'ifmap_config_conv1.json')
wght_config_conv1 =os.path.join(config_dir,'wght_config_conv1.json')
MXU_config_conv1  =os.path.join(config_dir,'MXU_config_conv1.json')

# conv2
ofmap_tile_conv2=tile_PE((1,14,14,16),is_fmap=True,wl=model_wl)
ifmap_tile_conv2=tile_PE((1,14,14,16),is_fmap=True,wl=model_wl)
wght_tile_conv2 =tile_PE((5,5,16,16),is_fmap=False,wl=model_wl)
ofmap_config_conv2=os.path.join(config_dir,'ofmap_config_conv2.json')
ifmap_config_conv2=os.path.join(config_dir,'ifmap_config_conv2.json')
wght_config_conv2 =os.path.join(config_dir,'wght_config_conv2.json')
MXU_config_conv2  =os.path.join(config_dir,'MXU_config_conv2.json')


## FC1
#ofmap_tile_fc1=tile_FC_PE((1,8),is_fmap=True,wl=model_wl)
#ifmap_tile_fc1=tile_FC_PE((1,882),is_fmap=True,wl=model_wl)
#wght_tile_fc1 =tile_FC_PE((882,8),is_fmap=False,wl=model_wl)
#ofmap_config_fc1=os.path.join(config_dir,'ofmap_config_fc1.json')
#ifmap_config_fc1=os.path.join(config_dir,'ifmap_config_fc1.json')
#wght_config_fc1 =os.path.join(config_dir,'wght_config_fc1.json')
#MXU_config_fc1  =os.path.join(config_dir,'MXU_config_fc1.json')

## FC2
#ofmap_tile_fc2=tile_FC_PE((1,10),is_fmap=True,wl=model_wl)
#ifmap_tile_fc2=tile_FC_PE((1,128),is_fmap=True,wl=model_wl)
#wght_tile_fc2 =tile_FC_PE((128,10),is_fmap=False,wl=model_wl)
#ofmap_config_fc2=os.path.join(config_dir,'ofmap_config_fc2.json')
#ifmap_config_fc2=os.path.join(config_dir,'ifmap_config_fc2.json')
#wght_config_fc2 =os.path.join(config_dir,'wght_config_fc2.json')
#MXU_config_fc2  =os.path.join(config_dir,'MXU_config_fc2.json')


#%% fault generation

def gen_model_PE_fault_dict(ref_model,faultloc,faultinfo,print_detail=False):
    model_mac_math_fault_dict_list=[None for i in range(8)] 

    # clear fault dictionary every iteration
    ofmap_tile_conv1.clear()
    ifmap_tile_conv1.clear()
    wght_tile_conv1.clear()
    ofmap_tile_conv2.clear()
    ifmap_tile_conv2.clear()
    wght_tile_conv2.clear()
    # ofmap_tile_fc1.clear()
    # ifmap_tile_fc1.clear()
    # wght_tile_fc1.clear()
    # ofmap_tile_fc2.clear()
    # ifmap_tile_fc2.clear()
    # wght_tile_fc2.clear()
    MXU.clear_all()
    
    PE_mapping_forward(ifmap_tile_conv1,wght_tile_conv1,ofmap_tile_conv1,MXU,
                       ifmap_config_conv1,wght_config_conv1,ofmap_config_conv1,MXU_config_conv1,
                       pre_plan=True,print_detail=True)
    MXU.gen_PEarray_permanent_fault_dict(faultloc, faultinfo, mac_config=True)
    model_mac_math_fault_dict_list[1] = PE_mapping_backward(ref_model.layers[1], MXU, print_detail=True)
    MXU.clear_all()
    
    PE_mapping_forward(ifmap_tile_conv2,wght_tile_conv2,ofmap_tile_conv2,MXU,
                       ifmap_config_conv2,wght_config_conv2,ofmap_config_conv2,MXU_config_conv2,
                       pre_plan=True,print_detail=True)
    MXU.gen_PEarray_permanent_fault_dict(faultloc, faultinfo, mac_config=True)
    model_mac_math_fault_dict_list[3] = PE_mapping_backward(ref_model.layers[3], MXU, print_detail=True)
    MXU.clear_all()
    
    #PE_mapping_forward(ifmap_tile_fc1,wght_tile_fc1,ofmap_tile_fc1,MXU,
    #                   ifmap_config_fc1,wght_config_fc1,ofmap_config_fc1,MXU_config_fc1,
    #                   pre_plan=True,print_detail=True)
    #MXU.gen_PEarray_permanent_fault_dict(faultloc, faultinfo, mac_config=True)
    #model_mac_math_fault_dict_list[6] = PE_mapping_backward(model.layers[7], MXU, print_detail=True)
    #MXU.clear_all()
    #
    #PE_mapping_forward(ifmap_tile_fc2,wght_tile_fc2,ofmap_tile_fc2,MXU,
    #                   ifmap_config_fc2,wght_config_fc2,ofmap_config_fc2,MXU_config_fc2,
    #                   pre_plan=True,print_detail=True)
    #MXU.gen_PEarray_permanent_fault_dict(faultloc, faultinfo, mac_config=True)
    #model_mac_math_fault_dict_list[7] = PE_mapping_backward(model.layers[7], MXU, print_detail=True)
    #MXU.clear_all()
    
    return model_mac_math_fault_dict_list

#%% test run
compile_augment={'loss':'categorical_crossentropy','optimizer':'adam','metrics':['accuracy',top2_acc]}

dataset_augment={'dataset':'mnist'}

FT_augment={'model_name':'lenet','loss_function':categorical_crossentropy,'metrics':['accuracy',top2_acc,acc_loss,relative_acc,pred_miss,top2_pred_miss,conf_score_vary_10,conf_score_vary_50]}    
    

model_augment=list()
for round_id in range(test_rounds):
    print('|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|')
    print('|=|        Test Round %d/%d'%(round_id,test_rounds))
    print('|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|')
    ref_model=call_model()
    # fault generation
    model_mac_math_fdl=gen_model_PE_fault_dict(ref_model,fault_locs[round_id],fault_infos[round_id],print_detail=True)
    K.clear_session()
    
    model_augment.append({'nbits':model_word_length,
                          'fbits':model_fractional_bit,
                          'rounding_method':'nearest',
                          'batch_size':batch_size,
                          'quant_mode':'hybrid',
                          'ofmap_fault_dict_list':model_mac_math_fdl,
                          'mac_unit':PE})
    
    # inference test
    result_save_file=os.path.join(result_save_folder,dataflow_type,'metric.csv')
    inference_scheme(quantized_lenet5, 
                     model_augment, 
                     compile_augment, 
                     dataset_augment, 
                     result_save_file, 
                     append_save_file=True,
                     weight_load=True, 
                     weight_name=weight_name, 
                     save_runtime=True,
                     FT_evaluate=True, 
                     FT_augment=FT_augment)

