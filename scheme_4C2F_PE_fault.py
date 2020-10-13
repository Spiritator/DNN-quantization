# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 12:17:59 2020

@author: Yung-Yu Tsai

An example of using inference scheme to arange analysis and save result.
evaluate PE array fault injection testing result of 4C2F CNN
"""

import os,csv,pickle
import tensorflow.keras.backend as K

from simulator.inference.scheme import inference_scheme
from simulator.models.model_library import quantized_4C2F
from simulator.metrics.topk_metrics import top2_acc
from simulator.metrics.FT_metrics import acc_loss,relative_acc,pred_miss,top2_pred_miss,conf_score_vary_10,conf_score_vary_50
from tensorflow.keras.losses import categorical_crossentropy
from simulator.comp_unit.tile import tile_PE, tile_FC_PE
from simulator.comp_unit.PEarray import PEarray
from simulator.comp_unit.mac import mac_unit, preprocess_model_mac_fault
from simulator.comp_unit.mapping_flow import PE_mapping_forward,PE_mapping_backward
from simulator.models.model_mods import make_ref_model

#%% setting parameter

noise_inject=False
report_filename='metric'

result_save_folder=os.path.join('..','test_result','mnist_lenet5_PE_fault')
dataflow_type='ws'
weight_name=os.path.join('..','cifar10_4C2FBN_weight_fused_BN.h5')
model_word_length=16
model_fractional_bit=12
rounding_method=['down','nearest','down']
quant_mode='hybrid'
batch_size=20
# PE array fault simulation parameter
config_dir=os.path.join('..','pe_mapping_config')
network_dir='4C2F'
dataflow_dir='ws'
PEarraysize='8x8'
#PEarraysize='16x16'
#PEarraysize='32x32'
config_dir=os.path.join(config_dir, network_dir, dataflow_dir, PEarraysize)
mac_config=os.path.join(config_dir,'mac_unit_config.json')
model_wl=model_word_length
mapping_verbose=5

test_rounds=200

#%% model & fault information setup

# model for get configuration
ref_model=make_ref_model(model=quantized_4C2F(nbits=model_word_length,
                                              fbits=model_fractional_bit,
                                              rounding_method=rounding_method,
                                              batch_size=batch_size,
                                              quant_mode=None))

# PE represent computation unit
PE=mac_unit(mac_config, noise_inject=noise_inject)

# PE array
MXU=PEarray(8,8,mac_config=PE)
#MXU=PEarray(16,16,mac_config=PE)
#MXU=PEarray(32,32,mac_config=PE)

# assign fault dictionary
fault_locs=list()
fault_infos=list()
for i in range(test_rounds):
    loc_tmp,info_tmp=MXU.make_single_SA_fault(n_bit=model_wl, fault_type='flip')
    fault_locs.append(loc_tmp)
    fault_infos.append(info_tmp)
# Read in fault dictionary
# with open('../test_fault_dictionary_stuff/validate_mac_math_lenet_fault_locs_8x8.pickle', 'rb') as fdfile:
#     fault_locs = pickle.load(fdfile)
# with open('../test_fault_dictionary_stuff/validate_mac_math_lenet_fault_infos_8x8.pickle', 'rb') as fdfile:
#     fault_infos = pickle.load(fdfile)

# Read in distribution info
with open('../test_fault_dictionary_stuff/wght_distribution_info_cnn4C2FfusedBN.pickle', 'rb') as fdfile:
    c4f2fusedBN_wght_distribution_info = pickle.load(fdfile)
with open('../test_fault_dictionary_stuff/ifmap_distribution_info_cnn4C2FfusedBN.pickle', 'rb') as fdfile:
    c4f2fusedBN_ifmap_distribution_info = pickle.load(fdfile)

#%% 8x8 PE

# conv1
ofmap_tile_conv1=tile_PE((1,16,16,16),is_fmap=True,wl=model_wl)
ifmap_tile_conv1=tile_PE((1,16,16,3),is_fmap=True,wl=model_wl)
wght_tile_conv1 =tile_PE((3,3,3,16),is_fmap=False,wl=model_wl)
ofmap_config_conv1=os.path.join(config_dir,'ofmap_config_conv1.json')
ifmap_config_conv1=os.path.join(config_dir,'ifmap_config_conv1.json')
wght_config_conv1 =os.path.join(config_dir,'wght_config_conv1.json')
MXU_config_conv1  =os.path.join(config_dir,'MXU_config_conv1.json')

# conv2
ofmap_tile_conv2=tile_PE((1,15,15,16),is_fmap=True,wl=model_wl)
ifmap_tile_conv2=tile_PE((1,17,17,16),is_fmap=True,wl=model_wl)
wght_tile_conv2 =tile_PE((3,3,16,16),is_fmap=False,wl=model_wl)
ofmap_config_conv2=os.path.join(config_dir,'ofmap_config_conv2.json')
ifmap_config_conv2=os.path.join(config_dir,'ifmap_config_conv2.json')
wght_config_conv2 =os.path.join(config_dir,'wght_config_conv2.json')
MXU_config_conv2  =os.path.join(config_dir,'MXU_config_conv2.json')

# conv3
ofmap_tile_conv3=tile_PE((1,15,15,16),is_fmap=True,wl=model_wl)
ifmap_tile_conv3=tile_PE((1,15,15,16),is_fmap=True,wl=model_wl)
wght_tile_conv3 =tile_PE((3,3,16,16),is_fmap=False,wl=model_wl)
ofmap_config_conv3=os.path.join(config_dir,'ofmap_config_conv3.json')
ifmap_config_conv3=os.path.join(config_dir,'ifmap_config_conv3.json')
wght_config_conv3 =os.path.join(config_dir,'wght_config_conv3.json')
MXU_config_conv3  =os.path.join(config_dir,'MXU_config_conv3.json')

# conv4
ofmap_tile_conv4=tile_PE((1,13,13,16),is_fmap=True,wl=model_wl)
ifmap_tile_conv4=tile_PE((1,15,15,16),is_fmap=True,wl=model_wl)
wght_tile_conv4 =tile_PE((3,3,16,16),is_fmap=False,wl=model_wl)
ofmap_config_conv4=os.path.join(config_dir,'ofmap_config_conv4.json')
ifmap_config_conv4=os.path.join(config_dir,'ifmap_config_conv4.json')
wght_config_conv4 =os.path.join(config_dir,'wght_config_conv4.json')
MXU_config_conv4  =os.path.join(config_dir,'MXU_config_conv4.json')

# FC1
ofmap_tile_fc1=tile_FC_PE((1,8),is_fmap=True,wl=model_wl)
ifmap_tile_fc1=tile_FC_PE((1,288),is_fmap=True,wl=model_wl)
wght_tile_fc1 =tile_FC_PE((288,8),is_fmap=False,wl=model_wl)
ofmap_config_fc1=os.path.join(config_dir,'ofmap_config_fc1.json')
ifmap_config_fc1=os.path.join(config_dir,'ifmap_config_fc1.json')
wght_config_fc1 =os.path.join(config_dir,'wght_config_fc1.json')
MXU_config_fc1  =os.path.join(config_dir,'MXU_config_fc1.json')

# FC2
ofmap_tile_fc2=tile_FC_PE((1,10),is_fmap=True,wl=model_wl)
ifmap_tile_fc2=tile_FC_PE((1,176),is_fmap=True,wl=model_wl)
wght_tile_fc2 =tile_FC_PE((176,10),is_fmap=False,wl=model_wl)
ofmap_config_fc2=os.path.join(config_dir,'ofmap_config_fc2.json')
ifmap_config_fc2=os.path.join(config_dir,'ifmap_config_fc2.json')
wght_config_fc2 =os.path.join(config_dir,'wght_config_fc2.json')
MXU_config_fc2  =os.path.join(config_dir,'MXU_config_fc2.json')

#%% 16x16 PE

# # conv1
# ofmap_tile_conv1=tile_PE((1,16,16,32),is_fmap=True,wl=model_wl)
# ifmap_tile_conv1=tile_PE((1,16,16,3),is_fmap=True,wl=model_wl)
# wght_tile_conv1 =tile_PE((3,3,3,32),is_fmap=False,wl=model_wl)
# ofmap_config_conv1=os.path.join(config_dir,'ofmap_config_conv1.json')
# ifmap_config_conv1=os.path.join(config_dir,'ifmap_config_conv1.json')
# wght_config_conv1 =os.path.join(config_dir,'wght_config_conv1.json')
# MXU_config_conv1  =os.path.join(config_dir,'MXU_config_conv1.json')

# # conv2
# ofmap_tile_conv2=tile_PE((1,15,15,32),is_fmap=True,wl=model_wl)
# ifmap_tile_conv2=tile_PE((1,17,17,32),is_fmap=True,wl=model_wl)
# wght_tile_conv2 =tile_PE((3,3,32,32),is_fmap=False,wl=model_wl)
# ofmap_config_conv2=os.path.join(config_dir,'ofmap_config_conv2.json')
# ifmap_config_conv2=os.path.join(config_dir,'ifmap_config_conv2.json')
# wght_config_conv2 =os.path.join(config_dir,'wght_config_conv2.json')
# MXU_config_conv2  =os.path.join(config_dir,'MXU_config_conv2.json')

# # conv3
# ofmap_tile_conv3=tile_PE((1,15,15,32),is_fmap=True,wl=model_wl)
# ifmap_tile_conv3=tile_PE((1,15,15,32),is_fmap=True,wl=model_wl)
# wght_tile_conv3 =tile_PE((3,3,32,32),is_fmap=False,wl=model_wl)
# ofmap_config_conv3=os.path.join(config_dir,'ofmap_config_conv3.json')
# ifmap_config_conv3=os.path.join(config_dir,'ifmap_config_conv3.json')
# wght_config_conv3 =os.path.join(config_dir,'wght_config_conv3.json')
# MXU_config_conv3  =os.path.join(config_dir,'MXU_config_conv3.json')

# # conv4
# ofmap_tile_conv4=tile_PE((1,13,13,32),is_fmap=True,wl=model_wl)
# ifmap_tile_conv4=tile_PE((1,15,15,32),is_fmap=True,wl=model_wl)
# wght_tile_conv4 =tile_PE((3,3,32,32),is_fmap=False,wl=model_wl)
# ofmap_config_conv4=os.path.join(config_dir,'ofmap_config_conv4.json')
# ifmap_config_conv4=os.path.join(config_dir,'ifmap_config_conv4.json')
# wght_config_conv4 =os.path.join(config_dir,'wght_config_conv4.json')
# MXU_config_conv4  =os.path.join(config_dir,'MXU_config_conv4.json')

# # FC1
# ofmap_tile_fc1=tile_FC_PE((1,16),is_fmap=True,wl=model_wl)
# ifmap_tile_fc1=tile_FC_PE((1,576),is_fmap=True,wl=model_wl)
# wght_tile_fc1 =tile_FC_PE((576,16),is_fmap=False,wl=model_wl)
# ofmap_config_fc1=os.path.join(config_dir,'ofmap_config_fc1.json')
# ifmap_config_fc1=os.path.join(config_dir,'ifmap_config_fc1.json')
# wght_config_fc1 =os.path.join(config_dir,'wght_config_fc1.json')
# MXU_config_fc1  =os.path.join(config_dir,'MXU_config_fc1.json')

# # FC2
# ofmap_tile_fc2=tile_FC_PE((1,10),is_fmap=True,wl=model_wl)
# ifmap_tile_fc2=tile_FC_PE((1,512),is_fmap=True,wl=model_wl)
# wght_tile_fc2 =tile_FC_PE((512,10),is_fmap=False,wl=model_wl)
# ofmap_config_fc2=os.path.join(config_dir,'ofmap_config_fc2.json')
# ifmap_config_fc2=os.path.join(config_dir,'ifmap_config_fc2.json')
# wght_config_fc2 =os.path.join(config_dir,'wght_config_fc2.json')
# MXU_config_fc2  =os.path.join(config_dir,'MXU_config_fc2.json')

#%% 32x32 PE

# # conv1
# ofmap_tile_conv1=tile_PE((1,32,32,32),is_fmap=True,wl=model_wl)
# ifmap_tile_conv1=tile_PE((1,32,32,3),is_fmap=True,wl=model_wl)
# wght_tile_conv1 =tile_PE((3,3,3,32),is_fmap=False,wl=model_wl)
# ofmap_config_conv1=os.path.join(config_dir,'ofmap_config_conv1.json')
# ifmap_config_conv1=os.path.join(config_dir,'ifmap_config_conv1.json')
# wght_config_conv1 =os.path.join(config_dir,'wght_config_conv1.json')
# MXU_config_conv1  =os.path.join(config_dir,'MXU_config_conv1.json')

# # conv2
# ofmap_tile_conv2=tile_PE((1,30,30,32),is_fmap=True,wl=model_wl)
# ifmap_tile_conv2=tile_PE((1,32,32,32),is_fmap=True,wl=model_wl)
# wght_tile_conv2 =tile_PE((3,3,32,32),is_fmap=False,wl=model_wl)
# ofmap_config_conv2=os.path.join(config_dir,'ofmap_config_conv2.json')
# ifmap_config_conv2=os.path.join(config_dir,'ifmap_config_conv2.json')
# wght_config_conv2 =os.path.join(config_dir,'wght_config_conv2.json')
# MXU_config_conv2  =os.path.join(config_dir,'MXU_config_conv2.json')

# # conv3
# ofmap_tile_conv3=tile_PE((1,15,15,64),is_fmap=True,wl=model_wl)
# ifmap_tile_conv3=tile_PE((1,15,15,32),is_fmap=True,wl=model_wl)
# wght_tile_conv3 =tile_PE((3,3,32,64),is_fmap=False,wl=model_wl)
# ofmap_config_conv3=os.path.join(config_dir,'ofmap_config_conv3.json')
# ifmap_config_conv3=os.path.join(config_dir,'ifmap_config_conv3.json')
# wght_config_conv3 =os.path.join(config_dir,'wght_config_conv3.json')
# MXU_config_conv3  =os.path.join(config_dir,'MXU_config_conv3.json')

# # conv4
# ofmap_tile_conv4=tile_PE((1,13,13,32),is_fmap=True,wl=model_wl)
# ifmap_tile_conv4=tile_PE((1,15,15,64),is_fmap=True,wl=model_wl)
# wght_tile_conv4 =tile_PE((3,3,64,32),is_fmap=False,wl=model_wl)
# ofmap_config_conv4=os.path.join(config_dir,'ofmap_config_conv4.json')
# ifmap_config_conv4=os.path.join(config_dir,'ifmap_config_conv4.json')
# wght_config_conv4 =os.path.join(config_dir,'wght_config_conv4.json')
# MXU_config_conv4  =os.path.join(config_dir,'MXU_config_conv4.json')

# # FC1
# ofmap_tile_fc1=tile_FC_PE((1,32),is_fmap=True,wl=model_wl)
# ifmap_tile_fc1=tile_FC_PE((1,576),is_fmap=True,wl=model_wl)
# wght_tile_fc1 =tile_FC_PE((576,32),is_fmap=False,wl=model_wl)
# ofmap_config_fc1=os.path.join(config_dir,'ofmap_config_fc1.json')
# ifmap_config_fc1=os.path.join(config_dir,'ifmap_config_fc1.json')
# wght_config_fc1 =os.path.join(config_dir,'wght_config_fc1.json')
# MXU_config_fc1  =os.path.join(config_dir,'MXU_config_fc1.json')

# # FC2
# ofmap_tile_fc2=tile_FC_PE((1,10),is_fmap=True,wl=model_wl)
# ifmap_tile_fc2=tile_FC_PE((1,512),is_fmap=True,wl=model_wl)
# wght_tile_fc2 =tile_FC_PE((512,10),is_fmap=False,wl=model_wl)
# ofmap_config_fc2=os.path.join(config_dir,'ofmap_config_fc2.json')
# ifmap_config_fc2=os.path.join(config_dir,'ifmap_config_fc2.json')
# wght_config_fc2 =os.path.join(config_dir,'wght_config_fc2.json')
# MXU_config_fc2  =os.path.join(config_dir,'MXU_config_fc2.json')

#%% fault generation

def gen_model_PE_fault_dict(ref_model,fault_loc,fault_info,verbose):
    model_mac_fault_dict_list=[None for i in range(14)] 
    psidx_cnt=0
    
    # clear fault dictionary every iteration
    ofmap_tile_conv1.clear()
    ifmap_tile_conv1.clear()
    wght_tile_conv1.clear()
    ofmap_tile_conv2.clear()
    ifmap_tile_conv2.clear()
    wght_tile_conv2.clear()
    ofmap_tile_conv3.clear()
    ifmap_tile_conv3.clear()
    wght_tile_conv3.clear()
    ofmap_tile_conv4.clear()
    ifmap_tile_conv4.clear()
    wght_tile_conv4.clear()
    ofmap_tile_fc1.clear()
    ifmap_tile_fc1.clear()
    wght_tile_fc1.clear()
    ofmap_tile_fc2.clear()
    ifmap_tile_fc2.clear()
    wght_tile_fc2.clear()
    MXU.clear_all()
    
    PE_mapping_forward(ifmap_tile_conv1,wght_tile_conv1,ofmap_tile_conv1,MXU,
                        ifmap_config_conv1,wght_config_conv1,ofmap_config_conv1,MXU_config_conv1,
                        pre_plan=True,verbose=verbose)
    MXU.gen_PEarray_permanent_fault_dict(fault_loc, fault_info, mac_config=True)
    model_mac_fault_dict_list[1], psidx_tmp = PE_mapping_backward(ref_model.layers[1], MXU, verbose=verbose, return_detail=True)
    MXU.clear_all()
    
    PE_mapping_forward(ifmap_tile_conv2,wght_tile_conv2,ofmap_tile_conv2,MXU,
                       ifmap_config_conv2,wght_config_conv2,ofmap_config_conv2,MXU_config_conv2,
                       pre_plan=True,verbose=verbose)
    MXU.gen_PEarray_permanent_fault_dict(fault_loc, fault_info, mac_config=True)
    model_mac_fault_dict_list[2], psidx_tmp = PE_mapping_backward(ref_model.layers[2], MXU, verbose=verbose, return_detail=True)
    MXU.clear_all()
    
    PE_mapping_forward(ifmap_tile_conv3,wght_tile_conv3,ofmap_tile_conv3,MXU,
                        ifmap_config_conv3,wght_config_conv3,ofmap_config_conv3,MXU_config_conv3,
                        pre_plan=True,verbose=verbose)
    MXU.gen_PEarray_permanent_fault_dict(fault_loc, fault_info, mac_config=True)
    model_mac_fault_dict_list[5], psidx_tmp = PE_mapping_backward(ref_model.layers[5], MXU, verbose=verbose, return_detail=True)
    MXU.clear_all()
    
    PE_mapping_forward(ifmap_tile_conv4,wght_tile_conv4,ofmap_tile_conv4,MXU,
                       ifmap_config_conv4,wght_config_conv4,ofmap_config_conv4,MXU_config_conv4,
                       pre_plan=True,verbose=verbose)
    MXU.gen_PEarray_permanent_fault_dict(fault_loc, fault_info, mac_config=True)
    model_mac_fault_dict_list[6], psidx_tmp = PE_mapping_backward(ref_model.layers[6], MXU, verbose=verbose, return_detail=True)
    MXU.clear_all()
    
    PE_mapping_forward(ifmap_tile_fc1,wght_tile_fc1,ofmap_tile_fc1,MXU,
                      ifmap_config_fc1,wght_config_fc1,ofmap_config_fc1,MXU_config_fc1,
                      pre_plan=True,verbose=verbose)
    MXU.gen_PEarray_permanent_fault_dict(fault_loc, fault_info, mac_config=True)
    model_mac_fault_dict_list[10], psidx_tmp = PE_mapping_backward(ref_model.layers[10], MXU, verbose=verbose, return_detail=True)
    MXU.clear_all()
    
    PE_mapping_forward(ifmap_tile_fc2,wght_tile_fc2,ofmap_tile_fc2,MXU,
                      ifmap_config_fc2,wght_config_fc2,ofmap_config_fc2,MXU_config_fc2,
                      pre_plan=True,verbose=verbose)
    MXU.gen_PEarray_permanent_fault_dict(fault_loc, fault_info, mac_config=True)
    model_mac_fault_dict_list[12], psidx_tmp = PE_mapping_backward(ref_model.layers[12], MXU, verbose=verbose, return_detail=True)
    MXU.clear_all()
    
    # make preprocess data
    model_mac_fault_dict_list=preprocess_model_mac_fault(ref_model, PE, model_mac_fault_dict_list,
                                                         model_fmap_dist_stat_list=c4f2fusedBN_ifmap_distribution_info,
                                                         model_wght_dist_stat_list=c4f2fusedBN_wght_distribution_info)
    
    return model_mac_fault_dict_list, psidx_cnt

#%% test run
compile_argument={'loss':'categorical_crossentropy','optimizer':'adam','metrics':['accuracy',top2_acc]}

dataset_argument={'dataset':'cifar10'}

FT_argument={'model_name':'4c2f','loss_function':categorical_crossentropy,'metrics':['accuracy',top2_acc,acc_loss,relative_acc,pred_miss,top2_pred_miss,conf_score_vary_10,conf_score_vary_50]}        

for round_id in range(test_rounds):
    print('======================================')
    print('        Test Round %d/%d'%(round_id,test_rounds))
    print('======================================')
    # fault generation
    model_mac_math_fdl, psidx_count=gen_model_PE_fault_dict(ref_model,fault_locs[round_id],fault_infos[round_id],verbose=mapping_verbose)
    K.clear_session()
    
    info_add_on={'PE y':[fault_locs[round_id][0]],
                 'PE x':[fault_locs[round_id][1]],
                 'param':[fault_infos[round_id]['param']],
                 'SA type':[fault_infos[round_id]['SA_type']],
                 'SA bit':[fault_infos[round_id]['SA_bit']],
                 'num psidx':[psidx_count]}
    
    model_argument=[{'nbits':model_word_length,
                    'fbits':model_fractional_bit,
                    'rounding_method':rounding_method,
                    'batch_size':batch_size,
                    'quant_mode':quant_mode,
                    'ofmap_fault_dict_list':model_mac_math_fdl,
                    'mac_unit':PE}]
    
    # inference test
    result_save_file=os.path.join(result_save_folder, dataflow_type, report_filename+'.csv')
    inference_scheme(quantized_4C2F, 
                     model_argument, 
                     compile_argument, 
                     dataset_argument, 
                     result_save_file, 
                     append_save_file=True,
                     weight_load_name=weight_name, 
                     save_runtime=True,
                     FT_evaluate_argument=FT_argument,
                     save_file_add_on=info_add_on,
                     verbose=4)

