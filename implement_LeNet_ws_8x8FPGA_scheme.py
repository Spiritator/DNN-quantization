# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 16:02:22 2021

@author: Yung-Yu Tsai

The test and verification for run scheme LeNet-5 inference on FPGA with weight stationary 8x8 systolic PE array.
"""

import os,csv,pickle,time
import tensorflow.keras.backend as K

from simulator.inference.scheme import inference_scheme
from simulator.models.model_library import quantized_lenet5
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

result_save_folder=os.path.join('..','test_result','LeNet5_ws_8x8_FPGA')
weight_name=os.path.join('..','mnist_lenet5_weight.h5')
model_word_length=8
model_fractional_bit=3
rounding_method=['down','nearest','down']
quant_mode='hybrid'
batch_size=20
# PE array fault simulation parameter
config_dir=os.path.join('..','pe_mapping_config','LeNet5_ws_8x8_FPGA')
mac_config=os.path.join(config_dir,'mac_unit_config.json')
model_wl=model_word_length
mapping_verbose=5

test_rounds=100

#%% model & fault information setup

# model for get configuration
ref_model=make_ref_model(quantized_lenet5(nbits=model_word_length,
                                          fbits=model_fractional_bit,
                                          batch_size=batch_size,
                                          quant_mode=None,
                                          verbose=False))

# PE represent computation unit
PE=mac_unit(mac_config, noise_inject=noise_inject)

# PE array
MXU=PEarray(8,8,mac_config=PE)

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


#%% PE mapping setup 8x8

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


# FC1
ofmap_tile_fc1=tile_FC_PE((1,8),is_fmap=True,wl=model_wl)
ifmap_tile_fc1=tile_FC_PE((1,984),is_fmap=True,wl=model_wl)
wght_tile_fc1 =tile_FC_PE((984,8),is_fmap=False,wl=model_wl)
ofmap_config_fc1=os.path.join(config_dir,'ofmap_config_fc1.json')
ifmap_config_fc1=os.path.join(config_dir,'ifmap_config_fc1.json')
wght_config_fc1 =os.path.join(config_dir,'wght_config_fc1.json')
MXU_config_fc1  =os.path.join(config_dir,'MXU_config_fc1.json')

# FC2
ofmap_tile_fc2=tile_FC_PE((1,10),is_fmap=True,wl=model_wl)
ifmap_tile_fc2=tile_FC_PE((1,128),is_fmap=True,wl=model_wl)
wght_tile_fc2 =tile_FC_PE((128,10),is_fmap=False,wl=model_wl)
ofmap_config_fc2=os.path.join(config_dir,'ofmap_config_fc2.json')
ifmap_config_fc2=os.path.join(config_dir,'ifmap_config_fc2.json')
wght_config_fc2 =os.path.join(config_dir,'wght_config_fc2.json')
MXU_config_fc2  =os.path.join(config_dir,'MXU_config_fc2.json')


#%% fault generation

def gen_model_PE_fault_dict(ref_model,faultloc,faultinfo,verbose):
    t=time.time()
    model_mac_fault_dict_list=[None for i in range(8)] 
    psidx_cnt=0
    
    # clear fault dictionary every iteration
    ofmap_tile_conv1.clear()
    ifmap_tile_conv1.clear()
    wght_tile_conv1.clear()
    ofmap_tile_conv2.clear()
    ifmap_tile_conv2.clear()
    wght_tile_conv2.clear()
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
    MXU.gen_PEarray_permanent_fault_dict(faultloc, faultinfo, mac_config=True)
    model_mac_fault_dict_list[1], psidx_tmp = PE_mapping_backward(ref_model.layers[1], MXU, verbose=verbose, return_detail=True)
    psidx_cnt+=psidx_tmp['num_layer_psum_idx']
    MXU.clear_all()
    
    PE_mapping_forward(ifmap_tile_conv2,wght_tile_conv2,ofmap_tile_conv2,MXU,
                        ifmap_config_conv2,wght_config_conv2,ofmap_config_conv2,MXU_config_conv2,
                        pre_plan=True,verbose=verbose)
    MXU.gen_PEarray_permanent_fault_dict(faultloc, faultinfo, mac_config=True)
    model_mac_fault_dict_list[3], psidx_tmp = PE_mapping_backward(ref_model.layers[3], MXU, verbose=verbose, return_detail=True)
    psidx_cnt+=psidx_tmp['num_layer_psum_idx']
    MXU.clear_all()
    
    PE_mapping_forward(ifmap_tile_fc1,wght_tile_fc1,ofmap_tile_fc1,MXU,
                      ifmap_config_fc1,wght_config_fc1,ofmap_config_fc1,MXU_config_fc1,
                      pre_plan=True,verbose=verbose)
    MXU.gen_PEarray_permanent_fault_dict(faultloc, faultinfo, mac_config=True)
    model_mac_fault_dict_list[6], psidx_tmp = PE_mapping_backward(ref_model.layers[7], MXU, verbose=verbose, return_detail=True)
    psidx_cnt+=psidx_tmp['num_layer_psum_idx']
    MXU.clear_all()
    
    PE_mapping_forward(ifmap_tile_fc2,wght_tile_fc2,ofmap_tile_fc2,MXU,
                      ifmap_config_fc2,wght_config_fc2,ofmap_config_fc2,MXU_config_fc2,
                      pre_plan=True,verbose=verbose)
    MXU.gen_PEarray_permanent_fault_dict(faultloc, faultinfo, mac_config=True)
    model_mac_fault_dict_list[7], psidx_tmp = PE_mapping_backward(ref_model.layers[7], MXU, verbose=verbose, return_detail=True)
    psidx_cnt+=psidx_tmp['num_layer_psum_idx']
    MXU.clear_all()
    
    # make preprocess data
    model_mac_fault_dict_list=preprocess_model_mac_fault(ref_model, PE, model_mac_fault_dict_list)
    t=time.time()-t
    if verbose>0:
        print('mapping time : %f s'%t)
        
    return model_mac_fault_dict_list, psidx_cnt

#%% test run
compile_argument={'loss':'categorical_crossentropy','optimizer':'adam','metrics':['accuracy',top2_acc]}

dataset_argument={'dataset':'mnist'}

FT_argument={'model_name':'lenet','loss_function':categorical_crossentropy,'metrics':['accuracy',top2_acc,acc_loss,relative_acc,pred_miss,top2_pred_miss,conf_score_vary_10,conf_score_vary_50]}    
    

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
    result_save_file=os.path.join(result_save_folder, report_filename+'.csv')
    inference_scheme(quantized_lenet5, 
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
    
    del model_mac_math_fdl
        
