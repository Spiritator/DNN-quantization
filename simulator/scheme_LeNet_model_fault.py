# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 16:46:08 2019

@author: Yung-Yu Tsai

An example of using inference scheme to arange analysis and save result.
"""

from inference.scheme import inference_scheme
from models.model_library import quantized_lenet5
from metrics.topk_metrics import top2_acc
from metrics.FT_metrics import acc_loss,relative_acc,pred_miss,top2_pred_miss,pred_vary_10,pred_vary_20
from keras.losses import categorical_crossentropy
from testing.fault_list import generate_model_stuck_fault


#%%
# setting parameter

result_save_folder='../../test_result/mnist_lenet5_model_fault_rate'
weight_name='../../mnist_lenet5_weight.h5'
test_rounds=5
model_word_length=8
model_factorial_bit=3
batch_size=20
# memory fault simulation parameter
fault_rate_list=[1e-5,2e-5,5e-5]


#%%
# fault generation

# model for get configuration
ref_model=quantized_lenet5(nbits=model_word_length,
                       fbits=model_factorial_bit,
                       batch_size=batch_size,
                       quant_mode=None)

# fault parameter setting
param_set=list()
for fr in fault_rate_list:
    param_set.append({'model':ref_model,
                      'fault_rate':fr,
                      'batch_size':batch_size,
                      'model_word_length':model_word_length,
                      'layer_wise':False,
                      'coor_distribution':'uniform',
                      'coor_pois_lam':None,
                      'bit_loc_distribution':'uniform',
                      'bit_loc_pois_lam':None,
                      'fault_type':'flip'})

#%%
# test
fault_rate_scan_data=dict()

for fr,param in enumerate(param_set):
    model_augment=list()
    for i in range(test_rounds):
        model_ifmap_fdl,model_ofmap_fdl,model_weight_fdl=generate_model_stuck_fault( **param)
        model_augment.append({'nbits':model_word_length,
                              'fbits':model_factorial_bit,
                              'rounding_method':'nearest',
                              'batch_size':batch_size,
                              'quant_mode':'hybrid',
                              'ifmap_fault_dict_list':model_ifmap_fdl,
                              'ofmap_fault_dict_list':model_ofmap_fdl,
                              'weight_fault_dict_list':model_weight_fdl})
    
    fault_rate_scan_data[fault_rate_list[fr]]=model_augment

compile_augment={'loss':'categorical_crossentropy','optimizer':'adam','metrics':['accuracy',top2_acc]}

dataset_augment={'dataset':'mnist'}

FT_augment={'model_name':'lenet','loss_function':categorical_crossentropy,'metrics':['accuracy',top2_acc,acc_loss,relative_acc,pred_miss,top2_pred_miss,pred_vary_10,pred_vary_20]}

for fr in fault_rate_list:
    result_save_file=result_save_folder+'/'+str(fr)+'.csv'
    inference_scheme(quantized_lenet5, fault_rate_scan_data[fr], compile_augment, dataset_augment, result_save_file, weight_load=True, weight_name=weight_name, FT_evaluate=True, FT_augment=FT_augment)


