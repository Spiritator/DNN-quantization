# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 16:46:08 2019

@author: Yung-Yu Tsai

An example of using inference scheme to arange analysis and save result.
"""

from inference.scheme import inference_scheme
from models.model_library import quantized_lenet5
from metrics.topk_metrics import top2_acc
from metrics.FT_metrics import acc_loss,relative_acc,pred_miss,top2_pred_miss,conf_score_vary_10,conf_score_vary_50
from keras.losses import categorical_crossentropy
from testing.fault_list import generate_model_stuck_fault
from testing.fault_core import generate_model_modulator


#%%
# setting parameter

result_save_folder='../../test_result/mnist_lenet5_model_fault_rate'
weight_name='../../mnist_lenet5_weight.h5'
test_rounds=200
model_word_length=8
model_fractional_bit=3
batch_size=20
# model fault simulation parameter
fault_rate_list=[5e-7,1e-6,2e-6,5e-6,1e-5,2e-5,5e-5,1e-4,2e-4,5e-4,1e-3,2e-3,5e-3,1e-2,2e-2,5e-2,1e-1]


#%%

# model for get configuration
def call_model():
    return quantized_lenet5(nbits=model_word_length,
                            fbits=model_fractional_bit,
                            batch_size=batch_size,
                            quant_mode=None)


#%%
# test
compile_augment={'loss':'categorical_crossentropy','optimizer':'adam','metrics':['accuracy',top2_acc]}

dataset_augment={'dataset':'mnist'}

FT_augment={'model_name':'lenet','loss_function':categorical_crossentropy,'metrics':['accuracy',top2_acc,acc_loss,relative_acc,pred_miss,top2_pred_miss,conf_score_vary_10,conf_score_vary_50]}    

for fr in fault_rate_list:
    print('|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|')
    print('|=|        Test Bit Fault Rate %s'%str(fr))
    print('|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|')
    ref_model=call_model()
    
    # fault parameter setting
    param={'model':ref_model,
           'fault_rate':fr,
           'batch_size':batch_size,
           'model_word_length':model_word_length,
           'layer_wise':False,
           'coor_distribution':'uniform',
           'coor_pois_lam':None,
           'bit_loc_distribution':'uniform',
           'bit_loc_pois_lam':None,
           'fault_type':'flip',
           'print_detail':False}
    
    # fault generation
    model_augment=list()
    for i in range(test_rounds):
        print('Generating fault for test round %d...'%i)
        model_ifmap_fdl,model_ofmap_fdl,model_weight_fdl=generate_model_stuck_fault( **param)
        
        model_ifmap_fdl, model_ofmap_fdl, model_weight_fdl\
        =generate_model_modulator(ref_model,
                                  model_word_length,
                                  model_fractional_bit,
                                  model_ifmap_fdl, 
                                  model_ofmap_fdl, 
                                  model_weight_fdl)
        
        model_augment.append({'nbits':model_word_length,
                              'fbits':model_fractional_bit,
                              'rounding_method':'nearest',
                              'batch_size':batch_size,
                              'quant_mode':'hybrid',
                              'ifmap_fault_dict_list':model_ifmap_fdl,
                              'ofmap_fault_dict_list':model_ofmap_fdl,
                              'weight_fault_dict_list':model_weight_fdl})

    result_save_file=result_save_folder+'/'+str(fr)+'.csv'
    inference_scheme(quantized_lenet5, 
                     model_augment, 
                     compile_augment, 
                     dataset_augment, 
                     result_save_file, 
                     weight_load=True, 
                     weight_name=weight_name, 
                     FT_evaluate=True, 
                     FT_augment=FT_augment, 
                     name_tag='fault rate '+str(fr))


