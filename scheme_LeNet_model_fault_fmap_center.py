# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 16:46:08 2019

@author: Yung-Yu Tsai

evaluate fault injection testing result of LeNet-5
Using inference scheme to arange analysis and save result. Evaluate the FT difference of center or edge of feature maps.
"""
import os

from simulator.inference.scheme import inference_scheme
from simulator.models.model_library import quantized_lenet5
from simulator.metrics.topk_metrics import top2_acc
from simulator.metrics.FT_metrics import acc_loss,relative_acc,pred_miss,top2_pred_miss,conf_score_vary_10,conf_score_vary_50
from keras.losses import categorical_crossentropy
from simulator.testing.fault_list import generate_model_stuck_fault

#%%
# setting parameter

result_save_folder='../test_result/mnist_lenet5_model_fault_rate_fmc'
weight_name='../mnist_lenet5_weight.h5'
model_word_length=8
model_fractional_bit=3
batch_size=20
# model fault simulation parameter
fault_rate_list=  [5e-7,1e-6,2e-6,5e-6,1e-5,2e-5,5e-5,1e-4,2e-4,5e-4,1e-3,2e-3,5e-3,1e-2,2e-2,5e-2,1e-1]
test_rounds_lists=[200 ,200 ,200 ,200 ,200 ,200 ,200 ,200 ,200 ,200 ,200 ,100 ,100 ,100 ,100 ,50  ,50  ]
# different level of concentration on center of feature maps
concentration_list=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

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

for concen in concentration_list:
    
    if not os.path.isdir(result_save_folder+'/'+str(concen)):
        os.mkdir(result_save_folder+'/'+str(concen))
    
    for test_rounds,fr in enumerate(fault_rate_list):
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
               'param_filter':[True,False,True],
               'fast_gen':True,
               'return_modulator':True,
               'coor_distribution':'center',
               'concentration':concen,
               'bit_loc_distribution':'uniform',
               'fault_type':'flip',
               'print_detail':False}
        
        # fault generation
        model_augment=list()
        for i in range(test_rounds_lists[test_rounds]):
            print('Generating fault for test round %d...'%(i+1))
            model_ifmap_fdl,model_ofmap_fdl,model_weight_fdl=generate_model_stuck_fault( **param)
            
#            model_ifmap_fdl, model_ofmap_fdl, model_weight_fdl\
#            =generate_model_modulator(ref_model,
#                                      model_word_length,
#                                      model_fractional_bit,
#                                      model_ifmap_fdl, 
#                                      model_ofmap_fdl, 
#                                      model_weight_fdl,
#                                      fast_gen=True)
            
            model_augment.append({'nbits':model_word_length,
                                  'fbits':model_fractional_bit,
                                  'rounding_method':'nearest',
                                  'batch_size':batch_size,
                                  'quant_mode':'hybrid',
                                  'ifmap_fault_dict_list':model_ifmap_fdl,
                                  'ofmap_fault_dict_list':model_ofmap_fdl,
                                  'weight_fault_dict_list':model_weight_fdl})
    
        result_save_file=result_save_folder+'/'+str(concen)+'/'+str(fr)+'.csv'
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


