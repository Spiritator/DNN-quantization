# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 16:46:08 2019

@author: Yung-Yu Tsai

evaluate fault injection testing result of LeNet-5
Using inference scheme to arange analysis and save result layer by layer.
"""
import os

from simulator.inference.scheme import inference_scheme
from simulator.models.model_library import quantized_lenet5
from simulator.metrics.topk_metrics import top2_acc
from simulator.metrics.FT_metrics import acc_loss,relative_acc,pred_miss,top2_pred_miss,conf_score_vary_10,conf_score_vary_50
from tensorflow.keras.losses import categorical_crossentropy
from simulator.fault.fault_list import generate_model_stuck_fault
from simulator.approximation.estimate import get_model_param_size
from simulator.inference.scheme import gen_test_round_list


#%%
# setting parameter

result_save_folder='../test_result/mnist_lenet5_model_fault_rate_lbl'
weight_name='../mnist_lenet5_weight.h5'
model_word_length=8
model_fractional_bit=3
batch_size=20
# model fault simulation parameter
test_round_upper_bound=200
test_round_lower_bound=50

#%%

# model for get configuration
def call_model():
    return quantized_lenet5(nbits=model_word_length,
                            fbits=model_fractional_bit,
                            batch_size=batch_size,
                            quant_mode=None)

# layer by layer information
ref_model=call_model()
param_size_report=get_model_param_size(ref_model,batch_size)

param_layers=list()
for j in range(len(ref_model.layers)):
    if param_size_report['input_params'][j]!=0:
        param_layers.append(j)

#%%
# test
compile_augment={'loss':'categorical_crossentropy','optimizer':'adam','metrics':['accuracy',top2_acc]}

dataset_augment={'dataset':'mnist'}

FT_augment={'model_name':'lenet','loss_function':categorical_crossentropy,'metrics':['accuracy',top2_acc,acc_loss,relative_acc,pred_miss,top2_pred_miss,conf_score_vary_10,conf_score_vary_50]}    

for layer_id in param_layers:
    input_bits=param_size_report['input_bits'][layer_id]
    if isinstance(input_bits,list):
        input_bits=max(input_bits)
    output_bits=param_size_report['output_bits'][layer_id]
    if isinstance(output_bits,list):
        output_bits=max(output_bits)
    weight_bits=max(param_size_report['weight_bits'][layer_id])
    num_bits=max(input_bits,output_bits,weight_bits)
    
    fault_rate_list,test_rounds_lists=gen_test_round_list(num_bits,test_round_upper_bound,test_round_lower_bound)
    
    if not os.path.isdir(result_save_folder+'/'+str(layer_id)):
        os.mkdir(result_save_folder+'/'+str(layer_id))
    
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
               'param_filter':[True,True,True],
               'fast_gen':True,
               'return_modulator':True,
               'coor_distribution':'uniform',
               'coor_pois_lam':None,
               'bit_loc_distribution':'uniform',
               'bit_loc_pois_lam':None,
               'fault_type':'flip',
               'print_detail':False,
               'layer_gen_list':[layer_id]}
        
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
    
        result_save_file=result_save_folder+'/'+str(layer_id)+'/'+str(fr)+'.csv'
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


