# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 14:32:50 2018

@author: Yung-Yu Tsai

evaluate fault injection testing result of MobileNetV1
Using inference scheme to arange analysis and save result layer by layer.
"""
import os

from simulator.inference.scheme import inference_scheme
from simulator.models.mobilenet import QuantizedMobileNetV1FusedBN, preprocess_input
from simulator.metrics.topk_metrics import top5_acc
from tensorflow.keras.losses import categorical_crossentropy
from simulator.fault.fault_list import generate_model_stuck_fault
from simulator.approximation.estimate import get_model_param_size
from simulator.inference.scheme import gen_test_round_list
from simulator.metrics.FT_metrics import acc_loss, relative_acc, pred_miss, top5_pred_miss, conf_score_vary_10, conf_score_vary_50
from simulator.models.model_mods import make_ref_model

# dimensions of our images.
img_width, img_height = 224, 224


result_save_folder='../test_result/imagenet_mobilenet_model_fault_rate_lbl'
weight_name='../mobilenet_1_0_224_tf_fused_BN.h5'
set_size=2
batch_size=40
model_word_length=16
model_fractional_bit=9
rounding_method='nearest'
if set_size in [50,'full',None]:
    validation_data_dir = '../../dataset/imagenet_val_imagedatagenerator'
else:
    validation_data_dir = '../../dataset/imagenet_val_imagedatagenerator_setsize_%d'%set_size
    
# model fault simulation parameter
test_round_upper_bound=100
test_round_lower_bound=2

#%%

# model for get configuration
ref_model=make_ref_model(QuantizedMobileNetV1FusedBN(weights=weight_name, 
                                                     nbits=model_word_length,
                                                     fbits=model_fractional_bit, 
                                                     rounding_method=rounding_method,
                                                     batch_size=batch_size,
                                                     quant_mode=None,
                                                     verbose=False))

# layer by layer information
param_size_report=get_model_param_size(ref_model,batch_size)

param_layers=list()
for j in range(len(ref_model.layers)):
    if param_size_report['input_params'][j]!=0:
        param_layers.append(j)


#%%
# test
compile_argument={'loss':'categorical_crossentropy','optimizer':'adam','metrics':['accuracy',top5_acc]}

dataset_argument={'dataset':'ImageDataGenerator','img_rows':img_width,'img_cols':img_height,'batch_size':batch_size,'data_augmentation':False,'data_dir':validation_data_dir,'preprocessing_function':preprocess_input}

FT_argument={'model_name':'mobilenet','loss_function':categorical_crossentropy,'metrics':['accuracy',top5_acc,acc_loss,relative_acc,pred_miss,top5_pred_miss,conf_score_vary_10,conf_score_vary_50],'fuseBN':True,'setsize':set_size}    

for layer_id in param_layers:
    input_bits=param_size_report['input_bits'][layer_id]
    if isinstance(input_bits,list):
        input_bits=max(input_bits)
    output_bits=param_size_report['output_bits'][layer_id]
    if isinstance(output_bits,list):
        output_bits=max(output_bits)
    weight_bits=max(param_size_report['weight_bits'][layer_id])
    num_bits=max(input_bits,output_bits,weight_bits)
    
    fault_rate_list,test_rounds_lists=gen_test_round_list(num_bits,test_round_upper_bound,test_round_lower_bound,left_bound=-3,right_bound=-2)    
    
    if not os.path.isdir(result_save_folder+'/'+str(layer_id)):
        os.mkdir(result_save_folder+'/'+str(layer_id))
    
    for test_rounds,fr in enumerate(fault_rate_list):
        print('|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|')
        print('|=|        Test Bit Fault Rate %s'%str(fr))
        print('|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|')
        
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
        model_argument=list()
        n_round=test_rounds_lists[test_rounds]
        for i in range(n_round):
            print('\rGenerating fault for test round %d/%d...'%(i+1,n_round),end='')
            model_ifmap_fdl,model_ofmap_fdl,model_weight_fdl=generate_model_stuck_fault( **param)
            
#            model_ifmap_fdl, model_ofmap_fdl, model_weight_fdl\
#            =generate_model_modulator(ref_model,
#                                      model_word_length,
#                                      model_fractional_bit,
#                                      model_ifmap_fdl, 
#                                      model_ofmap_fdl, 
#                                      model_weight_fdl,
#                                      fast_gen=True)
            
            model_argument.append({'weights':weight_name,
                                  'nbits':model_word_length,
                                  'fbits':model_fractional_bit,
                                  'rounding_method':rounding_method,
                                  'batch_size':batch_size,
                                  'quant_mode':'hybrid',
                                  'ifmap_fault_dict_list':model_ifmap_fdl,
                                  'ofmap_fault_dict_list':model_ofmap_fdl,
                                  'weight_fault_dict_list':model_weight_fdl})
    
        result_save_file=result_save_folder+'/'+str(layer_id)+'/'+str(fr)+'.csv'
        inference_scheme(QuantizedMobileNetV1FusedBN, 
                         model_argument, 
                         compile_argument, 
                         dataset_argument, 
                         result_save_file, 
                         FT_evaluate_argument=FT_argument, 
                         name_tag='fault rate '+str(fr))
        
        del model_argument


