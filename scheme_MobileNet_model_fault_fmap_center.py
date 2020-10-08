# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 14:32:50 2018

@author: Yung-Yu Tsai

evaluate fault injection testing result of MobileNetV1
Using inference scheme to arange analysis and save result. Evaluate the FT difference of center or edge of feature maps.
"""
import os

from simulator.inference.scheme import inference_scheme
from tensorflow.keras.utils import multi_gpu_model,to_categorical
from simulator.models.mobilenet import QuantizedMobileNetV1FusedBN, preprocess_input
from simulator.metrics.topk_metrics import top5_acc
from tensorflow.keras.losses import categorical_crossentropy
from simulator.fault.fault_list import generate_model_stuck_fault
from simulator.metrics.FT_metrics import acc_loss, relative_acc, pred_miss, top5_pred_miss, conf_score_vary_10, conf_score_vary_50

#%%
# setting parameter

# dimensions of our images.
img_width, img_height = 224, 224


result_save_folder='../test_result/imagenet_mobilenet_model_fault_rate_fmc'
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
    
fault_rate_list=  [1e-9,2e-9,5e-9,1e-8,2e-8,5e-8,1e-7,2e-7,5e-7,1e-6,2e-6,5e-6,1e-5,2e-5,5e-5,1e-4,2e-4,5e-4,1e-3,2e-3,5e-3,1e-2,2e-2,5e-2,1e-1]
test_rounds_lists=[200 ,200 ,200 ,200 ,200 ,200 ,200 ,200 ,200 ,200 ,200 ,100 ,100 ,100 ,100 ,50  ,50  ,50  ,10  ,10  ,10  ,4   ,2   ,2   ,2   ]

# different level of concentration on center of feature maps
concentration_list=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

#%%

# model for get configuration
def call_model():
    return QuantizedMobileNetV1FusedBN(weights=weight_name, 
                                       nbits=model_word_length,
                                       fbits=model_fractional_bit, 
                                       rounding_method=rounding_method,
                                       batch_size=batch_size,
                                       quant_mode=None)


#%%
# test
compile_argument={'loss':'categorical_crossentropy','optimizer':'adam','metrics':['accuracy',top5_acc]}

dataset_argument={'dataset':'ImageDataGenerator','img_rows':img_width,'img_cols':img_height,'batch_size':batch_size,'data_augmentation':False,'data_dir':validation_data_dir,'preprocessing_function':preprocess_input}

FT_argument={'model_name':'mobilenet','loss_function':categorical_crossentropy,'metrics':['accuracy',top5_acc,acc_loss,relative_acc,pred_miss,top5_pred_miss,conf_score_vary_10,conf_score_vary_50],'fuseBN':True,'setsize':set_size}    

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
        model_argument=list()
        for i in range(test_rounds_lists[test_rounds]):
            print('Generating fault for test round %d...'%(i+1))
            model_ifmap_fdl,model_ofmap_fdl,model_weight_fdl=generate_model_stuck_fault( **param)
                        
            model_argument.append({'weights':weight_name,
                                  'nbits':model_word_length,
                                  'fbits':model_fractional_bit,
                                  'rounding_method':rounding_method,
                                  'batch_size':batch_size,
                                  'quant_mode':'hybrid',
                                  'ifmap_fault_dict_list':model_ifmap_fdl,
                                  'ofmap_fault_dict_list':model_ofmap_fdl,
                                  'weight_fault_dict_list':model_weight_fdl})
    
        result_save_file=result_save_folder+'/'+str(concen)+'/'+str(fr)+'.csv'
        inference_scheme(QuantizedMobileNetV1FusedBN, 
                         model_argument, 
                         compile_argument, 
                         dataset_argument, 
                         result_save_file, 
                         FT_evaluate=True, 
                         FT_argument=FT_argument, 
                         name_tag='fault rate '+str(fr))
        
        del model_argument


