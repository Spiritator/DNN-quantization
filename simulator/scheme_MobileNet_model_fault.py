# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 14:32:50 2018

@author: Yung-Yu Tsai

An example of using inference scheme to arange analysis and save result.
"""

from inference.scheme import inference_scheme
from keras.utils import multi_gpu_model,to_categorical
from models.mobilenet import QuantizedMobileNetV1FusedBN, preprocess_input
from metrics.topk_metrics import top5_acc
from keras.losses import categorical_crossentropy
from testing.fault_list import generate_model_stuck_fault
from testing.fault_core import generate_model_modulator
from metrics.FT_metrics import acc_loss, relative_acc, pred_miss, top5_pred_miss, conf_score_vary_10, conf_score_vary_50

# dimensions of our images.
img_width, img_height = 224, 224


result_save_folder='../../test_result/imagenet_mobilenet_model_fault_rate'
weight_name='../../mobilenet_1_0_224_tf_fused_BN.h5'
test_rounds=200
set_size=2
class_number=1000
batch_size=40
model_word_length=16
model_fractional_bit=9
rounding_method='nearest'
fault_rate=1e-8
if set_size in [50,'full',None]:
    validation_data_dir = '../../../dataset/imagenet_val_imagedatagenerator'
else:
    validation_data_dir = '../../../dataset/imagenet_val_imagedatagenerator_setsize_%d'%set_size
    
fault_rate_list=[1e-9,2e-9,5e-9,1e-8,2e-8,5e-8,1e-7,2e-7,5e-7,1e-6,2e-6,5e-6,1e-5,2e-5,5e-5,1e-4,2e-4,5e-4,1e-3,2e-3,5e-3,1e-2,2e-2,5e-2,1e-1]


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
compile_augment={'loss':'categorical_crossentropy','optimizer':'adam','metrics':['accuracy',top5_acc]}

dataset_augment={'dataset':'ImageDataGenerator','img_rows':img_width,'img_cols':img_height,'batch_size':batch_size,'data_augmentation':False,'data_dir':validation_data_dir,'preprocessing_function':preprocess_input}

FT_augment={'model_name':'mobilenet','loss_function':categorical_crossentropy,'metrics':['accuracy',top5_acc,acc_loss,relative_acc,pred_miss,top5_pred_miss,conf_score_vary_10,conf_score_vary_50],'fuseBN':True,'setsize':set_size}    

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
        
        model_augment.append({'weights':weight_name,
                              'nbits':model_word_length,
                              'fbits':model_fractional_bit,
                              'rounding_method':'nearest',
                              'batch_size':batch_size,
                              'quant_mode':'hybrid',
                              'ifmap_fault_dict_list':model_ifmap_fdl,
                              'ofmap_fault_dict_list':model_ofmap_fdl,
                              'weight_fault_dict_list':model_weight_fdl})

    result_save_file=result_save_folder+'/'+str(fr)+'.csv'
    inference_scheme(QuantizedMobileNetV1FusedBN, 
                     model_augment, 
                     compile_augment, 
                     dataset_augment, 
                     result_save_file, 
                     FT_evaluate=True, 
                     FT_augment=FT_augment, 
                     name_tag='fault rate '+str(fr))


