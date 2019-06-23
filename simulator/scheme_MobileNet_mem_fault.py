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
from memory.mem_bitmap import bitmap
from memory.tile import tile, tile_FC, generate_layer_memory_mapping
from testing.fault_core import generate_model_modulator
from metrics.FT_metrics import acc_loss, relative_acc, pred_miss, top5_pred_miss, conf_score_vary_10, conf_score_vary_50

# dimensions of our images.
img_width, img_height = 224, 224

result_save_folder='../../test_result/imagenet_mobilenet_memory_fault_rate'
weight_name='../../mobilenet_1_0_224_tf_fused_BN.h5'
set_size=2
class_number=1000
batch_size=20
model_word_length=16
model_fractional_bit=9
rounding_method='nearest'
if set_size in [50,'full',None]:
    validation_data_dir = '../../../dataset/imagenet_val_imagedatagenerator'
else:
    validation_data_dir = '../../../dataset/imagenet_val_imagedatagenerator_setsize_%d'%set_size

# memory fault simulation parameter
row_ifmap=98
col_ifmap=16*8
word_ifmap=4

row_ofmap=98
col_ofmap=16*8
word_ofmap=4

row_wght=64+1
col_wght=16*8
word_wght=4

model_wl=model_word_length

memory_column_priority=['Tm','Tc','Tr','Tn']
memory_row_priority=['Tr','Tm','Tc','Tn']

# memory fault simulation parameter
fault_rate_list=  [2e-6,5e-6,1e-5,2e-5,5e-5,1e-4,2e-4,5e-4,1e-3,2e-3,5e-3,1e-2,2e-2,5e-2,1e-1]
test_rounds_lists=[200 ,200 ,200 ,200 ,200 ,100, 100 ,100 ,50 ,50 ,10 ,10 ,10 ,5  ,2  ,2  ,2  ]

#%%
# fault generation

# model for get configuration
def call_model():
    return QuantizedMobileNetV1FusedBN(weights=weight_name, 
                                       nbits=model_word_length,
                                       fbits=model_fractional_bit, 
                                       rounding_method=rounding_method,
                                       batch_size=batch_size,
                                       quant_mode=None)

# memory mapping
GLB_wght=bitmap(row_wght, col_wght*word_wght*model_wl, wl=model_wl)  # 65KB
GLB_ifmap=bitmap(row_ifmap, col_ifmap*word_ifmap*model_wl, wl=model_wl) # 98KB
GLB_ofmap=bitmap(row_ofmap, col_ofmap*word_ofmap*model_wl, wl=model_wl) # 98KB

#%%
# tile setting

# standard conv1
ofmap_tile_conv1=tile((1,38,38,32),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_conv1=tile((1,76,76,4),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_conv1 =tile((3,3,4,32),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
# DW conv2
ofmap_tile_DW2=tile((1,112,112,4),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_DW2=tile((1,112,112,4),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_DW2 =tile((3,3,4,1),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
# PW conv3
ofmap_tile_PW3=tile((1,38,38,32),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_PW3=tile((1,38,38,16),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_PW3 =tile((1,1,16,32),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
# DW conv4
ofmap_tile_DW4=tile((1,28,28,16),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_DW4=tile((1,56,56,16),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_DW4 =tile((3,3,16,1),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
# PW conv5
ofmap_tile_PW5=tile((1,28,28,64),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_PW5=tile((1,28,28,64),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_PW5 =tile((1,1,64,64),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
# DW conv6
ofmap_tile_DW6=tile((1,56,56,16),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_DW6=tile((1,56,56,16),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_DW6 =tile((3,3,16,1),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
# PW conv7
ofmap_tile_PW7=tile((1,56,56,16),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_PW7=tile((1,56,56,8),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_PW7 =tile((1,1,8,16),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
# DW conv8
ofmap_tile_DW8=tile((1,28,28,16),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_DW8=tile((1,56,56,16),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_DW8 =tile((3,3,16,1),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
# PW conv9
ofmap_tile_PW9=tile((1,28,28,32),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_PW9=tile((1,28,28,64),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_PW9 =tile((1,1,64,32),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
# DW conv10
ofmap_tile_DW10=tile((1,28,28,64),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_DW10=tile((1,28,28,64),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_DW10 =tile((3,3,64,1),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
# PW conv11
ofmap_tile_PW11=tile((1,28,28,32),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_PW11=tile((1,28,28,64),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_PW11 =tile((1,1,64,32),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
# DW conv12
ofmap_tile_DW12=tile((1,14,14,64),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_DW12=tile((1,28,28,64),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_DW12 =tile((3,3,64,1),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
# PW conv13
ofmap_tile_PW13=tile((1,14,14,128),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_PW13=tile((1,14,14,256),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_PW13 =tile((1,1,256,128),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
# DW conv14
ofmap_tile_DW14=tile((1,14,14,256),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_DW14=tile((1,14,14,256),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_DW14 =tile((3,3,256,1),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
# PW conv15
ofmap_tile_PW15=tile((1,14,14,128),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_PW15=tile((1,14,14,256),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_PW15 =tile((1,1,256,128),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
# DW conv16
ofmap_tile_DW16=tile((1,14,14,256),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_DW16=tile((1,14,14,256),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_DW16 =tile((3,3,256,1),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
# PW conv17
ofmap_tile_PW17=tile((1,14,14,128),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_PW17=tile((1,14,14,256),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_PW17 =tile((1,1,256,128),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
# DW conv18
ofmap_tile_DW18=tile((1,14,14,256),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_DW18=tile((1,14,14,256),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_DW18 =tile((3,3,256,1),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
# PW conv19
ofmap_tile_PW19=tile((1,14,14,128),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_PW19=tile((1,14,14,256),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_PW19 =tile((1,1,256,128),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
# DW conv20
ofmap_tile_DW20=tile((1,14,14,256),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_DW20=tile((1,14,14,256),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_DW20 =tile((3,3,256,1),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
# PW conv21
ofmap_tile_PW21=tile((1,14,14,128),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_PW21=tile((1,14,14,256),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_PW21 =tile((1,1,256,128),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
# DW conv22
ofmap_tile_DW22=tile((1,14,14,256),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_DW22=tile((1,14,14,256),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_DW22 =tile((3,3,256,1),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
# PW conv23
ofmap_tile_PW23=tile((1,14,14,128),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_PW23=tile((1,14,14,256),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_PW23 =tile((1,1,256,128),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
# DW conv24
ofmap_tile_DW24=tile((1,7,7,256),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_DW24=tile((1,14,14,256),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_DW24 =tile((3,3,256,1),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
# PW conv25
ofmap_tile_PW25=tile((1,7,7,128),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_PW25=tile((1,7,7,256),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_PW25 =tile((1,1,256,128),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
# DW conv26
ofmap_tile_DW26=tile((1,7,7,512),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_DW26=tile((1,7,7,512),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_DW26 =tile((3,3,512,1),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
# PW conv27
ofmap_tile_PW27=tile((1,7,7,128),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_PW27=tile((1,7,7,256),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_PW27 =tile((1,1,256,128),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
# Pred conv (fc1000)
ofmap_tile_fc=tile((1,1,1,128),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_fc=tile((1,1,1,256),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_fc =tile((1,1,256,128),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)

#%%

def gen_model_mem_fault_dict(ref_model,fault_rate,print_detail=False,fast_mode=True):
    model_ifmap_fault_dict_list=[None for i in range(75)]
    model_ofmap_fault_dict_list=[None for i in range(75)] 
    model_weight_fault_dict_list=[[None,None] for i in range(75)]
    
    GLB_ifmap.clear()
    GLB_ofmap.clear()
    GLB_wght.clear()
    ofmap_tile_conv1.clear()
    ifmap_tile_conv1.clear()
    wght_tile_conv1.clear()
    ofmap_tile_DW2.clear()
    ifmap_tile_DW2.clear()
    wght_tile_DW2.clear()
    ofmap_tile_PW3.clear()
    ifmap_tile_PW3.clear()
    wght_tile_PW3.clear()
    ofmap_tile_DW4.clear()
    ifmap_tile_DW4.clear()
    wght_tile_DW4.clear()
    ofmap_tile_PW5.clear()
    ifmap_tile_PW5.clear()
    wght_tile_PW5.clear()
    ofmap_tile_DW6.clear()
    ifmap_tile_DW6.clear()
    wght_tile_DW6.clear()
    ofmap_tile_PW7.clear()
    ifmap_tile_PW7.clear()
    wght_tile_PW7.clear()
    ofmap_tile_DW8.clear()
    ifmap_tile_DW8.clear()
    wght_tile_DW8.clear()
    ofmap_tile_PW9.clear()
    ifmap_tile_PW9.clear()
    wght_tile_PW9.clear()
    ofmap_tile_DW10.clear()
    ifmap_tile_DW10.clear()
    wght_tile_DW10.clear()
    ofmap_tile_PW11.clear()
    ifmap_tile_PW11.clear()
    wght_tile_PW11.clear()
    ofmap_tile_DW12.clear()
    ifmap_tile_DW12.clear()
    wght_tile_DW12.clear()
    ofmap_tile_PW13.clear()
    ifmap_tile_PW13.clear()
    wght_tile_PW13.clear()
    ofmap_tile_DW14.clear()
    ifmap_tile_DW14.clear()
    wght_tile_DW14.clear()
    ofmap_tile_PW15.clear()
    ifmap_tile_PW15.clear()
    wght_tile_PW15.clear()
    ofmap_tile_DW16.clear()
    ifmap_tile_DW16.clear()
    wght_tile_DW16.clear()
    ofmap_tile_PW17.clear()
    ifmap_tile_PW17.clear()
    wght_tile_PW17.clear()
    ofmap_tile_DW18.clear()
    ifmap_tile_DW18.clear()
    wght_tile_DW18.clear()
    ofmap_tile_PW19.clear()
    ifmap_tile_PW19.clear()
    wght_tile_PW19.clear()
    ofmap_tile_DW20.clear()
    ifmap_tile_DW20.clear()
    wght_tile_DW20.clear()
    ofmap_tile_PW21.clear()
    ifmap_tile_PW21.clear()
    wght_tile_PW21.clear()
    ofmap_tile_DW22.clear()
    ifmap_tile_DW22.clear()
    wght_tile_DW22.clear()
    ofmap_tile_PW23.clear()
    ifmap_tile_PW23.clear()
    wght_tile_PW23.clear()
    ofmap_tile_DW24.clear()
    ifmap_tile_DW24.clear()
    wght_tile_DW24.clear()
    ofmap_tile_PW25.clear()
    ifmap_tile_PW25.clear()
    wght_tile_PW25.clear()
    ofmap_tile_DW26.clear()
    ifmap_tile_DW26.clear()
    wght_tile_DW26.clear()
    ofmap_tile_PW27.clear()
    ifmap_tile_PW27.clear()
    wght_tile_PW27.clear()
    ofmap_tile_fc.clear()
    ifmap_tile_fc.clear()
    wght_tile_fc.clear()

    # assign fault dictionary
    GLB_wght.gen_bitmap_SA_fault_dict(fault_rate,fast_gen=True)
    GLB_ifmap.gen_bitmap_SA_fault_dict(fault_rate,fast_gen=True)
    GLB_ofmap.gen_bitmap_SA_fault_dict(fault_rate,fast_gen=True)
    
    # generate fault dictionary
    model_ifmap_fault_dict_list[2],model_ofmap_fault_dict_list[2],model_weight_fault_dict_list[2]\
    =generate_layer_memory_mapping(ref_model.layers[2],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_conv1,wght_tile_conv1,ofmap_tile_conv1,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[5],model_ofmap_fault_dict_list[5],model_weight_fault_dict_list[5]\
    =generate_layer_memory_mapping(ref_model.layers[5],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_DW2,wght_tile_DW2,ofmap_tile_DW2,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[7],model_ofmap_fault_dict_list[7],model_weight_fault_dict_list[7]\
    =generate_layer_memory_mapping(ref_model.layers[7],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_PW3,wght_tile_PW3,ofmap_tile_PW3,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[10],model_ofmap_fault_dict_list[10],model_weight_fault_dict_list[10]\
    =generate_layer_memory_mapping(ref_model.layers[10],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_DW4,wght_tile_DW4,ofmap_tile_DW4,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[12],model_ofmap_fault_dict_list[12],model_weight_fault_dict_list[12]\
    =generate_layer_memory_mapping(ref_model.layers[12],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_PW5,wght_tile_PW5,ofmap_tile_PW5,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[15],model_ofmap_fault_dict_list[15],model_weight_fault_dict_list[15]\
    =generate_layer_memory_mapping(ref_model.layers[15],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_DW6,wght_tile_DW6,ofmap_tile_DW6,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[17],model_ofmap_fault_dict_list[17],model_weight_fault_dict_list[17]\
    =generate_layer_memory_mapping(ref_model.layers[17],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_PW7,wght_tile_PW7,ofmap_tile_PW7,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[20],model_ofmap_fault_dict_list[20],model_weight_fault_dict_list[20]\
    =generate_layer_memory_mapping(ref_model.layers[20],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_DW8,wght_tile_DW8,ofmap_tile_DW8,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[22],model_ofmap_fault_dict_list[22],model_weight_fault_dict_list[22]\
    =generate_layer_memory_mapping(ref_model.layers[22],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_PW9,wght_tile_PW9,ofmap_tile_PW9,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[25],model_ofmap_fault_dict_list[25],model_weight_fault_dict_list[25]\
    =generate_layer_memory_mapping(ref_model.layers[25],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_DW10,wght_tile_DW10,ofmap_tile_DW10,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[27],model_ofmap_fault_dict_list[27],model_weight_fault_dict_list[27]\
    =generate_layer_memory_mapping(ref_model.layers[27],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_PW11,wght_tile_PW11,ofmap_tile_PW11,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[30],model_ofmap_fault_dict_list[30],model_weight_fault_dict_list[30]\
    =generate_layer_memory_mapping(ref_model.layers[30],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_DW12,wght_tile_DW12,ofmap_tile_DW12,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[32],model_ofmap_fault_dict_list[32],model_weight_fault_dict_list[32]\
    =generate_layer_memory_mapping(ref_model.layers[32],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_PW13,wght_tile_PW13,ofmap_tile_PW13,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[35],model_ofmap_fault_dict_list[35],model_weight_fault_dict_list[35]\
    =generate_layer_memory_mapping(ref_model.layers[35],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_DW14,wght_tile_DW14,ofmap_tile_DW14,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[37],model_ofmap_fault_dict_list[37],model_weight_fault_dict_list[37]\
    =generate_layer_memory_mapping(ref_model.layers[37],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_PW15,wght_tile_PW15,ofmap_tile_PW15,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[40],model_ofmap_fault_dict_list[40],model_weight_fault_dict_list[40]\
    =generate_layer_memory_mapping(ref_model.layers[40],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_DW16,wght_tile_DW16,ofmap_tile_DW16,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[42],model_ofmap_fault_dict_list[42],model_weight_fault_dict_list[42]\
    =generate_layer_memory_mapping(ref_model.layers[42],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_PW17,wght_tile_PW17,ofmap_tile_PW17,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[45],model_ofmap_fault_dict_list[45],model_weight_fault_dict_list[45]\
    =generate_layer_memory_mapping(ref_model.layers[45],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_DW18,wght_tile_DW18,ofmap_tile_DW18,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[47],model_ofmap_fault_dict_list[47],model_weight_fault_dict_list[47]\
    =generate_layer_memory_mapping(ref_model.layers[47],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_PW19,wght_tile_PW19,ofmap_tile_PW19,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[50],model_ofmap_fault_dict_list[50],model_weight_fault_dict_list[50]\
    =generate_layer_memory_mapping(ref_model.layers[50],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_DW20,wght_tile_DW20,ofmap_tile_DW20,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[52],model_ofmap_fault_dict_list[52],model_weight_fault_dict_list[52]\
    =generate_layer_memory_mapping(ref_model.layers[52],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_PW21,wght_tile_PW21,ofmap_tile_PW21,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[55],model_ofmap_fault_dict_list[55],model_weight_fault_dict_list[55]\
    =generate_layer_memory_mapping(ref_model.layers[55],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_DW22,wght_tile_DW22,ofmap_tile_DW22,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[57],model_ofmap_fault_dict_list[57],model_weight_fault_dict_list[57]\
    =generate_layer_memory_mapping(ref_model.layers[57],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_PW23,wght_tile_PW23,ofmap_tile_PW23,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[60],model_ofmap_fault_dict_list[60],model_weight_fault_dict_list[60]\
    =generate_layer_memory_mapping(ref_model.layers[60],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_DW24,wght_tile_DW24,ofmap_tile_DW24,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[62],model_ofmap_fault_dict_list[62],model_weight_fault_dict_list[62]\
    =generate_layer_memory_mapping(ref_model.layers[62],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_PW25,wght_tile_PW25,ofmap_tile_PW25,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[65],model_ofmap_fault_dict_list[65],model_weight_fault_dict_list[65]\
    =generate_layer_memory_mapping(ref_model.layers[65],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_DW26,wght_tile_DW26,ofmap_tile_DW26,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[67],model_ofmap_fault_dict_list[67],model_weight_fault_dict_list[67]\
    =generate_layer_memory_mapping(ref_model.layers[67],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_PW27,wght_tile_PW27,ofmap_tile_PW27,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[72],model_ofmap_fault_dict_list[72],model_weight_fault_dict_list[72]\
    =generate_layer_memory_mapping(ref_model.layers[72],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_fc,wght_tile_fc,ofmap_tile_fc,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    
    return model_ifmap_fault_dict_list,model_ofmap_fault_dict_list,model_weight_fault_dict_list


#%%
# test
compile_augment={'loss':'categorical_crossentropy','optimizer':'adam','metrics':['accuracy',top5_acc]}

dataset_augment={'dataset':'ImageDataGenerator','img_rows':img_width,'img_cols':img_height,'batch_size':batch_size,'data_augmentation':False,'data_dir':validation_data_dir,'preprocessing_function':preprocess_input}

FT_augment={'model_name':'mobilenet','loss_function':categorical_crossentropy,'metrics':['accuracy',top5_acc,acc_loss,relative_acc,pred_miss,top5_pred_miss,conf_score_vary_10,conf_score_vary_50],'fuseBN':True,'setsize':set_size}    

for test_rounds,fr in enumerate(fault_rate_list):
    print('|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|')
    print('|=|        Test Bit Fault Rate %s'%str(fr))
    print('|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|')
    ref_model=call_model()
        
    # fault generation
    model_augment=list()
    for i in range(test_rounds_lists[test_rounds]):
        print('Generating fault for test round %d...'%(i+1))
        model_ifmap_fdl,model_ofmap_fdl,model_weight_fdl=gen_model_mem_fault_dict(ref_model,fr)
        
        model_ifmap_fdl, model_ofmap_fdl, model_weight_fdl\
        =generate_model_modulator(ref_model,
                                  model_word_length,
                                  model_fractional_bit,
                                  model_ifmap_fdl, 
                                  model_ofmap_fdl, 
                                  model_weight_fdl,
                                  fast_gen=True)
        
        model_augment.append({'weights':weight_name,
                              'nbits':model_word_length,
                              'fbits':model_fractional_bit,
                              'rounding_method':rounding_method,
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
    
    del model_augment

