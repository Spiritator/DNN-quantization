# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 14:32:50 2018

@author: Yung-Yu Tsai

evaluate memory fault injection testing result of MobileNetV1
"""

from tensorflow.keras.utils import multi_gpu_model,to_categorical
from simulator.models.mobilenet import QuantizedMobileNetV1FusedBN
from simulator.utils_tool.dataset_setup import dataset_setup
from simulator.utils_tool.confusion_matrix import show_confusion_matrix
from simulator.metrics.topk_metrics import top5_acc
import time
from simulator.memory.mem_bitmap import bitmap
from simulator.memory.tile import tile, tile_FC, generate_layer_memory_mapping
from simulator.testing.fault_core import generate_model_modulator
from simulator.metrics.FT_metrics import acc_loss, relative_acc, pred_miss, top5_pred_miss, conf_score_vary_10, conf_score_vary_50
from simulator.inference.evaluate import evaluate_FT

# dimensions of our images.
img_width, img_height = 224, 224

set_size=2
class_number=1000
batch_size=20
model_word_length=16
model_fractional_bit=9
rounding_method='nearest'
if set_size in [50,'full',None]:
    validation_data_dir = '../../dataset/imagenet_val_imagedatagenerator'
else:
    validation_data_dir = '../../dataset/imagenet_val_imagedatagenerator_setsize_%d'%set_size

# memory fault simulation parameter
fault_rate=0.0001

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

fast_mode=True

#%%
# fault generation

# model for get configuration
model = QuantizedMobileNetV1FusedBN(weights='../mobilenet_1_0_224_tf_fused_BN.h5', 
                                    nbits=model_word_length,
                                    fbits=model_fractional_bit, 
                                    rounding_method=rounding_method,
                                    batch_size=batch_size,
                                    quant_mode=None)

model_ifmap_fault_dict_list=[None for i in range(75)]
model_ofmap_fault_dict_list=[None for i in range(75)] 
model_weight_fault_dict_list=[[None,None] for i in range(75)]

# memory mapping
GLB_wght=bitmap(row_wght, col_wght*word_wght*model_wl, wl=model_wl)  # 65KB
GLB_ifmap=bitmap(row_ifmap, col_ifmap*word_ifmap*model_wl, wl=model_wl) # 98KB
GLB_ofmap=bitmap(row_ofmap, col_ofmap*word_ofmap*model_wl, wl=model_wl) # 98KB

# assign fault dictionary
GLB_wght.gen_bitmap_SA_fault_dict(fault_rate,fast_gen=True)
GLB_ifmap.gen_bitmap_SA_fault_dict(fault_rate,fast_gen=True)
GLB_ofmap.gen_bitmap_SA_fault_dict(fault_rate,fast_gen=True)

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
# generate fault dictionary
model_ifmap_fault_dict_list[2],model_ofmap_fault_dict_list[2],model_weight_fault_dict_list[2]\
=generate_layer_memory_mapping(model.layers[2],
                               GLB_ifmap,GLB_wght,GLB_ofmap,
                               ifmap_tile_conv1,wght_tile_conv1,ofmap_tile_conv1,
                               fast_mode=fast_mode)
model_ifmap_fault_dict_list[5],model_ofmap_fault_dict_list[5],model_weight_fault_dict_list[5]\
=generate_layer_memory_mapping(model.layers[5],
                               GLB_ifmap,GLB_wght,GLB_ofmap,
                               ifmap_tile_DW2,wght_tile_DW2,ofmap_tile_DW2,
                               fast_mode=fast_mode)
model_ifmap_fault_dict_list[7],model_ofmap_fault_dict_list[7],model_weight_fault_dict_list[7]\
=generate_layer_memory_mapping(model.layers[7],
                               GLB_ifmap,GLB_wght,GLB_ofmap,
                               ifmap_tile_PW3,wght_tile_PW3,ofmap_tile_PW3,
                               fast_mode=fast_mode)
model_ifmap_fault_dict_list[10],model_ofmap_fault_dict_list[10],model_weight_fault_dict_list[10]\
=generate_layer_memory_mapping(model.layers[10],
                               GLB_ifmap,GLB_wght,GLB_ofmap,
                               ifmap_tile_DW4,wght_tile_DW4,ofmap_tile_DW4,
                               fast_mode=fast_mode)
model_ifmap_fault_dict_list[12],model_ofmap_fault_dict_list[12],model_weight_fault_dict_list[12]\
=generate_layer_memory_mapping(model.layers[12],
                               GLB_ifmap,GLB_wght,GLB_ofmap,
                               ifmap_tile_PW5,wght_tile_PW5,ofmap_tile_PW5,
                               fast_mode=fast_mode)
model_ifmap_fault_dict_list[15],model_ofmap_fault_dict_list[15],model_weight_fault_dict_list[15]\
=generate_layer_memory_mapping(model.layers[15],
                               GLB_ifmap,GLB_wght,GLB_ofmap,
                               ifmap_tile_DW6,wght_tile_DW6,ofmap_tile_DW6,
                               fast_mode=fast_mode)
model_ifmap_fault_dict_list[17],model_ofmap_fault_dict_list[17],model_weight_fault_dict_list[17]\
=generate_layer_memory_mapping(model.layers[17],
                               GLB_ifmap,GLB_wght,GLB_ofmap,
                               ifmap_tile_PW7,wght_tile_PW7,ofmap_tile_PW7,
                               fast_mode=fast_mode)
model_ifmap_fault_dict_list[20],model_ofmap_fault_dict_list[20],model_weight_fault_dict_list[20]\
=generate_layer_memory_mapping(model.layers[20],
                               GLB_ifmap,GLB_wght,GLB_ofmap,
                               ifmap_tile_DW8,wght_tile_DW8,ofmap_tile_DW8,
                               fast_mode=fast_mode)
model_ifmap_fault_dict_list[22],model_ofmap_fault_dict_list[22],model_weight_fault_dict_list[22]\
=generate_layer_memory_mapping(model.layers[22],
                               GLB_ifmap,GLB_wght,GLB_ofmap,
                               ifmap_tile_PW9,wght_tile_PW9,ofmap_tile_PW9,
                               fast_mode=fast_mode)
model_ifmap_fault_dict_list[25],model_ofmap_fault_dict_list[25],model_weight_fault_dict_list[25]\
=generate_layer_memory_mapping(model.layers[25],
                               GLB_ifmap,GLB_wght,GLB_ofmap,
                               ifmap_tile_DW10,wght_tile_DW10,ofmap_tile_DW10,
                               fast_mode=fast_mode)
model_ifmap_fault_dict_list[27],model_ofmap_fault_dict_list[27],model_weight_fault_dict_list[27]\
=generate_layer_memory_mapping(model.layers[27],
                               GLB_ifmap,GLB_wght,GLB_ofmap,
                               ifmap_tile_PW11,wght_tile_PW11,ofmap_tile_PW11,
                               fast_mode=fast_mode)
model_ifmap_fault_dict_list[30],model_ofmap_fault_dict_list[30],model_weight_fault_dict_list[30]\
=generate_layer_memory_mapping(model.layers[30],
                               GLB_ifmap,GLB_wght,GLB_ofmap,
                               ifmap_tile_DW12,wght_tile_DW12,ofmap_tile_DW12,
                               fast_mode=fast_mode)
model_ifmap_fault_dict_list[32],model_ofmap_fault_dict_list[32],model_weight_fault_dict_list[32]\
=generate_layer_memory_mapping(model.layers[32],
                               GLB_ifmap,GLB_wght,GLB_ofmap,
                               ifmap_tile_PW13,wght_tile_PW13,ofmap_tile_PW13,
                               fast_mode=fast_mode)
model_ifmap_fault_dict_list[35],model_ofmap_fault_dict_list[35],model_weight_fault_dict_list[35]\
=generate_layer_memory_mapping(model.layers[35],
                               GLB_ifmap,GLB_wght,GLB_ofmap,
                               ifmap_tile_DW14,wght_tile_DW14,ofmap_tile_DW14,
                               fast_mode=fast_mode)
model_ifmap_fault_dict_list[37],model_ofmap_fault_dict_list[37],model_weight_fault_dict_list[37]\
=generate_layer_memory_mapping(model.layers[37],
                               GLB_ifmap,GLB_wght,GLB_ofmap,
                               ifmap_tile_PW15,wght_tile_PW15,ofmap_tile_PW15,
                               fast_mode=fast_mode)
model_ifmap_fault_dict_list[40],model_ofmap_fault_dict_list[40],model_weight_fault_dict_list[40]\
=generate_layer_memory_mapping(model.layers[40],
                               GLB_ifmap,GLB_wght,GLB_ofmap,
                               ifmap_tile_DW16,wght_tile_DW16,ofmap_tile_DW16,
                               fast_mode=fast_mode)
model_ifmap_fault_dict_list[42],model_ofmap_fault_dict_list[42],model_weight_fault_dict_list[42]\
=generate_layer_memory_mapping(model.layers[42],
                               GLB_ifmap,GLB_wght,GLB_ofmap,
                               ifmap_tile_PW17,wght_tile_PW17,ofmap_tile_PW17,
                               fast_mode=fast_mode)
model_ifmap_fault_dict_list[45],model_ofmap_fault_dict_list[45],model_weight_fault_dict_list[45]\
=generate_layer_memory_mapping(model.layers[45],
                               GLB_ifmap,GLB_wght,GLB_ofmap,
                               ifmap_tile_DW18,wght_tile_DW18,ofmap_tile_DW18,
                               fast_mode=fast_mode)
model_ifmap_fault_dict_list[47],model_ofmap_fault_dict_list[47],model_weight_fault_dict_list[47]\
=generate_layer_memory_mapping(model.layers[47],
                               GLB_ifmap,GLB_wght,GLB_ofmap,
                               ifmap_tile_PW19,wght_tile_PW19,ofmap_tile_PW19,
                               fast_mode=fast_mode)
model_ifmap_fault_dict_list[50],model_ofmap_fault_dict_list[50],model_weight_fault_dict_list[50]\
=generate_layer_memory_mapping(model.layers[50],
                               GLB_ifmap,GLB_wght,GLB_ofmap,
                               ifmap_tile_DW20,wght_tile_DW20,ofmap_tile_DW20,
                               fast_mode=fast_mode)
model_ifmap_fault_dict_list[52],model_ofmap_fault_dict_list[52],model_weight_fault_dict_list[52]\
=generate_layer_memory_mapping(model.layers[52],
                               GLB_ifmap,GLB_wght,GLB_ofmap,
                               ifmap_tile_PW21,wght_tile_PW21,ofmap_tile_PW21,
                               fast_mode=fast_mode)
model_ifmap_fault_dict_list[55],model_ofmap_fault_dict_list[55],model_weight_fault_dict_list[55]\
=generate_layer_memory_mapping(model.layers[55],
                               GLB_ifmap,GLB_wght,GLB_ofmap,
                               ifmap_tile_DW22,wght_tile_DW22,ofmap_tile_DW22,
                               fast_mode=fast_mode)
model_ifmap_fault_dict_list[57],model_ofmap_fault_dict_list[57],model_weight_fault_dict_list[57]\
=generate_layer_memory_mapping(model.layers[57],
                               GLB_ifmap,GLB_wght,GLB_ofmap,
                               ifmap_tile_PW23,wght_tile_PW23,ofmap_tile_PW23,
                               fast_mode=fast_mode)
model_ifmap_fault_dict_list[60],model_ofmap_fault_dict_list[60],model_weight_fault_dict_list[60]\
=generate_layer_memory_mapping(model.layers[60],
                               GLB_ifmap,GLB_wght,GLB_ofmap,
                               ifmap_tile_DW24,wght_tile_DW24,ofmap_tile_DW24,
                               fast_mode=fast_mode)
model_ifmap_fault_dict_list[62],model_ofmap_fault_dict_list[62],model_weight_fault_dict_list[62]\
=generate_layer_memory_mapping(model.layers[62],
                               GLB_ifmap,GLB_wght,GLB_ofmap,
                               ifmap_tile_PW25,wght_tile_PW25,ofmap_tile_PW25,
                               fast_mode=fast_mode)
model_ifmap_fault_dict_list[65],model_ofmap_fault_dict_list[65],model_weight_fault_dict_list[65]\
=generate_layer_memory_mapping(model.layers[65],
                               GLB_ifmap,GLB_wght,GLB_ofmap,
                               ifmap_tile_DW26,wght_tile_DW26,ofmap_tile_DW26,
                               fast_mode=fast_mode)
model_ifmap_fault_dict_list[67],model_ofmap_fault_dict_list[67],model_weight_fault_dict_list[67]\
=generate_layer_memory_mapping(model.layers[67],
                               GLB_ifmap,GLB_wght,GLB_ofmap,
                               ifmap_tile_PW27,wght_tile_PW27,ofmap_tile_PW27,
                               fast_mode=fast_mode)
model_ifmap_fault_dict_list[72],model_ofmap_fault_dict_list[72],model_weight_fault_dict_list[72]\
=generate_layer_memory_mapping(model.layers[72],
                               GLB_ifmap,GLB_wght,GLB_ofmap,
                               ifmap_tile_fc,wght_tile_fc,ofmap_tile_fc,
                               fast_mode=fast_mode)

#%%
# generate modulator

model_ifmap_fault_dict_list, model_ofmap_fault_dict_list, model_weight_fault_dict_list\
=generate_model_modulator(model,
                          model_word_length,
                          model_fractional_bit,
                          model_ifmap_fault_dict_list, 
                          model_ofmap_fault_dict_list, 
                          model_weight_fault_dict_list,
                          fast_gen=True)

#%%
# model setup

print('Building model...')

t = time.time()

model = QuantizedMobileNetV1FusedBN(weights='../mobilenet_1_0_224_tf_fused_BN.h5', 
                                    nbits=model_word_length,
                                    fbits=model_fractional_bit, 
                                    rounding_method=rounding_method,
                                    batch_size=batch_size,
                                    quant_mode='hybrid',
                                    ifmap_fault_dict_list=model_ifmap_fault_dict_list,
                                    ofmap_fault_dict_list=model_ofmap_fault_dict_list,
                                    weight_fault_dict_list=model_weight_fault_dict_list)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', top5_acc])

t = time.time()-t

model.summary()

print('model build time: %f s'%t)

# multi GPU model

#print('Building multi GPU model...')
#
#t = time.time()
#parallel_model = multi_gpu_model(model, gpus=2)
#parallel_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', top5_acc])
#
#parallel_model.summary()
#
#t = time.time()-t
#
#print('multi GPU model build time: %f s'%t)

#%%
#dataset setup

print('preparing dataset...')
x_train, x_test, y_train, y_test, class_indices, datagen, input_shape = dataset_setup('ImageDataGenerator', img_rows = img_width, img_cols = img_height, batch_size = batch_size, data_augmentation = False, data_dir = validation_data_dir)
print('dataset ready')


#%%
# test

t = time.time()
print('evaluating...')

from tensorflow.keras.losses import categorical_crossentropy
#prediction = parallel_model.predict_generator(datagen, verbose=1, steps=len(datagen))
prediction = model.predict_generator(datagen, verbose=1, steps=len(datagen))
test_result = evaluate_FT('mobilenet',prediction=prediction,test_label=to_categorical(datagen.classes,1000),loss_function=categorical_crossentropy,metrics=['accuracy',top5_acc,acc_loss,relative_acc,pred_miss,top5_pred_miss,conf_score_vary_10,conf_score_vary_50],fuseBN=True,setsize=set_size)

t = time.time()-t
print('\nruntime: %f s'%t)
for key in test_result.keys():
    print('Test %s\t:'%key, test_result[key])

#%%
# draw confusion matrix

#print('\n')
#prediction = model.predict_generator(datagen, verbose=1, steps=len(datagen))
#prediction = np.argmax(prediction, axis=1)
#
#show_confusion_matrix(datagen.classes,prediction,datagen.class_indices.keys(),'Confusion Matrix',figsize=(10,8),normalize=False,big_matrix=True)

