# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 16:46:08 2019

@author: Yung-Yu Tsai

An example of using inference scheme to arange analysis and save result.
"""

from inference.scheme import inference_scheme
from models.model_library import quantized_4C2F
from metrics.topk_metrics import top2_acc,top5_acc
from memory.mem_bitmap import bitmap
from memory.tile import tile, tile_FC, generate_layer_memory_mapping


#%%
# setting parameter

result_save_file='../../test_result/cifar10_4C2F_mem_fault.csv'
weight_name='../../cifar10_4C2FBN_weight_fused_BN.h5'
test_rounds=100
model_word_length=16
model_factorial_bit=12
batch_size=20
# memory fault simulation parameter
fault_rate=0.00001
# buffer size 25.6KB
row=80
col=40
# buffer size 80KB
#row=100
#col=100
word=4
model_wl=model_word_length

memory_column_priority=['Tm','Tc','Tr','Tn']
memory_row_priority=['Tr','Tm','Tc','Tn']


#%%
# fault generation

# model for get configuration
ref_model=quantized_4C2F(nbits=model_word_length,
                         fbits=model_factorial_bit,
                         batch_size=batch_size,
                         quant_mode=None)

# memory mapping
GLB_wght=bitmap(row, col*word*model_wl, wl=model_wl)
GLB_ifmap=bitmap(row, col*word*model_wl, wl=model_wl)
GLB_ofmap=bitmap(row, col*word*model_wl, wl=model_wl)

# conv1
ofmap_tile_conv1=tile((1,32,32,11),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_conv1=tile((1,32,32,3),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_conv1 =tile((3,3,3,11),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)

# conv2
ofmap_tile_conv2=tile((1,30,30,11),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_conv2=tile((1,32,32,11),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_conv2 =tile((3,3,11,11),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)

# conv3
ofmap_tile_conv3=tile((1,15,15,32),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_conv3=tile((1,15,15,32),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_conv3 =tile((3,3,32,32),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)

# conv4
ofmap_tile_conv4=tile((1,13,13,32),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_conv4=tile((1,15,15,32),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_conv4 =tile((3,3,32,32),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)

# FC1
ofmap_tile_fc1=tile_FC((1,5),is_fmap=True,wl=model_wl)
ifmap_tile_fc1=tile_FC((1,2304),is_fmap=True,wl=model_wl)
wght_tile_fc1 =tile_FC((2304,5),is_fmap=False,wl=model_wl)

# FC2
ofmap_tile_fc2=tile_FC((1,10),is_fmap=True,wl=model_wl)
ifmap_tile_fc2=tile_FC((1,512),is_fmap=True,wl=model_wl)
wght_tile_fc2 =tile_FC((512,10),is_fmap=False,wl=model_wl)

## conv1
#ofmap_tile_conv1=tile((1,32,32,32),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
#ifmap_tile_conv1=tile((1,32,32,3),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
#wght_tile_conv1 =tile((3,3,3,32),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
#
## conv2
#ofmap_tile_conv2=tile((1,30,30,32),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
#ifmap_tile_conv2=tile((1,32,32,32),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
#wght_tile_conv2 =tile((3,3,32,32),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
#
## conv3
#ofmap_tile_conv3=tile((1,15,15,64),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
#ifmap_tile_conv3=tile((1,15,15,32),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
#wght_tile_conv3 =tile((3,3,32,64),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
#
## conv4
#ofmap_tile_conv4=tile((1,13,13,64),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
#ifmap_tile_conv4=tile((1,15,15,64),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
#wght_tile_conv4 =tile((3,3,64,64),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
#
## FC1
#ofmap_tile_fc1=tile_FC((1,17),is_fmap=True,wl=model_wl)
#ifmap_tile_fc1=tile_FC((1,2304),is_fmap=True,wl=model_wl)
#wght_tile_fc1 =tile_FC((2304,17),is_fmap=False,wl=model_wl)
#
## FC2
#ofmap_tile_fc2=tile_FC((1,10),is_fmap=True,wl=model_wl)
#ifmap_tile_fc2=tile_FC((1,512),is_fmap=True,wl=model_wl)
#wght_tile_fc2 =tile_FC((512,10),is_fmap=False,wl=model_wl)


def gen_model_mem_fault_dict():
    model_ifmap_fault_dict_list=[None for i in range(14)]
    model_ofmap_fault_dict_list=[None for i in range(14)] 
    model_weight_fault_dict_list=[[None,None] for i in range(14)]

    # clear fault dictionary every iteration
    GLB_wght.clear()
    GLB_ifmap.clear()
    GLB_ofmap.clear()
    ofmap_tile_conv1.clear()
    ifmap_tile_conv1.clear()
    wght_tile_conv1.clear()
    ofmap_tile_conv2.clear()
    ifmap_tile_conv2.clear()
    wght_tile_conv2.clear()
    ofmap_tile_conv3.clear()
    ifmap_tile_conv3.clear()
    wght_tile_conv3.clear()
    ofmap_tile_conv4.clear()
    ifmap_tile_conv4.clear()
    wght_tile_conv4.clear()
    ofmap_tile_fc1.clear()
    ifmap_tile_fc1.clear()
    wght_tile_fc1.clear()
    ofmap_tile_fc2.clear()
    ifmap_tile_fc2.clear()
    wght_tile_fc2.clear()
    
    # assign fault dictionary
    GLB_wght.gen_bitmap_SA_fault_dict(fault_rate)
    GLB_ifmap.gen_bitmap_SA_fault_dict(fault_rate)
    GLB_ofmap.gen_bitmap_SA_fault_dict(fault_rate)
        
    # generate fault dictionary
    model_ifmap_fault_dict_list[1],model_ofmap_fault_dict_list[1],model_weight_fault_dict_list[1]\
    =generate_layer_memory_mapping(ref_model.layers[1],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_conv1,wght_tile_conv1,ofmap_tile_conv1)
    
    model_ifmap_fault_dict_list[2],model_ofmap_fault_dict_list[2],model_weight_fault_dict_list[2]\
    =generate_layer_memory_mapping(ref_model.layers[2],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_conv2,wght_tile_conv2,ofmap_tile_conv2)
    
    model_ifmap_fault_dict_list[5],model_ofmap_fault_dict_list[5],model_weight_fault_dict_list[5]\
    =generate_layer_memory_mapping(ref_model.layers[5],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_conv3,wght_tile_conv3,ofmap_tile_conv3)
    
    model_ifmap_fault_dict_list[6],model_ofmap_fault_dict_list[6],model_weight_fault_dict_list[6]\
    =generate_layer_memory_mapping(ref_model.layers[6],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_conv4,wght_tile_conv4,ofmap_tile_conv4)
    
    model_ifmap_fault_dict_list[10],model_ofmap_fault_dict_list[10],model_weight_fault_dict_list[10]\
    =generate_layer_memory_mapping(ref_model.layers[10],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_fc1,wght_tile_fc1,ofmap_tile_fc1)
    
    model_ifmap_fault_dict_list[13],model_ofmap_fault_dict_list[13],model_weight_fault_dict_list[13]\
    =generate_layer_memory_mapping(ref_model.layers[13],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_fc2,wght_tile_fc2,ofmap_tile_fc2)
    
    return model_ifmap_fault_dict_list,model_ofmap_fault_dict_list,model_weight_fault_dict_list

#%%
# test

model_augment=list()

for i in range(test_rounds):
    model_ifmap_fdl,model_ofmap_fdl,model_weight_fdl=gen_model_mem_fault_dict()
    model_augment.append({'nbits':model_word_length,
                          'fbits':model_factorial_bit,
                          'rounding_method':'nearest',
                          'batch_size':batch_size,
                          'quant_mode':'hybrid',
                          'ifmap_fault_dict_list':model_ifmap_fdl,
                          'ofmap_fault_dict_list':model_ofmap_fdl,
                          'weight_fault_dict_list':model_weight_fdl})

compile_augment={'loss':'categorical_crossentropy','optimizer':'adam','metrics':['accuracy',top2_acc]}

dataset_augment={'dataset':'cifar10'}


inference_scheme(quantized_4C2F, model_augment, compile_augment, dataset_augment, result_save_file, weight_load=True, weight_name=weight_name)

