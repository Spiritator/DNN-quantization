# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 14:32:50 2018

@author: Yung-Yu Tsai

evaluate quantized testing result with custom Keras quantize layer 
"""

from simulator.inference.scheme import inference_scheme
from keras.utils import multi_gpu_model,to_categorical
from simulator.models.resnet50 import QuantizedResNet50FusedBN,preprocess_input
from simulator.metrics.topk_metrics import top5_acc
from keras.losses import categorical_crossentropy
from simulator.memory.mem_bitmap import bitmap
from simulator.memory.tile import tile, tile_FC, generate_layer_memory_mapping
from simulator.testing.fault_core import generate_model_modulator
from simulator.metrics.FT_metrics import acc_loss, relative_acc, pred_miss, top5_pred_miss, conf_score_vary_10, conf_score_vary_50
from simulator.inference.evaluate import evaluate_FT

# dimensions of our images.
img_width, img_height = 224, 224

result_save_folder='../test_result/imagenet_resnet_memory_fault_rate'
weight_name='../resnet50_weights_tf_dim_ordering_tf_kernels_fused_BN.h5'
set_size=2
class_number=1000
batch_size=20
model_word_length=[16,16,16]
model_fractional_bit=[8,12,8]
rounding_method='nearest'
if set_size in [50,'full',None]:
    validation_data_dir = '../../dataset/imagenet_val_imagedatagenerator'
else:
    validation_data_dir = '../../dataset/imagenet_val_imagedatagenerator_setsize_%d'%set_size

# memory fault simulation parameter
row_ifmap=392
col_ifmap=16*4
word_ifmap=4

row_ofmap=392
col_ofmap=16*4
word_ofmap=4

row_wght=1024+4
col_wght=16*4
word_wght=4

model_wl=16

memory_column_priority=['Tm','Tc','Tr','Tn']
memory_row_priority=['Tr','Tm','Tc','Tn']

# memory fault simulation parameter
fault_rate_list=  [5e-7,1e-6,2e-6,5e-6,1e-5,2e-5,5e-5,1e-4,2e-4,5e-4,1e-3,2e-3,5e-3,1e-2,2e-2,5e-2,1e-1]
test_rounds_lists=[200 ,200 ,200 ,200 ,200 ,100 ,100 ,100, 50  ,50  ,10  ,10  ,10  ,2   ,2   ,2   ,2   ]

#%%
# fault generation

# model for get configuration
def call_model():
    return QuantizedResNet50FusedBN(weights=weight_name, 
                                    nbits=model_word_length,
                                    fbits=model_fractional_bit, 
                                    rounding_method=rounding_method,
                                    batch_size=batch_size,
                                    quant_mode=None)


# memory mapping
GLB_wght=bitmap(row_wght, col_wght*word_wght*model_wl, wl=model_wl)  # 514KB
GLB_ifmap=bitmap(row_ifmap, col_ifmap*word_ifmap*model_wl, wl=model_wl) # 196KB
GLB_ofmap=bitmap(row_ofmap, col_ofmap*word_ofmap*model_wl, wl=model_wl) # 196KB

#%%
# tile setting

# conv1
ofmap_tile_conv1=tile((1,56,56,32),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_conv1=tile((1,115,115,3),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_conv1 =tile((7,7,3,32),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
# r2a_b1
ofmap_tile_r2a_b1=tile((1,55,55,32),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_r2a_b1=tile((1,55,55,32),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_r2a_b1 =tile((1,1,32,32),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
# r2a_b2a
ofmap_tile_r2a_b2a=tile((1,55,55,32),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_r2a_b2a=tile((1,55,55,32),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_r2a_b2a =tile((1,1,32,32),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
# r2bc_b2a
ofmap_tile_r2bc_b2a=tile((1,55,55,32),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_r2bc_b2a=tile((1,55,55,32),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_r2bc_b2a =tile((1,1,32,32),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
# r2abc_b2b
ofmap_tile_r2abc_b2b=tile((1,55,55,32),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_r2abc_b2b=tile((1,55,55,32),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_r2abc_b2b =tile((3,3,32,32),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
# r2abc_b2c
ofmap_tile_r2abc_b2c=tile((1,55,55,32),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_r2abc_b2c=tile((1,55,55,32),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_r2abc_b2c =tile((1,1,32,32),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
# r3a_b1
ofmap_tile_r3a_b1=tile((1,14,14,512),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_r3a_b1=tile((1,28,28,128),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_r3a_b1 =tile((1,1,128,512),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
# r3a_b2a
ofmap_tile_r3a_b2a=tile((1,14,14,128),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_r3a_b2a=tile((1,28,28,128),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_r3a_b2a =tile((1,1,128,128),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
# r3bc_b2a
ofmap_tile_r3bcd_b2a=tile((1,28,28,128),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_r3bcd_b2a=tile((1,28,28,128),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_r3bcd_b2a =tile((1,1,128,128),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
# r3abc_b2b
ofmap_tile_r3abcd_b2b=tile((1,28,28,128),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_r3abcd_b2b=tile((1,28,28,128),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_r3abcd_b2b =tile((3,3,128,128),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
# r3abc_b2c
ofmap_tile_r3abcd_b2c=tile((1,28,28,128),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_r3abcd_b2c=tile((1,28,28,128),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_r3abcd_b2c =tile((1,1,128,128),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
# r4a_b1
ofmap_tile_r4a_b1=tile((1,7,7,512),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_r4a_b1=tile((1,14,14,512),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_r4a_b1 =tile((1,1,512,512),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
# r4a_b2a
ofmap_tile_r4a_b2a=tile((1,7,7,256),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_r4a_b2a=tile((1,14,14,512),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_r4a_b2a =tile((1,1,512,256),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
# r4bcdef_b2a
ofmap_tile_r4bcdef_b2a=tile((1,14,14,256),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_r4bcdef_b2a=tile((1,14,14,512),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_r4bcdef_b2a =tile((1,1,512,256),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
# r4abc_b2b
ofmap_tile_r4abcdef_b2b=tile((1,14,14,110),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_r4abcdef_b2b=tile((1,14,14,256),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_r4abcdef_b2b =tile((3,3,256,110),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
# r4abc_b2c
ofmap_tile_r4abcdef_b2c=tile((1,14,14,512),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_r4abcdef_b2c=tile((1,14,14,256),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_r4abcdef_b2c =tile((1,1,256,512),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
# r5a_b1
ofmap_tile_r5a_b1=tile((1,7,7,512),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_r5a_b1=tile((1,14,14,512),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_r5a_b1 =tile((1,1,512,512),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
# r5a_b2a
ofmap_tile_r5a_b2a=tile((1,7,7,512),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_r5a_b2a=tile((1,14,14,512),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_r5a_b2a =tile((1,1,512,512),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
# r5bc_b2a
ofmap_tile_r5bc_b2a=tile((1,7,7,512),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_r5bc_b2a=tile((1,7,7,512),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_r5bc_b2a =tile((1,1,512,512),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
# r5abc_b2b
ofmap_tile_r5abc_b2b=tile((1,7,7,128),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_r5abc_b2b=tile((1,7,7,171),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_r5abc_b2b =tile((3,3,171,128),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
# r5abc_b2c
ofmap_tile_r5abc_b2c=tile((1,7,7,512),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_r5abc_b2c=tile((1,7,7,512),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_r5abc_b2c =tile((1,1,512,512),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
# fc1000
ofmap_tile_fc1000=tile_FC((1,250),is_fmap=True,wl=model_wl)
ifmap_tile_fc1000=tile_FC((1,1024),is_fmap=True,wl=model_wl)
wght_tile_fc1000 =tile_FC((1024,250),is_fmap=False,wl=model_wl)

#%%

def gen_model_mem_fault_dict(ref_model,fault_rate,print_detail=False,fast_mode=True):
    model_ifmap_fault_dict_list=[None for i in range(124)]
    model_ofmap_fault_dict_list=[None for i in range(124)] 
    model_weight_fault_dict_list=[[None,None] for i in range(124)]
    
    GLB_wght.clear()
    GLB_ifmap.clear()
    GLB_ofmap.clear()
    ofmap_tile_conv1.clear()
    ifmap_tile_conv1.clear()
    wght_tile_conv1.clear()
    ofmap_tile_r2a_b1.clear()
    ifmap_tile_r2a_b1.clear()
    wght_tile_r2a_b1.clear()
    ofmap_tile_r2a_b2a.clear()
    ifmap_tile_r2a_b2a.clear()
    wght_tile_r2a_b2a.clear()
    ofmap_tile_r2bc_b2a.clear()
    ifmap_tile_r2bc_b2a.clear()
    wght_tile_r2bc_b2a.clear()
    ofmap_tile_r2abc_b2b.clear()
    ifmap_tile_r2abc_b2b.clear()
    wght_tile_r2abc_b2b.clear()
    ofmap_tile_r2abc_b2c.clear()
    ifmap_tile_r2abc_b2c.clear()
    wght_tile_r2abc_b2c.clear()
    ofmap_tile_r3a_b1.clear()
    ifmap_tile_r3a_b1.clear()
    wght_tile_r3a_b1.clear()
    ofmap_tile_r3a_b2a.clear()
    ifmap_tile_r3a_b2a.clear()
    wght_tile_r3a_b2a.clear()
    ofmap_tile_r3bcd_b2a.clear()
    ifmap_tile_r3bcd_b2a.clear()
    wght_tile_r3bcd_b2a.clear()
    ofmap_tile_r3abcd_b2b.clear()
    ifmap_tile_r3abcd_b2b.clear()
    wght_tile_r3abcd_b2b.clear()
    ofmap_tile_r3abcd_b2c.clear()
    ifmap_tile_r3abcd_b2c.clear()
    wght_tile_r3abcd_b2c.clear()
    ofmap_tile_r4a_b1.clear()
    ifmap_tile_r4a_b1.clear()
    wght_tile_r4a_b1.clear()
    ofmap_tile_r4a_b2a.clear()
    ifmap_tile_r4a_b2a.clear()
    wght_tile_r4a_b2a.clear()
    ofmap_tile_r4bcdef_b2a.clear()
    ifmap_tile_r4bcdef_b2a.clear()
    wght_tile_r4bcdef_b2a.clear()
    ofmap_tile_r4abcdef_b2b.clear()
    ifmap_tile_r4abcdef_b2b.clear()
    wght_tile_r4abcdef_b2b.clear()
    ofmap_tile_r4abcdef_b2c.clear()
    ifmap_tile_r4abcdef_b2c.clear()
    wght_tile_r4abcdef_b2c.clear()
    ofmap_tile_r5a_b1.clear()
    ifmap_tile_r5a_b1.clear()
    wght_tile_r5a_b1.clear()
    ofmap_tile_r5a_b2a.clear()
    ifmap_tile_r5a_b2a.clear()
    wght_tile_r5a_b2a.clear()
    ofmap_tile_r5bc_b2a.clear()
    ifmap_tile_r5bc_b2a.clear()
    wght_tile_r5bc_b2a.clear()
    ofmap_tile_r5abc_b2b.clear()
    ifmap_tile_r5abc_b2b.clear()
    wght_tile_r5abc_b2b.clear()
    ofmap_tile_r5abc_b2c.clear()
    ifmap_tile_r5abc_b2c.clear()
    wght_tile_r5abc_b2c.clear()
    ofmap_tile_fc1000.clear()
    ifmap_tile_fc1000.clear()
    wght_tile_fc1000.clear()

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
    #r2
    model_ifmap_fault_dict_list[5],model_ofmap_fault_dict_list[5],model_weight_fault_dict_list[5]\
    =generate_layer_memory_mapping(ref_model.layers[5],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r2a_b2a,wght_tile_r2a_b2a,ofmap_tile_r2a_b2a,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[7],model_ofmap_fault_dict_list[7],model_weight_fault_dict_list[7]\
    =generate_layer_memory_mapping(ref_model.layers[7],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r2abc_b2b,wght_tile_r2abc_b2b,ofmap_tile_r2abc_b2b,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[9],model_ofmap_fault_dict_list[9],model_weight_fault_dict_list[9]\
    =generate_layer_memory_mapping(ref_model.layers[9],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r2abc_b2c,wght_tile_r2abc_b2c,ofmap_tile_r2abc_b2c,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[10],model_ofmap_fault_dict_list[10],model_weight_fault_dict_list[10]\
    =generate_layer_memory_mapping(ref_model.layers[10],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r2a_b1,wght_tile_r2a_b1,ofmap_tile_r2a_b1,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[13],model_ofmap_fault_dict_list[13],model_weight_fault_dict_list[13]\
    =generate_layer_memory_mapping(ref_model.layers[13],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r2bc_b2a,wght_tile_r2bc_b2a,ofmap_tile_r2bc_b2a,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[15],model_ofmap_fault_dict_list[15],model_weight_fault_dict_list[15]\
    =generate_layer_memory_mapping(ref_model.layers[15],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r2abc_b2b,wght_tile_r2abc_b2b,ofmap_tile_r2abc_b2b,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[17],model_ofmap_fault_dict_list[17],model_weight_fault_dict_list[17]\
    =generate_layer_memory_mapping(ref_model.layers[17],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r2abc_b2c,wght_tile_r2abc_b2c,ofmap_tile_r2abc_b2c,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[20],model_ofmap_fault_dict_list[20],model_weight_fault_dict_list[20]\
    =generate_layer_memory_mapping(ref_model.layers[20],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r2bc_b2a,wght_tile_r2bc_b2a,ofmap_tile_r2bc_b2a,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[22],model_ofmap_fault_dict_list[22],model_weight_fault_dict_list[22]\
    =generate_layer_memory_mapping(ref_model.layers[22],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r2abc_b2b,wght_tile_r2abc_b2b,ofmap_tile_r2abc_b2b,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[24],model_ofmap_fault_dict_list[24],model_weight_fault_dict_list[24]\
    =generate_layer_memory_mapping(ref_model.layers[24],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r2abc_b2c,wght_tile_r2abc_b2c,ofmap_tile_r2abc_b2c,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    #r3
    model_ifmap_fault_dict_list[27],model_ofmap_fault_dict_list[27],model_weight_fault_dict_list[27]\
    =generate_layer_memory_mapping(ref_model.layers[27],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r3a_b2a,wght_tile_r3a_b2a,ofmap_tile_r3a_b2a,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[29],model_ofmap_fault_dict_list[29],model_weight_fault_dict_list[29]\
    =generate_layer_memory_mapping(ref_model.layers[29],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r3abcd_b2b,wght_tile_r3abcd_b2b,ofmap_tile_r3abcd_b2b,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[31],model_ofmap_fault_dict_list[31],model_weight_fault_dict_list[31]\
    =generate_layer_memory_mapping(ref_model.layers[31],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r3abcd_b2c,wght_tile_r3abcd_b2c,ofmap_tile_r3abcd_b2c,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[32],model_ofmap_fault_dict_list[32],model_weight_fault_dict_list[32]\
    =generate_layer_memory_mapping(ref_model.layers[32],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r3a_b1,wght_tile_r3a_b1,ofmap_tile_r3a_b1,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[35],model_ofmap_fault_dict_list[35],model_weight_fault_dict_list[35]\
    =generate_layer_memory_mapping(ref_model.layers[35],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r3bcd_b2a,wght_tile_r3bcd_b2a,ofmap_tile_r3bcd_b2a,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[37],model_ofmap_fault_dict_list[37],model_weight_fault_dict_list[37]\
    =generate_layer_memory_mapping(ref_model.layers[37],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r3abcd_b2b,wght_tile_r3abcd_b2b,ofmap_tile_r3abcd_b2b,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[39],model_ofmap_fault_dict_list[39],model_weight_fault_dict_list[39]\
    =generate_layer_memory_mapping(ref_model.layers[39],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r3abcd_b2c,wght_tile_r3abcd_b2c,ofmap_tile_r3abcd_b2c,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[42],model_ofmap_fault_dict_list[42],model_weight_fault_dict_list[42]\
    =generate_layer_memory_mapping(ref_model.layers[42],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r3bcd_b2a,wght_tile_r3bcd_b2a,ofmap_tile_r3bcd_b2a,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[44],model_ofmap_fault_dict_list[44],model_weight_fault_dict_list[44]\
    =generate_layer_memory_mapping(ref_model.layers[44],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r3abcd_b2b,wght_tile_r3abcd_b2b,ofmap_tile_r3abcd_b2b,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[46],model_ofmap_fault_dict_list[46],model_weight_fault_dict_list[46]\
    =generate_layer_memory_mapping(ref_model.layers[46],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r3abcd_b2c,wght_tile_r3abcd_b2c,ofmap_tile_r3abcd_b2c,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[49],model_ofmap_fault_dict_list[49],model_weight_fault_dict_list[49]\
    =generate_layer_memory_mapping(ref_model.layers[49],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r3bcd_b2a,wght_tile_r3bcd_b2a,ofmap_tile_r3bcd_b2a,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[51],model_ofmap_fault_dict_list[51],model_weight_fault_dict_list[51]\
    =generate_layer_memory_mapping(ref_model.layers[51],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r3abcd_b2b,wght_tile_r3abcd_b2b,ofmap_tile_r3abcd_b2b,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[53],model_ofmap_fault_dict_list[53],model_weight_fault_dict_list[53]\
    =generate_layer_memory_mapping(ref_model.layers[53],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r3abcd_b2c,wght_tile_r3abcd_b2c,ofmap_tile_r3abcd_b2c,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    #r4
    model_ifmap_fault_dict_list[56],model_ofmap_fault_dict_list[56],model_weight_fault_dict_list[56]\
    =generate_layer_memory_mapping(ref_model.layers[56],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r4a_b2a,wght_tile_r4a_b2a,ofmap_tile_r4a_b2a,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[58],model_ofmap_fault_dict_list[58],model_weight_fault_dict_list[58]\
    =generate_layer_memory_mapping(ref_model.layers[58],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r4abcdef_b2b,wght_tile_r3abcd_b2b,ofmap_tile_r3abcd_b2b,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[60],model_ofmap_fault_dict_list[60],model_weight_fault_dict_list[60]\
    =generate_layer_memory_mapping(ref_model.layers[60],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r3abcd_b2c,wght_tile_r3abcd_b2c,ofmap_tile_r3abcd_b2c,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[61],model_ofmap_fault_dict_list[61],model_weight_fault_dict_list[61]\
    =generate_layer_memory_mapping(ref_model.layers[61],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r4a_b1,wght_tile_r4a_b1,ofmap_tile_r4a_b1,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[64],model_ofmap_fault_dict_list[64],model_weight_fault_dict_list[64]\
    =generate_layer_memory_mapping(ref_model.layers[64],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r4bcdef_b2a,wght_tile_r4bcdef_b2a,ofmap_tile_r4bcdef_b2a,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[66],model_ofmap_fault_dict_list[66],model_weight_fault_dict_list[66]\
    =generate_layer_memory_mapping(ref_model.layers[66],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r4abcdef_b2b,wght_tile_r4abcdef_b2b,ofmap_tile_r4abcdef_b2b,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[68],model_ofmap_fault_dict_list[68],model_weight_fault_dict_list[68]\
    =generate_layer_memory_mapping(ref_model.layers[68],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r4abcdef_b2c,wght_tile_r4abcdef_b2c,ofmap_tile_r4abcdef_b2c,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[71],model_ofmap_fault_dict_list[71],model_weight_fault_dict_list[71]\
    =generate_layer_memory_mapping(ref_model.layers[71],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r4bcdef_b2a,wght_tile_r4bcdef_b2a,ofmap_tile_r4bcdef_b2a,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[73],model_ofmap_fault_dict_list[73],model_weight_fault_dict_list[73]\
    =generate_layer_memory_mapping(ref_model.layers[73],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r4abcdef_b2b,wght_tile_r4abcdef_b2b,ofmap_tile_r4abcdef_b2b,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[75],model_ofmap_fault_dict_list[75],model_weight_fault_dict_list[75]\
    =generate_layer_memory_mapping(ref_model.layers[75],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r4abcdef_b2c,wght_tile_r4abcdef_b2c,ofmap_tile_r4abcdef_b2c,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[78],model_ofmap_fault_dict_list[78],model_weight_fault_dict_list[78]\
    =generate_layer_memory_mapping(ref_model.layers[78],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r4bcdef_b2a,wght_tile_r4bcdef_b2a,ofmap_tile_r4bcdef_b2a,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[80],model_ofmap_fault_dict_list[80],model_weight_fault_dict_list[80]\
    =generate_layer_memory_mapping(ref_model.layers[80],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r4abcdef_b2b,wght_tile_r4abcdef_b2b,ofmap_tile_r4abcdef_b2b,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[82],model_ofmap_fault_dict_list[82],model_weight_fault_dict_list[82]\
    =generate_layer_memory_mapping(ref_model.layers[82],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r4abcdef_b2c,wght_tile_r4abcdef_b2c,ofmap_tile_r4abcdef_b2c,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[85],model_ofmap_fault_dict_list[85],model_weight_fault_dict_list[85]\
    =generate_layer_memory_mapping(ref_model.layers[85],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r4bcdef_b2a,wght_tile_r4bcdef_b2a,ofmap_tile_r4bcdef_b2a,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[87],model_ofmap_fault_dict_list[87],model_weight_fault_dict_list[87]\
    =generate_layer_memory_mapping(ref_model.layers[87],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r4abcdef_b2b,wght_tile_r4abcdef_b2b,ofmap_tile_r4abcdef_b2b,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[89],model_ofmap_fault_dict_list[89],model_weight_fault_dict_list[89]\
    =generate_layer_memory_mapping(ref_model.layers[89],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r4abcdef_b2c,wght_tile_r4abcdef_b2c,ofmap_tile_r4abcdef_b2c,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[92],model_ofmap_fault_dict_list[92],model_weight_fault_dict_list[92]\
    =generate_layer_memory_mapping(ref_model.layers[92],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r4bcdef_b2a,wght_tile_r4bcdef_b2a,ofmap_tile_r4bcdef_b2a,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[94],model_ofmap_fault_dict_list[94],model_weight_fault_dict_list[94]\
    =generate_layer_memory_mapping(ref_model.layers[94],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r4abcdef_b2b,wght_tile_r4abcdef_b2b,ofmap_tile_r4abcdef_b2b,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[96],model_ofmap_fault_dict_list[96],model_weight_fault_dict_list[96]\
    =generate_layer_memory_mapping(ref_model.layers[96],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r4abcdef_b2c,wght_tile_r4abcdef_b2c,ofmap_tile_r4abcdef_b2c,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    #r5
    model_ifmap_fault_dict_list[99],model_ofmap_fault_dict_list[99],model_weight_fault_dict_list[99]\
    =generate_layer_memory_mapping(ref_model.layers[99],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r5a_b2a,wght_tile_r5a_b2a,ofmap_tile_r5a_b2a,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[101],model_ofmap_fault_dict_list[101],model_weight_fault_dict_list[101]\
    =generate_layer_memory_mapping(ref_model.layers[101],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r5abc_b2b,wght_tile_r5abc_b2b,ofmap_tile_r5abc_b2b,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[103],model_ofmap_fault_dict_list[103],model_weight_fault_dict_list[103]\
    =generate_layer_memory_mapping(ref_model.layers[103],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r5abc_b2c,wght_tile_r5abc_b2c,ofmap_tile_r5abc_b2c,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[104],model_ofmap_fault_dict_list[104],model_weight_fault_dict_list[104]\
    =generate_layer_memory_mapping(ref_model.layers[104],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r5a_b1,wght_tile_r5a_b1,ofmap_tile_r5a_b1,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[107],model_ofmap_fault_dict_list[107],model_weight_fault_dict_list[107]\
    =generate_layer_memory_mapping(ref_model.layers[107],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r5bc_b2a,wght_tile_r5bc_b2a,ofmap_tile_r5bc_b2a,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[109],model_ofmap_fault_dict_list[109],model_weight_fault_dict_list[109]\
    =generate_layer_memory_mapping(ref_model.layers[109],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r5abc_b2b,wght_tile_r5abc_b2b,ofmap_tile_r5abc_b2b,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[111],model_ofmap_fault_dict_list[111],model_weight_fault_dict_list[111]\
    =generate_layer_memory_mapping(ref_model.layers[111],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r5abc_b2c,wght_tile_r5abc_b2c,ofmap_tile_r5abc_b2c,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[114],model_ofmap_fault_dict_list[114],model_weight_fault_dict_list[114]\
    =generate_layer_memory_mapping(ref_model.layers[114],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r5bc_b2a,wght_tile_r5bc_b2a,ofmap_tile_r5bc_b2a,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[116],model_ofmap_fault_dict_list[116],model_weight_fault_dict_list[116]\
    =generate_layer_memory_mapping(ref_model.layers[116],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r5abc_b2b,wght_tile_r5abc_b2b,ofmap_tile_r5abc_b2b,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[118],model_ofmap_fault_dict_list[118],model_weight_fault_dict_list[118]\
    =generate_layer_memory_mapping(ref_model.layers[118],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_r5abc_b2c,wght_tile_r5abc_b2c,ofmap_tile_r5abc_b2c,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    model_ifmap_fault_dict_list[123],model_ofmap_fault_dict_list[123],model_weight_fault_dict_list[123]\
    =generate_layer_memory_mapping(ref_model.layers[123],
                                   GLB_ifmap,GLB_wght,GLB_ofmap,
                                   ifmap_tile_fc1000,wght_tile_fc1000,ofmap_tile_fc1000,
                                   print_detail=print_detail,
                                   fast_mode=fast_mode)
    
    return model_ifmap_fault_dict_list,model_ofmap_fault_dict_list,model_weight_fault_dict_list

#%%
# test
compile_augment={'loss':'categorical_crossentropy','optimizer':'adam','metrics':['accuracy',top5_acc]}

dataset_augment={'dataset':'ImageDataGenerator','img_rows':img_width,'img_cols':img_height,'batch_size':batch_size,'data_augmentation':False,'data_dir':validation_data_dir,'preprocessing_function':preprocess_input}

FT_augment={'model_name':'resnet','loss_function':categorical_crossentropy,'metrics':['accuracy',top5_acc,acc_loss,relative_acc,pred_miss,top5_pred_miss,conf_score_vary_10,conf_score_vary_50],'fuseBN':True,'setsize':set_size}    

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
    inference_scheme(QuantizedResNet50FusedBN, 
                     model_augment, 
                     compile_augment, 
                     dataset_augment, 
                     result_save_file, 
                     FT_evaluate=True, 
                     FT_augment=FT_augment, 
                     name_tag='fault rate '+str(fr))
    
    del model_augment

