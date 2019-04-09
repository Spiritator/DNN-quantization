# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 11:33:23 2018

@author: Yung-Yu Tsai

evaluate quantized testing result with custom Keras quantize layer 
"""

# setup

import keras
import numpy as np
import keras.backend as K
import time
import scipy


from models.model_library import quantized_lenet5
from utils_tool.weight_conversion import convert_original_weight_layer_name
from utils_tool.dataset_setup import dataset_setup
from utils_tool.confusion_matrix import show_confusion_matrix
from metrics.topk_metrics import top2_acc
from memory.mem_bitmap import bitmap
from memory.tile import tile, tile_FC, generate_layer_memory_mapping
#from testing.fault_list import generate_model_stuck_fault

#%%
# setting parameter

weight_name='../../mnist_lenet5_weight.h5'
model_word_length=8
model_factorial_bit=4
rounding_method='nearest'
batch_size=20
# memory fault simulation parameter
fault_rate=0.0001
row=80
col=20
word=4
model_wl=model_word_length

memory_column_priority=['Tm','Tc','Tr','Tn']
memory_row_priority=['Tr','Tm','Tc','Tn']


#%%
# fault generation

# model for get configuration
model=quantized_lenet5(nbits=model_word_length,
                       fbits=model_factorial_bit,
                       rounding_method=rounding_method,
                       batch_size=batch_size,
                       quant_mode=None)

model_ifmap_fault_dict_list=[None for i in range(8)]
model_ofmap_fault_dict_list=[None for i in range(8)] 
model_weight_fault_dict_list=[[None,None] for i in range(8)]

# memory mapping
GLB_wght=bitmap(row, col*word*model_wl, wl=model_wl)
GLB_ifmap=bitmap(row, col*word*model_wl, wl=model_wl)
GLB_ofmap=bitmap(row, col*word*model_wl, wl=model_wl)

GLB_wght.gen_bitmap_SA_fault_dict(fault_rate)
GLB_ifmap.gen_bitmap_SA_fault_dict(fault_rate)
GLB_ofmap.gen_bitmap_SA_fault_dict(fault_rate)

# conv1
ofmap_tile_conv1=tile((1,28,28,8),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_conv1=tile((1,28,28,1),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_conv1 =tile((5,5,1,8),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)

# conv2
ofmap_tile_conv2=tile((1,14,14,12),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_conv2=tile((1,14,14,16),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_conv2 =tile((5,5,16,12),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)

# FC1
ofmap_tile_fc1=tile_FC((1,128),is_fmap=True,wl=model_wl)
ifmap_tile_fc1=tile_FC((1,1764),is_fmap=True,wl=model_wl)
wght_tile_fc1 =tile_FC((1764,3),is_fmap=False,wl=model_wl)

# FC2
ofmap_tile_fc2=tile_FC((1,10),is_fmap=True,wl=model_wl)
ifmap_tile_fc2=tile_FC((1,128),is_fmap=True,wl=model_wl)
wght_tile_fc2 =tile_FC((128,10),is_fmap=False,wl=model_wl)

# generate fault dictionary
model_ifmap_fault_dict_list[1],model_ofmap_fault_dict_list[1],model_weight_fault_dict_list[1]\
=generate_layer_memory_mapping(model.layers[1],
                               GLB_ifmap,GLB_wght,GLB_ofmap,
                               ifmap_tile_conv1,wght_tile_conv1,ofmap_tile_conv1)

model_ifmap_fault_dict_list[3],model_ofmap_fault_dict_list[3],model_weight_fault_dict_list[3]\
=generate_layer_memory_mapping(model.layers[3],
                               GLB_ifmap,GLB_wght,GLB_ofmap,
                               ifmap_tile_conv2,wght_tile_conv2,ofmap_tile_conv2)

model_ifmap_fault_dict_list[6],model_ofmap_fault_dict_list[6],model_weight_fault_dict_list[6]\
=generate_layer_memory_mapping(model.layers[6],
                               GLB_ifmap,GLB_wght,GLB_ofmap,
                               ifmap_tile_fc1,wght_tile_fc1,ofmap_tile_fc1)

model_ifmap_fault_dict_list[7],model_ofmap_fault_dict_list[7],model_weight_fault_dict_list[7]\
=generate_layer_memory_mapping(model.layers[7],
                               GLB_ifmap,GLB_wght,GLB_ofmap,
                               ifmap_tile_fc2,wght_tile_fc2,ofmap_tile_fc2)


#%%
# model setup

t = time.time()
model=quantized_lenet5(nbits=model_word_length,
                       fbits=model_factorial_bit,
                       rounding_method=rounding_method,
                       batch_size=batch_size,
                       quant_mode='hybrid',
                       ifmap_fault_dict_list=model_ifmap_fault_dict_list,
                       ofmap_fault_dict_list=model_ofmap_fault_dict_list,
                       weight_fault_dict_list=model_weight_fault_dict_list)
t = time.time()-t
print('\nModel build time: %f s'%t)

print('Model compiling...')
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy',top2_acc])
print('Model compiled !')
model.load_weights(weight_name)
print('orginal weight loaded')

#%%
#dataset setup

x_train, x_test, y_train, y_test, class_indices, datagen, input_shape = dataset_setup('mnist')

#%%
# view test result

t = time.time()

test_result = model.evaluate(x_test, y_test, verbose=1, batch_size=batch_size)

t = time.time()-t
  
print('\nruntime: %f s'%t)
print('\nTest loss:', test_result[0])
print('Test top1 accuracy:', test_result[1])
print('Test top2 accuracy:', test_result[2])

#%%
# draw confusion matrix

print('\n')
prediction = model.predict(x_test, verbose=1, batch_size=batch_size)
prediction = np.argmax(prediction, axis=1)

show_confusion_matrix(np.argmax(y_test, axis=1),prediction,class_indices,'Confusion Matrix',normalize=False)

