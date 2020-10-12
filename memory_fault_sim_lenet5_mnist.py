# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 11:33:23 2018

@author: Yung-Yu Tsai

evaluate memory fault injection testing result of LeNet-5
"""

import numpy as np
import tensorflow.keras.backend as K
import time


from simulator.models.model_library import quantized_lenet5
from simulator.utils_tool.dataset_setup import dataset_setup
from simulator.utils_tool.confusion_matrix import show_confusion_matrix
from simulator.metrics.topk_metrics import top2_acc
from simulator.memory.mem_bitmap import bitmap
from simulator.memory.tile import tile, tile_FC, generate_layer_memory_mapping
from simulator.fault.fault_core import generate_model_modulator
from tensorflow.keras.losses import categorical_crossentropy
from simulator.metrics.FT_metrics import acc_loss, relative_acc, pred_miss, top2_pred_miss, conf_score_vary_10, conf_score_vary_50
from simulator.inference.evaluate import evaluate_FT
#from simulator.fault.fault_list import generate_model_stuck_fault

#%%
# setting parameter

weight_name='../mnist_lenet5_weight.h5'
model_word_length=8
model_fractional_bit=4
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

fast_mode=True

#%%
# fault generation

# model for get configuration
model=quantized_lenet5(nbits=model_word_length,
                       fbits=model_fractional_bit,
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

# assign fault dictionary
GLB_wght.gen_bitmap_SA_fault_dict(fault_rate,fast_gen=True)
GLB_ifmap.gen_bitmap_SA_fault_dict(fault_rate,fast_gen=True)
GLB_ofmap.gen_bitmap_SA_fault_dict(fault_rate,fast_gen=True)

# conv1
ofmap_tile_conv1=tile((1,28,28,8),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_conv1=tile((1,28,28,1),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_conv1 =tile((5,5,1,8),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)

# conv2
ofmap_tile_conv2=tile((1,14,14,12),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
ifmap_tile_conv2=tile((1,14,14,16),is_fmap=True,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)
wght_tile_conv2 =tile((5,5,16,12),is_fmap=False,wl=model_wl,row_prior=memory_row_priority,col_prior=memory_column_priority)

# FC1
ofmap_tile_fc1=tile_FC((1,3),is_fmap=True,wl=model_wl)
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
                               ifmap_tile_conv1,wght_tile_conv1,ofmap_tile_conv1,
                               fast_mode=fast_mode)

model_ifmap_fault_dict_list[3],model_ofmap_fault_dict_list[3],model_weight_fault_dict_list[3]\
=generate_layer_memory_mapping(model.layers[3],
                               GLB_ifmap,GLB_wght,GLB_ofmap,
                               ifmap_tile_conv2,wght_tile_conv2,ofmap_tile_conv2,
                               fast_mode=fast_mode)

model_ifmap_fault_dict_list[6],model_ofmap_fault_dict_list[6],model_weight_fault_dict_list[6]\
=generate_layer_memory_mapping(model.layers[6],
                               GLB_ifmap,GLB_wght,GLB_ofmap,
                               ifmap_tile_fc1,wght_tile_fc1,ofmap_tile_fc1,
                               fast_mode=fast_mode)

model_ifmap_fault_dict_list[7],model_ofmap_fault_dict_list[7],model_weight_fault_dict_list[7]\
=generate_layer_memory_mapping(model.layers[7],
                               GLB_ifmap,GLB_wght,GLB_ofmap,
                               ifmap_tile_fc2,wght_tile_fc2,ofmap_tile_fc2,
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

t = time.time()
model=quantized_lenet5(nbits=model_word_length,
                       fbits=model_fractional_bit,
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

#test_result = model.evaluate(x_test, y_test, verbose=1, batch_size=batch_size)
prediction = model.predict(x_test, verbose=1,batch_size=batch_size)
test_result = evaluate_FT('lenet',prediction=prediction,test_label=y_test,loss_function=categorical_crossentropy,metrics=['accuracy',top2_acc,acc_loss,relative_acc,pred_miss,top2_pred_miss,conf_score_vary_10,conf_score_vary_50])

t = time.time()-t
print('\nruntime: %f s'%t)
for key in test_result.keys():
    print('Test %s\t:'%key, test_result[key])

#%%
# draw confusion matrix

print('\n')
#prediction = model.predict(x_test, verbose=1, batch_size=batch_size)
prediction = np.argmax(prediction, axis=1)

show_confusion_matrix(np.argmax(y_test, axis=1),prediction,class_indices,'Confusion Matrix',normalize=False)


