# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 11:49:48 2020

@author: Yung-Yu Tsai

PE MAC unit setup example
This file shows example of PE MAC unit setup with high level control and read in configuration from json file
Also, the example of how the MAC math fault injection work
"""

#%%

from simulator.comp_unit.mac import mac_unit
from simulator.layers.quantized_ops import quantizer
from simulator.fault.fault_mac import mac_fault_injector

#%% setup MAC unit

PE=mac_unit(quantizers=quantizer(nb=8,
                                 fb=6,
                                 rounding_method='nearest'),
            quant_mode='intrinsic',
            ifmap_io={'type':'io_pair', 
                      'dimension':'PE_x', 
                      'direction':'forward'},
            wght_io={'type':'io_pair',
                     'dimension':'PE_y',
                     'direction':'forward'},
            psum_io={'type':'io_pair', 
                     'dimension':'PE_y', 
                     'direction':'forward'}
            )

PE=mac_unit('../pe_mapping_config/mac_unit_config.json')

fault_loc_orig=(4,3)
fault_loc_prop=PE.propagated_idx_list('ifmap_in', fault_loc_orig, (8,8))

#%% MAC unit math fault injection

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from simulator.layers.quantized_layers import QuantizedConv2D

import tensorflow as tf
import pickle
import numpy as np

#%% create test model & layer data 

qtz_info={'nb':8,'fb':6,'rounding_method':'nearest'}
qtz=quantizer(**qtz_info)

ifmap=np.divide(np.random.randint(-32,31,[4,56,56,32]),2**6,dtype='float32')
weight=np.divide(np.random.randint(-32,31,[3,3,32,64]),2**6,dtype='float32')
weightT=tf.Variable(weight)
ofmap=np.zeros([4,56,56,64],dtype='float32')

# load fault dictionary 
with open('../pe_mapping_config/fault_dict_solved_layer_wghtin.pickle', 'rb') as fdfile:
    fault_dict_solved_layer = pickle.load(fdfile)


#%% test inject mac math fault | tf2.x
    
# # mac_unit preprocess
# preprocess_data=PE.preprocess_mac_fault_caller(fault_dict_solved_layer, noise_inject=False, fast_gen=True,
#                                                layer_type='Conv2D',padding='same')

#%% tf.function decorator

#TODO
# tf.function decorator not working -> fuction passing preprocessed_data transform argument into Tensor which contradict with inner tf.constant() call

# @tf.function
# def layer_mac_math_FI(macunit,inputs,weights,outputs,faultdict,qtzr,layer_type,padding='valid', noise_inject=False, fast_gen=True):
#     output_alter=tf.add(outputs,tf.constant(0.))
    
#     #qtzr=quantizer(**qtzr)
#     output_alter=mac_fault_injector(macunit)(output_alter, faultdict,
#                                               inputs, weights, 
#                                               noise_inject=noise_inject,
#                                               fast_gen=fast_gen,
#                                               layer_type=layer_type,
#                                               padding=padding)
#     return output_alter

# ofmap_alter=layer_mac_math_FI(PE,ifmap, weightT, ofmap, preprocess_data, qtz_info, 'Conv2D', padding='same', noise_inject=False, fast_gen=True)
# ofmap_alter=ofmap_alter.numpy()

#%% tf EagerTensor call

# ifmapT=tf.add(ifmap,tf.constant(1.))
# ofmapT=tf.add(ofmap,tf.constant(1.))

# ofmap_alter=mac_fault_injector(PE)(ofmapT, preprocess_data, ifmapT, weightT,
#                                    noise_inject=False, fast_gen=True,
#                                    layer_type='Conv2D', padding='same')
# ofmap_alter=ofmap_alter.numpy()


#%% test inject mac math fault not fast gen

# with open('../pe_mapping_config/fault_dict_solved_layer_scatter.pickle', 'rb') as fdfile:
#     fault_dict_solved_scatter = pickle.load(fdfile)
    
# # mac_unit preprocess
# preprocess_data=PE.preprocess_mac_fault_caller(fault_dict_solved_scatter, noise_inject=False, fast_gen=False,
#                                                layer_type='Conv2D',padding='same')

# ifmapT=tf.add(ifmap,tf.constant(1.))
# ofmapT=tf.add(ofmap,tf.constant(1.))

# ofmap_alt_scatter=mac_fault_injector(PE)(ofmapT, preprocess_data, ifmapT, weightT,
#                                          noise_inject=False, fast_gen=False,
#                                          layer_type='Conv2D', padding='same')

# ofmap_alt_scatter=ofmap_alt_scatter.numpy()


#%% layer mac macth FI test

# PE=mac_unit(quantizers=quantizer(nb=8,
#                                   fb=6,
#                                   rounding_method='nearest'),
#             quant_mode='hybrid',
#             ifmap_io={'type':'io_pair', 
#                       'dimension':'PE_x', 
#                       'direction':'forward'},
#             wght_io={'type':'io_pair',
#                       'dimension':'PE_y',
#                       'direction':'forward'},
#             psum_io={'type':'io_pair', 
#                       'dimension':'PE_y', 
#                       'direction':'forward'},
#             noise_inject=False,
#             fast_gen=True
#             )

# # mac_unit preprocess
# preprocess_data=PE.preprocess_mac_fault_caller(fault_dict_solved_layer, layer_type='Conv2D',padding='same')

# ifmap=np.divide(np.random.randint(-32,31,[4,56,56,32]),2**6,dtype='float32')
# weight=np.divide(np.random.randint(-32,31,[3,3,32,64]),2**6,dtype='float32')

# input_shape=Input(batch_shape=(4,56,56,32))
# x=QuantizedConv2D(filters=64,
#                   quantizers=qtz,
#                   kernel_size=(3,3),
#                   padding='same',
#                   strides=(1, 1),                              
#                   activation='relu',
#                   use_bias=False,
#                   quant_mode='hybrid',
#                   ofmap_sa_fault_injection=preprocess_data,
#                   mac_unit=PE)(input_shape)
# model=Model(inputs=input_shape, outputs=x, name='test_model')

# model.layers[1].set_weights([weight])

# # ofmap_alter=model.predict(ifmap,verbose=1,batch_size=4)
# ofmap_alter=model(ifmap)
# ofmap_alter=ofmap_alter.numpy()


#%% layer mac noise FI test

PE=mac_unit(quantizers=quantizer(nb=8,
                                  fb=6,
                                  rounding_method='nearest'),
            quant_mode='hybrid',
            ifmap_io={'type':'io_pair', 
                      'dimension':'PE_x', 
                      'direction':'forward'},
            wght_io={'type':'io_pair',
                      'dimension':'PE_y',
                      'direction':'forward'},
            psum_io={'type':'io_pair', 
                      'dimension':'PE_y', 
                      'direction':'forward'},
            noise_inject=True,
            fast_gen=True,
            amp_factor_fmap=0.28,
            amp_factor_wght=0.28
            )

# mac_unit preprocess
preprocess_data=PE.preprocess_mac_fault_caller(fault_dict_solved_layer, ofmap_shape=(4,56,56,64))

ifmap=np.divide(np.random.randint(-32,31,[4,56,56,32]),2**6,dtype='float32')
weight=np.divide(np.random.randint(-32,31,[3,3,32,64]),2**6,dtype='float32')

input_shape=Input(batch_shape=(4,56,56,32))
x=QuantizedConv2D(filters=64,
                  quantizers=qtz,
                  kernel_size=(3,3),
                  padding='same',
                  strides=(1, 1),                              
                  activation='relu',
                  use_bias=False,
                  quant_mode='hybrid',
                  ofmap_sa_fault_injection=preprocess_data,
                  mac_unit=PE)(input_shape)
model=Model(inputs=input_shape, outputs=x, name='test_model')

model.layers[1].set_weights([weight])

# ofmap_alter=model.predict(ifmap,verbose=1,batch_size=4)
ofmap_alter=model(ifmap)
ofmap_alter=ofmap_alter.numpy()
