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

from keras.models import Model
from keras.layers import Input
from simulator.layers.quantized_layers import QuantizedConv2D

import pickle

#%% create test model

input_shape=Input(batch_shape=(4,56,56,32))
x=QuantizedConv2D(filters=64,
                  quantizers=quantizer(8,6),
                  kernel_size=(3,3),
                  padding='same',
                  strides=(1, 1),                              
                  activation='relu',
                  quant_mode='hybrid')(input_shape)
model=Model(inputs=input_shape, outputs=x, name='test_model')


with open('../pe_mapping_config/fault_dict_solved_layer.pickle', 'rb') as fdfile:
    fault_dict_solved_layer = pickle.load(fdfile)

