# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:34:38 2020

@author: Yung-Yu Tsai

PE array mapping example
This file shows example of PE array mapping with high level control and read in configuration from json file
"""

from keras.models import Model
from keras.layers import Input
from simulator.layers.quantized_layers import QuantizedConv2D, quantizer

from simulator.comp_unit.PEarray import PEarray, PE_mapping_forward, PE_mapping_backward
from simulator.comp_unit.tile import tile_PE
from simulator.comp_unit.mac import mac_unit

# Test example using TPU-like vecter mac

#%% mapping PE fault to tile - setup

# dataflow pre-plan
# one run of Tile to PE with out tile fault dictionary, only mapping configure.

#%% setup tile
# weight tile and feature map tile
wght_tile=tile_PE((3,3,16,32),is_fmap=False,wl=8)
ifmap_tile=tile_PE((1,28,28,16),is_fmap=True,wl=8)
ofmap_tile=tile_PE((1,28,28,32),is_fmap=True,wl=8)

ofmap_config='../pe_mapping_config/ofmap_config.json'
wght_config='../pe_mapping_config/wght_config.json'
ifmap_config='../pe_mapping_config/ifmap_config.json'

#%% setup MAC unit

PE=mac_unit('../pe_mapping_config/mac_unit_config.json')

#%% setup PEarray

MXU=PEarray(16,16,ofmap_tile=ofmap_tile,wght_tile=wght_tile,ifmap_tile=ifmap_tile,mac_config=PE)

MXU_config='../pe_mapping_config/MXU_config.json'

#%% pre-plan PE mapping forward with file read-in

PE_mapping_forward(ifmap_tile,wght_tile,ofmap_tile,MXU,ifmap_config,wght_config,ofmap_config,MXU_config,pre_plan=True,print_detail=True)

#%% generate PE array fault dictionary

#MXU.gen_PEarray_SA_fault_dict(n_bit=8, fault_type='flip')

MXU.gen_PEarray_SA_fault_dict(n_bit=8, fault_type='flip', mac_config=True)
PE_fault_dict=MXU.fault_dict

# figure out this case
#fault_info={'SA_type':'flip','SA_bit':0,'param':'wght_in'}
#MXU.gen_PEarray_permanent_fault_dict((12,0), fault_info, mac_config=True)
#PE_fault_dict=MXU.fault_dict

#fault_info={'SA_type':'flip','SA_bit':6,'param':'wght_out'}
#MXU.gen_PEarray_permanent_fault_dict((9,7), fault_info, mac_config=True)
#PE_fault_dict=MXU.fault_dict

#fault_info={'SA_type':'flip','SA_bit':3,'param':'wght_in'}
#MXU.gen_PEarray_permanent_fault_dict((0,0), fault_info, mac_config=None)
#PE_fault_dict=MXU.fault_dict

#%% PE mapping backward

# create test model
input_shape=Input(batch_shape=(1,28,28,16))
x=QuantizedConv2D(filters=32,
                  quantizers=quantizer(8,6),
                  kernel_size=(3,3),
                  padding='same',
                  strides=(1, 1),                              
                  activation='relu',
                  quant_mode='hybrid')(input_shape)
model=Model(inputs=input_shape, outputs=x, name='test_model')

# backward mapping
PE_mac_fault_dict = PE_mapping_backward(model.layers[1], MXU, print_detail=True)


#%% PE mapping backward (fault propagation and tile2layer)

## create test model
#input_shape=Input(batch_shape=(2,56,56,32))
#x=QuantizedConv2D(filters=64,
#                  quantizers=quantizer(8,6),
#                  kernel_size=(3,3),
#                  padding='same',
#                  strides=(1, 1),                              
#                  activation='relu',
#                  quant_mode='hybrid')(input_shape)
#model=Model(inputs=input_shape, outputs=x, name='test_model')
#
## backward mapping
#PE_mac_fault_dict = PE_mapping_backward(model.layers[1], MXU, print_detail=True)

#%% PE mapping backward (fault propagation, tile2layer and uneven tile cut)

## create test model
#input_shape=Input(batch_shape=(2,70,70,32))
#x=QuantizedConv2D(filters=64,
#                  quantizers=quantizer(8,6),
#                  kernel_size=(3,3),
#                  padding='same',
#                  strides=(1, 1),                              
#                  activation='relu',
#                  quant_mode='hybrid')(input_shape)
#model=Model(inputs=input_shape, outputs=x, name='test_model')
#
## backward mapping
#PE_mac_fault_dict = PE_mapping_backward(model.layers[1], MXU, print_detail=True)


