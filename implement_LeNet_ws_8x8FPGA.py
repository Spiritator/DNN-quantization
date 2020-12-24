# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 22:17:00 2020

@author: Yung-Yu Tsai

The test, verification and analysis for LeNet-5 inference on FPGA with weight stationary 8x8 systolic PE array.
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from simulator.layers.quantized_layers import QuantizedConv2D, quantizer

from simulator.comp_unit.PEarray import PEarray
from simulator.comp_unit.tile import tile_PE
from simulator.comp_unit.mac import mac_unit
from simulator.comp_unit.mapping_flow import PE_mapping_forward, PE_mapping_backward

# Test example using TPU-like vecter mac

#%% mapping PE fault to tile - setup

# dataflow pre-plan
# one run of Tile to PE with out tile fault dictionary, only mapping configure.

#%% setup tile
# weight tile and feature map tile
wght_tile=tile_PE((5,5,1,8),is_fmap=False,wl=8)
ifmap_tile=tile_PE((1,28,28,1),is_fmap=True,wl=8)
ofmap_tile=tile_PE((1,28,28,8),is_fmap=True,wl=8)

ofmap_config='../pe_mapping_config/LeNet5_ws_8x8_FPGA/ofmap_config_conv1.json'
wght_config='../pe_mapping_config/LeNet5_ws_8x8_FPGA/wght_config_conv1.json'
ifmap_config='../pe_mapping_config/LeNet5_ws_8x8_FPGA/ifmap_config_conv1.json'

#%% setup MAC unit

PE=mac_unit('../pe_mapping_config/LeNet5_ws_8x8_FPGA/mac_unit_config.json')

#%% setup PEarray

MXU=PEarray(8,8,ofmap_tile=ofmap_tile,wght_tile=wght_tile,ifmap_tile=ifmap_tile,mac_config=PE)

MXU_config='../pe_mapping_config/LeNet5_ws_8x8_FPGA/MXU_config_conv1.json'

#%% pre-plan PE mapping forward with file read-in

PE_mapping_forward(ifmap_tile,wght_tile,ofmap_tile,MXU,ifmap_config,wght_config,ofmap_config,MXU_config,pre_plan=True,verbose=4)

#%% generate PE array fault dictionary

#MXU.gen_PEarray_SA_fault_dict(n_bit=8, fault_type='flip')

# PE_fault_dict=MXU.gen_PEarray_SA_fault_dict(n_bit=8, fault_type='flip', mac_config=True)

# figure out this case
fault_info={'SA_type':'flip','SA_bit':0,'param':'wght_in'}
PE_fault_dict=MXU.gen_PEarray_permanent_fault_dict((3,0), fault_info, mac_config=True)

#fault_info={'SA_type':'flip','SA_bit':6,'param':'wght_out'}
#PE_fault_dict=MXU.gen_PEarray_permanent_fault_dict((9,7), fault_info, mac_config=True)

#fault_info={'SA_type':'flip','SA_bit':3,'param':'wght_in'}
#PE_fault_dict=MXU.gen_PEarray_permanent_fault_dict((0,0), fault_info, mac_config=None)

#%% PE mapping backward

# create test model
input_shape=Input(batch_shape=(1,28,28,1))
x=QuantizedConv2D(filters=16,
                  quantizers=quantizer(8,3),
                  kernel_size=(5,5),
                  padding='same',
                  strides=(1, 1),                              
                  activation='relu',
                  quant_mode='hybrid')(input_shape)
model=Model(inputs=input_shape, outputs=x, name='test_model')

# backward mapping
PE_mac_fault_dict = PE_mapping_backward(model.layers[1], MXU, verbose=4)


#%% PE mapping backward (fault propagation and tile2layer)

# # create test model
# input_shape=Input(batch_shape=(4,56,56,32))
# x=QuantizedConv2D(filters=64,
#                   quantizers=quantizer(8,6),
#                   kernel_size=(3,3),
#                   padding='same',
#                   strides=(1, 1),                              
#                   activation='relu',
#                   quant_mode='hybrid')(input_shape)
# model=Model(inputs=input_shape, outputs=x, name='test_model')

# # backward mapping
# PE_mac_fault_dict = PE_mapping_backward(model.layers[1], MXU, verbose=4)

#%% PE mapping backward (fault propagation, tile2layer and uneven tile cut)

# # create test model
# input_shape=Input(batch_shape=(2,70,70,32))
# x=QuantizedConv2D(filters=64,
#                   quantizers=quantizer(8,6),
#                   kernel_size=(3,3),
#                   padding='same',
#                   strides=(1, 1),                              
#                   activation='relu',
#                   quant_mode='hybrid')(input_shape)
# model=Model(inputs=input_shape, outputs=x, name='test_model')

# # backward mapping
# PE_mac_fault_dict = PE_mapping_backward(model.layers[1], MXU, verbose=4)


