# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 11:18:30 2020

@author: Yung-Yu Tsai

PE array mapping example
This file shows example of mapping from Tile to PEarray
"""

from simulator.comp_unit.PEarray import PEarray
from simulator.comp_unit.tile import tile_PE

import numpy as np

# Test example using TPU-like vecter mac

#%% mapping PE fault to tile - setup

# dataflow pre-plan
# one run of Tile to PE with out tile fault dictionary, only mapping configure.

#%% setup tile
# weight tile and feature map tile
wght_tile=tile_PE((3,3,16,32),is_fmap=False,wl=8)
ifmap_tile=tile_PE((1,28,28,16),is_fmap=True,wl=8)
ofmap_tile=tile_PE((1,28,28,32),is_fmap=True,wl=8)

# pre-plan tile reshape and slice
ofmap_tile.expand_reshape_data(orig_prior=[3,0,1,2],
                               expect_shape=(784,32),
                               reshape_prior=[0,1],
                               slicing_dims=(784,16),
                               slices_permute=[0,1],
                               tilting=True, 
                               tilt_axis=1, 
                               tilt_direction=0,
                               dataflow_pre_plan=True)

wght_tile.expand_reshape_data(orig_prior=[0,1,2,3],
                              expect_shape=(144,32),
                              reshape_prior=[0,1],
                              slicing_dims=(16,16),
                              slices_permute=[0,1],
                              dataflow_pre_plan=True)

ifmap_tile.expand_extract_patches(ksizes=(1,3,3,1),
                                  strides=(1,1,1,1),
                                  dilation_rates=(1,1,1,1),
                                  padding='same',
                                  edge_fill=False,
                                  patches_unravel=[0,1,2],
                                  reshape_patches=True,
                                  patches_prior=[3,1,2,0],
                                  expect_shape=(1*28*28,144),
                                  reshape_prior=[1,0],
                                  slicing_dims=(1*28*28,16),
                                  slices_permute=[1,0],
                                  tilting=True, 
                                  tilt_axis=1, 
                                  tilt_direction=0,
                                  dataflow_pre_plan=True)

wght_tile.expand_slice_bias(bias_slice_width=16,
                            dataflow_pre_plan=True)

#%% setup PEarray

MXU=PEarray(16,16,ofmap_tile=ofmap_tile,wght_tile=wght_tile,ifmap_tile=ifmap_tile)

MXU.setup_dataflow(o_permute_info={'PE_required_axes_prior':['PE_x','t_clk'],
                                   'tile_mapping_prior':[0,1,2]}, 
                   o_fixed_info={'PE_fix_axis':'PE_y',
                                 'indice':-1}, 
                   o_broadcast_info=None, 
                   o_streaming_info=None, 
                   o_repeat=9, 
                   o_duplicate=0, 
                   o_pack_size=1,
                   o_stall_latency=17+15,
                   
                   w_permute_info={'PE_required_axes_prior':['PE_x','PE_y','t_clk'],
                                   'tile_mapping_prior':[0,1,2]}, 
                   w_fixed_info=None, 
                   w_broadcast_info=None, 
                   w_streaming_info=None, 
                   w_repeat=799+15, 
                   w_duplicate=0, 
                   w_pack_size=799+15,
                   w_stall_latency=17,
                   
                   i_permute_info={'PE_required_axes_prior':['PE_y','t_clk'],
                                   'tile_mapping_prior':[0,1,2]}, 
                   i_fixed_info=None, 
                   i_broadcast_info=None, 
                   i_streaming_info={'PE_stream_axis':'PE_x',
                                     'tile_direction':'forward',
                                     'PE_direction':'forward'}, 
                   i_repeat=0, 
                   i_duplicate=2, 
                   i_pack_size=1,
                   i_stall_latency=17,
                   
                   p_permute_info={'PE_required_axes_prior':['PE_x','t_clk'],
                                   'tile_mapping_prior':[0,1,2]}, 
                   p_fixed_info=None, 
                   p_broadcast_info=None, 
                   p_streaming_info={'PE_stream_axis':'PE_y',
                                     'tile_direction':'forward',
                                     'PE_direction':'forward'}, 
                   p_repeat=9, 
                   p_duplicate=0, 
                   p_pack_size=1,
                   p_stall_latency=17,
                   
                   b_permute_info={'PE_required_axes_prior':['PE_x','t_clk'],
                                   'tile_mapping_prior':[0,1]}, 
                   b_fixed_info={'PE_fix_axis':'PE_y',
                                 'indice':0}, 
                   b_broadcast_info=None, 
                   b_streaming_info=None, 
                   b_repeat=799+15, 
                   b_duplicate=0, 
                   b_pack_size=799+15,
                   b_stall_latency=17,
                   b_dummy_pack_insert='post_each',
                   b_dummy_pack_n=8)


#%% pre-plan dataflow

# pre-plan premapping
MXU.premapping_tile('ofmap', dataflow_pre_plan=True)
MXU.premapping_tile('wght', dataflow_pre_plan=True)
MXU.premapping_tile('ifmap', dataflow_pre_plan=True)
MXU.premapping_tile('bias', dataflow_pre_plan=True)
MXU.premapping_tile('psum', dataflow_pre_plan=True)
# pre-plan duplication
MXU.duplicate_mapping('ofmap', dataflow_pre_plan=True)
MXU.duplicate_mapping('wght', dataflow_pre_plan=True)
MXU.duplicate_mapping('ifmap', dataflow_pre_plan=True)
MXU.duplicate_mapping('bias', dataflow_pre_plan=True)
MXU.duplicate_mapping('psum', dataflow_pre_plan=True)
# pre-plan alignment
MXU.align_slice_pack(dataflow_pre_plan=True)

#%% generate PE array fault dictionary

#MXU.fault_dict={(6,15,77):{'SA_type':'flip','SA_bit':3,'param':'ifmap_in'},
#                (9,3,833):{'SA_type':'flip','SA_bit':5,'param':'wght_in'},
#                (7,12,12664):{'SA_type':'flip','SA_bit':0,'param':'psum_out'},
#                (6,6,863):{'SA_type':'flip','SA_bit':5,'param':'psum_out'},
#                (4,4,862):{'SA_type':'flip','SA_bit':5,'param':'wght_in'},
#                (15,4,666):{'SA_type':'flip','SA_bit':2,'param':'psum_out'},
#                (13,6,11111):{'SA_type':'flip','SA_bit':6,'param':'psum_in'},
#                (3,13,777):{'SA_type':'flip','SA_bit':7,'param':'ifmap_out'},
#                (9,4,8766):{'SA_type':'flip','SA_bit':2,'param':'wght_out'},
#                (0,0,444):{'SA_type':'flip','SA_bit':1,'param':'psum_in'},
#                (15,15,8787):{'SA_type':'flip','SA_bit':3,'param':'ifmap_out'},
#                (13,3,8553):{'SA_type':'flip','SA_bit':7,'param':'ifmap_in'},# ans (0,15,6,3)
#                (3,10,2444):{'SA_type':'flip','SA_bit':4,'param':'wght_in'},# ans (2,2,3,10)
#                (15,6,5207):{'SA_type':'flip','SA_bit':7,'param':'psum_out'}}# ans (0,15,6,6)
#
#MXU.fault_dict=MXU.assign_id(MXU.fault_dict)
#PE_fault_dict=MXU.fault_dict

#MXU.fault_dict=MXU.neighbor_io_fault_dict_coors(MXU.fault_dict)
#PE_fault_dict_neighbor=MXU.fault_dict

MXU.gen_PEarray_SA_fault_dict(n_bit=8, fault_type='flip')
PE_fault_dict=MXU.fault_dict

#MXU.gen_PEarray_transient_fault_dict(n_bit=8, fault_num=20, fault_type='flip')
#PE_fault_dict=MXU.fault_dict

#%% test decomposed slice pack

mapped_fault_dict_ifmap,mapped_fault_dict_wght,mapped_fault_dict_ofmap,mapped_fault_dict_bias,mapped_fault_dict_psum \
= MXU.decompose_slice_pack()

#%% reduce mapping from duplicate data reuse or accumulate partial sum

# ofmap reduce mapping
reduced_fault_dict_ofmap=MXU.reduce_mapping('ofmap')
# weight reduce mapping
reduced_fault_dict_wght=MXU.reduce_mapping('wght')
# ifmap reduce mapping
reduced_fault_dict_ifmap=MXU.reduce_mapping('ifmap')
# bias reduce mapping
reduced_fault_dict_bias=MXU.reduce_mapping('bias')
# psum reduce mapping
reduced_fault_dict_psum=MXU.reduce_mapping('psum')

#%% test stream flowback 

flowback_coors_in=np.array([[13,7,94],
                            [6,17,211],
                            [9,25,42]])
flowback_coors_out_f=MXU.stream_flowback_idx(flowback_coors_in,
                                             data_shape=(784,16), 
                                             data_stream_axis=0,  
                                             window_shape=(28,28,784), 
                                             window_stream_axis=1)

flowback_coors_out_b=MXU.stream_flowback_idx(flowback_coors_in,
                                             data_shape=(784,16), 
                                             data_stream_axis=0,  
                                             window_shape=(28,28,784), 
                                             window_stream_axis=1,
                                             data_flow_direction='backward', 
                                             window_flow_direction='backward')

#flowback_coors_in=np.array([[7,13,94],
#                          [17,6,211],
#                          [25,9,42]])
#stream_coors_out_f=MXU.stream_flowback_idx(flowback_coors_in,
#                                           data_shape=(16,784), 
#                                           data_stream_axis=1,  
#                                           window_shape=(28,28,784), 
#                                           window_stream_axis=0)

#%% test retract broadcast

narrowcast_coors_in0=np.array([[7,15,188],
                               [13,3,174],
                               [2,6,339]])
narrowcast_coors_out0=MXU.narrowcast_idx(narrowcast_coors_in0,
                                         data_shape=(16,691), 
                                         target_shape=(16,16,691), 
                                         broadcast_dims=0)
narrowcast_coors_in1=np.array([[9,10,188],
                               [0,3,174],
                               [14,7,339]])
narrowcast_coors_out1=MXU.narrowcast_idx(narrowcast_coors_in1,
                                         data_shape=(691,), 
                                         target_shape=(16,16,691), 
                                         broadcast_dims=[0,1])

#%% test retract fix

unfix_coors_in0=np.array([[-1,14,178],
                          [-1,0,199],
                          [-1,11,449]])
unfix_coors_out0=MXU.unfix_idx(unfix_coors_in0,
                               indice_fix=-1,
                               fix_dims=0, 
                               target_shape=(16,16,691))

unfix_coors_in1=np.array([[-1,0,178],
                          [-1,0,199],
                          [-1,0,449]])
unfix_coors_out1=MXU.unfix_idx(unfix_coors_in1,
                               indice_fix=[-1,0],
                               fix_dims=[0,1], 
                               target_shape=(16,16,691))

unfix_coors_in2=np.array([[15,14,178],
                          [7,0,199],
                          [15,11,449]])
unfix_coors_out2,condidxfix=MXU.unfix_idx(unfix_coors_in2,
                                          indice_fix=-1,
                                          fix_dims=0, 
                                          target_shape=(16,16,691),
                                          get_cond_idx=True)

#%% demapping return fault dictionary to Tile

# ofmap reduce mapping
demapped_fault_dict_ofmap=MXU.demapping_tile('ofmap')
# weight reduce mapping
demapped_fault_dict_wght=MXU.demapping_tile('wght')
# ifmap reduce mapping
demapped_fault_dict_ifmap=MXU.demapping_tile('ifmap')
# bias reduce mapping
demapped_fault_dict_bias=MXU.demapping_tile('bias')
# psum reduce mapping
demapped_fault_dict_psum=MXU.demapping_tile('psum')

#%% test assemble slices 

assembled_coor=wght_tile.assemble_slice_idx((3,10,48),
                                            orig_shape=(3,3,16,32),
                                            slicing_dims=(0,0,12,12),
                                            slices_permute=[0,1,3,2])

sliced_coors=np.array([[3,10,48],
                       [1,4,11],
                       [3,3,39],
                       [5,9,18],
                       [3,1,48],
                       [9,0,34],
                       [3,6,36],
                       [11,5,20],
                       [7,6,46],
                       [6,3,8]])
assembled_coors=wght_tile.assemble_slice_idx(sliced_coors,
                                             orig_shape=(3,3,16,32),
                                             slicing_dims=(0,0,12,12),
                                             slices_permute=[0,1,3,2])

#%% test return patches

returned_coor1=ifmap_tile.return_patches_idx((0,0,0,18),
                                             fmap_shape=(1,28,28,16),
                                             ksizes=(1,3,3,1),
                                             strides=(1,1,1,1),
                                             dilation_rates=(1,1,1,1),
                                             padding='valid',
                                             edge_fill=False)

returned_coor2=ifmap_tile.return_patches_idx((0,4,8,31),
                                             fmap_shape=(1,28,28,16),
                                             ksizes=(1,3,3,1),
                                             strides=(1,2,2,1),
                                             dilation_rates=(1,1,1,1),
                                             padding='same',
                                             edge_fill=True)

extracted_coors1=np.array([[  0,  24,   2,   9],
                           [  0,   0,   5,  47],
                           [  0,  25,  14,   0],
                           [  0,   9,  19, 134],
                           [  0,  13,  12,  92],
                           [  0,  12,  14,  93],
                           [  0,   4,   1, 131],
                           [  0,   0,  24, 130],
                           [  0,  13,  21,  48],
                           [  0,  24,  11, 130]])
returned_coors1=ifmap_tile.return_patches_idx(extracted_coors1,
                                              fmap_shape=(1,28,28,16),
                                              ksizes=(1,3,3,1),
                                              strides=(1,1,1,1),
                                              dilation_rates=(1,1,1,1),
                                              padding='valid',
                                              edge_fill=False)

extracted_coors2=np.array([[  0,   5,   7,  29],
                           [  0,   9,   8,  64],
                           [  0,   3,   2, 100],
                           [  0,   0,  10, 134],
                           [  0,   5,   2,  96],
                           [  0,   0,   0,  53],
                           [  0,  12,   5, 131],
                           [  0,   6,  12,  64],
                           [  0,   3,   4,  23],
                           [  0,   5,   4,  75]])
returned_coors2=ifmap_tile.return_patches_idx(extracted_coors2,
                                              fmap_shape=(1,28,28,16),
                                              ksizes=(1,3,3,1),
                                              strides=(1,2,2,1),
                                              dilation_rates=(1,1,1,1),
                                              padding='same',
                                              edge_fill=True)

#%% test shrink tile back to original shape

shrink_fault_dict_reshape_o=ofmap_tile.shrink_reshape_data()
shrink_fault_dict_reshape_p=ofmap_tile.shrink_reshape_data(psum=True)

shrink_fault_dict_reshape_w=wght_tile.shrink_reshape_data()
shrink_fault_dict_bias=wght_tile.shrink_slice_bias()

# shrink return patches
shrink_fault_dict_returned_i=ifmap_tile.shrink_return_patches()

#%%
"""
io_data_solver example
The demonstration of how the io_data_solver work with tile2layer
"""
from simulator.comp_unit.tile import io_data_solver

from keras.models import Model
from keras.layers import Input
from simulator.layers.quantized_layers import QuantizedConv2D, quantizer

#%% test io_data_solver

# organize fault dict and give partial sum index
solver=io_data_solver(ofmap_tile,wght_tile,ifmap_tile)
fault_dict_solved=solver.solve_correspond_io(print_detail=True)

# create test model
input_shape=Input(batch_shape=(4,56,56,32))
x=QuantizedConv2D(filters=64,
                  quantizers=quantizer(8,6),
                  kernel_size=(3,3),
                  padding='same',
                  strides=(1, 1),                              
                  activation='relu',
                  quant_mode='hybrid')(input_shape)
model=Model(inputs=input_shape, outputs=x, name='test_model')

# transform fault dictionary from tile to layer
fault_dict_layer=solver.tile2layer(fault_dict_solved, based_tile='ofmap', layer=model.layers[1])


