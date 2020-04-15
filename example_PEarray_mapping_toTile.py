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
import json

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
ofmap_tile.expand_reshape_data(orig_prior=[0,3,2,1],
                               expect_shape=(784,32),
                               reshape_prior=[1,0],
                               slicing_dims=(784,16),
                               slices_permute=[1,0],
                               tilting=True, 
                               tilt_axis=1, 
                               tilt_direction=0,
                               dataflow_pre_plan=True)

wght_tile.expand_reshape_data(orig_prior=[3,2,1,0],
                              expect_shape=(144,32),
                              reshape_prior=[1,0],
                              slicing_dims=(16,16),
                              slices_permute=[1,0],
                              dataflow_pre_plan=True)

ifmap_tile.expand_extract_patches(ksizes=(1,3,3,1),
                                  strides=(1,1,1,1),
                                  dilation_rates=(1,1,1,1),
                                  padding='same',
                                  edge_fill=False,
                                  reshape_patches=True,
                                  patches_prior=[0,1,2,3],
                                  expect_shape=(1*28*28,144),
                                  reshape_prior=[0,1],
                                  slicing_dims=(1*28*28,16),
                                  slices_permute=[0,1],
                                  tilting=True, 
                                  tilt_axis=1, 
                                  tilt_direction=0,
                                  dataflow_pre_plan=True)

wght_tile.expand_slice_bias(slice_width=16,
                            dataflow_pre_plan=True)

#%% setup PEarray

MXU=PEarray(16,16,ofmap_tile=ofmap_tile,wght_tile=wght_tile,ifmap_tile=ifmap_tile)

MXU.setup_dataflow(o_permute_info={'PE_required_axes_prior':['t_clk','PE_x'],
                                   'tile_mapping_prior':[2,1,0]}, 
                   o_fixed_info={'PE_fix_axis':'PE_y',
                                 'indice':-1}, 
                   o_broadcast_info=None, 
                   o_streaming_info=None, 
                   o_repeat=9, 
                   o_duplicate=0, 
                   o_pack_size=1,
                   o_stall_latency=17+15,
                   
                   w_permute_info={'PE_required_axes_prior':['t_clk','PE_y','PE_x'],
                                   'tile_mapping_prior':[2,1,0]}, 
                   w_fixed_info=None, 
                   w_broadcast_info=None, 
                   w_streaming_info=None, 
                   w_repeat=799+15, 
                   w_duplicate=0, 
                   w_pack_size=799+15,
                   w_stall_latency=17,
                   
                   i_permute_info={'PE_required_axes_prior':['t_clk','PE_y'],
                                   'tile_mapping_prior':[2,1,0]}, 
                   i_fixed_info=None, 
                   i_broadcast_info=None, 
                   i_streaming_info={'PE_stream_axis':'PE_x',
                                     'tile_direction':'forward',
                                     'PE_direction':'forward'}, 
                   i_repeat=0, 
                   i_duplicate=2, 
                   i_pack_size=1,
                   i_stall_latency=17,
                   
                   p_permute_info={'PE_required_axes_prior':['t_clk','PE_x'],
                                   'tile_mapping_prior':[2,1,0]}, 
                   p_fixed_info=None, 
                   p_broadcast_info=None, 
                   p_streaming_info={'PE_stream_axis':'PE_y',
                                     'tile_direction':'forward',
                                     'PE_direction':'forward'}, 
                   p_repeat=9, 
                   p_duplicate=0, 
                   p_pack_size=1,
                   p_stall_latency=17,
                   
                   b_permute_info={'PE_required_axes_prior':['t_clk','PE_x'],
                                   'tile_mapping_prior':[1,0]}, 
                   b_fixed_info={'PE_fix_axis':'PE_y',
                                 'indice':0}, 
                   b_broadcast_info=None, 
                   b_streaming_info=None, 
                   b_repeat=9, 
                   b_duplicate=0, 
                   b_pack_size=1,
                   b_stall_latency=17)

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

#%% test decomposed slice pack

MXU.fault_dict={(6,15,77):{'SA_type':'flip','SA_bit':3,'param':'ifmap_in'},
                (9,3,833):{'SA_type':'flip','SA_bit':5,'param':'wght_in'},
                (7,12,12664):{'SA_type':'flip','SA_bit':0,'param':'psum_out'},
                (6,6,863):{'SA_type':'flip','SA_bit':5,'param':'psum_out'},
                (4,4,862):{'SA_type':'flip','SA_bit':5,'param':'wght_in'}}

mapped_fault_dict_ifmap,mapped_fault_dict_wght,mapped_fault_dict_ofmap,mapped_fault_dict_bias,mapped_fault_dict_psum = MXU.decompose_slice_pack()

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

