# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 13:47:58 2020

@author: Yung-Yu Tsai

PE array mapping example
This file shows example of mapping from Tile to PEarray
"""

from simulator.comp_unit.PEarray import PEarray
from simulator.comp_unit.tile import tile_PE

import numpy as np
import json, pickle

#%% Test example using TPU-like vecter mac

# weight tile and feature map tile
wght_tile=tile_PE((3,3,16,32),is_fmap=False,wl=8)
ifmap_tile=tile_PE((1,28,28,16),is_fmap=True,wl=8)
ofmap_tile=tile_PE((1,28,28,32),is_fmap=True,wl=8)

wght_tile.fault_dict = {(2, 2, 3, 10): {'SA_type': 'flip', 'SA_bit': 4}}
with open('../pe_mapping_config/fault_dict_wght_tile.pickle', 'rb') as fdfile:
    wght_tile.fault_dict = pickle.load(fdfile)
orig_coors_w=np.array(list(wght_tile.fault_dict.keys()))

ifmap_tile.fault_dict = {(0, 15, 6, 3): {'SA_type': 'flip', 'SA_bit': 7}}
with open('../pe_mapping_config/fault_dict_ifmap_tile.pickle', 'rb') as fdfile:
    ifmap_tile.fault_dict  = pickle.load(fdfile)
orig_coors_if=np.array(list(ifmap_tile.fault_dict.keys()))

ofmap_tile.fault_dict = {(0, 15, 6, 6): {'SA_type': 'flip', 'SA_bit': 7}}
with open('../pe_mapping_config/fault_dict_ofmap_tile.pickle', 'rb') as fdfile:
    ofmap_tile.fault_dict  = pickle.load(fdfile)

wght_tile.use_bias=True
wght_tile.bias_fault_dict={(7,): {'SA_type': 'flip', 'SA_bit': 4}, (28,): {'SA_type': 'flip', 'SA_bit': 5}, (15,): {'SA_type': 'flip', 'SA_bit': 7}}

#%% test reshape index transformation

reshaped_coor=wght_tile.reshape_ravel_idx((2,2,3,10),
                                          source_shape=wght_tile.tile_shape,
                                          source_prior=[0,1,2,3],
                                          target_shape=(144,32),
                                          target_prior=[0,1])

reshaped_coors=wght_tile.reshape_ravel_idx(orig_coors_w,
                                           source_shape=wght_tile.tile_shape,
                                           source_prior=[0,1,2,3],
                                           target_shape=(144,32),
                                           target_prior=[0,1])

#%% test slice permute index

permuted_coor=wght_tile.slice_permute_idx((131,10),
                                          orig_shape=(144,32),
                                          slicing_dims=(16,16),
                                          slices_permute=[0,1])

#permuted_coor=wght_tile.slice_permute_idx((2,2,3,10),
#                                          orig_shape=(3,3,16,32),
#                                          slicing_dims=(0,0,12,12),
#                                          slices_permute=[3,2,0,1])
#
#permuted_coors=wght_tile.slice_permute_idx(orig_coors_w,
#                                           orig_shape=(3,3,16,32),
#                                           slicing_dims=(0,0,12,12),
#                                           slices_permute=[3,2,0,1])

#%% test extract patches index transform

extracted_shape1=ifmap_tile.get_extracted_shape(fmap_shape=(1,28,28,16),
                                                ksizes=(1,3,3,1),
                                                strides=(1,1,1,1),
                                                dilation_rates=(1,1,1,1),
                                                padding='valid',
                                                edge_fill=False)

extracted_shape2=ifmap_tile.get_extracted_shape(fmap_shape=(1,28,28,16),
                                                ksizes=(1,3,3,1),
                                                strides=(1,2,2,1),
                                                dilation_rates=(1,1,1,1),
                                                padding='same',
                                                edge_fill=True)

extracted_coor1=ifmap_tile.extract_patches_idx((0,0,0,2),
                                               fmap_shape=(1,28,28,16),
                                               ksizes=(1,3,3,1),
                                               strides=(1,1,1,1),
                                               dilation_rates=(1,1,1,1),
                                               padding='valid',
                                               edge_fill=False)

extracted_coor2=ifmap_tile.extract_patches_idx((0,9,17,3),
                                               fmap_shape=(1,28,28,16),
                                               ksizes=(1,3,3,1),
                                               strides=(1,2,2,1),
                                               dilation_rates=(1,1,1,1),
                                               padding='same',
                                               edge_fill=True)

extracted_coors1=ifmap_tile.extract_patches_idx(orig_coors_if,
                                                fmap_shape=(1,28,28,16),
                                                ksizes=(1,3,3,1),
                                                strides=(1,1,1,1),
                                                dilation_rates=(1,1,1,1),
                                                padding='valid',
                                                edge_fill=False)

extracted_coors2=ifmap_tile.extract_patches_idx(orig_coors_if,
                                                fmap_shape=(1,28,28,16),
                                                ksizes=(1,3,3,1),
                                                strides=(1,2,2,1),
                                                dilation_rates=(1,1,1,1),
                                                padding='same',
                                                edge_fill=True)

#%% test tile expand data

# expand reshape and slice/permute

expand_fault_dict_reshape_o=ofmap_tile.expand_reshape_data(orig_prior=[3,0,1,2],
                                                           expect_shape=(784,32),
                                                           reshape_prior=[0,1],
                                                           slicing_dims=(784,16),
                                                           slices_permute=[0,1],
                                                           tilting=True, 
                                                           tilt_axis=1, 
                                                           tilt_direction=0)

expand_fault_dict_reshape_w=wght_tile.expand_reshape_data(orig_prior=[0,1,2,3],
                                                          expect_shape=(144,32),
                                                          reshape_prior=[0,1],
                                                          slicing_dims=(16,16),
                                                          slices_permute=[0,1])

expand_fault_dict_extracted=ifmap_tile.expand_extract_patches(ksizes=(1,3,3,1),
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
                                                              tilt_direction=0)

expand_bias_fault_dict_slice=wght_tile.expand_slice_bias(bias_slice_width=16)

#%% test PE array mapping

MXU=PEarray(16,16,ofmap_tile=ofmap_tile,wght_tile=wght_tile,ifmap_tile=ifmap_tile)

#%% test streaming mapping

stream_coors_in=np.array([[87,13],
                          [194,6],
                          [17,9]])
stream_coors_out_f=MXU.stream_capture_idx(stream_coors_in,
                                          data_shape=(784,16), 
                                          data_stream_axis=0,  
                                          window_shape=(28,28,784), 
                                          window_stream_axis=1)

stream_coors_out_b=MXU.stream_capture_idx(stream_coors_in,
                                          data_shape=(784,16), 
                                          data_stream_axis=0,  
                                          window_shape=(28,28,784), 
                                          window_stream_axis=1,
                                          data_flow_direction='backward', 
                                          window_flow_direction='backward')

#stream_coors_in=np.array([[7,137],
#                          [9,664],
#                          [1,98]])
#stream_coors_out_f=MXU.stream_capture_idx(stream_coors_in,
#                                          data_shape=(16,784), 
#                                          data_stream_axis=1,  
#                                          window_shape=(28,28,784), 
#                                          window_stream_axis=0)

#%% test broadcast array mapping

broadcast_coors_in0=np.array([[15,188],
                              [3,174],
                              [6,339]])
broadcast_coors_out0=MXU.broadcast_idx(broadcast_coors_in0,
                                       data_shape=(16,691), 
                                       target_shape=(16,16,691), 
                                       broadcast_dims=0)
broadcast_coors_in1=np.array([[188],
                              [174],
                              [339]])
broadcast_coors_out1=MXU.broadcast_idx(broadcast_coors_in1,
                                       data_shape=(691,), 
                                       target_shape=(16,16,691), 
                                       broadcast_dims=[0,1])

#%% test fixed array mapping

fixed_coors_in0=np.array([[14,178],
                          [0,199],
                          [11,449]])
fixed_coors_out0=MXU.fixed_idx(fixed_coors_in0,
                               indice_fix=-1,
                               fix_dims=0, 
                               target_shape=(16,16,691))
fixed_coors_in1=np.array([[178],
                          [199],
                          [449]])
fixed_coors_out1=MXU.fixed_idx(fixed_coors_in1,
                               indice_fix=[-1,0],
                               fix_dims=[0,1], 
                               target_shape=(16,16,691))

#%% setup mapping configuration

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
                   i_stall_latency=17)


#%% pre-mapping tiles to PE

# ofmap pre-mapping
mapped_fault_dict_ofmap=MXU.premapping_tile('ofmap')
# weight pre-mapping
mapped_fault_dict_wght=MXU.premapping_tile('wght')
# ifmap pre-mappingde
mapped_fault_dict_ifmap=MXU.premapping_tile('ifmap')

#%% duplicate mapping for data reuse or accumulate partial sum

# ofmap duplicate mapping
duped_fault_dict_ofmap=MXU.duplicate_mapping('ofmap')
# weight duplicate mapping
duped_fault_dict_wght=MXU.duplicate_mapping('wght')
# ifmap duplicate mapping
duped_fault_dict_ifmap=MXU.duplicate_mapping('ifmap')

#%% align clock cycle and make final mapping fault dictionary

PE_fault_dict_final=MXU.align_slice_pack()

#%% assign psum and bias mapping configuration

MXU.clear_fd()
MXU.clear_map_config()

MXU.setup_dataflow(p_permute_info={'PE_required_axes_prior':['PE_x','t_clk'],
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

#%% read in mapping configuration

with open('config_PEarray.json', 'r') as config_file:
    PEconfig=json.load(config_file)

MXU.setup_dataflow(** PEconfig)


