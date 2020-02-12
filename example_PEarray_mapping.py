# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 13:47:58 2020

@author: Yung-Yu Tsai

PE array mapping example
"""

from simulator.comp_unit.PEarray import PEarray
from simulator.comp_unit.tile import tile_PE

import numpy as np

#%% 
# Test example using TPU-like vecter mac

# weight tile and feature map tile
wght_tile=tile_PE((3,3,16,32),is_fmap=False,wl=8,required_axes=['PE_x','PE_y','t_clk'])
ifmap_tile=tile_PE((1,28,28,16),is_fmap=True,wl=8,required_axes=['PE_y','t_clk'])
ofmap_tile=tile_PE((1,28,28,32),is_fmap=True,wl=8,required_axes=['PE_x','t_clk'])

wght_tile.fault_dict = {(2, 2, 3, 10): {'SA_type': 'flip', 'SA_bit': 4}, (0, 1, 13, 28): {'SA_type': 'flip', 'SA_bit': 5}, (2, 0, 15, 15): {'SA_type': 'flip', 'SA_bit': 4}, (1, 0, 5, 9): {'SA_type': 'flip', 'SA_bit': 5}, (2, 2, 3, 1): {'SA_type': 'flip', 'SA_bit': 7}, (1, 2, 9, 24): {'SA_type': 'flip', 'SA_bit': 6}, (2, 0, 3, 6): {'SA_type': 'flip', 'SA_bit': 4}, (1, 0, 11, 17): {'SA_type': 'flip', 'SA_bit': 4}, (2, 1, 7, 30): {'SA_type': 'flip', 'SA_bit': 3}, (0, 1, 6, 15): {'SA_type': 'flip', 'SA_bit': 4}, (2, 2, 11, 18): {'SA_type': 'flip', 'SA_bit': 5}, (0, 1, 5, 18): {'SA_type': 'flip', 'SA_bit': 0}, (2, 1, 1, 20): {'SA_type': 'flip', 'SA_bit': 6}, (0, 0, 6, 24): {'SA_type': 'flip', 'SA_bit': 6}, (1, 2, 4, 5): {'SA_type': 'flip', 'SA_bit': 5}, (0, 2, 7, 19): {'SA_type': 'flip', 'SA_bit': 3}, (0, 0, 12, 30): {'SA_type': 'flip', 'SA_bit': 0}, (1, 1, 13, 1): {'SA_type': 'flip', 'SA_bit': 4}, (2, 2, 13, 8): {'SA_type': 'flip', 'SA_bit': 3}, (0, 1, 0, 25): {'SA_type': 'flip', 'SA_bit': 6}, (0, 0, 10, 26): {'SA_type': 'flip', 'SA_bit': 0}, (0, 1, 10, 4): {'SA_type': 'flip', 'SA_bit': 4}, (0, 1, 11, 0): {'SA_type': 'flip', 'SA_bit': 7}, (2, 0, 3, 25): {'SA_type': 'flip', 'SA_bit': 1}, (2, 0, 12, 6): {'SA_type': 'flip', 'SA_bit': 7}, (0, 2, 5, 29): {'SA_type': 'flip', 'SA_bit': 7}, (0, 2, 14, 26): {'SA_type': 'flip', 'SA_bit': 7}, (2, 2, 6, 26): {'SA_type': 'flip', 'SA_bit': 3}, (2, 1, 15, 24): {'SA_type': 'flip', 'SA_bit': 1}, (0, 1, 11, 6): {'SA_type': 'flip', 'SA_bit': 2}, (0, 1, 13, 2): {'SA_type': 'flip', 'SA_bit': 1}, (1, 2, 8, 0): {'SA_type': 'flip', 'SA_bit': 4}, (2, 2, 6, 28): {'SA_type': 'flip', 'SA_bit': 2}, (2, 1, 11, 17): {'SA_type': 'flip', 'SA_bit': 2}, (1, 2, 12, 14): {'SA_type': 'flip', 'SA_bit': 2}, (1, 1, 9, 29): {'SA_type': 'flip', 'SA_bit': 7}}
orig_coors_w=np.array(list(wght_tile.fault_dict.keys()))
ifmap_tile.fault_dict = {(0, 14, 11, 27): {'SA_type': 'flip', 'SA_bit': 5}, (0, 11, 10, 16): {'SA_type': 'flip', 'SA_bit': 0}, (0, 11, 23, 0): {'SA_type': 'flip', 'SA_bit': 0}, (0, 23, 20, 26): {'SA_type': 'flip', 'SA_bit': 2}, (0, 6, 7, 23): {'SA_type': 'flip', 'SA_bit': 7}, (0, 9, 23, 1): {'SA_type': 'flip', 'SA_bit': 2}, (0, 3, 9, 7): {'SA_type': 'flip', 'SA_bit': 1}, (0, 24, 12, 25): {'SA_type': 'flip', 'SA_bit': 6}, (0, 4, 6, 6): {'SA_type': 'flip', 'SA_bit': 7}, (0, 16, 16, 10): {'SA_type': 'flip', 'SA_bit': 0}, (0, 7, 18, 8): {'SA_type': 'flip', 'SA_bit': 0}, (0, 5, 20, 20): {'SA_type': 'flip', 'SA_bit': 0}, (0, 1, 18, 11): {'SA_type': 'flip', 'SA_bit': 6}, (0, 8, 16, 20): {'SA_type': 'flip', 'SA_bit': 1}, (0, 26, 20, 19): {'SA_type': 'flip', 'SA_bit': 0}, (0, 9, 13, 15): {'SA_type': 'flip', 'SA_bit': 0}, (0, 7, 17, 30): {'SA_type': 'flip', 'SA_bit': 1}, (0, 3, 20, 30): {'SA_type': 'flip', 'SA_bit': 5}, (0, 0, 18, 6): {'SA_type': 'flip', 'SA_bit': 6}, (0, 26, 23, 29): {'SA_type': 'flip', 'SA_bit': 2}, (0, 24, 19, 8): {'SA_type': 'flip', 'SA_bit': 0}, (0, 26, 13, 24): {'SA_type': 'flip', 'SA_bit': 0}, (0, 17, 7, 7): {'SA_type': 'flip', 'SA_bit': 4}, (0, 14, 2, 30): {'SA_type': 'flip', 'SA_bit': 6}, (0, 3, 2, 15): {'SA_type': 'flip', 'SA_bit': 0}, (0, 3, 11, 26): {'SA_type': 'flip', 'SA_bit': 3}, (0, 5, 16, 30): {'SA_type': 'flip', 'SA_bit': 6}, (0, 6, 1, 5): {'SA_type': 'flip', 'SA_bit': 0}, (0, 9, 18, 6): {'SA_type': 'flip', 'SA_bit': 1}, (0, 25, 24, 6): {'SA_type': 'flip', 'SA_bit': 4}, (0, 21, 8, 23): {'SA_type': 'flip', 'SA_bit': 0}, (0, 5, 15, 8): {'SA_type': 'flip', 'SA_bit': 5}, (0, 22, 2, 4): {'SA_type': 'flip', 'SA_bit': 2}, (0, 14, 16, 27): {'SA_type': 'flip', 'SA_bit': 1}, (0, 0, 5, 31): {'SA_type': 'flip', 'SA_bit': 1}, (0, 22, 24, 29): {'SA_type': 'flip', 'SA_bit': 3}, (0, 25, 0, 20): {'SA_type': 'flip', 'SA_bit': 3}, (0, 11, 23, 10): {'SA_type': 'flip', 'SA_bit': 0}, (0, 17, 2, 8): {'SA_type': 'flip', 'SA_bit': 2}, (0, 0, 4, 7): {'SA_type': 'flip', 'SA_bit': 3}, (0, 21, 26, 18): {'SA_type': 'flip', 'SA_bit': 2}, (0, 13, 16, 2): {'SA_type': 'flip', 'SA_bit': 1}, (0, 14, 11, 31): {'SA_type': 'flip', 'SA_bit': 1}, (0, 19, 24, 13): {'SA_type': 'flip', 'SA_bit': 2}, (0, 1, 25, 6): {'SA_type': 'flip', 'SA_bit': 4}, (0, 10, 22, 16): {'SA_type': 'flip', 'SA_bit': 5}, (0, 12, 19, 10): {'SA_type': 'flip', 'SA_bit': 3}, (0, 4, 9, 17): {'SA_type': 'flip', 'SA_bit': 6}, (0, 9, 27, 13): {'SA_type': 'flip', 'SA_bit': 4}, (0, 13, 18, 13): {'SA_type': 'flip', 'SA_bit': 5}, (0, 3, 25, 1): {'SA_type': 'flip', 'SA_bit': 0}, (0, 17, 10, 2): {'SA_type': 'flip', 'SA_bit': 1}, (0, 15, 7, 29): {'SA_type': 'flip', 'SA_bit': 6}, (0, 19, 1, 23): {'SA_type': 'flip', 'SA_bit': 5}, (0, 13, 20, 27): {'SA_type': 'flip', 'SA_bit': 2}, (0, 0, 20, 21): {'SA_type': 'flip', 'SA_bit': 0}, (0, 1, 5, 0): {'SA_type': 'flip', 'SA_bit': 1}, (0, 18, 15, 27): {'SA_type': 'flip', 'SA_bit': 1}, (0, 17, 25, 17): {'SA_type': 'flip', 'SA_bit': 6}, (0, 16, 8, 27): {'SA_type': 'flip', 'SA_bit': 7}, (0, 11, 5, 8): {'SA_type': 'flip', 'SA_bit': 6}, (0, 3, 26, 31): {'SA_type': 'flip', 'SA_bit': 2}, (0, 4, 2, 23): {'SA_type': 'flip', 'SA_bit': 5}, (0, 3, 8, 16): {'SA_type': 'flip', 'SA_bit': 6}, (0, 2, 6, 25): {'SA_type': 'flip', 'SA_bit': 6}, (0, 11, 5, 6): {'SA_type': 'flip', 'SA_bit': 1}, (0, 10, 15, 31): {'SA_type': 'flip', 'SA_bit': 4}, (0, 27, 9, 4): {'SA_type': 'flip', 'SA_bit': 6}, (0, 20, 6, 23): {'SA_type': 'flip', 'SA_bit': 2}, (0, 12, 24, 31): {'SA_type': 'flip', 'SA_bit': 1}, (0, 13, 2, 16): {'SA_type': 'flip', 'SA_bit': 0}, (0, 14, 16, 19): {'SA_type': 'flip', 'SA_bit': 7}, (0, 2, 0, 13): {'SA_type': 'flip', 'SA_bit': 2}, (0, 24, 2, 12): {'SA_type': 'flip', 'SA_bit': 1}, (0, 25, 1, 20): {'SA_type': 'flip', 'SA_bit': 3}, (0, 17, 24, 16): {'SA_type': 'flip', 'SA_bit': 0}, (0, 22, 20, 25): {'SA_type': 'flip', 'SA_bit': 4}, (0, 12, 17, 19): {'SA_type': 'flip', 'SA_bit': 6}, (0, 15, 4, 28): {'SA_type': 'flip', 'SA_bit': 3}, (0, 20, 9, 0): {'SA_type': 'flip', 'SA_bit': 0}, (0, 5, 11, 19): {'SA_type': 'flip', 'SA_bit': 4}, (0, 24, 10, 23): {'SA_type': 'flip', 'SA_bit': 1}, (0, 23, 16, 16): {'SA_type': 'flip', 'SA_bit': 7}, (0, 18, 25, 16): {'SA_type': 'flip', 'SA_bit': 3}, (0, 27, 6, 12): {'SA_type': 'flip', 'SA_bit': 4}, (0, 7, 5, 12): {'SA_type': 'flip', 'SA_bit': 3}, (0, 8, 14, 3): {'SA_type': 'flip', 'SA_bit': 7}, (0, 21, 6, 16): {'SA_type': 'flip', 'SA_bit': 5}, (0, 0, 8, 15): {'SA_type': 'flip', 'SA_bit': 2}, (0, 27, 27, 10): {'SA_type': 'flip', 'SA_bit': 5}, (0, 6, 17, 6): {'SA_type': 'flip', 'SA_bit': 7}, (0, 6, 3, 12): {'SA_type': 'flip', 'SA_bit': 7}, (0, 0, 20, 14): {'SA_type': 'flip', 'SA_bit': 1}, (0, 2, 1, 10): {'SA_type': 'flip', 'SA_bit': 7}, (0, 17, 7, 13): {'SA_type': 'flip', 'SA_bit': 5}, (0, 15, 18, 29): {'SA_type': 'flip', 'SA_bit': 1}, (0, 0, 4, 21): {'SA_type': 'flip', 'SA_bit': 4}, (0, 27, 15, 18): {'SA_type': 'flip', 'SA_bit': 5}, (0, 8, 12, 22): {'SA_type': 'flip', 'SA_bit': 7}, (0, 1, 9, 0): {'SA_type': 'flip', 'SA_bit': 2}, (0, 1, 25, 29): {'SA_type': 'flip', 'SA_bit': 7}, (0, 27, 1, 28): {'SA_type': 'flip', 'SA_bit': 5}, (0, 3, 1, 23): {'SA_type': 'flip', 'SA_bit': 3}, (0, 25, 27, 24): {'SA_type': 'flip', 'SA_bit': 4}, (0, 9, 11, 19): {'SA_type': 'flip', 'SA_bit': 7}, (0, 4, 4, 26): {'SA_type': 'flip', 'SA_bit': 4}, (0, 13, 3, 14): {'SA_type': 'flip', 'SA_bit': 7}, (0, 18, 23, 29): {'SA_type': 'flip', 'SA_bit': 7}, (0, 21, 7, 18): {'SA_type': 'flip', 'SA_bit': 1}, (0, 27, 18, 22): {'SA_type': 'flip', 'SA_bit': 6}, (0, 24, 20, 27): {'SA_type': 'flip', 'SA_bit': 2}, (0, 17, 11, 11): {'SA_type': 'flip', 'SA_bit': 3}, (0, 16, 10, 5): {'SA_type': 'flip', 'SA_bit': 4}, (0, 3, 27, 4): {'SA_type': 'flip', 'SA_bit': 5}, (0, 7, 4, 26): {'SA_type': 'flip', 'SA_bit': 5}, (0, 11, 8, 5): {'SA_type': 'flip', 'SA_bit': 1}, (0, 4, 22, 22): {'SA_type': 'flip', 'SA_bit': 6}, (0, 18, 14, 17): {'SA_type': 'flip', 'SA_bit': 5}, (0, 23, 3, 30): {'SA_type': 'flip', 'SA_bit': 6}, (0, 2, 15, 10): {'SA_type': 'flip', 'SA_bit': 1}, (0, 14, 17, 16): {'SA_type': 'flip', 'SA_bit': 3}, (0, 13, 20, 28): {'SA_type': 'flip', 'SA_bit': 0}, (0, 4, 18, 10): {'SA_type': 'flip', 'SA_bit': 3}, (0, 19, 4, 2): {'SA_type': 'flip', 'SA_bit': 6}, (0, 7, 26, 6): {'SA_type': 'flip', 'SA_bit': 0}, (0, 6, 20, 29): {'SA_type': 'flip', 'SA_bit': 7}, (0, 12, 10, 28): {'SA_type': 'flip', 'SA_bit': 6}, (0, 5, 17, 9): {'SA_type': 'flip', 'SA_bit': 4}, (0, 25, 0, 23): {'SA_type': 'flip', 'SA_bit': 0}, (0, 22, 19, 13): {'SA_type': 'flip', 'SA_bit': 7}, (0, 9, 14, 23): {'SA_type': 'flip', 'SA_bit': 1}, (0, 0, 16, 9): {'SA_type': 'flip', 'SA_bit': 6}, (0, 2, 12, 22): {'SA_type': 'flip', 'SA_bit': 5}, (0, 5, 8, 25): {'SA_type': 'flip', 'SA_bit': 6}, (0, 15, 24, 16): {'SA_type': 'flip', 'SA_bit': 5}, (0, 12, 23, 31): {'SA_type': 'flip', 'SA_bit': 3}, (0, 18, 19, 18): {'SA_type': 'flip', 'SA_bit': 6}, (0, 2, 9, 6): {'SA_type': 'flip', 'SA_bit': 1}, (0, 3, 15, 6): {'SA_type': 'flip', 'SA_bit': 0}, (0, 11, 22, 7): {'SA_type': 'flip', 'SA_bit': 4}, (0, 3, 12, 24): {'SA_type': 'flip', 'SA_bit': 1}, (0, 21, 14, 9): {'SA_type': 'flip', 'SA_bit': 7}, (0, 18, 0, 9): {'SA_type': 'flip', 'SA_bit': 2}, (0, 27, 2, 27): {'SA_type': 'flip', 'SA_bit': 1}, (0, 13, 6, 29): {'SA_type': 'flip', 'SA_bit': 5}, (0, 16, 27, 4): {'SA_type': 'flip', 'SA_bit': 7}, (0, 2, 14, 29): {'SA_type': 'flip', 'SA_bit': 4}, (0, 14, 25, 5): {'SA_type': 'flip', 'SA_bit': 3}, (0, 24, 14, 18): {'SA_type': 'flip', 'SA_bit': 0}, (0, 24, 12, 28): {'SA_type': 'flip', 'SA_bit': 3}, (0, 26, 18, 13): {'SA_type': 'flip', 'SA_bit': 0}, (0, 23, 13, 28): {'SA_type': 'flip', 'SA_bit': 2}, (0, 2, 10, 21): {'SA_type': 'flip', 'SA_bit': 0}, (0, 4, 3, 5): {'SA_type': 'flip', 'SA_bit': 3}, (0, 3, 5, 21): {'SA_type': 'flip', 'SA_bit': 3}, (0, 10, 24, 5): {'SA_type': 'flip', 'SA_bit': 1}, (0, 18, 3, 26): {'SA_type': 'flip', 'SA_bit': 4}, (0, 27, 27, 26): {'SA_type': 'flip', 'SA_bit': 6}, (0, 1, 25, 3): {'SA_type': 'flip', 'SA_bit': 1}, (0, 4, 9, 29): {'SA_type': 'flip', 'SA_bit': 5}, (0, 8, 21, 1): {'SA_type': 'flip', 'SA_bit': 6}, (0, 11, 6, 12): {'SA_type': 'flip', 'SA_bit': 5}, (0, 2, 18, 2): {'SA_type': 'flip', 'SA_bit': 6}, (0, 26, 9, 9): {'SA_type': 'flip', 'SA_bit': 3}, (0, 23, 17, 0): {'SA_type': 'flip', 'SA_bit': 2}, (0, 3, 2, 24): {'SA_type': 'flip', 'SA_bit': 7}, (0, 25, 12, 18): {'SA_type': 'flip', 'SA_bit': 6}, (0, 27, 16, 11): {'SA_type': 'flip', 'SA_bit': 3}, (0, 15, 7, 31): {'SA_type': 'flip', 'SA_bit': 5}, (0, 0, 20, 19): {'SA_type': 'flip', 'SA_bit': 4}, (0, 12, 19, 17): {'SA_type': 'flip', 'SA_bit': 2}, (0, 25, 19, 29): {'SA_type': 'flip', 'SA_bit': 5}, (0, 7, 10, 19): {'SA_type': 'flip', 'SA_bit': 2}, (0, 2, 14, 1): {'SA_type': 'flip', 'SA_bit': 1}, (0, 16, 1, 31): {'SA_type': 'flip', 'SA_bit': 7}, (0, 14, 5, 30): {'SA_type': 'flip', 'SA_bit': 3}, (0, 4, 2, 12): {'SA_type': 'flip', 'SA_bit': 7}, (0, 3, 16, 3): {'SA_type': 'flip', 'SA_bit': 0}, (0, 11, 7, 1): {'SA_type': 'flip', 'SA_bit': 2}, (0, 9, 11, 30): {'SA_type': 'flip', 'SA_bit': 2}, (0, 8, 15, 14): {'SA_type': 'flip', 'SA_bit': 3}, (0, 15, 25, 6): {'SA_type': 'flip', 'SA_bit': 0}, (0, 8, 24, 27): {'SA_type': 'flip', 'SA_bit': 6}, (0, 8, 2, 12): {'SA_type': 'flip', 'SA_bit': 3}, (0, 6, 17, 18): {'SA_type': 'flip', 'SA_bit': 2}, (0, 25, 18, 12): {'SA_type': 'flip', 'SA_bit': 2}, (0, 26, 15, 4): {'SA_type': 'flip', 'SA_bit': 2}, (0, 22, 2, 8): {'SA_type': 'flip', 'SA_bit': 7}, (0, 12, 14, 29): {'SA_type': 'flip', 'SA_bit': 1}, (0, 14, 7, 0): {'SA_type': 'flip', 'SA_bit': 7}, (0, 20, 24, 28): {'SA_type': 'flip', 'SA_bit': 7}, (0, 26, 14, 29): {'SA_type': 'flip', 'SA_bit': 0}, (0, 16, 5, 1): {'SA_type': 'flip', 'SA_bit': 5}, (0, 12, 1, 0): {'SA_type': 'flip', 'SA_bit': 1}, (0, 25, 19, 12): {'SA_type': 'flip', 'SA_bit': 7}, (0, 19, 12, 6): {'SA_type': 'flip', 'SA_bit': 6}, (0, 10, 14, 21): {'SA_type': 'flip', 'SA_bit': 1}}
orig_coors_if=np.array(list(ifmap_tile.fault_dict.keys()))

#%%
# test reshape index transformation
reshaped_coor=wght_tile.reshape_ravel_idx((2,2,3,10),
                                          source_shape=wght_tile.tile_shape,
                                          source_prior=[3,2,1,0],
                                          target_shape=(144,32),
                                          target_prior=[1,0])

reshaped_coors=wght_tile.reshape_ravel_idx(orig_coors_w,
                                           source_shape=wght_tile.tile_shape,
                                           source_prior=[3,2,1,0],
                                           target_shape=(144,32),
                                           target_prior=[1,0])

#%%
# test slice permute index
permuted_coor=wght_tile.slice_permute_idx((2,2,3,10),
                                          orig_shape=(3,3,16,32),
                                          slice_dims=(0,0,12,12),
                                          slices_permute=[0,1,3,2])

permuted_coors=wght_tile.slice_permute_idx(orig_coors_w,
                                           orig_shape=(3,3,16,32),
                                           slice_dims=(0,0,12,12),
                                           slices_permute=[0,1,3,2])

#%%
# test extract patches index transform
extracted_shape,extracted_coor1=ifmap_tile.extract_patches_idx((0,0,0,2),
                                                              fmap_shape=(1,28,28,16),
                                                              ksizes=(1,3,3,1),
                                                              strides=(1,1,1,1),
                                                              dilation_rates=(1,1,1,1),
                                                              padding='same',
                                                              edge_fill=True)

extracted_shape,extracted_coor2=ifmap_tile.extract_patches_idx((0,9,17,3),
                                                              fmap_shape=(1,28,28,16),
                                                              ksizes=(1,3,3,1),
                                                              strides=(1,2,2,1),
                                                              dilation_rates=(1,1,1,1),
                                                              padding='same',
                                                              edge_fill=True)

extracted_shape,extracted_coors1=ifmap_tile.extract_patches_idx(orig_coors_if,
                                                                fmap_shape=(1,28,28,16),
                                                                ksizes=(1,3,3,1),
                                                                strides=(1,1,1,1),
                                                                dilation_rates=(1,1,1,1),
                                                                padding='same',
                                                                edge_fill=True)

extracted_shape,extracted_coors2=ifmap_tile.extract_patches_idx(orig_coors_if,
                                                                fmap_shape=(1,28,28,16),
                                                                ksizes=(1,3,3,1),
                                                                strides=(1,2,2,1),
                                                                dilation_rates=(1,1,1,1),
                                                                padding='same',
                                                                edge_fill=True)
