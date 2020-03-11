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
wght_tile=tile_PE((3,3,16,32),is_fmap=False,wl=8)
ifmap_tile=tile_PE((1,28,28,16),is_fmap=True,wl=8)
ofmap_tile=tile_PE((1,28,28,32),is_fmap=True,wl=8)

wght_tile.fault_dict = {(2, 2, 3, 10): {'SA_type': 'flip', 'SA_bit': 4}, (0, 1, 13, 28): {'SA_type': 'flip', 'SA_bit': 5}, (2, 0, 15, 15): {'SA_type': 'flip', 'SA_bit': 4}, (1, 0, 5, 9): {'SA_type': 'flip', 'SA_bit': 5}, (2, 2, 3, 1): {'SA_type': 'flip', 'SA_bit': 7}, (1, 2, 9, 24): {'SA_type': 'flip', 'SA_bit': 6}, (2, 0, 3, 6): {'SA_type': 'flip', 'SA_bit': 4}, (1, 0, 11, 17): {'SA_type': 'flip', 'SA_bit': 4}, (2, 1, 7, 30): {'SA_type': 'flip', 'SA_bit': 3}, (0, 1, 6, 15): {'SA_type': 'flip', 'SA_bit': 4}, (2, 2, 11, 18): {'SA_type': 'flip', 'SA_bit': 5}, (0, 1, 5, 18): {'SA_type': 'flip', 'SA_bit': 0}, (2, 1, 1, 20): {'SA_type': 'flip', 'SA_bit': 6}, (0, 0, 6, 24): {'SA_type': 'flip', 'SA_bit': 6}, (1, 2, 4, 5): {'SA_type': 'flip', 'SA_bit': 5}, (0, 2, 7, 19): {'SA_type': 'flip', 'SA_bit': 3}, (0, 0, 12, 30): {'SA_type': 'flip', 'SA_bit': 0}, (1, 1, 13, 1): {'SA_type': 'flip', 'SA_bit': 4}, (2, 2, 13, 8): {'SA_type': 'flip', 'SA_bit': 3}, (0, 1, 0, 25): {'SA_type': 'flip', 'SA_bit': 6}, (0, 0, 10, 26): {'SA_type': 'flip', 'SA_bit': 0}, (0, 1, 10, 4): {'SA_type': 'flip', 'SA_bit': 4}, (0, 1, 11, 0): {'SA_type': 'flip', 'SA_bit': 7}, (2, 0, 3, 25): {'SA_type': 'flip', 'SA_bit': 1}, (2, 0, 12, 6): {'SA_type': 'flip', 'SA_bit': 7}, (0, 2, 5, 29): {'SA_type': 'flip', 'SA_bit': 7}, (0, 2, 14, 26): {'SA_type': 'flip', 'SA_bit': 7}, (2, 2, 6, 26): {'SA_type': 'flip', 'SA_bit': 3}, (2, 1, 15, 24): {'SA_type': 'flip', 'SA_bit': 1}, (0, 1, 11, 6): {'SA_type': 'flip', 'SA_bit': 2}, (0, 1, 13, 2): {'SA_type': 'flip', 'SA_bit': 1}, (1, 2, 8, 0): {'SA_type': 'flip', 'SA_bit': 4}, (2, 2, 6, 28): {'SA_type': 'flip', 'SA_bit': 2}, (2, 1, 11, 17): {'SA_type': 'flip', 'SA_bit': 2}, (1, 2, 12, 14): {'SA_type': 'flip', 'SA_bit': 2}, (1, 1, 9, 29): {'SA_type': 'flip', 'SA_bit': 7}}
orig_coors_w=np.array(list(wght_tile.fault_dict.keys()))
ifmap_tile.fault_dict = {(0, 15, 6, 3): {'SA_type': 'flip', 'SA_bit': 7}, (0, 5, 8, 10): {'SA_type': 'flip', 'SA_bit': 5}, (0, 25, 12, 14): {'SA_type': 'flip', 'SA_bit': 0}, (0, 8, 12, 0): {'SA_type': 'flip', 'SA_bit': 2}, (0, 27, 6, 0): {'SA_type': 'flip', 'SA_bit': 4}, (0, 27, 14, 9): {'SA_type': 'flip', 'SA_bit': 4}, (0, 27, 12, 15): {'SA_type': 'flip', 'SA_bit': 2}, (0, 11, 21, 14): {'SA_type': 'flip', 'SA_bit': 2}, (0, 20, 23, 10): {'SA_type': 'flip', 'SA_bit': 3}, (0, 2, 22, 14): {'SA_type': 'flip', 'SA_bit': 1}, (0, 23, 15, 12): {'SA_type': 'flip', 'SA_bit': 4}, (0, 19, 1, 14): {'SA_type': 'flip', 'SA_bit': 5}, (0, 6, 15, 2): {'SA_type': 'flip', 'SA_bit': 5}, (0, 5, 6, 9): {'SA_type': 'flip', 'SA_bit': 1}, (0, 2, 4, 14): {'SA_type': 'flip', 'SA_bit': 2}, (0, 9, 3, 9): {'SA_type': 'flip', 'SA_bit': 4}, (0, 3, 13, 0): {'SA_type': 'flip', 'SA_bit': 4}, (0, 23, 14, 2): {'SA_type': 'flip', 'SA_bit': 6}, (0, 16, 24, 12): {'SA_type': 'flip', 'SA_bit': 1}, (0, 16, 6, 4): {'SA_type': 'flip', 'SA_bit': 0}, (0, 0, 23, 5): {'SA_type': 'flip', 'SA_bit': 5}, (0, 18, 24, 7): {'SA_type': 'flip', 'SA_bit': 7}, (0, 24, 24, 3): {'SA_type': 'flip', 'SA_bit': 6}, (0, 4, 20, 10): {'SA_type': 'flip', 'SA_bit': 5}, (0, 7, 24, 1): {'SA_type': 'flip', 'SA_bit': 6}, (0, 3, 20, 4): {'SA_type': 'flip', 'SA_bit': 2}, (0, 13, 1, 15): {'SA_type': 'flip', 'SA_bit': 0}, (0, 2, 2, 5): {'SA_type': 'flip', 'SA_bit': 2}, (0, 26, 9, 7): {'SA_type': 'flip', 'SA_bit': 5}, (0, 26, 5, 15): {'SA_type': 'flip', 'SA_bit': 7}, (0, 15, 9, 4): {'SA_type': 'flip', 'SA_bit': 6}, (0, 22, 3, 6): {'SA_type': 'flip', 'SA_bit': 0}, (0, 26, 8, 7): {'SA_type': 'flip', 'SA_bit': 4}, (0, 21, 27, 5): {'SA_type': 'flip', 'SA_bit': 6}, (0, 21, 4, 0): {'SA_type': 'flip', 'SA_bit': 6}, (0, 27, 9, 10): {'SA_type': 'flip', 'SA_bit': 7}, (0, 5, 27, 8): {'SA_type': 'flip', 'SA_bit': 0}, (0, 24, 17, 0): {'SA_type': 'flip', 'SA_bit': 4}, (0, 14, 21, 5): {'SA_type': 'flip', 'SA_bit': 5}, (0, 17, 5, 14): {'SA_type': 'flip', 'SA_bit': 5}, (0, 5, 3, 1): {'SA_type': 'flip', 'SA_bit': 1}, (0, 24, 9, 14): {'SA_type': 'flip', 'SA_bit': 5}, (0, 16, 22, 9): {'SA_type': 'flip', 'SA_bit': 1}, (0, 15, 16, 0): {'SA_type': 'flip', 'SA_bit': 3}, (0, 5, 3, 14): {'SA_type': 'flip', 'SA_bit': 3}, (0, 27, 0, 10): {'SA_type': 'flip', 'SA_bit': 1}, (0, 10, 23, 12): {'SA_type': 'flip', 'SA_bit': 5}, (0, 23, 18, 5): {'SA_type': 'flip', 'SA_bit': 5}, (0, 12, 7, 5): {'SA_type': 'flip', 'SA_bit': 0}, (0, 4, 16, 8): {'SA_type': 'flip', 'SA_bit': 3}, (0, 25, 14, 0): {'SA_type': 'flip', 'SA_bit': 3}, (0, 6, 5, 11): {'SA_type': 'flip', 'SA_bit': 4}, (0, 2, 8, 3): {'SA_type': 'flip', 'SA_bit': 7}, (0, 20, 0, 15): {'SA_type': 'flip', 'SA_bit': 0}, (0, 20, 0, 7): {'SA_type': 'flip', 'SA_bit': 5}, (0, 19, 3, 2): {'SA_type': 'flip', 'SA_bit': 1}, (0, 7, 10, 2): {'SA_type': 'flip', 'SA_bit': 3}, (0, 23, 1, 8): {'SA_type': 'flip', 'SA_bit': 2}, (0, 16, 8, 7): {'SA_type': 'flip', 'SA_bit': 4}, (0, 1, 25, 14): {'SA_type': 'flip', 'SA_bit': 4}, (0, 12, 25, 7): {'SA_type': 'flip', 'SA_bit': 5}, (0, 6, 6, 10): {'SA_type': 'flip', 'SA_bit': 7}, (0, 15, 20, 4): {'SA_type': 'flip', 'SA_bit': 1}, (0, 26, 3, 8): {'SA_type': 'flip', 'SA_bit': 2}, (0, 10, 3, 5): {'SA_type': 'flip', 'SA_bit': 6}, (0, 23, 0, 4): {'SA_type': 'flip', 'SA_bit': 7}, (0, 0, 7, 5): {'SA_type': 'flip', 'SA_bit': 1}, (0, 19, 3, 1): {'SA_type': 'flip', 'SA_bit': 0}, (0, 22, 7, 6): {'SA_type': 'flip', 'SA_bit': 5}, (0, 24, 13, 4): {'SA_type': 'flip', 'SA_bit': 5}, (0, 27, 12, 8): {'SA_type': 'flip', 'SA_bit': 0}, (0, 11, 17, 1): {'SA_type': 'flip', 'SA_bit': 6}, (0, 5, 11, 14): {'SA_type': 'flip', 'SA_bit': 0}, (0, 13, 0, 5): {'SA_type': 'flip', 'SA_bit': 2}, (0, 20, 10, 10): {'SA_type': 'flip', 'SA_bit': 1}, (0, 24, 2, 1): {'SA_type': 'flip', 'SA_bit': 5}, (0, 20, 2, 6): {'SA_type': 'flip', 'SA_bit': 3}, (0, 6, 27, 1): {'SA_type': 'flip', 'SA_bit': 7}, (0, 1, 25, 8): {'SA_type': 'flip', 'SA_bit': 1}, (0, 2, 9, 15): {'SA_type': 'flip', 'SA_bit': 4}, (0, 12, 21, 15): {'SA_type': 'flip', 'SA_bit': 7}, (0, 5, 27, 11): {'SA_type': 'flip', 'SA_bit': 4}, (0, 2, 27, 3): {'SA_type': 'flip', 'SA_bit': 7}, (0, 13, 14, 10): {'SA_type': 'flip', 'SA_bit': 7}, (0, 18, 8, 11): {'SA_type': 'flip', 'SA_bit': 2}, (0, 23, 23, 11): {'SA_type': 'flip', 'SA_bit': 6}, (0, 6, 21, 10): {'SA_type': 'flip', 'SA_bit': 2}, (0, 9, 3, 13): {'SA_type': 'flip', 'SA_bit': 2}, (0, 10, 16, 3): {'SA_type': 'flip', 'SA_bit': 5}, (0, 0, 5, 14): {'SA_type': 'flip', 'SA_bit': 4}, (0, 6, 16, 4): {'SA_type': 'flip', 'SA_bit': 1}, (0, 5, 15, 9): {'SA_type': 'flip', 'SA_bit': 7}, (0, 12, 4, 10): {'SA_type': 'flip', 'SA_bit': 3}, (0, 14, 0, 13): {'SA_type': 'flip', 'SA_bit': 7}, (0, 16, 19, 4): {'SA_type': 'flip', 'SA_bit': 7}, (0, 18, 17, 7): {'SA_type': 'flip', 'SA_bit': 6}, (0, 20, 1, 4): {'SA_type': 'flip', 'SA_bit': 1}, (0, 3, 15, 5): {'SA_type': 'flip', 'SA_bit': 2}, (0, 11, 8, 8): {'SA_type': 'flip', 'SA_bit': 4}, (0, 24, 5, 5): {'SA_type': 'flip', 'SA_bit': 7}}
orig_coors_if=np.array(list(ifmap_tile.fault_dict.keys()))
ofmap_tile.fault_dict = {(0, 15, 6, 6): {'SA_type': 'flip', 'SA_bit': 7}, (0, 5, 8, 20): {'SA_type': 'flip', 'SA_bit': 5}, (0, 25, 12, 28): {'SA_type': 'flip', 'SA_bit': 0}, (0, 8, 12, 0): {'SA_type': 'flip', 'SA_bit': 2}, (0, 27, 6, 0): {'SA_type': 'flip', 'SA_bit': 4}, (0, 27, 14, 18): {'SA_type': 'flip', 'SA_bit': 4}, (0, 27, 12, 30): {'SA_type': 'flip', 'SA_bit': 2}, (0, 11, 21, 28): {'SA_type': 'flip', 'SA_bit': 2}, (0, 20, 23, 20): {'SA_type': 'flip', 'SA_bit': 3}, (0, 2, 22, 28): {'SA_type': 'flip', 'SA_bit': 1}, (0, 23, 15, 24): {'SA_type': 'flip', 'SA_bit': 4}, (0, 19, 1, 28): {'SA_type': 'flip', 'SA_bit': 5}, (0, 6, 15, 4): {'SA_type': 'flip', 'SA_bit': 5}, (0, 5, 6, 18): {'SA_type': 'flip', 'SA_bit': 1}, (0, 2, 4, 28): {'SA_type': 'flip', 'SA_bit': 2}, (0, 9, 3, 18): {'SA_type': 'flip', 'SA_bit': 4}, (0, 3, 13, 0): {'SA_type': 'flip', 'SA_bit': 4}, (0, 23, 14, 4): {'SA_type': 'flip', 'SA_bit': 6}, (0, 16, 24, 12): {'SA_type': 'flip', 'SA_bit': 1}, (0, 16, 6, 8): {'SA_type': 'flip', 'SA_bit': 0}, (0, 0, 23, 10): {'SA_type': 'flip', 'SA_bit': 5}, (0, 18, 24, 7): {'SA_type': 'flip', 'SA_bit': 7}, (0, 24, 24, 3): {'SA_type': 'flip', 'SA_bit': 6}, (0, 4, 20, 21): {'SA_type': 'flip', 'SA_bit': 5}, (0, 7, 24, 1): {'SA_type': 'flip', 'SA_bit': 6}, (0, 3, 20, 9): {'SA_type': 'flip', 'SA_bit': 2}, (0, 13, 1, 31): {'SA_type': 'flip', 'SA_bit': 0}, (0, 2, 2, 5): {'SA_type': 'flip', 'SA_bit': 2}, (0, 26, 9, 7): {'SA_type': 'flip', 'SA_bit': 5}, (0, 26, 5, 19): {'SA_type': 'flip', 'SA_bit': 7}, (0, 15, 9, 4): {'SA_type': 'flip', 'SA_bit': 6}, (0, 22, 3, 13): {'SA_type': 'flip', 'SA_bit': 0}, (0, 26, 8, 7): {'SA_type': 'flip', 'SA_bit': 4}, (0, 21, 27, 5): {'SA_type': 'flip', 'SA_bit': 6}, (0, 21, 4, 0): {'SA_type': 'flip', 'SA_bit': 6}, (0, 27, 9, 1): {'SA_type': 'flip', 'SA_bit': 7}, (0, 5, 27, 17): {'SA_type': 'flip', 'SA_bit': 0}, (0, 24, 17, 0): {'SA_type': 'flip', 'SA_bit': 4}, (0, 14, 21, 5): {'SA_type': 'flip', 'SA_bit': 5}, (0, 17, 5, 27): {'SA_type': 'flip', 'SA_bit': 5}, (0, 5, 3, 1): {'SA_type': 'flip', 'SA_bit': 1}, (0, 24, 9, 14): {'SA_type': 'flip', 'SA_bit': 5}, (0, 16, 22, 9): {'SA_type': 'flip', 'SA_bit': 1}, (0, 15, 16, 0): {'SA_type': 'flip', 'SA_bit': 3}, (0, 5, 3, 14): {'SA_type': 'flip', 'SA_bit': 3}, (0, 27, 0, 10): {'SA_type': 'flip', 'SA_bit': 1}, (0, 10, 23, 12): {'SA_type': 'flip', 'SA_bit': 5}, (0, 23, 18, 5): {'SA_type': 'flip', 'SA_bit': 5}, (0, 12, 7, 5): {'SA_type': 'flip', 'SA_bit': 0}, (0, 4, 16, 8): {'SA_type': 'flip', 'SA_bit': 3}, (0, 25, 14, 0): {'SA_type': 'flip', 'SA_bit': 3}, (0, 6, 5, 11): {'SA_type': 'flip', 'SA_bit': 4}, (0, 2, 8, 3): {'SA_type': 'flip', 'SA_bit': 7}, (0, 20, 0, 15): {'SA_type': 'flip', 'SA_bit': 0}, (0, 20, 0, 7): {'SA_type': 'flip', 'SA_bit': 5}, (0, 19, 3, 2): {'SA_type': 'flip', 'SA_bit': 1}, (0, 7, 10, 2): {'SA_type': 'flip', 'SA_bit': 3}, (0, 23, 1, 8): {'SA_type': 'flip', 'SA_bit': 2}, (0, 16, 8, 7): {'SA_type': 'flip', 'SA_bit': 4}, (0, 1, 25, 14): {'SA_type': 'flip', 'SA_bit': 4}, (0, 12, 25, 7): {'SA_type': 'flip', 'SA_bit': 5}, (0, 6, 6, 10): {'SA_type': 'flip', 'SA_bit': 7}, (0, 15, 20, 4): {'SA_type': 'flip', 'SA_bit': 1}, (0, 26, 3, 8): {'SA_type': 'flip', 'SA_bit': 2}, (0, 10, 3, 5): {'SA_type': 'flip', 'SA_bit': 6}, (0, 23, 0, 4): {'SA_type': 'flip', 'SA_bit': 7}, (0, 0, 7, 5): {'SA_type': 'flip', 'SA_bit': 1}, (0, 19, 3, 1): {'SA_type': 'flip', 'SA_bit': 0}, (0, 22, 7, 6): {'SA_type': 'flip', 'SA_bit': 5}, (0, 24, 13, 4): {'SA_type': 'flip', 'SA_bit': 5}, (0, 27, 12, 8): {'SA_type': 'flip', 'SA_bit': 0}, (0, 11, 17, 1): {'SA_type': 'flip', 'SA_bit': 6}, (0, 5, 11, 14): {'SA_type': 'flip', 'SA_bit': 0}, (0, 13, 0, 5): {'SA_type': 'flip', 'SA_bit': 2}, (0, 20, 10, 10): {'SA_type': 'flip', 'SA_bit': 1}, (0, 24, 2, 1): {'SA_type': 'flip', 'SA_bit': 5}, (0, 20, 2, 6): {'SA_type': 'flip', 'SA_bit': 3}, (0, 6, 27, 1): {'SA_type': 'flip', 'SA_bit': 7}, (0, 1, 25, 8): {'SA_type': 'flip', 'SA_bit': 1}, (0, 2, 9, 15): {'SA_type': 'flip', 'SA_bit': 4}, (0, 12, 21, 15): {'SA_type': 'flip', 'SA_bit': 7}, (0, 5, 27, 11): {'SA_type': 'flip', 'SA_bit': 4}, (0, 2, 27, 3): {'SA_type': 'flip', 'SA_bit': 7}, (0, 13, 14, 10): {'SA_type': 'flip', 'SA_bit': 7}, (0, 18, 8, 11): {'SA_type': 'flip', 'SA_bit': 2}, (0, 23, 23, 11): {'SA_type': 'flip', 'SA_bit': 6}, (0, 6, 21, 10): {'SA_type': 'flip', 'SA_bit': 2}, (0, 9, 3, 13): {'SA_type': 'flip', 'SA_bit': 2}, (0, 10, 16, 3): {'SA_type': 'flip', 'SA_bit': 5}, (0, 0, 5, 14): {'SA_type': 'flip', 'SA_bit': 4}, (0, 6, 16, 4): {'SA_type': 'flip', 'SA_bit': 1}, (0, 5, 15, 9): {'SA_type': 'flip', 'SA_bit': 7}, (0, 12, 4, 10): {'SA_type': 'flip', 'SA_bit': 3}, (0, 14, 0, 13): {'SA_type': 'flip', 'SA_bit': 7}, (0, 16, 19, 4): {'SA_type': 'flip', 'SA_bit': 7}, (0, 18, 17, 7): {'SA_type': 'flip', 'SA_bit': 6}, (0, 20, 1, 4): {'SA_type': 'flip', 'SA_bit': 1}, (0, 3, 15, 5): {'SA_type': 'flip', 'SA_bit': 2}, (0, 11, 8, 8): {'SA_type': 'flip', 'SA_bit': 4}, (0, 24, 5, 5): {'SA_type': 'flip', 'SA_bit': 7}}

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
                                          slicing_dims=(0,0,12,12),
                                          slices_permute=[0,1,3,2])

permuted_coors=wght_tile.slice_permute_idx(orig_coors_w,
                                           orig_shape=(3,3,16,32),
                                           slicing_dims=(0,0,12,12),
                                           slices_permute=[0,1,3,2])

#%%
# test extract patches index transform
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

#%%
# test tile expand data

# expand reshape and slice/permute

expand_fault_dict_reshape_o=ofmap_tile.expand_reshape_data(orig_prior=[0,3,2,1],
                                                           expect_shape=(784,32),
                                                           reshape_prior=[1,0],
                                                           slicing_dims=(784,16),
                                                           slices_permute=[1,0],
                                                           tilting=True, 
                                                           tilt_axis=1, 
                                                           tilt_direction=0)

expand_fault_dict_reshape_w=wght_tile.expand_reshape_data(orig_prior=[3,2,1,0],
                                                expect_shape=(144,32),
                                                reshape_prior=[1,0],
                                                slicing_dims=(16,16),
                                                slices_permute=[1,0])

expand_fault_dict_extracted=ifmap_tile.expand_extract_patches(ksizes=(1,3,3,1),
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
                                                              tilt_direction=0)

#%%
# test PE array mapping

MXU=PEarray(16,16,ofmap_tile=ofmap_tile,wght_tile=wght_tile,ifmap_tile=ifmap_tile)

#%%
# test streaming mapping

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
                                          window_flow_direction='backward',)

#%%
# test broadcast array mapping

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

#%%
# test fixed array mapping

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

#%%
# pre-mapping tiles to PE

# setup mapping configuration
MXU.setup_dataflow(o_permute_info={'PE_required_axes_prior':['t_clk','PE_x'],
                                   'tile_mapping_prior':[2,1,0]}, 
                   o_fixed_info={'PE_fix_axis':'PE_y',
                                 'indice':-1}, 
                   o_broadcast_info=None, 
                   o_streaming_info=None, 
                   o_repeat=0, 
                   o_duplicate=0, 
                   o_stall=0, 
                   o_latency=0,
                   w_permute_info={'PE_required_axes_prior':['t_clk','PE_y','PE_x'],
                                   'tile_mapping_prior':[2,1,0]}, 
                   w_fixed_info=None, 
                   w_broadcast_info=None, 
                   w_streaming_info=None, 
                   w_repeat=0, 
                   w_duplicate=0, 
                   w_stall=0, 
                   w_latency=0,
                   i_permute_info={'PE_required_axes_prior':['t_clk','PE_y'],
                                   'tile_mapping_prior':[2,1,0]}, 
                   i_fixed_info=None, 
                   i_broadcast_info=None, 
                   i_streaming_info={'PE_stream_axis':'PE_x',
                                     'tile_direction':'forward',
                                     'PE_direction':'forward'}, 
                   i_repeat=0, 
                   i_duplicate=0, 
                   i_stall=0, 
                   i_latency=0)

# ofmap pre-mapping
mapped_fault_dict_ofmap=MXU.premapping_tile('ofmap')
# weight pre-mapping
mapped_fault_dict_wght=MXU.premapping_tile('wght')
# ifmap pre-mapping
mapped_fault_dict_ifmap=MXU.premapping_tile('ifmap')

