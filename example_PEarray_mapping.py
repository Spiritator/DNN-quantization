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
orig_coors=np.array(list(wght_tile.fault_dict.keys()))

#%%
# test reshape index transformation
reshaped_coor=wght_tile.reshape_ravel_idx((2,2,3,10),
                                          source_shape=wght_tile.tile_shape,
                                          source_prior=[3,2,1,0],
                                          target_shape=(144,32),
                                          target_prior=[1,0])

reshaped_coors=wght_tile.reshape_ravel_idx(orig_coors,
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

permuted_coors=wght_tile.slice_permute_idx(orig_coors,
                                           orig_shape=(3,3,16,32),
                                           slice_dims=(0,0,12,12),
                                           slices_permute=[0,1,3,2])

#%%
# 
