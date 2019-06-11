# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 12:08:34 2019

@author: Yung-Yu Tsai

Memory mapping example
"""

from memory.mem_bitmap import bitmap
from memory.tile import tile, tile_FC

import numpy as np

#%%

fault_rate=0.001
row=80
col=20
word=4
model_wl=8

# a buffer memory for weights
GLB_wght=bitmap(row, col*word*model_wl, wl=model_wl)
# a buffer memory for feature maps
GLB_fmap=bitmap(row, col*word*model_wl, wl=model_wl)

# genrate fault on memory
mem_fault_dict,mem_fault_num=GLB_wght.gen_bitmap_SA_fault_dict(fault_rate)
#mem_fault_dict,mem_fault_num=GLB_wght.gen_bitmap_SA_fault_dict(fault_rate,addr_distribution='poisson',addr_pois_lam=(20,50))

# example of number tag conversion in memory
numtag_mem=GLB_wght.get_numtag((2,3))
addr_mem=GLB_wght.numtag2addr(numtag_mem)

numtag_mem_multi=GLB_wght.get_numtag(np.array([[2,3],
                                               [3,4]]))
addr_mem_multi=GLB_wght.numtag2addr(numtag_mem_multi)

# priority of memory mapping for tile
memory_column_priority=['Tm','Tc','Tr','Tn']
memory_row_priority=['Tr','Tm','Tc','Tn']

#%%

# weight tile and feature map tile
wght_tile=tile((3,3,16,32),is_fmap=False,wl=8,row_prior=memory_row_priority,col_prior=memory_column_priority)
fmap_tile=tile((16,3,3,32),is_fmap=True,wl=8,row_prior=memory_row_priority,col_prior=memory_column_priority)

# example of coordinate moving in tile
coor_wght_test=(2,2,4,8)
coor_wght_reduced,bit_reduced=wght_tile.coor_tile_move(coor_wght_test,5,1+2+32+8*16*4*2*8)
coor_wght_increased,bit_increased=wght_tile.coor_tile_move(coor_wght_test,5,1+5+8*12,increase=True)

coor_fmap_test=(8,2,2,4)
coor_fmap_reduced,bit_reduced=fmap_tile.coor_tile_move(coor_fmap_test,5,1+2+32+8*16*4*2*8)
coor_fmap_increased,bit_increased=fmap_tile.coor_tile_move(coor_fmap_test,5,1+5+8*12,increase=True)

# example of number tag conversion in tile
numtag_tile_wght=wght_tile.get_numtag(coor_wght_test,3)
coor_wght_val,bit_val=wght_tile.numtag2coor(numtag_tile_wght)

# example of memory mapping to tile
addr_fault=wght_tile.tile2bitmap(coor_wght_test,3,GLB_wght)
coor_fault,bit_fault=wght_tile.bitmap2tile(addr_fault,GLB_wght)

# example of memory mapping to tile using fault dictionary
tile_fault_dict_wght=wght_tile.fault_dict_bitmap2tile(GLB_wght,use_bias=True)
tile_fault_dict_fmap=fmap_tile.fault_dict_bitmap2tile(GLB_wght)

# example of tile mapping to memory using fault dictionary
GLB_wght.fault_dict=dict()
mem_fault_dict_wght=wght_tile.fault_dict_tile2bitmap(GLB_wght)
mem_fault_dict_fmap=fmap_tile.fault_dict_tile2bitmap(GLB_fmap)

# example of tile fault dictionary restore to layer fault dictionary
layer_shape=(30,30,40,64)
layer_fault_dict=wght_tile.fault_dict_tile2layer(layer_shape)

# example of memory fault dictionary restore to layer fault dictionary 
layer_fault_dict_val=wght_tile.gen_layer_fault_dict(layer_shape,GLB_wght)

#%%

# test bias fault
print('\ntest bias fault')
# a buffer memory for weights
GLB_wght_b=bitmap(row, col*word*model_wl, wl=model_wl)
#mem_fault_dict_b,mem_fault_num=GLB_wght_b.gen_bitmap_SA_fault_dict(fault_rate)
# weight tile and feature map tile
wght_tile_b=tile((3,3,15,32),is_fmap=False,wl=8,row_prior=memory_row_priority,col_prior=memory_column_priority)

# assign bias fault
wght_tile_b.bias_fault_dict={(3,):{'SA_type': 'flip', 'SA_bit': 0},(17,):{'SA_type': 'flip', 'SA_bit': 4},(28,):{'SA_type': 'flip', 'SA_bit': 7}}
# prepare bias fault
mem_fault_dict_b=wght_tile_b.fault_dict_tile2bitmap(GLB_wght_b,use_bias=True)
wght_tile_b.bias_fault_dict=dict()

tile_fault_dict_wght_b=wght_tile_b.fault_dict_bitmap2tile(GLB_wght_b,use_bias=True)
layer_fault_dict_bias=wght_tile_b.gen_layer_fault_dict(layer_shape,GLB_wght_b)


#%% 

# test FC layer memory mapping
print('\ntest FC layer memory mapping')

GLB_wght_fc=bitmap(row, col*word*model_wl, wl=model_wl)
GLB_fmap_fc=bitmap(row, col*word*model_wl, wl=model_wl)

wght_tile_fc=tile_FC((1764,3),is_fmap=False,wl=model_wl)
fmap_tile_fc=tile_FC((2,1764),is_fmap=True,wl=model_wl)

mem_fault_dict_fc,mem_fault_num=GLB_wght_fc.gen_bitmap_SA_fault_dict(fault_rate)

layer_fault_dict_fc=wght_tile_fc.gen_layer_fault_dict((1764,128),GLB_wght_fc,use_bias=True)
tile_fault_dict_fc=wght_tile_fc.fault_dict
tile_fault_dict_fc_bias=wght_tile_fc.bias_fault_dict


