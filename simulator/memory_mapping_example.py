# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 12:08:34 2019

@author: Yung-Yu Tsai

Memory mapping example
"""

from memory.mem_bitmap import bitmap
from memory.tile import tile

#%%

fault_rate=0.001
row=80
col=20
word=4
model_wl=8

# a buffer memory
GLB_wght=bitmap(row, col*word*model_wl, wl=model_wl)

# genrate fault on memory
mem_fault_dict,mem_fault_num=GLB_wght.gen_bitmap_SA_fault_dict(fault_rate)
#mem_fault_dict,mem_fault_num=GLB_wght.gen_bitmap_SA_fault_dict(fault_rate,addr_distribution='poisson',addr_pois_lam=(20,50))

memory_column_priority=['Tm','Tc','Tr','Tn']
memory_row_priority=['Tr','Tm','Tc','Tn']

#%%

# weight tile and feature map tile
wght_tile=tile(16,32,3,3,is_fmap=False,wl=8,row_prior=memory_row_priority,col_prior=memory_column_priority)
fmap_tile=tile(16,32,3,3,is_fmap=True,wl=8,row_prior=memory_row_priority,col_prior=memory_column_priority)

coor_wght_test=(2,2,4,8)
coor_wght_reduced,bit_reduced=wght_tile.coor_tile_move(coor_wght_test,5,1+2+32+8*16*4*2*8)
coor_wght_increased,bit_increased=wght_tile.coor_tile_move(coor_wght_test,5,1+5+8*12,increase=True)

coor_fmap_test=(8,2,2,4)
coor_fmap_reduced,bit_reduced=fmap_tile.coor_tile_move(coor_fmap_test,5,1+2+32+8*16*4*2*8)
coor_fmap_increased,bit_increased=fmap_tile.coor_tile_move(coor_fmap_test,5,1+5+8*12,increase=True)

