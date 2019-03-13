# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 12:08:34 2019

@author: Yung-Yu Tsai

Memory mapping example
"""

from memory.mem_bitmap import gen_bitmap_SA_fault_dict

fault_rate=0.001
row=80
col=20
word=4
model_wl=8

mem_fault_dict,mem_fault_num=gen_bitmap_SA_fault_dict(row,col*word*model_wl,fault_rate)
#mem_fault_dict,mem_fault_num=gen_bitmap_SA_fault_dict(row,col*word*model_wl,fault_rate,addr_distribution='poisson',addr_pois_lam=(20,50))

