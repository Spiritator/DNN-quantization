# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:54:08 2019

@author: Yung-Yu Tsai

Memory bitmap setting for memory fault mapping
"""

import numpy as np
    
def fault_num_gen_mem(row,col,fault_rate):
    fault_num=int(row * col * fault_rate)
    return fault_num 

    
def addr_gen_mem(row,col,distribution='uniform',poisson_lam=None):
    
    if distribution=='uniform':
        row_tmp=np.random.randint(row)
        col_tmp=np.random.randint(col)
    elif distribution=='poisson':
        if not isinstance(poisson_lam,tuple) or len(poisson_lam)!=2:
            raise TypeError('Poisson distribution lambda setting must be a tuple has length of 2 (row, col).')
        
        if isinstance(poisson_lam[0],int) and poisson_lam[0]>=0 and poisson_lam[0]<row:
            row_tmp=np.random.poisson(poisson_lam[0])
            while row_tmp>=row:
                row_tmp=np.random.poisson(poisson_lam[0])
        else:
            raise ValueError('Poisson distribution Lambda must within feature map shape. Feature map shape %s but got lambda input %s'%(str((row,col)),str(poisson_lam)))
        
        if isinstance(poisson_lam[1],int) and poisson_lam[1]>=0 and poisson_lam[1]<col:
            col_tmp=np.random.poisson(poisson_lam[1])
            while col_tmp>=col:
                col_tmp=np.random.poisson(poisson_lam[1])
        else:
            raise ValueError('Poisson distribution Lambda must within feature map shape. Feature map shape %s but got lambda input %s'%(str((row,col)),str(poisson_lam)))

    elif distribution=='normal':
        pass 
        '''TO BE DONE'''   
    else:
        raise NameError('Invalid type of random generation distribution. Please choose between uniform, poisson, normal.')
    
    return (row_tmp,col_tmp)

def gen_bitmap_SA_fault_dict(row,col,fault_rate,addr_distribution='uniform',addr_pois_lam=None,fault_type='flip',**kwargs):
    fault_count=0        
    fault_dict=dict()
    fault_num=fault_num_gen_mem(row,col,fault_rate)
            
    while fault_count<fault_num:
        addr=addr_gen_mem(row,col,distribution=addr_distribution,poisson_lam=addr_pois_lam,**kwargs)
        
        if addr in fault_dict.keys():
            continue
        else:
            fault_dict[addr]=fault_type
            fault_count += 1
        
    return fault_dict,fault_num




