# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 19:31:45 2018

@author: Yung-Yu Tsai

inject stuck at fault during model build phase

"""

from testing.fault_injection import generate_single_stuck_at_fault, generate_multiple_stuck_at_fault

def check_fault_dict(self,data,fault_dict):
    for key in fault_dict.keys():
        if len(fault_dict[key])!=len(data.shpae):
            raise ValueError('fault location %s has different length with data index %s'%(fault_dict[key],data.shape))
            
        if any([fault_dict[key][i]>=data.shape[i] for i in range(len(fault_dict[key]))]):
            raise ValueError('fault location %s is out of data index %s'%(fault_dict[key],data.shape))

class inject_sa_fault_QuantizedDense():
    def __init__(self):
        pass        
        
    def weight(self, data, fault_dict):
        for key in fault_dict.keys():
            if len(fault_dict[key])>1:
                data=generate_multiple_stuck_at_fault()
            else:
                data=generate_single_stuck_at_fault()
        
        return data
    
    def ifmap(self, data, fault_dict):
        for key in fault_dict.keys():
            if len(fault_dict[key])>1:
                data=generate_multiple_stuck_at_fault()
            else:
                data=generate_single_stuck_at_fault()
        
        return data
    
    def ofmap(self, data, fault_dict):
        for key in fault_dict.keys():
            if len(fault_dict[key])>1:
                data=generate_multiple_stuck_at_fault()
            else:
                data=generate_single_stuck_at_fault()
        
        return data
            
