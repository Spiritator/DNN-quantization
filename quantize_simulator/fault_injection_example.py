# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 16:09:29 2018

@author: Yung-Yu Tsai

fault injection test
"""

import numpy as np
from testing.fault_injection import generate_single_stuck_at_fault, generate_multiple_stuck_at_fault
from testing.layer_stuck_at_fault import inject_layer_sa_fault

original_weight=np.arange(1,100)

single_fault_weight=generate_single_stuck_at_fault(original_weight,10,3,3,'1')

multiple_fault_weight=generate_multiple_stuck_at_fault(original_weight,10,3,[3,2],['1','1'])



layer_original_weight=np.reshape(np.arange(1,101,dtype='float32'), (10,10))

fault_dict={(1,6):\
            {'fault_type':'1',
             'fault_bit':2},
            (0,1):\
            {'fault_type':['1','1'],
             'fault_bit':[3,2]}
            }
            
layer_fault_weight=inject_layer_sa_fault(layer_original_weight,fault_dict,10,3,rounding='nearest')
            