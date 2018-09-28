# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 16:09:29 2018

@author: Yung-Yu Tsai

fault injection test
"""

import tensorflow as tf
import keras.backend as K
import numpy as np
from testing.fault_injection import generate_single_stuck_at_fault, generate_multiple_stuck_at_fault
from testing.layer_stuck_at_fault import inject_layer_sa_fault_tensor, inject_layer_sa_fault_nparray

original_weight=np.arange(1,100,dtype='float32')

single_fault_weight=generate_single_stuck_at_fault(original_weight,10,3,3,'1',tensor_return=False)

multiple_fault_weight=generate_multiple_stuck_at_fault(original_weight,10,3,[3,2],['1','1'],tensor_return=False)



layer_original_weight=np.reshape(np.arange(1,101,dtype='float32'), (10,10))
layer_original_input=tf.Variable(layer_original_weight)

fault_dict={(1,6):\
            {'fault_type':'1',
             'fault_bit':2},
            (0,1):\
            {'fault_type':['1','1'],
             'fault_bit':[3,2]}
            }
            
layer_fault_weight=inject_layer_sa_fault_nparray(layer_original_weight,fault_dict,10,3,rounding='nearest')

layer_fault_input=inject_layer_sa_fault_tensor(layer_original_input,fault_dict,10,3,rounding='nearest')
layer_fault_input_array=K.eval(layer_fault_input)
            