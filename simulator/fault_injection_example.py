# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 16:09:29 2018

@author: Yung-Yu Tsai

fault injection test
"""

import tensorflow as tf
import keras.backend as K
import numpy as np
from testing.fault_core import generate_single_stuck_at_fault, generate_multiple_stuck_at_fault
from testing.fault_ops import inject_layer_sa_fault_tensor, inject_layer_sa_fault_nparray
from layers.quantized_ops import quantizer

# declare qunatizer setting
qt=quantizer(10,3,rounding_method='nearest')

# a numpy array of 0 ~ 99
original_weight=np.arange(1,100,dtype='float32')

# inject single SA fault to a parameter
single_fault_weight=generate_single_stuck_at_fault(original_weight,3,'1',qt,tensor_return=False)

# inject multiple SA fault to a parameter
multiple_fault_weight=generate_multiple_stuck_at_fault(original_weight,[3,2],['1','1'],qt,tensor_return=False)


# the Tensor of original parameter
layer_original_weight_pos=np.reshape(np.arange(1,101,dtype='float32'), (10,10))
layer_original_input_pos=tf.Variable(layer_original_weight_pos)
layer_original_weight_neg=np.reshape(np.arange(1,101,dtype='float32')*(-1), (10,10))
layer_original_input_neg=tf.Variable(layer_original_weight_neg)

# overflow simulation
layer_overflow_sim_pos=qt.quantize(tf.subtract(layer_original_input_pos,0.5),overflow_sim=True)
layer_overflow_sim_neg=qt.quantize(tf.add(layer_original_input_neg,0.5),overflow_sim=True)
layer_overflow_input_pos=K.eval(layer_overflow_sim_pos)
layer_overflow_input_neg=K.eval(layer_overflow_sim_neg)

# example of fault dictionary
fault_dict={(1,6):\
            {'SA_type':'1',
             'SA_bit':2},
            (1,4):\
            {'SA_type':'0',
             'SA_bit':3},
            (0,1):\
            {'SA_type':['1','flip'],
             'SA_bit':[3,2]},
            (0,5):\
            {'SA_type':['1','flip'],
             'SA_bit':[3,2]},
            (0,8):\
            {'SA_type':['0','flip'],
             'SA_bit':[3,2]}
            }
            
# inject fault to a numpy array
layer_fault_weight_pos=inject_layer_sa_fault_nparray(layer_original_weight_pos,fault_dict,qt)
layer_fault_weight_neg=inject_layer_sa_fault_nparray(layer_original_weight_neg,fault_dict,qt)

# inject fault to a Tensor
layer_fault_input_pos=inject_layer_sa_fault_tensor(layer_original_input_pos,fault_dict,qt)
layer_fault_input_array_pos=K.eval(layer_fault_input_pos)
layer_fault_input_neg=inject_layer_sa_fault_tensor(layer_original_input_neg,fault_dict,qt)
layer_fault_input_array_neg=K.eval(layer_fault_input_neg)


# inject fault to a Tensor with overflow simulation
qt_ovf=quantizer(10,3,rounding_method='nearest',overflow_mode=True)
layer_fault_input_pos=inject_layer_sa_fault_tensor(layer_original_input_pos,fault_dict,qt_ovf)
layer_fault_input_array_pos_ovf=K.eval(layer_fault_input_pos)
layer_fault_input_neg=inject_layer_sa_fault_tensor(layer_original_input_neg,fault_dict,qt_ovf)
layer_fault_input_array_neg_ovf=K.eval(layer_fault_input_neg)


            