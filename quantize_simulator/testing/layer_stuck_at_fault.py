# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 19:31:45 2018

@author: Yung-Yu Tsai

inject stuck at fault during model build phase

"""

import tensorflow as tf
import numpy as np
from testing.fault_injection import generate_single_stuck_at_fault, generate_multiple_stuck_at_fault

def check_fault_dict(data, fault_dict):
    for key in fault_dict.keys():
        if len(key)!=len(data.shape):
            raise ValueError('fault location %s has different length with data shape %s'%(key,data.shape))
            
        if any([key[i]>=data.shape[i] for i in range(len(key))]):
            raise ValueError('fault location %s is out of data index with shape %s'%(key,data.shape))
            
#        if len(fault_dict[key]['fault_type'])!=len(fault_dict[key]['fault_bit']):
#            raise ValueError('fault location %s has different number of fault types and fault bits'%key)


def inject_layer_sa_fault_nparray(data, fault_dict, word_width, factorial_bit, rounding='nearest'):
    check_fault_dict(data,fault_dict)
    for key in fault_dict.keys():
        if not isinstance(fault_dict[key]['fault_bit'],list):
            data[key]=generate_single_stuck_at_fault(data[key],word_width,factorial_bit,fault_dict[key]['fault_bit'],fault_dict[key]['fault_type'],rounding=rounding,tensor_return=False)
        else:
            data[key]=generate_multiple_stuck_at_fault(data[key],word_width,factorial_bit,fault_dict[key]['fault_bit'],fault_dict[key]['fault_type'],rounding=rounding,tensor_return=False)
            
    return data

def inject_layer_sa_fault_tensor(data, fault_dict, word_width, factorial_bit, rounding='nearest'):
    check_fault_dict(data,fault_dict)
    fault_indices=np.zeros((1,len(data.shape)),dtype=int)
    fault_values=tf.constant([0],dtype='float32')
    for key in fault_dict.keys():
        if not isinstance(fault_dict[key]['fault_bit'],list):
            fault_indices=np.append(fault_indices,[key],axis=0)
            fault_value=generate_single_stuck_at_fault(tf.gather_nd(data,[key]),word_width,factorial_bit,fault_dict[key]['fault_bit'],fault_dict[key]['fault_type'],rounding=rounding)
            fault_values=tf.concat([fault_values,fault_value],0)
            #data[key]=generate_single_stuck_at_fault(data[key],word_width,factorial_bit,fault_dict[key]['fault_bit'],fault_dict[key]['fault_type'],rounding=rounding)
        else:
            fault_indices=np.append(fault_indices,[key],axis=0)
            fault_value=generate_multiple_stuck_at_fault(tf.gather_nd(data,[key]),word_width,factorial_bit,fault_dict[key]['fault_bit'],fault_dict[key]['fault_type'],rounding=rounding)
            fault_values=tf.concat([fault_values,fault_value],0)
            #data[key]=generate_multiple_stuck_at_fault(data[key],word_width,factorial_bit,fault_dict[key]['fault_bit'],fault_dict[key]['fault_type'],rounding=rounding)    
            
    
    fault_indices=fault_indices[1:]
    fault_values=tf.slice(fault_values,[1],[len(fault_indices)])
    fault_indices=tf.constant(fault_indices,dtype='int32')
    data=tf.scatter_nd_update(data,fault_indices,fault_values)
    
    return data
