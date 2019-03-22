# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 19:31:45 2018

@author: Yung-Yu Tsai

inject stuck at fault during model build phase

"""

import tensorflow as tf
import keras.backend as K
import numpy as np
from testing.fault_core import generate_single_stuck_at_fault, generate_multiple_stuck_at_fault, generate_stuck_at_fault_modulator
from layers.quantized_ops import quantize_1half,quantize_2half

def check_fault_dict(data, fault_dict):
    """Check the fault dictionary is valid for the data or not.
        If not, raise error.
    """
    for key in fault_dict.keys():
        if len(key)!=len(data.shape):
            raise ValueError('fault location %s has different length with data shape %s'%(key,data.shape))
            
        if any([key[i]>=data.shape[i] for i in range(len(key))]):
            raise ValueError('fault location %s is out of data index with shape %s'%(key,data.shape))
            
#        if len(fault_dict[key]['fault_type'])!=len(fault_dict[key]['SA_bit']):
#            raise ValueError('fault location %s has different number of fault types and fault bits'%key)


def inject_layer_sa_fault_nparray(data_in, fault_dict, word_width, fractional_bit, rounding='nearest'):
    """Inject fault dictionary to numpy array.

    # Arguments
        data_in: Variable. The variable to be injected fault.
        fault_dict: Dictionary. The dictionary contain fault list information.
        word_width: Variable. The fix-point representation of the parameter word length.
        fractional_bits: Variable. Number of fractional bits in a fix-point parameter
        rounding: String. Rounding method of quantization, augment must be one of 'nearest' , 'zero' , 'down', 'stochastic'.

    # Returns
        The faulty numpy array.
    """
    data=data_in
    check_fault_dict(data,fault_dict)
    for key in fault_dict.keys():
        if not isinstance(fault_dict[key]['SA_bit'],list):
            data[key]=generate_single_stuck_at_fault(data[key],word_width,fractional_bit,fault_dict[key]['SA_bit'],fault_dict[key]['SA_type'],rounding=rounding,tensor_return=False)
        else:
            data[key]=generate_multiple_stuck_at_fault(data[key],word_width,fractional_bit,fault_dict[key]['SA_bit'],fault_dict[key]['SA_type'],rounding=rounding,tensor_return=False)
            
    return data

def inject_layer_sa_fault_tensor(data_in, fault_dict, word_width, fractional_bit, rounding='nearest'):
    """Inject fault dictionary to Tensor.

    # Arguments
        data_in: Tensor. The Tensor to be injected fault.
        fault_dict: Dictionary. The dictionary contain fault list information.
        word_width: Variable. The fix-point representation of the parameter word length.
        fractional_bits: Variable. Number of fractional bits in a fix-point parameter
        rounding: String. Rounding method of quantization, augment must be one of 'nearest' , 'zero' , 'down', 'stochastic'.

    # Returns
        The faulty Tensor.
    """
    data=data_in
    if isinstance(fault_dict,dict):
        shape=data.shape
    elif isinstance(fault_dict,list):
        # for the bypass method of keras flatten layer batch number bug
        shape=fault_dict[1]
        fault_dict=fault_dict[0]
    else:
        raise TypeError('wrong type of fault list being injected. The fault list is either dict (normal injection) or list (index 0 fault list, index 1 the data shape of being injected tensor.)')

        
    check_fault_dict(data,fault_dict)
    fault_indices=[np.zeros((1,len(shape)),dtype='int32') for i in range(3)]
    fault_modulators=[tf.constant([0],dtype='int32') for i in range(3)]
    
    for key in fault_dict.keys():
        modulator0,modulator1,modulatorF=generate_stuck_at_fault_modulator(word_width,fractional_bit,fault_dict[key]['SA_bit'],fault_dict[key]['SA_type'])
        if modulator0 is not None:
            fault_indices[0]=np.append(fault_indices[0],[key],axis=0)
            fault_modulators[0]=tf.concat([fault_modulators[0],[modulator0]],0)
        if modulator1 is not None:
            fault_indices[1]=np.append(fault_indices[1],[key],axis=0)
            fault_modulators[1]=tf.concat([fault_modulators[1],[modulator1]],0)
        if modulatorF is not None:
            fault_indices[2]=np.append(fault_indices[2],[key],axis=0)
            fault_modulators[2]=tf.concat([fault_modulators[2],[modulatorF]],0)
    
    for i in range(3):
        fault_indices[i]=fault_indices[i][1:]
        fault_modulators[i]=tf.slice(fault_modulators[i],[1],[len(fault_indices[i])])
        fault_indices[i]=tf.constant(fault_indices[i],dtype='int32')
        
        
    data=quantize_1half(data, nb = word_width, fb = fractional_bit, rounding_method = rounding)
    data=tf.cast(data,tf.int32)
    
    if fault_indices[0].shape[0]>0:
        modulater_tensor0=tf.Variable(np.ones(shape,dtype='int32')*(-1),dtype='int32')
        modulater_tensor0=tf.scatter_nd_update(modulater_tensor0,fault_indices[0],fault_modulators[0])
        data=tf.bitwise.bitwise_and(data,modulater_tensor0)
    if fault_indices[1].shape[0]>0:
        modulater_tensor1=tf.Variable(np.zeros(shape,dtype='int32'))
        modulater_tensor1=tf.scatter_nd_update(modulater_tensor1,fault_indices[1],fault_modulators[1])
        data=tf.bitwise.bitwise_or(data,modulater_tensor1)
    if fault_indices[2].shape[0]>0:
        modulater_tensorF=tf.Variable(np.zeros(shape,dtype='int32'))
        modulater_tensorF=tf.scatter_nd_update(modulater_tensorF,fault_indices[2],fault_modulators[2])
        data=tf.bitwise.bitwise_xor(data,modulater_tensorF)        

    data=tf.cast(data,tf.float32)    
    data=quantize_2half(data, nb = word_width, fb = fractional_bit)
    
    return data
