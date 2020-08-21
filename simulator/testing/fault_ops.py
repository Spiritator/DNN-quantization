# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 19:31:45 2018

@author: Yung-Yu Tsai

inject stuck at fault during model build phase

"""

import tensorflow as tf
from .fault_core import generate_single_stuck_at_fault, generate_multiple_stuck_at_fault, generate_tensor_modulator

def check_fault_dict(data, fault_dict):
    """Check the fault dictionary is valid for the data or not.
        If not, raise error.
    """
    fault_dict_filt=dict()
    for key in fault_dict.keys():
        if len(key)!=len(data.shape):
            raise ValueError('fault location %s has different length with data shape %s'%(key,data.shape))
            
        if any([key[i]>=data.shape[i] for i in range(1,len(key))]):
            raise ValueError('fault location %s is out of data index with shape %s'%(key,data.shape))
            
        if key[0] < data.shape[0]:
            fault_dict_filt[key]=fault_dict[key]
    
    return fault_dict_filt
            
def check_fault_modulator(data, fault_modulator):
    """Check the fault dictionary is valid for the data or not.
    If not, raise error.
    """
    if not isinstance(fault_modulator,list) or len(fault_modulator)!=3:
        raise ValueError('augment fault_modulator must be datatype list and length 3. [modulator0, modulator1, modulatorF]')
        
    for i in range(3):
        if fault_modulator[i] is not None:
            if data.shape[1:] != fault_modulator[i].shape[1:]:
                raise ValueError('fault modulator must have the same shape as data. Expect %s but get %s'%(str(data.shape),str(fault_modulator[i].shape)))
            
            if fault_modulator[i].shape[0] > data.shape[0]:
                fault_modulator[i]=fault_modulator[i][:data.shape[0]]
                
    return fault_modulator

def inject_layer_sa_fault_nparray(data_in, fault_dict, quantizer):
    """Inject fault dictionary to numpy array.

    # Arguments
        data_in: Variable. The variable to be injected fault.
        fault_dict: Dictionary. The dictionary contain fault list information.
        quantizer: Class. The quantizer class contain following quantize operation infromation.
            word_width: Variable. The fix-point representation of the parameter word length.
            fractional_bits: Variable. Number of fractional bits in a fix-point parameter
            rounding: String. Rounding method of quantization, augment must be one of 'nearest' , 'down', 'zero', 'stochastic'.

    # Returns
        The faulty numpy array.
    """
    data=data_in
    check_fault_dict(data,fault_dict)
    for key in fault_dict.keys():
        if not isinstance(fault_dict[key]['SA_bit'],list):
            data[key]=generate_single_stuck_at_fault(data[key],fault_dict[key]['SA_bit'],fault_dict[key]['SA_type'],quantizer,tensor_return=False)
        else:
            data[key]=generate_multiple_stuck_at_fault(data[key],fault_dict[key]['SA_bit'],fault_dict[key]['SA_type'],quantizer,tensor_return=False)
            
    return data

def inject_layer_sa_fault_tensor(data, fault_list, quantizer):
    """Inject fault dictionary to Tensor.

    # Arguments
        data: Tensor. The Tensor to be injected fault.
        fault_list: Dictionary or List. The dictionary contain fault list information. Or the list of fault modulator [modulator0, modulator1, modulatorF].
        quantizer: Class. The quantizer class contain following quantize operation infromation.
            word_width: Variable. The fix-point representation of the parameter word length.
            fractional_bits: Variable. Number of fractional bits in a fix-point parameter
            rounding: String. Rounding method of quantization, augment must be one of 'nearest' , 'down', 'zero', 'stochastic'.

    # Returns
        The faulty Tensor.
    """
    if isinstance(fault_list,dict):
        fault_list=check_fault_dict(data,fault_list)
        tensor_modulator0,tensor_modulator1,tensor_modulatorF=generate_tensor_modulator(data.shape,quantizer.nb,quantizer.fb,fault_list)
    elif isinstance(fault_list,list):
        fault_list=check_fault_modulator(data, fault_list)
        tensor_modulator0=fault_list[0]
        tensor_modulator1=fault_list[1]
        tensor_modulatorF=fault_list[2]
        
    data=quantizer.left_shift_2int(data)
    
    tensor_modulator0=tf.constant(tensor_modulator0)
    tensor_modulator1=tf.constant(tensor_modulator1)
    tensor_modulatorF=tf.constant(tensor_modulatorF)
    
    if tensor_modulator0 is not None:
        data=tf.bitwise.bitwise_and(data,tensor_modulator0)
    if tensor_modulator1 is not None:
        data=tf.bitwise.bitwise_or(data,tensor_modulator1)
    if tensor_modulatorF is not None:
        data=tf.bitwise.bitwise_xor(data,tensor_modulatorF)        

    data=quantizer.right_shift_back(data)
    
    return data
