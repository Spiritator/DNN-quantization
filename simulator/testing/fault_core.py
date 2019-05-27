# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 16:39:57 2018

@author: Yung-Yu Tsai

weight fault injection
"""

import numpy as np
import tensorflow as tf
import keras.backend as K

def generate_single_stuck_at_fault(original_value,fault_bit,stuck_at,quantizer,tensor_return=True):
    """Returns the a tensor or variable with single SA fault injected in each parameter.

    # Arguments
        original_value: Tensor or Variable. The variable to be injected fault.
        quantizer: Class. The quantizer class contain following quantize operation infromation.
            word_width: Variable. The fix-point representation of the parameter word length.
            fractional_bits: Variable. Number of fractional bits in a fix-point parameter
            rounding: String. Rounding method of quantization, augment must be one of 'nearest' , 'down', \'zero\', 'stochastic'.
        fault_bit: Variable. The index of the SA fault bit on a fix-point parameter
        stuck_at: String. The SA type of the faulty bit, input augment must be one of '1' , '0' or 'flip'.
        tensor_return: Condition. Return augment in Tensor dtype or nparray.

    # Returns
        A faulty Tensor or Numpy Array with single SA fault of each parameter.

    # Examples
    ```python
    
        original_weight=np.arange(1,100,dtype='float32')
        single_fault_weight=generate_single_stuck_at_fault(original_weight,10,3,3,'1',tensor_return=False)
        
    ```
    """
    if quantizer.nb<=quantizer.fb-1:
        raise ValueError('Not enough word width %d for fractional bits %d'%(quantizer.nb,quantizer.fb))
    
    if fault_bit<0 or fault_bit>quantizer.nb or not isinstance(fault_bit,int):
        raise ValueError('Fault bit must be integer between (include) %d and 0, %d is MSB, 0 is LSB.'%(quantizer.nb-1,quantizer.nb-1))
        
    if stuck_at!='1' and stuck_at!='0' and stuck_at!='flip':
        raise ValueError('You must stuck at \'0\' , \'1\' or \'flip\'.')
        
    fault_value=quantizer.quantize_1half(original_value)
    fault_value=tf.cast(fault_value,tf.int32)
    
    if stuck_at=='1':
        modulator=np.left_shift(1,fault_bit)
        fault_value=tf.bitwise.bitwise_or(fault_value,modulator)
    elif stuck_at=='0':
        modulator=-(np.left_shift(1,fault_bit)+1)
        fault_value=tf.bitwise.bitwise_and(fault_value,modulator)
    elif stuck_at=='flip':
        modulator=np.left_shift(1,fault_bit)
        fault_value=tf.bitwise.bitwise_xor(fault_value,modulator)
    
    fault_value=tf.cast(fault_value,tf.float32)    
    fault_value=quantizer.quantize_2half(fault_value)
    
    if tensor_return:
        return fault_value
    else:
        return K.eval(fault_value)

def generate_multiple_stuck_at_fault(original_value,fault_bit,stuck_at,quantizer,tensor_return=True):
    """Returns the a tensor or variable with multiple SA fault injected in each parameter.

    # Arguments
        original_value: Tensor or Variable. The variable to be injected fault.
        quantizer: Class. The quantizer class contain following quantize operation infromation.
            word_width: Variable. The fix-point representation of the parameter word length.
            fractional_bits: Variable. Number of fractional bits in a fix-point parameter
            rounding: String. Rounding method of quantization, augment must be one of 'nearest' , 'down', \'zero\', 'stochastic'.
        fault_bit: List of Variable. The index of the SA fault bit on a fix-point parameter
        stuck_at: List of String. The SA type of the faulty bit, augment must be one of '1' , '0' or 'flip'.
        tensor_return: Condition. Return augment in Tensor dtype or nparray.

    # Returns
        A faulty Tensor or Numpy Array with multiple SA fault of each parameter.

    # Examples
    ```python
    
        original_weight=np.arange(1,100,dtype='float32')
        multiple_fault_weight=generate_multiple_stuck_at_fault(original_weight,10,3,[3,2],['1','1'],tensor_return=False)
        
    ```
    """
    if quantizer.nb<=quantizer.fb-1:
        raise ValueError('Not enough word width %d for fractional bits %d'%(quantizer.nb,quantizer.fb))
    
    if any([fault_bit_iter<0 or fault_bit_iter>quantizer.nb or not isinstance(fault_bit_iter,int) for fault_bit_iter in fault_bit]):
        raise ValueError('Fault bit must be integer between (include) %d and 0, %d is MSB, 0 is LSB.'%(quantizer.nb-1,quantizer.nb-1))
        
    if any([stuck_at_iter!='1' and stuck_at_iter!='0' and stuck_at_iter!='flip' for stuck_at_iter in stuck_at]):
        raise ValueError('You must stuck at \'0\' , \'1\' or \'flip\'.')
        
    if len(fault_bit) != len(stuck_at):
        raise ValueError('Fault location list and stuck at type list must be the same length')
        
    fault_value=quantizer.quantize_1half(original_value)
    fault_value=tf.cast(fault_value,tf.int32)
    
    modulator0=-1
    modulator1=0
    modulatorF=0
    for i in range(len(fault_bit)):
        if stuck_at[i]=='1':
            modulator=np.left_shift(1,fault_bit[i])
            modulator1=np.bitwise_or(modulator1,modulator)
        elif stuck_at[i]=='0':
            modulator=-(np.left_shift(1,fault_bit[i])+1)
            modulator0=np.bitwise_and(modulator0,modulator)
        elif stuck_at[i]=='flip':
            modulator=np.left_shift(1,fault_bit[i])
            modulatorF=np.bitwise_xor(modulatorF,modulator)
    
    if modulator0 != -1:
        fault_value=tf.bitwise.bitwise_and(fault_value,modulator0)
    if modulator1 != 0:
        fault_value=tf.bitwise.bitwise_or(fault_value,modulator1)
    if modulatorF != 0:
        fault_value=tf.bitwise.bitwise_xor(fault_value,modulatorF)
    
    fault_value=tf.cast(fault_value,tf.float32)    
    fault_value=quantizer.quantize_2half(fault_value)
    
    if tensor_return:
        return fault_value
    else:
        return K.eval(fault_value)    
    
def generate_stuck_at_fault_modulator(word_width,fractional_bits,fault_bit,stuck_at):
    """Returns the fault modulator of SA0, SA1 and invert bit.

    # Arguments
        word_width: Variable. The fix-point representation of the parameter word length.
        fractional_bits: Variable. Number of fractional bits in a fix-point parameter
        fault_bit: List of Variable. The index of the SA fault bit on a fix-point parameter
        stuck_at: List of String. The SA type of the faulty bit, input augment must be one of '1' , '0' or 'flip'.

    # Returns
        The fault modulator of SA0, SA1 and invert bit respectively.
    """
    if isinstance(fault_bit,list):
        if any([fault_bit_iter<0 or fault_bit_iter>word_width or not isinstance(fault_bit_iter,int) for fault_bit_iter in fault_bit]):
            raise ValueError('Fault bit must be integer between (include) %d and 0, %d is MSB, 0 is LSB.'%(word_width-1,word_width-1))
            
        if any([stuck_at_iter!='1' and stuck_at_iter!='0' and stuck_at_iter!='flip' for stuck_at_iter in stuck_at]):
            raise ValueError('You must stuck at \'0\' , \'1\' or \'flip\'.')
    
        if len(fault_bit) != len(stuck_at):
            raise ValueError('Fault location list and stuck at type list must be the same length')
    
    else: 
        if fault_bit<0 or fault_bit>word_width or not isinstance(fault_bit,int):
            raise ValueError('Fault bit must be integer between (include) %d and 0, %d is MSB, 0 is LSB.'%(word_width-1,word_width-1))
            
        if stuck_at!='1' and stuck_at!='0' and stuck_at!='flip':
            raise ValueError('You must stuck at \'0\' , \'1\' or \'flip\'.')
    
    modulator0=-1
    modulator1=0
    modulatorF=0
    if isinstance(fault_bit,list):
        for i in range(len(fault_bit)):
            if stuck_at[i]=='1':
                modulator=np.left_shift(1,fault_bit[i])
                modulator1=np.bitwise_or(modulator1,modulator)
            elif stuck_at[i]=='0':
                modulator=-(np.left_shift(1,fault_bit[i])+1)
                modulator0=np.bitwise_and(modulator0,modulator)
            elif stuck_at[i]=='flip':
                modulator=np.left_shift(1,fault_bit[i])
                modulatorF=np.bitwise_or(modulatorF,modulator)
    else:
        if stuck_at=='1':
            modulator1=np.left_shift(1,fault_bit)
        elif stuck_at=='0':
            modulator0=-(np.left_shift(1,fault_bit)+1)
        elif stuck_at=='flip':
            modulatorF=np.left_shift(1,fault_bit)
            
    if modulator0==-1:
        modulator0=None
    if modulator1==0:
        modulator1=None
    if modulatorF==0:
        modulatorF=None
    
    return modulator0, modulator1, modulatorF

def generate_tensor_modulator(shape,nb,fb,fault_dict):
    tensor_modulator0=-np.ones(shape,dtype=np.int32)
    tensor_modulator1=np.zeros(shape,dtype=np.int32)
    tensor_modulatorF=np.zeros(shape,dtype=np.int32)
    
    inject0=False
    inject1=False
    injectF=False
    
    for key in fault_dict.keys():
        modulator0,modulator1,modulatorF=generate_stuck_at_fault_modulator(nb,fb,fault_dict[key]['SA_bit'],fault_dict[key]['SA_type'])
        if modulator0 is not None:
            tensor_modulator0[key]=modulator0
            inject0=True
        if modulator1 is not None:
            tensor_modulator1[key]=modulator1
            inject1=True
        if modulatorF is not None:
            tensor_modulatorF[key]=modulatorF
            injectF=True
            
    if not inject0:
        tensor_modulator0=None
    if not inject1:
        tensor_modulator1=None
    if not injectF:
        tensor_modulatorF=None
            
    return [tensor_modulator0,tensor_modulator1,tensor_modulatorF]

def generate_layer_modulator(layer,word_length,fractional_bit,ifmap_fault_dict,ofmap_fault_dict,wght_fault_dict):
    layer_input_shape=layer.input_shape
    layer_output_shape=layer.output_shape
    layer_weight_shape=[weight_shape.shape for weight_shape in layer.get_weights()]
    
    if ifmap_fault_dict is None:
        ifmap_modulator=None
    else:
        ifmap_modulator=generate_tensor_modulator(layer_input_shape,word_length,fractional_bit,ifmap_fault_dict)
    
    if ofmap_fault_dict is None:
        ofmap_modulator=None
    else:
        ofmap_modulator=generate_tensor_modulator(layer_output_shape,word_length,fractional_bit,ofmap_fault_dict)
    wght_modulator=list()
    for i,shape in enumerate(layer_weight_shape):
        if wght_fault_dict[i] is None:
            wght_modulator.append(None)
        else:
            wght_modulator.append(generate_tensor_modulator(shape,word_length,fractional_bit,wght_fault_dict[i]))
    
    return ifmap_modulator,ofmap_modulator,wght_modulator

def generate_model_modulator(model,word_length,fractional_bit,ifmap_fault_dict_list,ofmap_fault_dict_list,wght_fault_dict_list):
    model_depth=len(model.layers)
    model_ifmap_fault_modulator_list=[None for _ in range(model_depth)]
    model_ofmap_fault_modulator_list=[None for _ in range(model_depth)]
    model_wght_fault_modulator_list=[[None,None] for _ in range(model_depth)]

    for layer_num in range(1,model_depth):
        ifmap_modulator,ofmap_modulator,wght_modulator\
        =generate_layer_modulator(model.layers[layer_num],
                                  word_length,
                                  fractional_bit,
                                  ifmap_fault_dict_list[layer_num],
                                  ofmap_fault_dict_list[layer_num],
                                  wght_fault_dict_list[layer_num])
        
        model_ifmap_fault_modulator_list[layer_num]=ifmap_modulator
        model_ofmap_fault_modulator_list[layer_num]=ofmap_modulator
        model_wght_fault_modulator_list[layer_num]=wght_modulator
        
    return model_ifmap_fault_modulator_list,model_ofmap_fault_modulator_list,model_wght_fault_modulator_list

def multi_gpu_fault_modulator_convert(model, gpus=2):
    new_batch=model.input_shape[0]//gpus
    model_depth=len(model.layers)
    for layer_num in range(1,model_depth):
        layer_weight_shape=[weight_shape.shape for weight_shape in model.layers[layer_num].get_weights()]
        if len(layer_weight_shape)!=0:
            for i in range(3):
                if model.layers[layer_num].ifmap_sa_fault_injection[i] is not None:
                    model.layers[layer_num].ifmap_sa_fault_injection[i]=model.layers[layer_num].ifmap_sa_fault_injection[i][:new_batch]
                
                if model.layers[layer_num].ofmap_sa_fault_injection[i] is not None:
                    model.layers[layer_num].ofmap_sa_fault_injection[i]=model.layers[layer_num].ofmap_sa_fault_injection[i][:new_batch]
                
                for j in range(len(layer_weight_shape)):
                    if model.layers[layer_num].weight_sa_fault_injection[j][i] is not None:
                        model.layers[layer_num].weight_sa_fault_injection[j][i]=model.layers[layer_num].weight_sa_fault_injection[j][i][:new_batch]
                    
    return model


