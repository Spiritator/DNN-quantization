# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 16:39:57 2018

@author: Yung-Yu Tsai

weight fault injection
"""

import tensorflow as tf
import keras.backend as K

def generate_single_stuck_at_fault(original_value,fault_bit,stuck_at,quantizer,tensor_return=True):
    """Returns the a tensor or variable with single SA fault injected in each parameter.

    # Arguments
        original_value: Tensor or Variable. The variable to be injected fault.
        quantizer: Class. The quantizer class contain following quantize operation infromation.
            word_width: Variable. The fix-point representation of the parameter word length.
            fractional_bits: Variable. Number of fractional bits in a fix-point parameter
            rounding: String. Rounding method of quantization, augment must be one of 'nearest' , 'zero' , 'down', 'stochastic'.
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
        modulator=tf.bitwise.left_shift(1,fault_bit)
        fault_value=tf.bitwise.bitwise_or(fault_value,modulator)
    elif stuck_at=='0':
        modulator=-(tf.bitwise.left_shift(1,fault_bit)+1)
        fault_value=tf.bitwise.bitwise_and(fault_value,modulator)
    elif stuck_at=='flip':
        modulator=tf.bitwise.left_shift(1,fault_bit)
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
            rounding: String. Rounding method of quantization, augment must be one of 'nearest' , 'zero' , 'down', 'stochastic'.
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
    
    modulator=0
    for i in range(len(fault_bit)):
        if stuck_at[i]=='1':
            modulator=tf.bitwise.left_shift(1,fault_bit[i])
            fault_value=tf.bitwise.bitwise_or(fault_value,modulator)
        elif stuck_at[i]=='0':
            modulator=-(tf.bitwise.left_shift(1,fault_bit[i])+1)
            fault_value=tf.bitwise.bitwise_and(fault_value,modulator)
        elif stuck_at[i]=='flip':
            modulator=tf.bitwise.left_shift(1,fault_bit[i])
            fault_value=tf.bitwise.bitwise_xor(fault_value,modulator)
    
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
    if word_width<=fractional_bits-1:
        raise ValueError('Not enough word width %d for fractional bits %d'%(word_width,fractional_bits))
    
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
                modulator=tf.bitwise.left_shift(1,fault_bit[i])
                modulator1=tf.bitwise.bitwise_or(modulator1,modulator)
            elif stuck_at[i]=='0':
                modulator=-(tf.bitwise.left_shift(1,fault_bit[i])+1)
                modulator0=tf.bitwise.bitwise_and(modulator0,modulator)
            elif stuck_at[i]=='flip':
                modulator=tf.bitwise.left_shift(1,fault_bit[i])
                modulatorF=tf.bitwise.bitwise_or(modulatorF,modulator)
    else:
        if stuck_at=='1':
            modulator1=tf.bitwise.left_shift(1,fault_bit)
        elif stuck_at=='0':
            modulator0=-(tf.bitwise.left_shift(1,fault_bit)+1)
        elif stuck_at=='flip':
            modulatorF=tf.bitwise.left_shift(1,fault_bit)
            
    if modulator0==2**word_width-1:
        modulator0=None
        
    if modulator1==0:
        modulator1=None
        
    if modulatorF==0:
        modulatorF=None
    
    return modulator0, modulator1, modulatorF
