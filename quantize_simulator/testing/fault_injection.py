# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 16:39:57 2018

@author: Yung-Yu Tsai

weight fault injection
"""

import keras
from keras.models import Sequential, Model
from layers.quantized_ops import quantize_1half,quantize_2half
import numpy as np

def generate_single_stuck_at_fault(original_value,word_width,factorial_bits,fault_bit,stuck_at,rounding='nearest'):
    if word_width<=factorial_bits-1:
        raise ValueError('Not enough word width %d for factorial bits %d'%(word_width,factorial_bits))
    
    if fault_bit<0 or fault_bit>word_width or not isinstance(fault_bit,int):
        raise ValueError('Fault bit must be integer between (include) %d and 0, %d is MSB, 0 is LSB.'%(word_width-1,word_width-1))
        
    if stuck_at!='1' and stuck_at!='0' and stuck_at!='flip':
        raise ValueError('You must stuck at \'0\' , \'1\' or \'flip\'.')
        
    fault_value=quantize_1half(original_value, nb = word_width, fb = factorial_bits, rounding_method = rounding)
    fault_value=fault_value.astype(int)
    
    if stuck_at=='1':
        modulator=np.left_shift(1,fault_bit)
        fault_value=np.bitwise_or(fault_value,modulator)
    elif stuck_at=='0':
        modulator=-(np.left_shift(1,fault_bit)+1)
        fault_value=np.bitwise_and(fault_value,modulator)
    elif stuck_at=='flip':
        modulator=np.left_shift(1,fault_bit)
        fault_value=np.bitwise_xor(fault_value,modulator)
        
    fault_value=quantize_2half(fault_value, nb = word_width, fb = factorial_bits)
    
    return fault_value
    
