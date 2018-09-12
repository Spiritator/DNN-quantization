# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 16:39:57 2018

@author: Yung-Yu Tsai

weight fault injection
"""

import keras
from keras.models import Sequential, Model
import numpy as np

def generate_single_stuck_at_fault(original_value,word_width,factorial_bits,fault_bit,stuck_at):
    if word_width<=factorial_bits-1:
        raise ValueError('Not enough word width %d for factorial bits %d'%(word_width,factorial_bits))
    
    if fault_bit<0 or fault_bit>word_width or isinstance(fault_bit,int):
        raise ValueError('Fault bit must be integer between (include) %d and 0. %d is MSB, 0 is LSB.'%(word_width-1,word_width-1))
        
    if stuck_at!=1 or stuck_at!=0:
        raise ValueError('You must stuck at 0 or 1.')
