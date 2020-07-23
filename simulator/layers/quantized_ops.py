# -*- coding: utf-8 -*-

'''
reference: https://github.com/BertMoons/QuantizedNeuralNetworks-Keras-Tensorflow
all the credit refer to BertMoons on QuantizedNeuralNetworks-Keras-
    # Reference:
    - [QuantizedNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}

@author: Yung-Yu Tsai

'''

from __future__ import absolute_import
import keras.backend as K
import tensorflow as tf
import numpy as np

class quantizer:
    def __init__(self,nb,fb,rounding_method='nearest',overflow_mode=False,stop_gradient=False):
        """ Setup fixed-point quantization parameter

        # Arguments
            nb: Integer. The word-length of this fixed-point number.
            fb: Integer. The number fractional bits in this fixed-point number.
            rounding_method: String. Rounding method of quantization, augment must be one of 'nearest' , 'down', \'zero\', 'stochastic'. 
            overflow_mode: Bool. The method of handle overflow and underflow simulation.
                If True, the overflow and underflow value will wrap-around like in fixed-point number in RTL description.
                Else False, the overflow and underflow value will saturate at the max and min number this fixed-point number can represent.
            stop_gradient: Bool. Whether to let the gradient pass through the quantization function or not.
        
        """
        if not isinstance(nb,int) or not isinstance(fb,int):
            raise ValueError('The word width and fractional bits augment must be integer type!')
        if nb<=fb-1:
            raise ValueError('Not enough word width %d for fractional bits %d'%(nb,fb))
        self.nb=nb
        self.fb=fb
        self.rounding_method=rounding_method
        self.overflow_mode=overflow_mode
        self.stop_gradient=stop_gradient
        
        self.shift_factor = np.power(2.,fb)
        self.min_value=-np.power(2,nb-fb-1)
        self.max_value=np.power(2,nb-fb-1)-np.power(0.5,fb)
        self.ovf_val=np.power(2,nb-1)
        self.ovf_capper=np.power(2,nb)

    def round_through(self,x):
        '''Element-wise rounding to the closest integer with full gradient propagation.
        A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)
        '''
        def ceil_fn():
            return tf.ceil(x)
        
        def floor_fn():
            return tf.floor(x)

        
        if self.rounding_method == 'nearest':
            rounded = tf.rint(x)
        elif self.rounding_method == 'down':
            rounded = tf.floor(x)
        elif self.rounding_method == 'stochastic':
            rounded=tf.cond(tf.greater(tf.reduce_mean(x-tf.floor(x)), 0.5), ceil_fn, floor_fn)
        elif self.rounding_method == 'zero':
            neg_alter=tf.add(tf.multiply(tf.cast(tf.less(x,0),'float32'),-2.0),1.0)
            rounded=tf.multiply(tf.floor(tf.multiply(x,neg_alter)),neg_alter)
        else:
            print('Wrong Rounding Type\nChoose between \'nearest\' , \'down\', \'zero\', \'stochastic\' ')
            
        rounded_through = x + K.stop_gradient(rounded - x)
        return rounded_through
    
    
    def clip_through(self, x, min_val, max_val):
        '''Element-wise clipping with gradient propagation
        Analogue to round_through
        '''
        clipped = K.clip(x, min_val, max_val)
        clipped_through= x + K.stop_gradient(clipped-x)
        return clipped_through 
    
    
    def quantize(self, X, clip_through=None, overflow_sim=None):
        """ Quantize input X data """
        if clip_through is None:
            clip_through=self.stop_gradient
        if overflow_sim is None:
            overflow_sim=self.overflow_mode
        #sess = tf.InteractiveSession()
        Xq = tf.multiply(X,self.shift_factor)
        Xq = self.round_through(Xq)
        
        if not overflow_sim:
            if clip_through:
                Xq = clip_through(tf.divide(Xq,self.shift_factor), self.min_value, self.max_value)    
            else:
                Xq = K.clip(tf.divide(Xq,self.shift_factor), self.min_value, self.max_value)
        else:
            Xq=tf.add(Xq,self.ovf_val)
            Xq=tf.floormod(Xq,self.ovf_capper)
            Xq=tf.subtract(Xq,self.ovf_val)
            
            Xq=tf.divide(Xq,self.shift_factor)
            
        return Xq
        
        
    def left_shift_2int(self, X):
        """ 
        Shift left to the integer interval. 
        Notice that the input X should be previously quantized.
        Or else, there might be unexpected fault.
        Shifted data was casted to integer type for bitwise operation.
        Which should be right_shift_back to its original fixed-point state.
        """
        Xq = tf.multiply(X,self.shift_factor)
        Xq = tf.cast(Xq,tf.int32)
            
        return Xq
    
    def right_shift_back(self, X):
        """ 
        Shift back to fixed-point data decimal point with fractional value.
        Reverse the left_shift_2int function.
        """
        
        Xq = tf.cast(X,tf.float32)   
        Xq = tf.divide(Xq,self.shift_factor)
        
        return Xq
    

def build_layer_quantizer(nbits,fbits,rounding_method,overflow_mode,stop_gradient):
    """ Layer quantizer builder. For generate different setup for ifmap, weight, ofmap individually. """
    multi_setting=False
    
    if isinstance(nbits,list) or isinstance(fbits,list) or isinstance(rounding_method,list) or isinstance(overflow_mode,list) or isinstance(stop_gradient,list):
        multi_setting=True
        
    if isinstance(nbits,list) and len(nbits)==3:
        nb_qt=nbits
    elif multi_setting:
        nb_qt=[nbits, nbits, nbits]
        
    if isinstance(fbits,list) and len(fbits)==3:
        fb_qt=fbits
    elif multi_setting:
        fb_qt=[fbits, fbits, fbits]

    
    if isinstance(rounding_method,list) and len(rounding_method)==3:
        rm_qt=rounding_method
    elif multi_setting:
        rm_qt=[rounding_method, rounding_method, rounding_method]
        
    if isinstance(overflow_mode,list) and len(overflow_mode)==3:
        ovf_qt=overflow_mode
    elif multi_setting:
        ovf_qt=[overflow_mode, overflow_mode, overflow_mode]
        
    if multi_setting:
        return [quantizer(nb_qt[0],fb_qt[0],rm_qt[0],ovf_qt[0],stop_gradient),
                quantizer(nb_qt[1],fb_qt[1],rm_qt[1],ovf_qt[1],stop_gradient),
                quantizer(nb_qt[2],fb_qt[2],rm_qt[2],ovf_qt[2],stop_gradient)]
    else:
        return quantizer(nbits,fbits,rounding_method,overflow_mode,stop_gradient)
        
        
        
