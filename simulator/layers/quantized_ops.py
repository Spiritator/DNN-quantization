# -*- coding: utf-8 -*-

'''
reference: https://github.com/BertMoons/QuantizedNeuralNetworks-Keras-Tensorflow
all the credit refer to BertMoons on QuantizedNeuralNetworks-Keras-Tensorflow

@author: Yung-Yu Tsai

'''

from __future__ import absolute_import
import keras.backend as K
import tensorflow as tf
import numpy as np

class quantizer:
    def __init__(self,nb,fb,rounding_method='nearest',overflow_mode='saturation',stop_gradient=False):
        self.nb=nb
        self.fb=fb
        self.rounding_method=rounding_method
        self.overflow_mode=overflow_mode
        self.stop_gradient=stop_gradient

    def round_through(self,x):
        '''Element-wise rounding to the closest integer with full gradient propagation.
        A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)
        '''
        if self.rounding_method == 'nearest':
            rounded = tf.rint(x)
        elif self.rounding_method == 'zero':
            rounded = tf.trunc(x)
        elif self.rounding_method == 'down':
            rounded = tf.floor(x)
        elif self.rounding_method == 'stochastic':
            if tf.average(x-tf.floor(x)).eval() > 0.5:
                rounded = tf.ceil(x)
            else:
                rounded = tf.floor(x)
        else:
            print('Wrong Rounding Type\nChoose between \'nearest\' , \'zero\' , \'down\'')
            
        rounded_through = x + K.stop_gradient(rounded - x)
        return rounded_through
    
    
    def clip_through(self, x, min_val, max_val):
        '''Element-wise clipping with gradient propagation
        Analogue to round_through
        '''
        clipped = K.clip(x, min_val, max_val)
        clipped_through= x + K.stop_gradient(clipped-x)
        return clipped_through 
    
    
    def quantize(self, X, clip_through=None):
    
        '''The weights' binarization function, 
    
        # Reference:
        - [QuantizedNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}
    
        '''
        if clip_through is None:
            clip_through=self.stop_gradient
        #sess = tf.InteractiveSession()
        non_sign_bits = self.nb-self.fb-1
        m = K.pow(2.,self.fb)
        #W = tf.constant(W)
        Xq = tf.multiply(X,m)
        if clip_through:
            Xq = clip_through(tf.divide(self.round_through(Xq),m),-np.power(2,non_sign_bits), np.power(2,non_sign_bits)-np.power(0.5,self.fb))    
        else:
            Xq = K.clip(tf.divide(self.round_through(Xq),m),-np.power(2,non_sign_bits), np.power(2,non_sign_bits)-np.power(0.5,self.fb))
            
        return Xq
        
        
    def quantize_1half(self, X, clip_through=False):
    
        '''The weights' binarization function, 
    
        # Reference:
        - [QuantizedNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}
    
        '''
        if clip_through is None:
            clip_through=self.stop_gradient
        #sess = tf.InteractiveSession()
        non_sign_bits = self.nb-self.fb-1
        m = pow(2.,self.fb)
        #W = tf.constant(W)
        Xq = tf.multiply(X,m)
        if clip_through:
            Xq = clip_through(self.round_through(Xq),-np.power(2,non_sign_bits)*m, (np.power(2,non_sign_bits)-np.power(0.5,self.fb))*m)    
        else:
            Xq = K.clip(self.round_through(Xq),-np.power(2,non_sign_bits)*m, (np.power(2,non_sign_bits)-np.power(0.5,self.fb))*m)
        return Xq
    
    def quantize_2half(self, X):
    
        '''The weights' binarization function, 
    
        # Reference:
        - [QuantizedNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}
    
        '''
        
        #sess = tf.InteractiveSession()
        m = pow(2.,self.fb)
        #W = tf.constant(W)
        Xq = tf.divide(X,m)       
        return Xq
    
