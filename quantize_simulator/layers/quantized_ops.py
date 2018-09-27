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


def round_through(x,rounding_method):
    '''Element-wise rounding to the closest integer with full gradient propagation.
    A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)
    '''
    if rounding_method == 'nearest':
        rounded = tf.round(x)
    elif rounding_method == 'zero':
        rounded = tf.trunc(x)
    elif rounding_method == 'down':
        rounded = tf.floor(x)
    elif rounding_method == 'stochastic':
        if tf.average(x-tf.floor(x)).eval() > 0.5:
            rounded = tf.ceil(x)
        else:
            rounded = tf.floor(x)
    else:
        print('Wrong Rounding Type\nChoose between \'nearest\' , \'zero\' , \'down\'')
        
    '''older version of non parallel computing version'''            
#    if rounding_method == 'nearest':
#        rounded = K.round(x)
#    elif rounding_method == 'zero':
#        rounded = np.trunc(x)
#    elif rounding_method == 'down':
#        rounded = np.floor(x)
#    elif rounding_method == 'stochastic':
#        if np.average(x-np.floor(x)) > 0.5:
#            rounded = np.ceil(x)
#        else:
#            rounded = np.floor(x)
    rounded_through = x + K.stop_gradient(rounded - x)
    return rounded_through


def clip_through(x, min_val, max_val):
    '''Element-wise clipping with gradient propagation
    Analogue to round_through
    '''
    clipped = K.clip(x, min_val, max_val)
    clipped_through= x + K.stop_gradient(clipped-x)
    return clipped_through 


def clip_through(x, min, max):
    '''Element-wise rounding to the closest integer with full gradient propagation.
    A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)
    '''
    clipped = K.clip(x,min,max)
    return x + K.stop_gradient(clipped - x)


def _hard_sigmoid(x):
    '''Hard sigmoid different from the more conventional form (see definition of K.hard_sigmoid).

    # Reference:
    - [QuantizedNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}

    '''
    return K.clip((x+1)/2, 0, 1)




def quantize(W, nb = 16, fb = 8, rounding_method = 'nearest', clip_through=False):

    '''The weights' binarization function, 

    # Reference:
    - [QuantizedNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}

    '''
    
    #sess = tf.InteractiveSession()
    non_sign_bits = nb-fb-1
    m = K.pow(2.,fb)
    #W = tf.constant(W)
    Wq = tf.multiply(W,m)
    if clip_through:
        Wq = clip_through(tf.divide(round_through(Wq,rounding_method),m),-np.power(2,non_sign_bits), np.power(2,non_sign_bits)-np.power(0.5,fb))    
    else:
        Wq = K.clip(tf.divide(round_through(Wq,rounding_method),m),-np.power(2,non_sign_bits), np.power(2,non_sign_bits)-np.power(0.5,fb))
        
    return Wq
    
    '''Older version of non parallel computing code'''
    #    non_sign_bits = nb-fb-1
    #    m = pow(2,fb)
    #    Wq = W*m
    #    #W = tf.Print(W,[W],summarize=20)
    #    if clip_through:
    #        Wq = clip_through(round_through(Wq,rounding_method)/m,-np.power(2,non_sign_bits), np.power(2,non_sign_bits)-np.power(0.5,fb))
    #        
    #    else:
    #        Wq = K.clip(round_through(Wq,rounding_method)/m,-np.power(2,non_sign_bits), np.power(2,non_sign_bits)-np.power(0.5,fb))
    #    #Wq = tf.Print(Wq,[Wq],summarize=20)
    #    return Wq
    
def quantize_1half(W, nb = 16, fb = 8, rounding_method = 'nearest', clip_through=False):

    '''The weights' binarization function, 

    # Reference:
    - [QuantizedNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}

    '''
    
    #sess = tf.InteractiveSession()
    non_sign_bits = nb-fb-1
    m = pow(2.,fb)
    #W = tf.constant(W)
    Wq = tf.multiply(W,m)
    if clip_through:
        Wq = clip_through(round_through(Wq,rounding_method),-np.power(2,non_sign_bits)*m, (np.power(2,non_sign_bits)-np.power(0.5,fb))*m)    
    else:
        Wq = K.clip(round_through(Wq,rounding_method),-np.power(2,non_sign_bits)*m, (np.power(2,non_sign_bits)-np.power(0.5,fb))*m)
    return Wq

def quantize_2half(W, nb = 16, fb = 8):

    '''The weights' binarization function, 

    # Reference:
    - [QuantizedNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}

    '''
    
    #sess = tf.InteractiveSession()
    m = pow(2.,fb)
    #W = tf.constant(W)
    Wq = tf.div(W,m)       
    return Wq


def quantized_relu(W, nb=16):

    '''The weights' binarization function, 

    # Reference:
    - [QuantizedNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}

    '''
    #non_sign_bits = nb-1
    #m = pow(2,non_sign_bits)
    #Wq = K.clip(round_through(W*m),0,m-1)/m

    nb_bits = nb
    Wq = K.clip(2. * (round_through(_hard_sigmoid(W) * pow(2, nb_bits)) / pow(2, nb_bits)) - 1., 0,
                1 - 1.0 / pow(2, nb_bits - 1))
    return Wq


def quantized_tanh(W, nb=16):

    '''The weights' binarization function,

    # Reference:
    - [QuantizedNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}

    '''
    non_sign_bits = nb-1
    m = pow(2,non_sign_bits)
    #W = tf.Print(W,[W],summarize=20)
    Wq = K.clip(round_through(W*m),-m,m-1)/m
    #Wq = tf.Print(Wq,[Wq],summarize=20)
    return Wq

def quantized_leakyrelu(W, nb=16, alpha=0.1):

    '''The weights' binarization function, 

    # Reference:
    - [QuantizedNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}

    '''
    if alpha != 0.:
        negative_part = tf.nn.relu(-W)
    W = tf.nn.relu(W)
    if alpha != 0.:
        alpha = tf.cast(tf.convert_to_tensor(alpha), W.dtype.base_dtype)
        W -= alpha * negative_part

    non_sign_bits = nb-1
    m = pow(2,non_sign_bits)
    #W = tf.Print(W,[W],summarize=20)
    Wq = clip_through(round_through(W*m),-m,m-1)/m
    #Wq = tf.Print(Wq,[Wq],summarize=20)

    return Wq

def quantized_maxrelu(W, nb=16):

    '''The weights' binarization function, 

    # Reference:
    - [QuantizedNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}

    '''
    non_sign_bits = nb-1
    max_ = tf.reduce_max((W))
    #max_ = tf.Print(max_,[max_])
    max__ = tf.pow(2.0,tf.ceil(tf.log(max_)/tf.log(tf.cast(tf.convert_to_tensor(2.0), W.dtype.base_dtype))))
    #max__ = tf.Print(max__,[max__])
    m = pow(2,non_sign_bits)
    #W = tf.Print(W,[W],summarize=20)
    Wq = max__*clip_through(round_through(W/max__*(m)),0,m-1)/(m)
    #Wq = tf.Print(Wq,[Wq],summarize=20)

    return Wq

def quantized_leakymaxrelu(W, nb=16, alpha=0.1):

    '''The weights' binarization function, 

    # Reference:
    - [QuantizedNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}

    '''
    if alpha != 0.:
        negative_part = tf.nn.relu(-W)
    W = tf.nn.relu(W)
    if alpha != 0.:
        alpha = tf.cast(tf.convert_to_tensor(alpha), W.dtype.base_dtype)
        W -= alpha * negative_part

    max_ = tf.reduce_max((W))
    #max_ = tf.Print(max_,[max_])
    max__ = tf.pow(2.0,tf.ceil(tf.log(max_)/tf.log(tf.cast(tf.convert_to_tensor(2.0), W.dtype.base_dtype))))
    #max__ = tf.Print(max__,[max__])

    non_sign_bits = nb-1
    m = pow(2,non_sign_bits)
    #W = tf.Print(W,[W],summarize=20)
    Wq = max__* clip_through(round_through(W/max__*m),-m,m-1)/m
    #Wq = tf.Print(Wq,[Wq],summarize=20)

    return Wq



def _mean_abs(x, axis=None, keepdims=False):
    return K.stop_gradient(K.mean(K.abs(x), axis=axis, keepdims=keepdims))

    
def xnorize(W, H=1., axis=None, keepdims=False):
    Wb = quantize(W, H)
    Wa = _mean_abs(W, axis, keepdims)
    
    return Wa, Wb
