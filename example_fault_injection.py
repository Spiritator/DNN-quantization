# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 16:09:29 2018

@author: Yung-Yu Tsai

fault injection test
"""

import tensorflow as tf
import numpy as np
from simulator.fault.fault_core import generate_single_stuck_at_fault, generate_multiple_stuck_at_fault
from simulator.fault.fault_ops import inject_layer_sa_fault_tensor, inject_layer_sa_fault_nparray
from simulator.layers.quantized_ops import quantizer
#%%
####################
##  Quantization  ##
####################

# declare qunatizer setting
qtn=quantizer(7,2,rounding_method='nearest')
qtd=quantizer(7,2,rounding_method='down')
qts=quantizer(7,2,rounding_method='stochastic')
qtz=quantizer(7,2,rounding_method='zero')

unquant_param=np.reshape((np.arange(1,1001,dtype='float32')-500)/16,(-1,10))

quant_param_nearest=qtn.quantize(unquant_param).numpy()
quant_param_down=qtd.quantize(unquant_param).numpy()
quant_param_stochastic=qts.quantize(unquant_param).numpy()
quant_param_zero=qtz.quantize(unquant_param).numpy()


#%%
#####################
## Fault Injection ##
#####################

# declare qunatizer setting
qt=quantizer(10,3,rounding_method='nearest')

# a numpy array of 0 ~ 99
original_weight=np.arange(1,100,dtype='float32')
original_weight=qt.quantize(original_weight).numpy()

# inject single SA fault to a parameter
single_fault_weight=generate_single_stuck_at_fault(original_weight,3,'1',qt,tensor_return=False)

# inject multiple SA fault to a parameter
multiple_fault_weight=generate_multiple_stuck_at_fault(original_weight,[3,2],['1','1'],qt,tensor_return=False)


# the Tensor of original parameter
layer_original_array_pos=np.reshape(np.arange(1,101,dtype='float32'), (10,10))
layer_original_weight_pos=tf.Variable(layer_original_array_pos)
#layer_original_fmap_pos=tf.data()
layer_original_array_neg=np.reshape(np.arange(1,101,dtype='float32')*(-1), (10,10))
layer_original_weight_neg=tf.Variable(layer_original_array_neg)
#layer_original_fmap_neg=tf.data()

# overflow simulation
layer_overflow_sim_pos=qt.quantize(tf.subtract(layer_original_weight_pos,0.5),overflow_sim=True)
layer_overflow_sim_neg=qt.quantize(tf.add(layer_original_weight_neg,0.5),overflow_sim=True)
layer_overflow_input_pos=layer_overflow_sim_pos.numpy()
layer_overflow_input_neg=layer_overflow_sim_neg.numpy()

# example of fault dictionary
fault_dict={(1,6):\
            {'SA_type':'1',
             'SA_bit':2},
            (1,4):\
            {'SA_type':'0',
             'SA_bit':3},
            (0,1):\
            {'SA_type':['1','flip'],
             'SA_bit':[3,2]},
            (0,5):\
            {'SA_type':['1','flip'],
             'SA_bit':[3,2]},
            (0,8):\
            {'SA_type':['0','flip'],
             'SA_bit':[3,2]}
            }
            
# inject fault to a numpy array
layer_fault_array_pos=inject_layer_sa_fault_nparray(layer_original_array_pos,fault_dict,qt)
layer_fault_array_neg=inject_layer_sa_fault_nparray(layer_original_array_neg,fault_dict,qt)

# inject fault to a Tensor and Variable
@tf.function
def fault_inject(faultdict,wghtpos,wghtneg,fmappos,fmapneg,ovf):
    qtzr=quantizer(10,3,rounding_method='nearest',overflow_mode=ovf)
    
    cnst=tf.constant(1.)
    faultfmappos=tf.add(fmappos,cnst)
    faultfmapneg=tf.subtract(fmapneg,cnst)

    faultwghtpos=qtzr.quantize(wghtpos)
    faultwghtneg=qtzr.quantize(wghtneg)
    faultfmappos=qtzr.quantize(faultfmappos)
    faultfmapneg=qtzr.quantize(faultfmapneg)
    
    faultwghtpos=inject_layer_sa_fault_tensor(faultwghtpos,faultdict,qtzr)
    faultwghtneg=inject_layer_sa_fault_tensor(faultwghtneg,faultdict,qtzr)
    faultfmappos=inject_layer_sa_fault_tensor(faultfmappos,faultdict,qtzr)
    faultfmapneg=inject_layer_sa_fault_tensor(faultfmapneg,faultdict,qtzr)

    return faultwghtpos,faultwghtneg,faultfmappos,faultfmapneg


layer_fault_weight_pos,layer_fault_weight_neg,layer_fault_fmap_pos,layer_fault_fmap_neg=\
    fault_inject(fault_dict, 
                 layer_original_weight_pos, layer_original_weight_neg, 
                 layer_original_array_pos, layer_original_array_neg, 
                 ovf=False)
    
layer_fault_weight_pos=layer_fault_weight_pos.numpy()
layer_fault_weight_neg=layer_fault_weight_neg.numpy()
layer_fault_fmap_pos=layer_fault_fmap_pos.numpy()
layer_fault_fmap_neg=layer_fault_fmap_neg.numpy()

#%% inject fault to a Tensor with overflow simulation

layer_fault_weight_pos_ovf,layer_fault_weight_neg_ovf,layer_fault_fmap_pos_ovf,layer_fault_fmap_neg_ovf=\
    fault_inject(fault_dict, 
                 layer_original_weight_pos, layer_original_weight_neg, 
                 layer_original_array_pos, layer_original_array_neg,
                 ovf=True)

layer_fault_weight_pos_ovf=layer_fault_weight_pos_ovf.numpy()
layer_fault_weight_neg_ovf=layer_fault_weight_neg_ovf.numpy()
layer_fault_fmap_pos_ovf=layer_fault_fmap_pos_ovf.numpy()
layer_fault_fmap_neg_ovf=layer_fault_fmap_neg_ovf.numpy()
            