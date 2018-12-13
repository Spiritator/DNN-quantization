# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 13:16:26 2018

@author: Yung-Yu Tsai

Estimation for approximate computing
"""

import numpy as np

def comp_num_estimate(model):
    '''
    Input model: quantized keras model for number of computation estimation
    '''
    estimation_report=dict()
    layer_list=model.layers
    total_mult=0
    total_accum=0
    total_mac=0
    for layer in layer_list:
        layer_config=layer.get_config()
        if len(layer.weights) != 0:
            if 'conv' in layer.name:
                if layer_config['data_format']=='channels_last':
                    mult_num=np.prod(layer.output_shape[1:3])*np.prod(layer.weights[0].shape)
                    accum_num=np.prod(layer.output_shape[1:3])*(np.prod(layer.weights[0].shape[:-1])-1)*layer.weights[0].shape[-1]
                elif layer_config['data_format']=='channels_first':
                    mult_num=np.prod(layer.output_shape[2:])*np.prod(layer.weights[0].shape)
                    accum_num=np.prod(layer.output_shape[2:])*(np.prod(layer.weights[0].shape[:-1])-1)*layer.weights[0].shape[-1]
                
                if layer_config['use_bias']:
                    accum_num+=np.prod(layer.output_shape[1:])
                
            elif 'dense' in layer.name:
                mult_num=np.prod(layer.weights[0].shape)
                accum_num=(layer.weights[0].shape[0]-1)*layer.weights[0].shape[1]
                
                if layer_config['use_bias']:
                    accum_num+=layer.output_shape[-1]
                    
            estimation_report[layer.name]={'multiplication':mult_num.value,'accumulation':accum_num.value,'total_computation':mult_num.value+accum_num.value}
            total_mult+=mult_num.value
            total_accum+=accum_num.value
            total_mac+=mult_num.value+accum_num.value
            
    estimation_report['total_multiplication']=total_mult
    estimation_report['total_accumulation']=total_accum
    estimation_report['total_MAC']=total_mac
    
    return estimation_report
            
            
