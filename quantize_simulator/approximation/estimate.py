# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 13:16:26 2018

@author: Yung-Yu Tsai

Estimation for approximate computing
"""

import numpy as np

def comp_num_estimate(model,add_topo=None):
    '''
    Inputs
    model: quantized keras model for number of computation estimation
    add_topo: adder topology setting default=None (worst case)
    '''
    estimation_report=dict()
    layer_list=model.layers
    total_mult=0
    total_accum=0
    total_mac=0
    total_mult_bit=0
    total_accum_bit=0
    total_mac_bit=0
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
                
                if 'nb' in layer_config.keys():
                    if isinstance(layer_config['nb'],list) and isinstance(layer_config['fb'],list) and len(layer_config['nb'])==3 and len(layer_config['fb'])==3:
                        mult_bit=mult_num*layer_config['nb'][0]*layer_config['nb'][1]
                        accum_bit=accum_num*layer_config['nb'][0]*layer_config['nb'][1]
                    else:
                        mult_bit=mult_num*layer_config['nb']*layer_config['nb']
                        accum_bit=accum_num*layer_config['nb']*layer_config['nb']
                
                if layer_config['use_bias']:
                    accum_num+=np.prod(layer.output_shape[1:])
                    if 'nb' in layer_config.keys():
                        if isinstance(layer_config['nb'],list) and isinstance(layer_config['fb'],list) and len(layer_config['nb'])==3 and len(layer_config['fb'])==3:
                            accum_bit+=np.prod(layer.output_shape[1:])*layer_config['nb'][2]
                        else:
                            accum_bit+=np.prod(layer.output_shape[1:])*layer_config['nb']
                
            elif 'dense' in layer.name:
                mult_num=np.prod(layer.weights[0].shape)
                accum_num=(layer.weights[0].shape[0]-1)*layer.weights[0].shape[1]
                
                if 'nb' in layer_config.keys():
                    if isinstance(layer_config['nb'],list) and isinstance(layer_config['fb'],list) and len(layer_config['nb'])==3 and len(layer_config['fb'])==3:
                        mult_bit=mult_num*layer_config['nb'][0]*layer_config['nb'][1]
                        accum_bit=accum_num*layer_config['nb'][0]*layer_config['nb'][1]
                    else:
                        mult_bit=mult_num*layer_config['nb']*layer_config['nb']
                        accum_bit=accum_num*layer_config['nb']*layer_config['nb']
                
                if layer_config['use_bias']:
                    accum_num+=layer.output_shape[-1]
                    if 'nb' in layer_config.keys():
                        if isinstance(layer_config['nb'],list) and isinstance(layer_config['fb'],list) and len(layer_config['nb'])==3 and len(layer_config['fb'])==3:
                            accum_bit+=np.prod(layer.output_shape[1:])*layer_config['nb'][2]
                        else:
                            accum_bit+=np.prod(layer.output_shape[1:])*layer_config['nb']
                    
            estimation_report[layer.name]={'multiplications':mult_num.value,'accumulations':accum_num.value,'total_computations':mult_num.value+accum_num.value}
            if 'nb' in layer_config.keys():
                estimation_report[layer.name]['mult_bits']=mult_bit.value
                estimation_report[layer.name]['accum_bits']=accum_bit.value
                estimation_report[layer.name]['total_bits']=mult_bit.value+accum_bit.value
            total_mult+=mult_num.value
            total_accum+=accum_num.value
            total_mac+=mult_num.value+accum_num.value
            total_mult_bit+=mult_bit.value
            total_accum_bit+=accum_bit.value
            total_mac_bit+=mult_bit.value+accum_bit.value
            
    estimation_report['total_multiplication']=total_mult
    estimation_report['total_accumulation']=total_accum
    estimation_report['total_MAC']=total_mac
    estimation_report['total_mult_bits']=total_mult_bit
    estimation_report['total_accum_bits']=total_accum_bit
    estimation_report['total_MAC_bits']=total_mac_bit
    
    return estimation_report
            
            
