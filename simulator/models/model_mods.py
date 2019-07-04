# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 11:31:34 2019

@author: Yung-Yu Tsai

Modify existing models
"""

import keras
from keras.models import Model
import numpy as np

from ..layers.quantized_layers import QuantizedDistributedConv2D
from keras.layers import Activation, Add

def exchange_distributed_conv(model,target_layer_num,splits,fault_dict_conversion=False,ifmap_fault_dict_list=None,ofmap_fault_dict_list=None,wght_fault_dict_list=[None,None]):
    layers = [l for l in model.layers]
    if isinstance(target_layer_num,int):
        target_layer_num=[target_layer_num]
        splits=[splits]
    
    x = layers[0].output
    for i in range(1, len(layers)):
        if i in target_layer_num:
            original_layer=layers[i]
            splits_tmp=splits[target_layer_num.index(i)]
            
            if fault_dict_conversion:
                ifmap_fault_dict_list=original_layer.ifmap_sa_fault_injection
                wght_fault_dict_list=original_layer.weight_sa_fault_injection
                if original_layer.ofmap_sa_fault_injection is None:
                    ofmap_fault_dict_list=None
                else:
                    if isinstance(splits_tmp,int):
                        ofmap_fault_dict_list=[original_layer.ofmap_sa_fault_injection for i in range(splits_tmp)]
                    elif isinstance(splits_tmp,list):
                        ofmap_fault_dict_list=[original_layer.ofmap_sa_fault_injection for i in range(len(splits_tmp))]
            
            x = QuantizedDistributedConv2D(filters=original_layer.filters,
                                           splits=splits_tmp,
                                           quantizers=original_layer.quantizer,
                                           kernel_size=original_layer.kernel_size,
                                           padding=original_layer.padding,
                                           strides=original_layer.strides,
                                           use_bias=original_layer.use_bias,
                                           name=original_layer.name,
                                           ifmap_sa_fault_injection=ifmap_fault_dict_list,
                                           ofmap_sa_fault_injection=ofmap_fault_dict_list,
                                           weight_sa_fault_injection=wght_fault_dict_list,
                                           quant_mode=original_layer.quant_mode)(x)
            x = Add()(x)
            x = Activation(original_layer.activation)(x)
            
        else:
            x = layers[i](x)

    new_model = Model(inputs=layers[0].input, outputs=x)
    return new_model
    
    
    
