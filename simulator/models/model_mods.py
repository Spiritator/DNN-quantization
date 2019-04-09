# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 11:31:34 2019

@author: Yung-Yu Tsai

Modify existing models
"""

import keras
from keras.models import Model
import numpy as np

from layers.quantized_layers import QuantizedDistributedConv2D

def exchange_distributed_conv(model,target_layer_num,splits,fault_dict_list):
    node_upstream=model.layers[target_layer_num-1].output
    node_downstream=model.layers[target_layer_num+1].input
    
    node_downstream=QuantizedDistributedConv2D()(node_upstream)
