# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 11:31:34 2019

@author: Yung-Yu Tsai

Modify existing models
"""

from tensorflow.keras.models import Model
import numpy as np

from ..layers.quantized_layers import QuantizedDistributedConv2D
from tensorflow.keras.layers import Activation, Add

def exchange_distributed_conv(model,target_layer_num,fault_dict_conversion,split_type,splits,ifmap_fault_dict_list=None,ofmap_fault_dict_list=None,wght_fault_dict_list=None):
    """Swap original DNN model layers to distributed convolution for emulate hardware partial sum.

    # Arguments
        model: Keras model. The model wanted to be swapped.
        target_layer_num: Integer or List. The layers that will be swapped by distributed convolution layer.
        fault_dict_conversion: Bool. Whether convert the original layer fault dict to distributed convolution or not.
        
        split_type: String or List of String. Choose from 'channel', 'k_height' (kernel_height), 'k_width' (kernel_width), 'k_seq' (kernel_sequential).
            'k_seq' can't coexist with 'k_height' or 'k_width'.
        
        splits: 
            For one single split the splits argument will be an Integer or List of Integer. 
            The argument for setting splits on channel axis.
            Either a 0-D integer `Tensor` indicating the number of splits 
            along split_dim or a 1-D integer `Tensor` containing the sizes of 
            each output tensor along split_dim. If a scalar then it must evenly
            divide `value.shape[axis]`; otherwise the sum of sizes along the 
            split dimension must match that of the `value`.
            
            For splits on multiple splits typesthe splits argument will be List of (Integer or List of Integer).
            List length is the number of split types permute according to the split_type list order.
            
            For split on multiple layers the splits argument will be List of (List of (Integer or List of Integer)).
            List length of first level is the number of target layers.
            List length of second level is the number of split types permute according to the split_type list order.
            
        ifmap_fault_dict_list: List of Dictionarys. The fault dictionary list for input feature maps.
        ofmap_fault_dict_list: List of Dictionarys. The fault dictionary list for output feature maps.
        wght_fault_dict_list: List of Dictionarys. The fault dictionary list for weights.

    # Returns
        A Model, result of distributed convolution swap.
    """
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
                                           split_type=split_type,
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
    
    
class pseudo_model:
    '''The class like Keras Model for fault generation.
        Only store layer Shape information.
    
    '''
    def __init__(self,layers=None):
        if layers is not None:
            self.layers=layers
        else:
            self.layers=list()
    
class pseudo_layer:
    '''The class like Keras Layer for fault generation.
        Only store layer Shape information.
    
    '''
    def __init__(self,input_shape,output_shape,weight_shape):
        self.input_shape=input_shape
        self.output_shape=output_shape
        self.weight_shape=weight_shape
        
    def get_weights(self):
        weights=list()
        for shape in self.weight_shape:
            weights.append(np.zeros(shape))
        return weights
        
def make_ref_model(model):
    '''Make a reference model of psuedo_model class.
        With only layer shape information for fault generation.
        In case of K.clear_session() commmand clear tensor record.
        Reference model provide shapes for multiple iterative fault generation.
    
    '''
    layer_list=list()
    for i in range(len(model.layers)):
        layer=model.layers[i]
        ref_layer=pseudo_layer(layer.input_shape,layer.output_shape,[weight_shape.shape for weight_shape in layer.get_weights()])
        layer_list.append(ref_layer)
        
    ref_model=pseudo_model(layer_list)
    return ref_model
