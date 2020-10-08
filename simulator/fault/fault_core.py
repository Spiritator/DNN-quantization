# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 16:39:57 2018

@author: Yung-Yu Tsai

weight fault injection
"""

import numpy as np
import tensorflow as tf

def generate_single_stuck_at_fault(original_value,fault_bit,stuck_at,quantizer,tensor_return=True):
    """Returns the a tensor or variable with single SA fault injected in each parameter.

    Arguments
    ---------
    original_value: Tensor or Float. 
        The variable to be injected fault.
    quantizer: Class. 
        | The quantizer class contain following quantize operation infromation.
        | word_width: Integer. The fix-point representation of the parameter word length.
        | fractional_bits: Integer. Number of fractional bits in a fix-point parameter.
        | rounding: String. Rounding method of quantization, argument must be one of 'nearest' , 'down', \'zero\', 'stochastic'.
    fault_bit: Integer. 0 <= fault_bit < word length
        The index of the SA fault bit on a fix-point parameter.
    stuck_at: String. One of '1' , '0' or 'flip'.
        The SA type of the faulty bit, input argument must be one of '1' , '0' or 'flip'.
    tensor_return: Bool. 
        Return argument in Tensor or Ndarray.

    Returns
    -------
    A faulty Tensor or Numpy Array with single SA fault of each parameter.

    Examples
    --------
    ```python
    
        original_weight=np.arange(1,100,dtype='float32')
        single_fault_weight=generate_single_stuck_at_fault(original_weight,10,3,3,'1',tensor_return=False)
        
    ```
    """
    if quantizer.nb<=quantizer.fb-1:
        raise ValueError('Not enough word width %d for fractional bits %d'%(quantizer.nb,quantizer.fb))
    
    if fault_bit<0 or fault_bit>quantizer.nb or not isinstance(fault_bit,int):
        raise ValueError('Fault bit must be integer between (include) %d and 0, %d is MSB, 0 is LSB.'%(quantizer.nb-1,quantizer.nb-1))
        
    if stuck_at!='1' and stuck_at!='0' and stuck_at!='flip':
        raise ValueError('You must stuck at \'0\' , \'1\' or \'flip\'.')
        
    fault_value=quantizer.left_shift_2int(original_value)
    
    if stuck_at=='1':
        modulator=np.left_shift(1,fault_bit,dtype=np.int32)
        fault_value=tf.bitwise.bitwise_or(fault_value,modulator)
    elif stuck_at=='0':
        modulator=-(np.left_shift(1,fault_bit,dtype=np.int32)+1)
        fault_value=tf.bitwise.bitwise_and(fault_value,modulator)
    elif stuck_at=='flip':
        modulator=np.left_shift(1,fault_bit,dtype=np.int32)
        fault_value=tf.bitwise.bitwise_xor(fault_value,modulator)
    
    fault_value=quantizer.right_shift_back(fault_value)
    
    if tensor_return:
        return fault_value
    else:
        return fault_value.numpy()

def generate_multiple_stuck_at_fault(original_value,fault_bit,stuck_at,quantizer,tensor_return=True):
    """Returns the a tensor or variable with multiple SA fault injected in each parameter.

    Arguments
    ---------
    original_value: Tensor or Float. The variable to be injected fault.
    quantizer: Class. 
        | The quantizer class contain following quantize operation infromation.
        | word_width: Integer. The fix-point representation of the parameter word length.
        | fractional_bits: Integer. Number of fractional bits in a fix-point parameter.
        | rounding: String. Rounding method of quantization, argument must be one of 'nearest' , 'down', \'zero\', 'stochastic'.
    fault_bit: List of Integers. 
        The index of the SA fault bit on a fix-point parameter.
    stuck_at: List of String. 
        The SA type of the faulty bit, argument must be one of '1' , '0' or 'flip'.
    tensor_return: Bool. 
        Return argument in Tensor dtype or nparray.

    Returns
    -------
    A faulty Tensor or Numpy Array with multiple SA fault of each parameter.

    Examples
    --------
    ```python
    
        original_weight=np.arange(1,100,dtype='float32')
        multiple_fault_weight=generate_multiple_stuck_at_fault(original_weight,10,3,[3,2],['1','1'],tensor_return=False)
        
    ```
    """
    if quantizer.nb<=quantizer.fb-1:
        raise ValueError('Not enough word width %d for fractional bits %d'%(quantizer.nb,quantizer.fb))
    
    if any([fault_bit_iter<0 or fault_bit_iter>quantizer.nb or not isinstance(fault_bit_iter,int) for fault_bit_iter in fault_bit]):
        raise ValueError('Fault bit must be integer between (include) %d and 0, %d is MSB, 0 is LSB.'%(quantizer.nb-1,quantizer.nb-1))
        
    if any([stuck_at_iter!='1' and stuck_at_iter!='0' and stuck_at_iter!='flip' for stuck_at_iter in stuck_at]):
        raise ValueError('You must stuck at \'0\' , \'1\' or \'flip\'.')
        
    if len(fault_bit) != len(stuck_at):
        raise ValueError('Fault location list and stuck at type list must be the same length')
        
    fault_value=quantizer.left_shift_2int(original_value)
    
    modulator0=-1
    modulator1=0
    modulatorF=0
    for i in range(len(fault_bit)):
        if stuck_at[i]=='1':
            modulator=np.left_shift(1,fault_bit[i])
            modulator1=np.bitwise_or(modulator1,modulator,dtype=np.int32)
        elif stuck_at[i]=='0':
            modulator=-(np.left_shift(1,fault_bit[i])+1)
            modulator0=np.bitwise_and(modulator0,modulator,dtype=np.int32)
        elif stuck_at[i]=='flip':
            modulator=np.left_shift(1,fault_bit[i])
            modulatorF=np.bitwise_xor(modulatorF,modulator,dtype=np.int32)
    
    if modulator0 != -1:
        fault_value=tf.bitwise.bitwise_and(fault_value,modulator0)
    if modulator1 != 0:
        fault_value=tf.bitwise.bitwise_or(fault_value,modulator1)
    if modulatorF != 0:
        fault_value=tf.bitwise.bitwise_xor(fault_value,modulatorF)
    
    fault_value=quantizer.right_shift_back(fault_value)
    
    if tensor_return:
        return fault_value
    else:
        return fault_value.numpy()    
    
def generate_stuck_at_fault_modulator(word_width,fractional_bits,fault_bit,stuck_at):
    """ Returns the fault modulator of SA0, SA1 and invert bit.
        For loop based generation. One fault location at an iteration. Can have multiple faults on one parameter.

    Arguments
    ---------
    word_width: Integer. 
        The fix-point representation of the parameter word length.
    fractional_bits: Integer. 
        Number of fractional bits in a fix-point parameter.
    fault_bit: List of Integer. 
        The index of the SA fault bit on a fix-point parameter.
    stuck_at: List of String. '1' , '0' or 'flip'
        The SA type of the faulty bit, input argument must be one of '1' , '0' or 'flip'.

    Returns
    -------
    Tuple of Ndarrays. (modulator0, modulator1, modulatorF)
        The fault modulator of SA0, SA1 and invert bit respectively.
    """
    if isinstance(fault_bit,list):
        if any([fault_bit_iter<0 or fault_bit_iter>word_width or not isinstance(fault_bit_iter,int) for fault_bit_iter in fault_bit]):
            raise ValueError('Fault bit must be integer between (include) %d and 0, %d is MSB, 0 is LSB.'%(word_width-1,word_width-1))
            
        if any([stuck_at_iter!='1' and stuck_at_iter!='0' and stuck_at_iter!='flip' for stuck_at_iter in stuck_at]):
            raise ValueError('You must stuck at \'0\' , \'1\' or \'flip\'.')
    
        if len(fault_bit) != len(stuck_at):
            raise ValueError('Fault location list and stuck at type list must be the same length')
    
    else: 
        if fault_bit<0 or fault_bit>word_width:# or not isinstance(fault_bit,int):
            raise ValueError('Fault bit must be integer between (include) %d and 0, %d is MSB, 0 is LSB.'%(word_width-1,word_width-1))
            
        if stuck_at!='1' and stuck_at!='0' and stuck_at!='flip':
            raise ValueError('You must stuck at \'0\' , \'1\' or \'flip\'.')
    
    modulator0=-1
    modulator1=0
    modulatorF=0
    if isinstance(fault_bit,list):
        for i in range(len(fault_bit)):
            if stuck_at[i]=='1':
                modulator=np.left_shift(1,fault_bit[i])
                modulator1=np.bitwise_or(modulator1,modulator,dtype=np.int32)
            elif stuck_at[i]=='0':
                modulator=-(np.left_shift(1,fault_bit[i])+1)
                modulator0=np.bitwise_and(modulator0,modulator,dtype=np.int32)
            elif stuck_at[i]=='flip':
                modulator=np.left_shift(1,fault_bit[i])
                modulatorF=np.bitwise_or(modulatorF,modulator,dtype=np.int32)
    else:
        if stuck_at=='1':
            modulator1=np.left_shift(1,fault_bit,dtype=np.int32)
        elif stuck_at=='0':
            modulator0=-(np.left_shift(1,fault_bit,dtype=np.int32)+1)
        elif stuck_at=='flip':
            modulatorF=np.left_shift(1,fault_bit,dtype=np.int32)
            
    if modulator0==-1:
        modulator0=None
    if modulator1==0:
        modulator1=None
    if modulatorF==0:
        modulatorF=None
    
    return modulator0, modulator1, modulatorF

def generate_stuck_at_fault_modulator_fast(shape,coor,fault_type,fault_bit):
    """ Generates the fault modulator of SA0, SA1 and invert bit.
        Numpy array based generation. Create layer input, weight or output modulator at once. 
        Assume that only one fault on a parameter at once. 
        The fault type of this generation must be unified and specified.
        Therefore, this method is faster.

    Parameters
    ----------
    shape : Tuple of Integer
        The data shape of return modulator, the shape of data fault inject to.
    coor : List of Tuples of Integer or Ndarray
        | The coordinate of the fault location in data. Format:
        | List of Tuple : [(0,2,2,6),(3,5,4,2),...]
        | Ndarray : [[0,2,2,6],
        |            [3,5,4,2],
        |            ...]
    fault_type : String. One of '1' , '0' or 'flip'.
        The SA type of the faulty bit, input argument must be one of '1' , '0' or 'flip'.
    fault_bit : List or Ndarray. Each element 0 <= fault_bit < word length
        The index of the SA fault bit on a fix-point parameter.

    Returns
    -------
    tensor_modulator : Ndarray
        The modulator for parameter with given shape.

    """
    if len(coor) == 0:
        return None
    
    coor=np.array(coor)
    
    modulator=np.ones((coor.shape[0],),dtype=np.int32)
    modulator=np.left_shift(modulator,fault_bit)
    
    coor=np.transpose(coor)
    coor=tuple(np.split(coor,coor.shape[0],axis=0))
    
    if fault_type == '0':
        tensor_modulator=-np.ones(shape,dtype=np.int32)
        np.add.at(tensor_modulator,coor,modulator)
    elif fault_type == '1':
        tensor_modulator=np.zeros(shape,dtype=np.int32)
        np.add.at(tensor_modulator,coor,modulator)
    elif fault_type == 'flip':
        tensor_modulator=np.zeros(shape,dtype=np.int32)
        np.add.at(tensor_modulator,coor,modulator)
        
    return tensor_modulator

def generate_tensor_modulator(shape,nb,fb,fault_dict,fast_gen=False): 
    """ Generate modulator for a Tensor.
        The Tensor could be input, weight or output of a layer.
        Specify the generation method is numpy array based (fast gen) or for loop based.

    Parameters
    ----------
    shape :Tuple of Integer
        The data shape of data fault inject to.
    nb : Integer. 
        The fix-point representation of the parameter word length.
    fb : Integer. 
        Number of fractional bits in a fix-point parameter.
    fault_dict : Dictionary.
        The keys is fault location, value is fault information dictionary.
    fast_gen : Bool, optional
        Use numpy array based generation (fast gen) or not. The default is False.

    Returns
    -------
    List of Ndarray. [tensor_modulator0,tensor_modulator1,tensor_modulatorF]
        The tensor modulator for SA0, SA1, bit-flip respectively.

    """
    if len(fault_dict)==0:
        return [None,None,None]
    
    inject0=False
    inject1=False
    injectF=False
    
    if fast_gen:
        coor=list(fault_dict.keys())
        fault_type=fault_dict[coor[0]]['SA_type']
        fault_bit=[fault['SA_bit'] for fault in fault_dict.values()]
                
        if fault_type == '0':
            tensor_modulator0=generate_stuck_at_fault_modulator_fast(shape,coor,fault_type,fault_bit)
            inject0=True
        elif fault_type == '1':
            tensor_modulator1=generate_stuck_at_fault_modulator_fast(shape,coor,fault_type,fault_bit)
            inject1=True
        elif fault_type == 'flip':
            tensor_modulatorF=generate_stuck_at_fault_modulator_fast(shape,coor,fault_type,fault_bit)
            injectF=True
    else:
        tensor_modulator0=-np.ones(shape,dtype=np.int32)
        tensor_modulator1=np.zeros(shape,dtype=np.int32)
        tensor_modulatorF=np.zeros(shape,dtype=np.int32)
        
        for key in fault_dict.keys():
            modulator0,modulator1,modulatorF=generate_stuck_at_fault_modulator(nb,fb,fault_dict[key]['SA_bit'],fault_dict[key]['SA_type'])
            if modulator0 is not None:
                tensor_modulator0[key]=modulator0
                inject0=True
            if modulator1 is not None:
                tensor_modulator1[key]=modulator1
                inject1=True
            if modulatorF is not None:
                tensor_modulatorF[key]=modulatorF
                injectF=True
                        
    if not inject0:
        tensor_modulator0=None
    if not inject1:
        tensor_modulator1=None
    if not injectF:
        tensor_modulatorF=None
            
    return [tensor_modulator0,tensor_modulator1,tensor_modulatorF]

def generate_layer_modulator(layer,word_length,fractional_bit,ifmap_fault_dict,ofmap_fault_dict,wght_fault_dict,fast_gen=False):
    """ Generate modulator for a DNN layer.
        Layer must be the TensorFlow/Keras layer with weights and MAC operation.
        Specify the generation method is numpy array based (fast gen) or for loop based.

    Parameters
    ----------
    layer : tensorflow.keras.layer class
        The layer for generate modulator. Get the layer shape info.
    word_length : Integer
        The word length of layer data. Assume that all input, weight and output of layer have the same word length.
    fractional_bit : Integer
        The number of fractional bit of layer data. Assume that all input, weight and output of layer have the same fractional bits.
    ifmap_fault_dict : Dictionary
        Fault dctionary for input feature maps.
    ofmap_fault_dict : Dictionay
        Fault dctionary for output feature maps.
    wght_fault_dict : List of Dictionary. [kernal_fault_dict, bias_fault_dict]
        Fault dctionary for output feature maps.
    fast_gen : Bool, optional
        Use numpy array based generation (fast gen) or not. The default is False.

    Returns
    -------
    ifmap_modulator : List of Ndarray. [tensor_modulator0,tensor_modulator1,tensor_modulatorF]
        The tensor modulator for SA0, SA1, bit-flip respectively on input feature maps.
    ofmap_modulator : List of Ndarray. [tensor_modulator0,tensor_modulator1,tensor_modulatorF]
        The tensor modulator for SA0, SA1, bit-flip respectively on output feature maps.
    wght_modulator : List of Ndarray. [tensor_modulator0,tensor_modulator1,tensor_modulatorF]
        The tensor modulator for SA0, SA1, bit-flip respectively on weights.

    """
    layer_input_shape=layer.input_shape
    layer_output_shape=layer.output_shape
    layer_weight_shape=[weight_shape.shape for weight_shape in layer.get_weights()]
    
    if ifmap_fault_dict is None:
        ifmap_modulator=None
    else:
        ifmap_modulator=generate_tensor_modulator(layer_input_shape,word_length,fractional_bit,ifmap_fault_dict,fast_gen=fast_gen)
    
    if ofmap_fault_dict is None:
        ofmap_modulator=None
    else:
        ofmap_modulator=generate_tensor_modulator(layer_output_shape,word_length,fractional_bit,ofmap_fault_dict,fast_gen=fast_gen)
    
    wght_modulator=list()
    for i,shape in enumerate(layer_weight_shape):
        if wght_fault_dict[i] is None:
            wght_modulator.append(None)
        else:
            wght_modulator.append(generate_tensor_modulator(shape,word_length,fractional_bit,wght_fault_dict[i],fast_gen=fast_gen))
    if len(wght_modulator)==0:
        wght_modulator=[None,None]
    
    return ifmap_modulator,ofmap_modulator,wght_modulator

def generate_model_modulator(model,word_length,fractional_bit,ifmap_fault_dict_list,ofmap_fault_dict_list,wght_fault_dict_list,fast_gen=False):
    """ Generate modulator for a DNN model.
        Layer must be the TensorFlow/Keras model with convolution layers.
        Specify the generation method is numpy array based (fast gen) or for loop based.

    Parameters
    ----------
    model : tensorflow.keras.model
        The model for generate modulator. Get the layer shape info in model.
    word_length : Integer
        The word length of layer data. Assume that all input, weight and output of model have the same word length.
    fractional_bit : Integer
        The number of fractional bit of layer data. Assume that all input, weight and output of model have the same fractional bits.
    ifmap_fault_dict_list : List of Dictionary
        Fault dictionary list for input feature maps.    
        The list are the same order as the Keras model layer list. Each Dictionary in List is for its corresponding layer.
        The layers have no weight and MAC operation are setting its fault dictionary to None.
    ofmap_fault_dict_list : List of Dictionary
        Fault dctionary for output feature maps.
        The list are the same order as the Keras model layer list. Each Dictionary in List is for its corresponding layer.
        The layers have no weight and MAC operation are setting its fault dictionary to None.
    wght_fault_dict_list : List of Dictionary
        Fault dctionary for output feature maps.
        The list are the same order as the Keras model layer list. Each Dictionary in List is for its corresponding layer.
        The layers have no weight and MAC operation are setting its fault dictionary to None.
    fast_gen : Bool, optional
        Use numpy array based generation (fast gen) or not. The default is False.

    Returns
    -------
    model_ifmap_fault_modulator_list : List of List of Ndarray
        The modulator for DNN model input feature maps.
        The outer List is layer list order, the inner list is the [modulator SA0, modulator SA1, modulator bit-flip].
    model_ofmap_fault_modulator_list : List of List of Ndarray
        The modulator for DNN model output feature maps.
        The outer List is layer list order, the inner list is the [modulator SA0, modulator SA1, modulator bit-flip].
    model_wght_fault_modulator_list : List of List of List of Ndarray
        The modulator for DNN model weights.
        The outer List is layer list order, the middle list is [kernel, bias], the inner list is the [modulator SA0, modulator SA1, modulator bit-flip].

    """
    model_depth=len(model.layers)
    model_ifmap_fault_modulator_list=[None for _ in range(model_depth)]
    model_ofmap_fault_modulator_list=[None for _ in range(model_depth)]
    model_wght_fault_modulator_list=[[None,None] for _ in range(model_depth)]

    for layer_num in range(1,model_depth):
        ifmap_modulator,ofmap_modulator,wght_modulator\
        =generate_layer_modulator(model.layers[layer_num],
                                  word_length,
                                  fractional_bit,
                                  ifmap_fault_dict_list[layer_num],
                                  ofmap_fault_dict_list[layer_num],
                                  wght_fault_dict_list[layer_num],
                                  fast_gen=fast_gen)
        
        model_ifmap_fault_modulator_list[layer_num]=ifmap_modulator
        model_ofmap_fault_modulator_list[layer_num]=ofmap_modulator
        model_wght_fault_modulator_list[layer_num]=wght_modulator
        
    return model_ifmap_fault_modulator_list,model_ofmap_fault_modulator_list,model_wght_fault_modulator_list



