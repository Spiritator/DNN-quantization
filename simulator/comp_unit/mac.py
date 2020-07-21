# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 10:55:00 2020

@author: Yung-Yu Tsai

The MAC unit description for fault transfer to TF computation
"""

import tensorflow as tf
import numpy as np
import json
from ..layers.quantized_ops import quantizer

class mac_unit:
    """
    The mac unit information holder class. For describe PE I/O interconnection and math of faulty behavior.
    """
    def __init__(self, quantizers, quant_mode=None, ifmap_io=None, wght_io=None, psum_io=None):
        """
        # Arguments
            quantizers: Class or String. The quantize library of simulator. Or the file path to MAC unit setup (.json) file.
                Default ifmap input/output, weight input/output, partial sum input/output are the same configuration.
                If using setup (.json) file, this will include all the following arguments input.
            quant_mode: String. The quantization mode of MAC. Be either 'intrinsic' or 'hybrid'.
                'intrinsic' means truncate before accumulation. 'hybrid' means accumulate with the word length of multiplier output than truncate.
            ifmap_io: Dictionary. Input feature map data I/O description.
            wght_io: Dictionary. Weight data I/O description.
            psum_io: Dictionary. Partial sum data I/O description. Partial sum fault prapagation is through MAC operation.
            
        # I/O description: (Dictionary)
            data_io={'type':'io_pair', 'dimension':'PE_x', 'direction':'forward'}
            
            type: The type of I/O. 
                'io_pair' the data flow from a primary input to a primary output which forms a I/O pair. The I/O pair become a systolic stream in PE array at some point.
                'one_way_in' the data is a primary input of MAC unit. No further propagation at any computation process.
                'one_way_out' the data is a primary output of MAC unit. No direct data propagation chain lead to this output.
            
            dimension: the dimension of I/O pair connect in PE array. One of None, 'PE_x', 'PE_y'.
            
            direction: the direction of I/O pair deliver data on the given dimension. 
                'forward' data prapagate along the index of PEs. 
                'backward' data prapagate by the reversed order of PE index.
            
        """
        if not isinstance(quantizers,str):
            self.quantizer=quantizers
            if quant_mode not in ['intrinsic', 'hybrid']:
                raise ValueError('Quantization mode must be either \'intrinsic\' or \'hybrid\'.')
            self.quant_mode=quant_mode
            self.ifmap_io=ifmap_io
            self.wght_io=wght_io
            self.psum_io=psum_io
        else:
            self._setup(quantizers)
        
    def _setup(self, setup_file):
        """
        file read in setup for MAC unit
        """
        with open(setup_file, 'r') as config_file:
            setups=json.load(config_file)
            
        self.quantizer=setups['quantizers']
        self.quant_mode=setups['quant_mode']
        self.ifmap_io=setups['ifmap_io']
        self.wght_io=setups['wght_io']
        self.psum_io=setups['psum_io']
        
        if isinstance(self.quantizer,dict):
            self.quantizer=quantizer(**self.quantizer)
        elif isinstance(self.quantizer,list):
            self.quantizer=[quantizer(**qinfo) for qinfo in self.quantizer]
            
    def get_io(self, param):
        """ get the I/O description by parameter
        
        # Arguments
            param: String. The type of parameter has fault. 
                One of ['ifmap_in', 'ifmap_out', 'wght_in', 'wght_out', 'psum_in', 'psum_out']. 
        
        # Returns
            description of I/O (Dictionary)
        """
        if param in ['ifmap_in', 'ifmap_out']:
            return self.ifmap_io
        elif param in ['wght_in', 'wght_out']:
            return self.wght_io
        elif param in ['psum_in', 'psum_out']:
            return self.psum_io
        
    def propagated_idx_list(self, param, fault_loc, array_shape):
        """ get the contaminated PE index in array by the fault location and parameter.
        
        # Arguments
            param: String. The type of parameter has fault. 
                One of ['ifmap_in', 'ifmap_out', 'wght_in', 'wght_out', 'psum_in', 'psum_out']. 
            fault_loc: Ndarray or List. PE dataflow model coordinate represent as the fault location.
            array_shape: Tuple or List. The shape of PE array.
        
        # Returns
            Converted coordinates. Multiple coordinate return in 2D ndarray.
        """
        if param in ['psum_in', 'psum_out']:
            return fault_loc
        
        ioconfig=self.get_io(param)
        if ioconfig['type']=='io_pair':
            if ioconfig['dimension']=='PE_y':
                dim=0                 
            elif ioconfig['dimension']=='PE_x':
                dim=1
                
            if ioconfig['direction']=='forward':
                idxlist=np.arange(fault_loc[dim],array_shape[dim])
            elif ioconfig['direction']=='backward':
                idxlist=np.arange(0,fault_loc[dim]+1)
                
            fault_loc=np.tile(np.expand_dims(fault_loc,0),[len(idxlist),1])
            fault_loc[:,dim]=idxlist
            
            return fault_loc
                    
        else:
            return fault_loc

    def fault_injection(self, ifmap, wght, ofmap, fault_dict, quantizer):
        """ The fault injection mathematical model for used in TF GPU parallel computing
        
        # Arguments
            ifmap: Tensor. Quantized Tensor. Layer input.
            wght: Tensor. Quantized Tensor. Layer output.
            ofmap: Tensor. The Tensor to be injected fault by math alteration. Quantized Tensor. Layer output.
            fault_dict: Dictionary or List. The dictionary contain fault list information.
            quantizer: Class. The quantizer class contain following quantize operation infromation.
                word_width: Variable. The fix-point representation of the parameter word length.
                fractional_bits: Variable. Number of fractional bits in a fix-point parameter
                rounding: String. Rounding method of quantization, augment must be one of 'nearest' , 'down', 'zero', 'stochastic'.
        
        # Returns
            Tensor. The amount of adjustment apply to output feature map of a DNN layer which represent the faulty behabvior of MAC unit.

        """
        
    #TODO
    # the generation for mac fault injection

        
