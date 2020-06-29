# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 10:55:00 2020

@author: Yung-Yu Tsai

The MAC unit description for fault transfer to TF computation
"""

import json
from ..layers.quantized_ops import quantizer

class mac_unit:
    """
    The mac unit information holder class. For describe PE I/O interconnection and math of faulty behavior.
    """
    def __init__(self, quantizers, quant_mode, ifmap_io=None, wght_io=None, psum_io=None):
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
            self.setup(quantizers)
        
    def setup(self, setup_file):
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
            
    def fault_injection(self, ):
        """
        The fault injection mathematical model for used in TF GPU parallel computing
        
        # Arguments
        
        
        # Returns
            Tensor. The amount of adjustment apply to output feature map of a DNN layer which represent the faulty behabvior of MAC unit.

        """
        
    #TODO
    # the generation for mac fault injection

        
