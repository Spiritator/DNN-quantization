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
                Quantizer Parameters:
                    word_width: Variable. The fix-point representation of the parameter word length.
                    fractional_bits: Variable. Number of fractional bits in a fix-point parameter
                    rounding: String. Rounding method of quantization, augment must be one of 'nearest' , 'down', 'zero', 'stochastic'.

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

    def inject_mac_math_fault_ndarray(self, ifmap, wght, ofmap, fault_dict, quantizer, ksizes=(3,3), padding='valid', dilation_rates=(1,1), fast_gen=True):
        """ The fault injection mathematical model for used in numpy computing
        
        # Arguments
            ifmap: Ndarray. Quantized array. Layer input.
            wght: Ndarray. Quantized array. Layer output.
            ofmap: Ndarray. The array to be injected fault by math alteration. Quantized array. Layer output.
            fault_dict: Dictionary or List. The dictionary contain fault list information.
            quantizer: Class. The quantizer class contain following quantize operation infromation.
                word_width: Variable. The fix-point representation of the parameter word length.
                fractional_bits: Variable. Number of fractional bits in a fix-point parameter
                rounding: String. Rounding method of quantization, augment must be one of 'nearest' , 'down', 'zero', 'stochastic'.
            ksize: Tuple. Size 2, the kernel size (row, col).
            padding: String. 'same' or 'valid'. The type of padding algorithm to use.
            dilation_rate: Tuple. Size 2, the dilation rate (row, col).
            fast_gen: Bool. Use fast generation or not. Fast generation has the same fault bit and SA type for all coordinates.
                The fault dictionay is form by fault data contamination.
        
        # Returns
            Ndarray. The amount of adjustment apply to output feature map of a DNN layer which represent the faulty behabvior of MAC unit.
        
        # Reminder!!
            padding: Padding is tie to layer setting. If user treat padding as a seperate layer.
                For example x=layers.ZeroPadding2D(padding=(1, 1), name='conv1_pad')(inputs)
                The tile setting must be padding='valid' which means no padding.
                Or else there might be fault.
        """
        fd_coor=np.array(list(fault_dict.keys()))
        fd_value=np.array(list(fault_dict.values()))
        # (coor idx, num of psidx, psum idx)
        psum_idx_list=np.array([info['psum_idx'] for info in fd_value])
        psum_idx_ofmap=psum_idx_list[:,:,[0,2,3,1]]
        
        if fast_gen:
            fault_param=fd_value[0]['param']
            fault_type=fd_value[0]['SA_type']
            fault_bit=fd_value[0]['SA_bit']
            
            if fault_param in ['ifmap_in','ifmap_out','wght_in','wght_out']:    
                psum_idx_wght=psum_idx_list[:,:,[5,6,4,1]]
                psum_idx_ifmap=psum_idx_list[:,:,[0,7,8,4]]
            
                if padding=='same':
                    dilated_ksize_row_edge = (ksizes[0] + (ksizes[0]-1) * (dilation_rates[0] - 1))//2
                    dilated_ksize_col_edge = (ksizes[1] + (ksizes[1]-1) * (dilation_rates[1] - 1))//2
                    psum_idx_ifmap[:,:,1]=np.add(psum_idx_ifmap[:,:,1],dilated_ksize_row_edge)
                    psum_idx_ifmap[:,:,2]=np.add(psum_idx_ifmap[:,:,2],dilated_ksize_col_edge)
                
                    pad=((0,0),(dilated_ksize_row_edge,dilated_ksize_row_edge),(dilated_ksize_col_edge,dilated_ksize_col_edge),(0,0))
                    ifmap=np.pad(ifmap,pad,'constant',constant_values=0)
    
    
                ifmap_alloc=ifmap[psum_idx_ifmap[:,:,0],psum_idx_ifmap[:,:,1],psum_idx_ifmap[:,:,2],psum_idx_ifmap[:,:,3]]
                wght_alloc=wght[psum_idx_wght[:,:,0],psum_idx_wght[:,:,1],psum_idx_wght[:,:,2],psum_idx_wght[:,:,3]]
                
                ifmap_alloc=quantizer.left_shift_2int(ifmap_alloc)
                wght_alloc=quantizer.left_shift_2int(wght_alloc)
            
            ofmap_alloc=ofmap[psum_idx_ofmap[:,:,0],psum_idx_ofmap[:,:,1],psum_idx_ofmap[:,:,2],psum_idx_ofmap[:,:,3]]
            ofmap_alloc=quantizer.left_shift_2int(ofmap_alloc)

    #        if fault_param=='ifmap_in' or fault_param=='ifmap_out':
    #            FI_param=ifmapin
    #            wlpolar=wl
    #        elif fault_param=='wght_in' or fault_param=='wght_out':
    #            FI_param=wght_in
    #            wlpolar=wl
    #        elif fault_param=='psum_in' or fault_param=='psum_out':
    #            FI_param=psum_out_ff
    #            if quantize_mode=='intrinsic':
    #                wlpolar=wl        
    #            elif quantize_mode=='hybrid':
    #                wlpolar=2*wl
    #            
    #        if quantize_mode=='intrinsic':
    #            wlpsum=wl
    #        elif quantize_mode=='hybrid':
    #            wlpsum=2*wl
    #        
    #        if SA_type=='flip':
    #            if np.binary_repr(FI_param,wlpolar)[wlpolar-1-fault_bit]=='1':
    #                polarity_check=-1
    #            elif np.binary_repr(FI_param,wlpolar)[wlpolar-1-fault_bit]=='0':
    #                polarity_check=1
    #        elif SA_type=='0':    
    #            if np.binary_repr(FI_param,wlpolar)[wlpolar-1-fault_bit]=='1':
    #                polarity_check=-1
    #            else:
    #                polarity_check=0
    #        elif SA_type=='1':    
    #            if np.binary_repr(FI_param,wlpolar)[wlpolar-1-fault_bit]=='0':
    #                polarity_check=1
    #            else:
    #                polarity_check=0
    #                
    #        if fault_bit==wlpolar-1:
    #            polarity_check*=-1
    #        
    #        if fault_param=='ifmap_in' or fault_param=='ifmap_out':
    #            ifmap_out_mk=ifmap_out_ff+polarity_check*(2**fault_bit)
    #            wght_out_mk=wght_out_ff
    #        
    #            psum_out_mk=overflowcap(psum_out_ff + psum_holdmk, wlpsum)
    #        
    #            if polarity_check==0:
    #                psum_holdmk=0
    #            else:
    #                psum_holdmk= wght_in*(2**fault_bit)
    #                
    #                if quantize_mode=='intrinsic':
    #                    if polarity_check==-1:
    #                        truncarry=int(psum_holdmk%scale_factor > ifmapin*wght_in%scale_factor)
    #                    elif polarity_check==1:
    #                        truncarry=int(psum_holdmk%scale_factor + ifmapin*wght_in%scale_factor > scale_factor)
    #                    psum_holdmk= polarity_check * (overflowcap(psum_holdmk//scale_factor,wlpsum) + truncarry)
    #                elif quantize_mode=='hybrid':
    #                    psum_holdmk= polarity_check * overflowcap(psum_holdmk, wlpsum)
    #                
    #        elif fault_param=='wght_in' or fault_param=='wght_out':
    #            ifmap_out_mk=ifmap_out_ff
    #            wght_out_mk=wght_out_ff+polarity_check*(2**fault_bit)
    #        
    #            psum_out_mk=overflowcap(psum_out_ff + psum_holdmk, wlpsum)
    #        
    #            if polarity_check==0:
    #                psum_holdmk=0
    #            else:
    #                psum_holdmk= ifmapin*(2**fault_bit)
    #                
    #                if quantize_mode=='intrinsic':
    #                    if polarity_check==-1:
    #                        truncarry=int(psum_holdmk%scale_factor > ifmapin*wght_in%scale_factor)
    #                    elif polarity_check==1:
    #                        truncarry=int(psum_holdmk%scale_factor + ifmapin*wght_in%scale_factor > scale_factor)
    #                    psum_holdmk= polarity_check * (overflowcap(psum_holdmk//scale_factor,wlpsum) + truncarry)
    #                elif quantize_mode=='hybrid':
    #                    psum_holdmk= polarity_check * overflowcap(psum_holdmk, wlpsum)
    #                
    #        elif fault_param=='psum_in' or fault_param=='psum_out':
    #            ifmap_out_mk=ifmap_out_ff
    #            wght_out_mk=wght_out_ff
    #        
    #            psum_out_mk=overflowcap(psum_out_ff + polarity_check*(2**fault_bit), wlpsum)
    #        
    #            psum_holdmk= 0
        else:
            pass
    
        return 'pupu'

        
    def inject_mac_math_fault_tensor(self, ifmap, wght, ofmap, fault_dict, quantizer, fast_gen=False):
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
            fast_gen: Bool. Use fast generation or not. Fast generation has the same fault bit and SA type for all coordinates.
                The fault dictionay is form by fault data contamination.
        
        # Returns
            Tensor. The amount of adjustment apply to output feature map of a DNN layer which represent the faulty behabvior of MAC unit.

        """
        
        # tf.gather_nd
        
    #TODO
    # the generation for mac fault injection

        
