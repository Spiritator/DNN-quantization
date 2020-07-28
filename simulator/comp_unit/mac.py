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
        
    def _polarity_check(self, fault_type, fault_bit, FI_param, wl):
        """ Get the polarity of parameter 
            
        """
        modulator=np.left_shift(1,fault_bit)
        polarity=tf.bitwise.bitwise_and(FI_param,modulator)
        polarity=tf.math.sign(polarity)
        if fault_type=='flip':
            polarity=tf.math.add(tf.multiply(polarity,-2),1)
        elif fault_type=='0':    
            polarity=tf.math.negative(polarity)
        elif fault_type=='1':    
            polarity=tf.math.subtract(1,polarity)
                
        if fault_bit==wl-1:
            polarity=tf.math.negative(polarity)
            
        polarity=tf.cast(polarity,tf.float32)
            
        return polarity


    def inject_mac_math_fault_tensor(self, ifmap, wght, ofmap, fault_dict, 
                                     quantizer=None, 
                                     ksizes=(3,3), padding='valid', dilation_rates=(1,1), 
                                     sim_truncarry=False, fast_gen=True):
        """ The fault injection mathematical model for used in numpy computing
        
        # Arguments
            ifmap: Tensor. Quantized Tensor. Layer input.
            wght: Tensor. Quantized Tensor. Layer output.
            ofmap: Tensor. The Tensor to be injected fault by math alteration. Quantized Tensor. Layer output.
            fault_dict: Dictionary or List. The dictionary contain fault list information.
            quantizer: Class or List. The quantizer class, one or in list [input, weight, output]. The quantizer class contain following quantize operation infromation.
                word_width: Variable. The fix-point representation of the parameter word length.
                fractional_bits: Variable. Number of fractional bits in a fix-point parameter
                rounding: String. Rounding method of quantization, augment must be one of 'nearest' , 'down', 'zero', 'stochastic'.
            ksize: Tuple. Size 2, the kernel size (row, col).
            padding: String. 'same' or 'valid'. The type of padding algorithm to use.
            dilation_rate: Tuple. Size 2, the dilation rate (row, col).
            sim_truncarry: Bool. simulate the truncation carry during mac math fault injection. 
                The truncation carry is cause by the carry-out of thrown away fractional bits. 
                It could be 1 in addition or -1 in subtraction, when the unwanted fractional bits value overflow to the needed fractional bits.
                How ever this could cause huge time overhead for absolute bit-true model.
            fast_gen: Bool. Use fast generation or not. Fast generation has the same fault bit and SA type for all coordinates.
                The fault dictionay is form by fault data contamination.
        
        # Returns
            Tensor. The amount of adjustment apply to output feature map of a DNN layer which represent the faulty behabvior of MAC unit.
        
        # Reminder!!
            padding: Padding is tie to layer setting. If user treat padding as a seperate layer.
                For example x=layers.ZeroPadding2D(padding=(1, 1), name='conv1_pad')(inputs)
                The tile setting must be padding='valid' which means no padding.
                Or else there might be fault.
        """
        if quantizer is None:
            if isinstance(self.quantizer,list) and len(self.quantizer)==3:
                quantizer_input =self.quantizer[0]
                quantizer_weight =self.quantizer[1]
                quantizer_output =self.quantizer[2]
            else:
                quantizer_input =self.quantizer
                quantizer_weight =self.quantizer
                quantizer_output =self.quantizer
        else:
            if isinstance(quantizer,list) and len(quantizer)==3:
                quantizer_input =quantizer[0]
                quantizer_weight =quantizer[1]
                quantizer_output =quantizer[2]
            else:
                quantizer_input =quantizer
                quantizer_weight =quantizer
                quantizer_output =quantizer
        
        fd_coor=np.array(list(fault_dict.keys()))
        fd_value=np.array(list(fault_dict.values()))
        fdoutput_alloc=tf.gather_nd(ofmap,fd_coor)
        
        if fast_gen:
            # (coor idx, num of psidx, psum idx)
            psum_idx_list=np.array([info['psum_idx'] for info in fd_value])
            psum_idx_ofmap=psum_idx_list[:,:,[0,2,3,1]]
        
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
                
                    pad=[[0,0],[dilated_ksize_row_edge,dilated_ksize_row_edge],[dilated_ksize_col_edge,dilated_ksize_col_edge],[0,0]]
                    ifmap=tf.pad(ifmap,pad,'constant',constant_values=0)
    
                ifmap_alloc=tf.gather_nd(ifmap,psum_idx_ifmap)
                wght_alloc=tf.gather_nd(wght,psum_idx_wght) 
                
                ifmap_alloc=quantizer_input.left_shift_2int(ifmap_alloc)
                wght_alloc=quantizer_weight.left_shift_2int(wght_alloc)
            
            ofmap_alloc=tf.gather_nd(ofmap,psum_idx_ofmap)
            ofmap_alloc=quantizer_output.left_shift_2int(ofmap_alloc)

            if fault_param=='ifmap_in' or fault_param=='ifmap_out':
                FI_param = ifmap_alloc
                wlpolar = quantizer_input.nb
            elif fault_param=='wght_in' or fault_param=='wght_out':
                FI_param = wght_alloc
                wlpolar = quantizer_weight.nb
            elif fault_param=='psum_in' or fault_param=='psum_out':
                FI_param = ofmap_alloc
                if self.quant_mode=='intrinsic':
                    wlpolar = quantizer_output.nb
                elif self.quant_mode=='hybrid':
                    wlpolar = quantizer_input.nb+quantizer_weight.nb
            
            polarity=self._polarity_check(fault_type, fault_bit, FI_param, wlpolar)
                            
            if fault_param=='ifmap_in' or fault_param=='ifmap_out' or fault_param=='wght_in' or fault_param=='wght_out':
                if fault_param=='ifmap_in' or fault_param=='ifmap_out':
                    psum_alter= tf.multiply(wght_alloc,2**fault_bit)
                elif fault_param=='wght_in' or fault_param=='wght_out':
                    psum_alter= tf.multiply(ifmap_alloc,2**fault_bit)
                
                if self.quant_mode=='intrinsic':
                    if sim_truncarry:
                        trivia_alter= tf.floormod(psum_alter, quantizer_output.shift_factor)
                        trivia_conv= tf.floormod(tf.multiply(ifmap_alloc,wght_alloc), quantizer_output.shift_factor)
                        trivia_conv= tf.multiply(trivia_conv, polarity)
                        truncarry= tf.add(trivia_alter, trivia_conv)
                        
                        comparator= tf.multiply(tf.floordiv(tf.add(polarity,1),2), quantizer_output.shift_factor-1)
                        truncarry= tf.sign(tf.subtract(truncarry, comparator))
                    
                    psum_alter=quantizer_output.right_shift_back(psum_alter)
                    psum_alter=quantizer_output.round_through(psum_alter)
                    psum_alter=quantizer_output.capping(psum_alter)
                    if sim_truncarry:
                        psum_alter=tf.add(psum_alter,truncarry)
                    psum_alter=tf.multiply(psum_alter, polarity)
                    
                    # sum all psum_alter
                    psum_alter=tf.reduce_sum(psum_alter, axis=1)
                    psum_alter=quantizer_output.quantize_2half(psum_alter)
                    
                elif self.quant_mode=='hybrid':
                    psum_alter=tf.multiply(polarity, psum_alter)
                    
                    # sum all psum_alter
                    psum_alter=tf.reduce_sum(psum_alter, axis=1)
                    psum_alter=quantizer_output.right_shift_back(psum_alter)
                    psum_alter=quantizer_output.quantize_2half(psum_alter)
                
            #TODO
            # there is no way to know the psum of exact time frame
            # fault on psum do it the way in PE RTL test
            elif fault_param=='psum_in' or fault_param=='psum_out':
                psum_alter=tf.multiply(polarity,2**fault_bit)
                
                psum_alter=tf.reduce_sum(psum_alter, axis=1)
                psum_alter=quantizer_output.right_shift_back(psum_alter)
                psum_alter=quantizer_output.quantize_2half(psum_alter)

            # add psum_alter back to ofmap
            output=tf.add(fdoutput_alloc, psum_alter)
            output=quantizer_output.quantize(output)
            output=tf.scatter_nd_update(ofmap,fd_coor,output)

                
        else: # slow loop gen
            if isinstance(fd_value[0]['id'],int):
                state='normal'
                        
                psum_idx_list=list()
                fault_param=list()
                fault_type=list()
                fault_bit=list()
                for info in fd_value:
                    psum_idx_list.append(info['psum_idx'])
                    fault_param.append(info['param'])
                    fault_type.append(info['SA_type'])
                    fault_bit.append(info['SA_bit'])
                    
                # (coor idx, num of psidx, psum idx)
                psum_idx_list=np.array(psum_idx_list)
                psum_idx_ofmap=psum_idx_list[:,:,[0,2,3,1]]

                fault_param=np.array(fault_param)
                fault_type=np.array(fault_type)
                fault_bit=np.array(fault_bit)
            
            elif isinstance(fd_value[0]['id'],list):
                state='repeative'
                
                psum_idx_list=list()
                cnt_psidx=list()
                fault_param=list()
                fault_type=list()
                fault_bit=list()
                for info in fd_value:
                    cnt_psidx.append(len(info['psum_idx']))
                    psum_idx_list.append(info['psum_idx'])
                    fault_param.append(info['param'])
                    fault_type.append(info['SA_type'])
                    fault_bit.append(info['SA_bit'])
                
                cnt_psidx=np.cumsum(cnt_psidx)-1
                # (coor idx, num of fault, num of psidx, psum idx)
                psum_idx_list=np.array(psum_idx_list)
                # (coor idx * num of fault, num of psidx, psum idx)
                psum_idx_list=np.concatenate(psum_idx_list)
                psum_idx_ofmap=psum_idx_list[:,:,[0,2,3,1]]

                fault_param=np.array(fault_param)
                fault_type=np.array(fault_type)
                fault_bit=np.array(fault_bit)
                fault_param=np.concatenate(fault_param)
                fault_type=np.concatenate(fault_type)
                fault_bit=np.concatenate(fault_bit)
                
                    
    
        return output

                
    #TODO
    # the generation for mac fault injection

        
