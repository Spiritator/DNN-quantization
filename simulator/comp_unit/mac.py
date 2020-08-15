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
    def __init__(self, quantizers, quant_mode='hybrid', ifmap_io=None, wght_io=None, psum_io=None, sim_truncarry=False, fast_gen=True):
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
        
        self.sim_truncarry=sim_truncarry
        self.fast_gen=fast_gen
        
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
        
        self.quantizer=self._build_layer_quantizer(**self.quantizer)
    
    def _build_layer_quantizer(self,nb,fb,rounding_method,overflow_mode,stop_gradient):
        """ Layer quantizer builder. For generate different setup for ifmap, weight, ofmap individually. """
        multi_setting=False
        
        if isinstance(nb,list) or isinstance(fb,list) or isinstance(rounding_method,list) or isinstance(overflow_mode,list) or isinstance(stop_gradient,list):
            multi_setting=True
            
        if isinstance(nb,list) and len(nb)==3:
            nb_qt=nb
        elif multi_setting:
            nb_qt=[nb, nb, nb]
            
        if isinstance(fb,list) and len(fb)==3:
            fb_qt=fb
        elif multi_setting:
            fb_qt=[fb, fb, fb]
        
        if isinstance(rounding_method,list) and len(rounding_method)==3:
            rm_qt=rounding_method
        elif multi_setting:
            rm_qt=[rounding_method, rounding_method, rounding_method]
            
        if isinstance(overflow_mode,list) and len(overflow_mode)==3:
            ovf_qt=overflow_mode
        elif multi_setting:
            ovf_qt=[overflow_mode, overflow_mode, overflow_mode]
            
        if multi_setting:
            return [quantizer(nb_qt[0],fb_qt[0],rm_qt[0],ovf_qt[0],stop_gradient),
                    quantizer(nb_qt[1],fb_qt[1],rm_qt[1],ovf_qt[1],stop_gradient),
                    quantizer(nb_qt[2],fb_qt[2],rm_qt[2],ovf_qt[2],stop_gradient)]
        else:
            return quantizer(nb,fb,rounding_method,overflow_mode,stop_gradient)

            
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
                
        if isinstance(fault_bit,int):
            if fault_bit==wl-1:
                polarity=tf.math.negative(polarity)
        else:
            signbit=np.equal(fault_bit,wl-1)
            signbit=np.add(np.multiply(signbit,-2,dtype=np.int32),1)
            polarity=tf.math.multiply(polarity,signbit)
                        
        return polarity
    
    def _padding_ifmap_and_idx(self, ifmap, index, ksizes, dilation_rates):
        """ Preproccess ifmap data and index for padding situation """
        dilated_ksize_row_edge = (ksizes[0] + (ksizes[0]-1) * (dilation_rates[0] - 1))//2
        dilated_ksize_col_edge = (ksizes[1] + (ksizes[1]-1) * (dilation_rates[1] - 1))//2
        index[:,:,1]=np.add(index[:,:,1],dilated_ksize_row_edge)
        index[:,:,2]=np.add(index[:,:,2],dilated_ksize_col_edge)
    
        if ifmap is None:
            return None, index
            
        pad=[[0,0],[dilated_ksize_row_edge,dilated_ksize_row_edge],[dilated_ksize_col_edge,dilated_ksize_col_edge],[0,0]]
        ifmap=tf.pad(ifmap,pad,'constant',constant_values=0)
        
        return ifmap, index

    def mac_math_alter_make(self, psum_alter, polarity, quantizer_output, sim_truncarry=False, ifmap_alloc=None, wght_alloc=None):
        """ The core funciton of create mac math fault injection alteration Tensor
            This alteration will be later add onto ofmap tensor
            
            psum_alter: The product data by multiply coefficient data and fault bit order. That is:
                if fault_param=='ifmap_in' or fault_param=='ifmap_out':
                    psum_alter= tf.multiply(wght_alloc,2**fault_bit)
                elif fault_param=='wght_in' or fault_param=='wght_out':
                    psum_alter= tf.multiply(ifmap_alloc,2**fault_bit)
                
        """
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
            psum_alter=tf.multiply(psum_alter, tf.cast(polarity,tf.float32))
            
            # sum all psum_alter
            psum_alter=tf.reduce_sum(psum_alter, axis=1)
            psum_alter=quantizer_output.quantize_2half(psum_alter)
            
        elif self.quant_mode=='hybrid':
            psum_alter=tf.multiply(polarity, psum_alter)
            
            # sum all psum_alter
            psum_alter=tf.reduce_sum(psum_alter, axis=1)
            psum_alter=quantizer_output.right_shift_back(psum_alter)
            psum_alter=quantizer_output.quantize_2half(psum_alter)
            
        return psum_alter
    
    def _fault_value_extract_loop(self, fault_value, repetitive=False):
        """ Extract data from fault dictionary values 
            The fault value is in np.array(list(fault_dict.values())) format
            This is only for slow loop generation
        """
        psum_idx_list=list()
        fault_bit=list()
        param_ifmap=list()
        param_wght=list()
        param_ofmap=list()
        type0=list()
        type1=list()
        typef=list()
        
        if not repetitive:
            for i,info in enumerate(fault_value):
                psum_idx_list.append(info['psum_idx'])
                fault_bit.append(info['SA_bit'])
                
                param=info['param']
                if param=='ifmap_in' or param=='ifmap_out':
                    param_ifmap.append(i)
                elif param=='wght_in' or param=='wght_out':
                    param_wght.append(i)
                elif param=='psum_in' or param=='psum_out':
                    param_ofmap.append(i)
                    
                typee=info['SA_type']
                if typee=='0':
                    type0.append(i)
                elif typee=='1':
                    type1.append(i)
                elif typee=='flip':
                    typef.append(i)
                
            # (coor idx, num of psidx, psum idx)
            psum_idx_list=np.array(psum_idx_list)
            fault_bit=np.array(fault_bit)
            param_ifmap=np.array(param_ifmap)
            param_wght=np.array(param_wght)
            param_ofmap=np.array(param_ofmap)
            type0=np.array(type0)
            type1=np.array(type1)
            typef=np.array(typef)
            cnt_psidx=None
        
        else:
            cnt_psidx=list()
            fault_param=list()
            fault_type=list()
            for info in fault_value:
                cnt_psidx.append(len(info['psum_idx']))
                psum_idx_list.append(info['psum_idx'])
                fault_bit.append(info['SA_bit'])
                fault_param.append(info['param'])
                fault_type.append(info['SA_type'])
            
            cnt_psidx=np.cumsum(cnt_psidx)[:-1]
            # (coor idx, num of fault, num of psidx, psum idx)
            psum_idx_list=np.array(psum_idx_list)
            # (coor idx * num of fault, num of psidx, psum idx)
            psum_idx_list=np.concatenate(psum_idx_list)

            fault_param=np.array(fault_param)
            fault_type=np.array(fault_type)
            fault_bit=np.array(fault_bit)
            fault_param=np.concatenate(fault_param)
            fault_type=np.concatenate(fault_type)
            fault_bit=np.concatenate(fault_bit)
            
            for i in range(len(fault_param)):
                param=fault_param[i]
                if param=='ifmap_in' or param=='ifmap_out':
                    param_ifmap.append(i)
                elif param=='wght_in' or param=='wght_out':
                    param_wght.append(i)
                elif param=='psum_in' or param=='psum_out':
                    param_ofmap.append(i)
                    
                typee=fault_type[i]
                if typee=='0':
                    type0.append(i)
                elif typee=='1':
                    type1.append(i)
                elif typee=='flip':
                    typef.append(i)

        return psum_idx_list,cnt_psidx,fault_bit,param_ifmap,param_wght,param_ofmap,type0,type1,typef
    
    def _param_find_fault_type(self, param_idx, SA0idx, SA1idx, flipidx):
        """ Find the extraction index for SA0, SA1, bit-flip fault in a parameter """
        searched_SA0=np.intersect1d(param_idx,SA0idx)
        searched_SA1=np.intersect1d(param_idx,SA1idx)
        searched_flip=np.intersect1d(param_idx,flipidx)
        searched_SA0=np.searchsorted(param_idx,searched_SA0)
        searched_SA1=np.searchsorted(param_idx,searched_SA1)
        searched_flip=np.searchsorted(param_idx,searched_flip)
        searched_SA0=np.expand_dims(searched_SA0,-1)
        searched_SA1=np.expand_dims(searched_SA1,-1)
        searched_flip=np.expand_dims(searched_flip,-1)
        
        return searched_SA0, searched_SA1, searched_flip
    
    def _polarity_check_type_grouping(self, data_tensor, fault_bit, data_idx, type0, type1, typef, wlpolar):
        """ Polarity check with type grouping and return the arranged combined polarity """
        # type grouping
        type0_idx, type1_idx, typef_idx=self._param_find_fault_type(data_idx,type0,type1,typef)
        
        # allocate type subgroup data & check polarity
        if len(type0_idx)>0:
            alloc_type0=tf.gather_nd(data_tensor,type0_idx)
            faultbit_type0=np.tile(fault_bit[type0_idx],[1,data_tensor.shape.dims[1].value])
            polarity_type0=self._polarity_check('0', faultbit_type0, alloc_type0, wlpolar)
            
        if len(type1_idx)>0:
            alloc_type1=tf.gather_nd(data_tensor,type1_idx)
            faultbit_type1=np.tile(fault_bit[type1_idx],[1,data_tensor.shape.dims[1].value])
            polarity_type1=self._polarity_check('1', faultbit_type1, alloc_type1, wlpolar)
            
        if len(typef_idx)>0:
            alloc_typef=tf.gather_nd(data_tensor,typef_idx)
            faultbit_typef=np.tile(fault_bit[typef_idx],[1,data_tensor.shape.dims[1].value])
            polarity_typef=self._polarity_check('flip', faultbit_typef, alloc_typef, wlpolar)

        # construct arranged combined polarrity
        polarity=tf.Variable(tf.zeros(data_tensor.shape))
        if len(type0_idx)>0:
            polarity=tf.scatter_nd_update(polarity,type0_idx,polarity_type0)
        if len(type1_idx)>0:
            polarity=tf.scatter_nd_update(polarity,type1_idx,polarity_type1)
        if len(typef_idx)>0:
            polarity=tf.scatter_nd_update(polarity,typef_idx,polarity_typef)
            
        return polarity
    
    def _fault_bit_ext4mult(self, fault_bit, num_psum_idx):
        """ Extend fault bit for tf.multiply
            Add a number of psum index axis and make 2's exponential
        """
        fault_bit=np.expand_dims(fault_bit,-1)
        fault_bit=np.tile(fault_bit,[1,num_psum_idx])
        fault_bit=np.power(2,fault_bit)
        
        return fault_bit
    
    def _layer_coor_order(self, layer_type):
        """ Give the coordinate access index order based on the layer type """
        if layer_type=='Conv2D':
            order_get_psidx_o=np.array([0,2,3,1])
            order_get_psidx_w=np.array([5,6,4,1])
            order_get_psidx_i=np.array([0,7,8,4])
        elif layer_type=='DepthwiseConv2D':
            order_get_psidx_o=np.array([0,2,3,1])
            order_get_psidx_w=np.array([5,6,1,4])
            order_get_psidx_i=np.array([0,7,8,4])
        elif layer_type=='Dense':
            order_get_psidx_o=np.array([0,1])
            order_get_psidx_w=np.array([2,1])
            order_get_psidx_i=np.array([0,2])
        else:
            raise ValueError('layer type must be one of \'Conv2D\', \'Dense\', \'DepthwiseConv2D\'')
        
        return order_get_psidx_o,order_get_psidx_w,order_get_psidx_i

    def inject_mac_math_fault_tensor(self, ifmap, wght, ofmap, fault_dict, 
                                     quantizer=None, quant_mode=None, layer_type='Conv2D',
                                     ksizes=(3,3), padding='valid', dilation_rates=(1,1), 
                                     sim_truncarry=None, fast_gen=None):
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
            quant_mode: String. The quantization mode of MAC. Be either 'intrinsic' or 'hybrid'.
                'intrinsic' means truncate before accumulation. 'hybrid' means accumulate with the word length of multiplier output than truncate.
            layer_type: String. The type of layer this solver wants to convert partial sum index and mapping into.
                One of 'Conv2D', 'Dense', 'DepthwiseConv2D'.
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
                
        if quant_mode is not None:
            self.quant_mode=quant_mode
        if sim_truncarry is None:
            sim_truncarry=self.sim_truncarry
        if fast_gen is not None:
            self.fast_gen=fast_gen
            
        order_get_psidx_o,order_get_psidx_w,order_get_psidx_i=self._layer_coor_order(layer_type)
        
        fd_coor=np.array(list(fault_dict.keys()))
        fd_value=np.array(list(fault_dict.values()))
        fdoutput_alloc=tf.gather_nd(ofmap,fd_coor)
        
        if self.fast_gen:
            # data allocation
            # (coor idx, num of psidx, psum idx)
            psum_idx_list=np.array([info['psum_idx'] for info in fd_value])
            psum_idx_ofmap=psum_idx_list[:,:,order_get_psidx_o]
        
            fault_param=fd_value[0]['param']
            fault_type=fd_value[0]['SA_type']
            fault_bit=fd_value[0]['SA_bit']
            
            if fault_param in ['ifmap_in','ifmap_out','wght_in','wght_out']:    
                psum_idx_wght=psum_idx_list[:,:,order_get_psidx_w]
                psum_idx_ifmap=psum_idx_list[:,:,order_get_psidx_i]
            
                if padding=='same' and layer_type!='Dense':
                    ifmap, psum_idx_ifmap=self._padding_ifmap_and_idx(ifmap, 
                                                                      psum_idx_ifmap, 
                                                                      ksizes, 
                                                                      dilation_rates)
    
                ifmap_alloc=tf.gather_nd(ifmap,psum_idx_ifmap)
                wght_alloc=tf.gather_nd(wght,psum_idx_wght) 
                
                ifmap_alloc=quantizer_input.left_shift_2int(ifmap_alloc)
                wght_alloc=quantizer_weight.left_shift_2int(wght_alloc)
            
            ofmap_alloc=tf.gather_nd(ofmap,psum_idx_ofmap)
            ofmap_alloc=quantizer_output.left_shift_2int(ofmap_alloc)

            # check polarity
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
                        
            # fault injection
            if fault_param=='ifmap_in' or fault_param=='ifmap_out' or fault_param=='wght_in' or fault_param=='wght_out':
                # fault injection of ifmap and wght => mac math FI
                if fault_param=='ifmap_in' or fault_param=='ifmap_out':
                    psum_alter= tf.multiply(wght_alloc,2**fault_bit)
                elif fault_param=='wght_in' or fault_param=='wght_out':
                    psum_alter= tf.multiply(ifmap_alloc,2**fault_bit)
                
                psum_alter=self.mac_math_alter_make(psum_alter, 
                                                    polarity, 
                                                    quantizer_output, 
                                                    sim_truncarry, 
                                                    ifmap_alloc, 
                                                    wght_alloc)
                
            #TODO
            # there is no way to know the psum of exact time frame
            # fault on psum do it the way in PE RTL test
            
            # fault injection of ofmap
            elif fault_param=='psum_in' or fault_param=='psum_out':
                psum_alter=tf.multiply(polarity,2**fault_bit)
                
                psum_alter=tf.reduce_sum(psum_alter, axis=1)
                psum_alter=quantizer_output.right_shift_back(psum_alter)
                psum_alter=quantizer_output.quantize_2half(psum_alter)

            # add psum_alter back to ofmap
            output=tf.add(fdoutput_alloc, psum_alter)
            output=quantizer_output.quantize(output)
            output=tf.tensor_scatter_update(ofmap,fd_coor,output)

                
        else: # slow loop gen
            # loop data extraction
            if isinstance(fd_value[0]['id'],int):
                (psum_idx_list,
                 cnt_psidx,
                 fault_bit,
                 param_ifmap,
                 param_wght,
                 param_ofmap,
                 type0,
                 type1,
                 typef)=\
                self._fault_value_extract_loop(fd_value, repetitive=False)
            
            elif isinstance(fd_value[0]['id'],list):
                (psum_idx_list,
                 cnt_psidx,
                 fault_bit,
                 param_ifmap,
                 param_wght,
                 param_ofmap,
                 type0,
                 type1,
                 typef)=\
                self._fault_value_extract_loop(fd_value, repetitive=True)
                
            # ofmap fault
            if len(param_ofmap)>0:
                # data gathering
                psum_idx_ofmap=psum_idx_list[param_ofmap]
                idx_ofmap=psum_idx_ofmap[:,:,order_get_psidx_o]
                faultbit_ofmap=fault_bit[param_ofmap]
                
                ofmap_alloc=tf.gather_nd(ofmap,idx_ofmap)
                ofmap_alloc=quantizer_output.left_shift_2int(ofmap_alloc)
                
                # check polarity
                if self.quant_mode=='intrinsic':
                    wlpolar = quantizer_output.nb
                elif self.quant_mode=='hybrid':
                    wlpolar = quantizer_input.nb+quantizer_weight.nb   
                    
                polarity_ofmap=self._polarity_check_type_grouping(ofmap_alloc,
                                                                  faultbit_ofmap,
                                                                  param_ofmap,
                                                                  type0,type1,typef,
                                                                  wlpolar)
                
            # ifmap fault
            if len(param_ifmap)>0:
                # data gathering
                psum_idx_ifmap=psum_idx_list[param_ifmap]
                idx_ifmap_ifmap=psum_idx_ifmap[:,:,order_get_psidx_i]
                idx_ifmap_wght=psum_idx_ifmap[:,:,order_get_psidx_w]
                faultbit_ifmap=fault_bit[param_ifmap]
            
                if padding=='same' and layer_type!='Dense':
                    ifmap, idx_ifmap_ifmap=self._padding_ifmap_and_idx(ifmap, 
                                                                       idx_ifmap_ifmap, 
                                                                       ksizes, 
                                                                       dilation_rates)
                ifmap_alloc_i=tf.gather_nd(ifmap,idx_ifmap_ifmap)
                ifmap_alloc_i=quantizer_input.left_shift_2int(ifmap_alloc_i)
                ifmap_alloc_w=tf.gather_nd(wght,idx_ifmap_wght)
                ifmap_alloc_w=quantizer_input.left_shift_2int(ifmap_alloc_w)
                
                # check polarity
                wlpolar = quantizer_input.nb
                
                polarity_ifmap=self._polarity_check_type_grouping(ifmap_alloc_i,
                                                                  faultbit_ifmap,
                                                                  param_ifmap,
                                                                  type0,type1,typef,
                                                                  wlpolar)
            
            # wght fault
            if len(param_wght)>0:
                # data gathering
                psum_idx_wght=psum_idx_list[param_wght]
                idx_wght_wght=psum_idx_wght[:,:,order_get_psidx_w]
                idx_wght_ifmap=psum_idx_wght[:,:,order_get_psidx_i]
                faultbit_wght=fault_bit[param_wght]
                
                if padding=='same' and layer_type!='Dense':
                    if len(param_ifmap)>0:
                        _, idx_wght_ifmap=self._padding_ifmap_and_idx(None, 
                                                                      idx_wght_ifmap, 
                                                                      ksizes, 
                                                                      dilation_rates)
                    else:
                        ifmap, idx_wght_ifmap=self._padding_ifmap_and_idx(ifmap, 
                                                                          idx_wght_ifmap, 
                                                                          ksizes, 
                                                                          dilation_rates)
                wght_alloc_w=tf.gather_nd(wght,idx_wght_wght) 
                wght_alloc_w=quantizer_weight.left_shift_2int(wght_alloc_w)
                wght_alloc_i=tf.gather_nd(ifmap,idx_wght_ifmap) 
                wght_alloc_i=quantizer_weight.left_shift_2int(wght_alloc_i)

                # check polarity
                wlpolar = quantizer_weight.nb
                
                polarity_wght=self._polarity_check_type_grouping(wght_alloc_w,
                                                                 faultbit_wght,
                                                                 param_wght,
                                                                 type0,type1,typef,
                                                                 wlpolar)
                
            # fault injection
            
            # ofmap fault injection
            if len(param_ofmap)>0:
                faultbit_ofmap=self._fault_bit_ext4mult(faultbit_ofmap, polarity_ofmap.shape.dims[1].value)
                psum_alter_ofmap=tf.multiply(polarity_ofmap, faultbit_ofmap)
                
                psum_alter_ofmap=tf.reduce_sum(psum_alter_ofmap, axis=1)
                psum_alter_ofmap=quantizer_output.right_shift_back(psum_alter_ofmap)
                psum_alter_ofmap=quantizer_output.quantize_2half(psum_alter_ofmap)
                
            # ifmap fault injection
            if len(param_ifmap)>0:
                faultbit_ifmap=self._fault_bit_ext4mult(faultbit_ifmap, polarity_ifmap.shape.dims[1].value)
                psum_alter_ifmap=tf.multiply(ifmap_alloc_w, faultbit_ifmap)
                
                psum_alter_ifmap=self.mac_math_alter_make(psum_alter_ifmap, 
                                                          polarity_ifmap, 
                                                          quantizer_output, 
                                                          sim_truncarry, 
                                                          ifmap_alloc_i, 
                                                          ifmap_alloc_w)
                
            # wght fault injection
            if len(param_wght)>0:
                faultbit_wght=self._fault_bit_ext4mult(faultbit_wght, polarity_wght.shape.dims[1].value)
                psum_alter_wght=tf.multiply(wght_alloc_i, faultbit_wght)
                
                psum_alter_wght=self.mac_math_alter_make(psum_alter_wght, 
                                                         polarity_wght, 
                                                         quantizer_output, 
                                                         sim_truncarry, 
                                                         wght_alloc_i, 
                                                         wght_alloc_w)
                
            
            # re-construct layer wise psum_alter
            psum_alter=tf.Variable(tf.zeros(psum_idx_list.shape[0]))

            if len(param_ofmap)>0:
                param_ofmap=np.expand_dims(param_ofmap,-1)
                psum_alter=tf.scatter_nd_update(psum_alter, param_ofmap, psum_alter_ofmap)
            if len(param_ifmap)>0:
                param_ifmap=np.expand_dims(param_ifmap,-1)
                psum_alter=tf.scatter_nd_update(psum_alter, param_ifmap, psum_alter_ifmap)
            if len(param_wght)>0:
                param_wght=np.expand_dims(param_wght,-1)
                psum_alter=tf.scatter_nd_update(psum_alter, param_wght, psum_alter_wght)

            if cnt_psidx is not None:
                psum_alter=tf.split(psum_alter,cnt_psidx)
                for alteritem in psum_alter:
                    alteritem=tf.reduce_sum(alteritem)
                psum_alter=tf.stack(psum_alter)
                psum_alter=quantizer_output.quantize(psum_alter)
            
            # add psum_alter back to ofmap
            output=tf.add(fdoutput_alloc, psum_alter)
            output=quantizer_output.quantize(output)
            output=tf.tensor_scatter_update(ofmap,fd_coor,output)
            
        return output
    
    def consistency_check(self, quant_mode, quantizer):
        """ Check the consistency between MAC unit setup and layer quantization setup """
        if self.quant_mode!=quant_mode:
            raise ValueError('The quantization mode of Layer and MAC unit must be the same!!\nGot Layer %s, MAC %s'%(self.quant_mode,quant_mode))
        if type(self.quantizer)!=type(quantizer):
            raise TypeError('The type of quantizer of Layer and MAC unit should be the same!!\n Class quantizer for one quantize set up, List for [input, weight, output] respectively.')
        if self.quantizer!=quantizer:
            raise AttributeError('The attributes of Layer quantizer and MAC unit quantizer are different!!')

                
        
