# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 15:54:51 2020

@author: Yung-Yu Tsai

Fault Tensor operations for MAC faults. Including MAC math fault injection and MAC noise fault injection
"""
import tensorflow as tf

class mac_fault_injector:
    """ The mac fault injection class.
        Responsible for Tensor operations of mac math fault injection and mac noise fault injection.
        The fault information was pass in by mac_unit and fault_dict.
        
    Arguments
    ---------
    mac_unit: mac_unit Class.
    
        | The holder for fault injection configurations.
        | quantizer: The quantize library of simulator.
        | quant_mode: The quantization mode of MAC. 
        | noise_inject: Using mac noise fault injection method or not.
        | sim_truncarry: Simulate the truncation carry during mac math fault injection. 
        | fast_gen: Use fast generation or not.
     
    fault_dict: Dictionary.
        The holder for fault injection data i.e. modulators and arrays for tf.constant.
    
    fault_dict description: (Dictionary)
    -----------------------------------------
    | mac math fault injection unique fault
    >>> preprocess_data={'fd_coor': 2D Ndarray, #the coordinate of fault in layer
    ...                  'psum_idx_ofmap': 2D Ndarray, #faulty ofmap coordinate
    ...                  'fault_param': String, #the faulty param
    ...                  'fault_type': String, #the fault type
    ...                  'fault_bit': Integer, #the fault bit order
    ...                  'psum_idx_ifmap': 2D Ndarray, #faulty ifmap coordinate
    ...                  'psum_idx_wght': 2D Ndarray, #faulty wght coordinate
    ...                  'modulator': Ndarray, #fault_bit order coefficient
    ...                  'signbit': Bool or Ndarray, #fault_bit on sign bit indication
    ...                  }

    | mac math fault injection scatter fault
    >>> preprocess_data={'fd_coor': 2D Ndarray, #the coordinate of fault in layer
    ...                  'param_ofmap': 1D Ndarray, #the faulty ofmap param index
    ...                  'param_ifmap': 1D Ndarray, #the faulty ifmap param index
    ...                  'param_wght': 1D Ndarray, #the faulty wght param index
    ...                  'idx_ofmap': 2D Ndarray, #faulty ofmap coordinate
    ...                  'modulator_ofmap': List, #polarty data list for ofmap fault
    ...                  'idx_ifmap_ifmap': 2D Ndarray, #faulty ifmap coordinate
    ...                  'idx_ifmap_wght': 2D Ndarray, #faulty ifmap correspond wght coordinate
    ...                  'modulator_ifmap': List, #polarty data list for ifmap fault
    ...                  'idx_wght_wght': 2D Ndarray, #faulty wght coordinate
    ...                  'idx_wght_ifmap': 2D Ndarray, #faulty wght correspond ifmap coordinate
    ...                  'modulator_wght': List, #polarty data list for wght fault
    ...                  'faultbit_ofmap': 1D Ndarray, #the ofmap fault bit order
    ...                  'faultbit_ifmap': 1D Ndarray, #the ifmap fault bit order
    ...                  'faultbit_wght': 1D Ndarray, #the wght fault bit order
    ...                  'cnt_psidx': 1D Ndarray, #the counter for uneven psidx number on single ofmap pixel
    ...                  'psum_idx_list_len': Integer, #number of psum_idx
    ...                  }
    
    | mac noise fault injection unique fault
    >>> preprocess_data={'stddev_amp_ofmap': 4D Ndarray} 
    ... #the standard deviation amplifier mask, same shape as target ofmap

    | mac noise fault injection scatter fault
    >>> preprocess_data={'stddev_amp_ofmap': 4D Ndarray} 
    ... #the standard deviation amplifier mask, same shape as target ofmap

        
    """
    def __init__(self, mac_unit, fault_dict=None):
        """ mac fault injector initializer """
        self.fault_dict=fault_dict
        
        self.quantizer=mac_unit.quantizer
        self.quant_mode=mac_unit.quant_mode

        self.noise_inject=mac_unit.noise_inject
        self.sim_truncarry=mac_unit.sim_truncarry
        self.fast_gen=mac_unit.fast_gen
        
    def _padding_ifmap(self, ifmap, ksizes, dilation_rates):
        """ Preproccess ifmap data and index for padding situation """
        dilated_ksize_row_edge = (ksizes[0] + (ksizes[0]-1) * (dilation_rates[0] - 1))//2
        dilated_ksize_col_edge = (ksizes[1] + (ksizes[1]-1) * (dilation_rates[1] - 1))//2
                
        pad=[[0,0],[dilated_ksize_row_edge,dilated_ksize_row_edge],[dilated_ksize_col_edge,dilated_ksize_col_edge],[0,0]]
        ifmap=tf.pad(ifmap,pad,'constant',constant_values=0)
        
        return ifmap
    
    def _polarity_check(self, fault_type, FI_param, modulator, signbit):
        """ Get the polarity of parameter """
        modulator=tf.constant(modulator)
        polarity=tf.bitwise.bitwise_and(FI_param,modulator)
        polarity=tf.math.sign(polarity)
        if fault_type=='flip':
            polarity=tf.math.add(tf.multiply(polarity,tf.constant(-2)),tf.constant(1))
        elif fault_type=='0':    
            polarity=tf.math.negative(polarity)
        elif fault_type=='1':    
            polarity=tf.math.subtract(tf.constant(1),polarity)
                
        if isinstance(signbit,bool):
            if signbit:
                polarity=tf.math.negative(polarity)
        else:
            signbit=tf.constant(signbit)
            polarity=tf.math.multiply(polarity,signbit)
                        
        return polarity
    
    def _polarity_check_type_grouping(self, data_tensor, type0_idx, modulator0, signbit0, type1_idx, modulator1, signbit1, typef_idx, modulatorf, signbitf):
        """ Polarity check with type grouping and return the arranged combined polarity """
        if len(type0_idx)>0:
            type0FI=True
        else:
            type0FI=False
        if len(type1_idx)>0:
            type1FI=True
        else:
            type1FI=False
        if len(typef_idx)>0:
            typefFI=True
        else:
            typefFI=False
        
        # allocate type subgroup data & check polarity
        if type0FI:
            type0_idx=tf.constant(type0_idx)
            alloc_type0=tf.gather_nd(data_tensor,type0_idx)
            polarity_type0=self._polarity_check('0', alloc_type0, modulator0, signbit0)
            
        if type1FI:
            type1_idx=tf.constant(type1_idx)
            alloc_type1=tf.gather_nd(data_tensor,type1_idx)
            polarity_type1=self._polarity_check('1', alloc_type1, modulator1, signbit1)
            
        if typefFI:
            typef_idx=tf.constant(typef_idx)
            alloc_typef=tf.gather_nd(data_tensor,typef_idx)
            polarity_typef=self._polarity_check('flip', alloc_typef, modulatorf, signbitf)

        polarity=tf.zeros(data_tensor.shape, dtype=tf.int32)
        if type0FI:
            polarity=tf.tensor_scatter_nd_update(polarity,type0_idx,polarity_type0)
        if type1FI:
            polarity=tf.tensor_scatter_nd_update(polarity,type1_idx,polarity_type1)
        if typefFI:
            polarity=tf.tensor_scatter_nd_update(polarity,typef_idx,polarity_typef)
            
        return polarity    
    
    def mac_math_alter_make(self, psum_alter, polarity, quantizer_output, sim_truncarry=False, ifmap_alloc=None, wght_alloc=None):
        """ The core funciton of create mac math fault injection alteration Tensor
            This alteration will be later add onto ofmap tensor
            
            psum_alter: The product data by multiply coefficient data and fault bit order. That is:
                >>> if fault_param=='ifmap_in' or fault_param=='ifmap_out':
                ...     psum_alter= tf.multiply(wght_alloc,2**fault_bit)
                ... elif fault_param=='wght_in' or fault_param=='wght_out':
                ...     psum_alter= tf.multiply(ifmap_alloc,2**fault_bit)
        """

        if self.quant_mode=='intrinsic':
            if sim_truncarry:
                trivia_alter= tf.floormod(psum_alter, tf.constant(quantizer_output.shift_factor))
                trivia_conv= tf.floormod(tf.multiply(ifmap_alloc,wght_alloc), tf.constant(quantizer_output.shift_factor))
                trivia_conv= tf.multiply(trivia_conv, polarity)
                truncarry= tf.add(trivia_alter, trivia_conv)
                
                comparator= tf.multiply(tf.floordiv(tf.add(polarity,tf.constant(1)),tf.constant(2)), tf.constant(quantizer_output.shift_factor-1))
                truncarry= tf.sign(tf.subtract(truncarry, comparator))
            
            psum_alter=quantizer_output.right_shift_back(psum_alter)
            psum_alter=quantizer_output.round_through(psum_alter)
            psum_alter=quantizer_output.capping(psum_alter)
            if sim_truncarry:
                psum_alter=tf.add(psum_alter,truncarry)
            psum_alter=tf.multiply(psum_alter, tf.cast(polarity,tf.float32))
            
            # sum all psum_alter
            psum_alter=tf.reduce_sum(psum_alter, axis=1)
            psum_alter=quantizer_output.right_shift_back(psum_alter)
            
        elif self.quant_mode=='hybrid':
            psum_alter=tf.multiply(polarity, psum_alter)
            
            # sum all psum_alter
            psum_alter=tf.reduce_sum(psum_alter, axis=1)
            psum_alter=quantizer_output.right_shift_back(psum_alter)
            psum_alter=quantizer_output.right_shift_back(psum_alter)
            
        return psum_alter
        
    def inject_mac_math_fault_tensor(self, ifmap, wght, ofmap, fault_dict, 
                                     quantizer=None, quant_mode=None, layer_type='Conv2D',
                                     ksizes=(3,3), padding='valid', dilation_rates=(1,1), 
                                     sim_truncarry=None, fast_gen=None):
        """ The fault injection mathematical model for used in Layer Tensor computing.
            Include both fast generation (fast_gen is True) and scattered generation (fast_gen is False).
            Fast generation means the all the fault information dict are refer to the same source defect 
            with the same fault bit, SA type and param.
            Scattered generation means the faults are from distinct defect sources that need to be split into 
            different groups and generate respectively. Therefore it's slower.
            
            This function is for inject the preprocessed fault data in Keras Layer/Model call.
            Seperate the CPU and GPU processing. The preprocess function under mac_fault_injector class are for GPU process.
            
            This function is not suitable for tf.function decorator which needs to solve the Tensor Graph.
            It may take a unnecessary long time to solve and get invalid shape.
            Recommand using inject_mac_math_fault_uni or inject_mac_math_fault_scatter if you are sure about your
            fault dictionry defect source is unique or not.
        
        Arguments
        ---------
        ifmap: Tensor. 
            Quantized Tensor. Layer input.
        wght: Tensor. 
            Quantized Tensor. Layer output.
        ofmap: Tensor. 
            The Tensor to be injected fault by math alteration. Quantized Tensor. Layer output.
        fault_dict: Dictionary or List. 
            The dictionary contain fault list information.
        quantizer: Class or List. 
            | The quantizer class, one or in list [input, weight, output]. The quantizer class contain following quantize operation infromation.
            | word_width: Variable. The fix-point representation of the parameter word length.
            | fractional_bits: Variable. Number of fractional bits in a fix-point parameter
            | rounding: String. Rounding method of quantization, augment must be one of 'nearest' , 'down', 'zero', 'stochastic'.
        quant_mode: String. Be either 'intrinsic' or 'hybrid'.
            | The quantization mode of MAC.
            | 'intrinsic' means truncate before accumulation. 
            | 'hybrid' means accumulate with the word length of multiplier output than truncate.
        layer_type: String. One of 'Conv2D', 'Dense', 'DepthwiseConv2D'.
            The type of layer this solver wants to convert partial sum index and mapping into.
        ksize: Tuple. Size 2. 
            The kernel size (row, col).
        padding: String. 'same' or 'valid'. 
            The type of padding algorithm to use.
        dilation_rate: Tuple. Size 2. 
            The dilation rate (row, col).
        sim_truncarry: Bool. 
            Simulate the truncation carry during mac math fault injection. 
            The truncation carry is cause by the carry-out of thrown away fractional bits. 
            It could be 1 in addition or -1 in subtraction, when the unwanted fractional bits value overflow to the needed fractional bits.
            How ever this could cause huge time overhead for absolute bit-true model.
        fast_gen: Bool. 
            Use fast generation or not. Fast generation has the same fault bit and SA type for all coordinates.
            The fault dictionay is form by fault data contamination.
    
        Returns
        -------
        Tensor. 
            The amount of adjustment apply to output feature map of a DNN layer which represent the faulty behabvior of MAC unit.
        
        Reminder!!
        ----------
        padding: Padding is tie to layer setting. 
            If user treat padding as a seperate layer.
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
                    
        fd_coor=tf.constant(fault_dict['fd_coor'])
        fdoutput_alloc=tf.gather_nd(ofmap,fd_coor)
        
        if self.fast_gen:
            # data allocation
            # (coor idx, num of psidx, psum idx)
            psum_idx_ofmap=tf.constant(fault_dict['psum_idx_ofmap'])
        
            fault_param=fault_dict['fault_param']
            fault_type=fault_dict['fault_type']
            fault_bit=fault_dict['fault_bit']
            
            if fault_param in ['ifmap_in','ifmap_out','wght_in','wght_out']:               
                if padding=='same' and layer_type!='Dense':
                    ifmap=self._padding_ifmap(ifmap, ksizes, dilation_rates)
    
                psum_idx_ifmap=tf.constant(fault_dict['psum_idx_ifmap'])
                psum_idx_wght=tf.constant(fault_dict['psum_idx_wght'])
                
                ifmap_alloc=tf.gather_nd(ifmap,psum_idx_ifmap)
                wght_alloc=tf.gather_nd(wght,psum_idx_wght) 
                
                ifmap_alloc=quantizer_input.left_shift_2int(ifmap_alloc)
                wght_alloc=quantizer_weight.left_shift_2int(wght_alloc)
            
            ofmap_alloc=tf.gather_nd(ofmap,psum_idx_ofmap)
            ofmap_alloc=quantizer_output.left_shift_2int(ofmap_alloc)

            # check polarity
            if fault_param=='ifmap_in' or fault_param=='ifmap_out':
                FI_param = ifmap_alloc
            elif fault_param=='wght_in' or fault_param=='wght_out':
                FI_param = wght_alloc
            elif fault_param=='psum_in' or fault_param=='psum_out':
                FI_param = ofmap_alloc
                
            modulator=fault_dict['modulator']
            signbit=fault_dict['signbit']
            
            polarity=self._polarity_check(fault_type, FI_param, modulator, signbit)
                        
            # fault injection
            if fault_param=='ifmap_in' or fault_param=='ifmap_out' or fault_param=='wght_in' or fault_param=='wght_out':
                # fault injection of ifmap and wght => mac math FI
                if fault_param=='ifmap_in' or fault_param=='ifmap_out':
                    psum_alter= tf.multiply(wght_alloc,tf.constant(2**fault_bit))
                elif fault_param=='wght_in' or fault_param=='wght_out':
                    psum_alter= tf.multiply(ifmap_alloc,tf.constant(2**fault_bit))
                
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
                psum_alter=tf.multiply(polarity,tf.constant(2**fault_bit))
                
                psum_alter=tf.reduce_sum(psum_alter, axis=1)
                psum_alter=quantizer_output.right_shift_back(psum_alter)
                psum_alter=quantizer_output.right_shift_back(psum_alter)

            # add psum_alter back to ofmap
            output=tf.add(fdoutput_alloc, psum_alter)
            output=tf.tensor_scatter_nd_update(ofmap,fd_coor,output)

                
        else: # slow loop gen
            param_ofmap=fault_dict['param_ofmap']
            param_ifmap=fault_dict['param_ifmap']
            param_wght=fault_dict['param_wght']
            FI_ofmap=len(param_ofmap)>0
            FI_ifmap=len(param_ifmap)>0
            FI_wght=len(param_wght)>0
            
            # ofmap fault
            if FI_ofmap:
                # data gathering
                idx_ofmap=tf.constant(fault_dict['idx_ofmap'])
                
                ofmap_alloc=tf.gather_nd(ofmap,idx_ofmap)
                ofmap_alloc=quantizer_output.left_shift_2int(ofmap_alloc)
                
                # check polarity
                modulator_ofmap=fault_dict['modulator_ofmap']
                polarity_ofmap=self._polarity_check_type_grouping(ofmap_alloc, *modulator_ofmap)
                
            # ifmap fault
            if FI_ifmap:
                # data gathering            
                if padding=='same' and layer_type!='Dense':
                    ifmap=self._padding_ifmap(ifmap, ksizes, dilation_rates)
                
                idx_ifmap_ifmap=tf.constant(fault_dict['idx_ifmap_ifmap'])
                idx_ifmap_wght=tf.constant(fault_dict['idx_ifmap_wght'])
                
                ifmap_alloc_i=tf.gather_nd(ifmap,idx_ifmap_ifmap)
                ifmap_alloc_i=quantizer_input.left_shift_2int(ifmap_alloc_i)
                ifmap_alloc_w=tf.gather_nd(wght,idx_ifmap_wght)
                ifmap_alloc_w=quantizer_input.left_shift_2int(ifmap_alloc_w)
                
                # check polarity
                modulator_ifmap=fault_dict['modulator_ifmap']
                polarity_ifmap=self._polarity_check_type_grouping(ifmap_alloc_i, *modulator_ifmap)
            
            # wght fault
            if FI_wght:
                if padding=='same' and layer_type!='Dense':
                    if not FI_ifmap:
                        ifmap=self._padding_ifmap(ifmap, ksizes, dilation_rates)
                        
                idx_wght_wght=tf.constant(fault_dict['idx_wght_wght'])
                idx_wght_ifmap=tf.constant(fault_dict['idx_wght_ifmap'])
                        
                wght_alloc_w=tf.gather_nd(wght,idx_wght_wght) 
                wght_alloc_w=quantizer_weight.left_shift_2int(wght_alloc_w)
                wght_alloc_i=tf.gather_nd(ifmap,idx_wght_ifmap) 
                wght_alloc_i=quantizer_weight.left_shift_2int(wght_alloc_i)

                # check polarity
                modulator_wght=fault_dict['modulator_wght']
                polarity_wght=self._polarity_check_type_grouping(wght_alloc_w, *modulator_wght)
                
            # fault injection
            
            # ofmap fault injection
            if FI_ofmap:
                faultbit_ofmap=tf.constant(fault_dict['faultbit_ofmap'])
                psum_alter_ofmap=tf.multiply(polarity_ofmap, faultbit_ofmap)
                
                psum_alter_ofmap=tf.reduce_sum(psum_alter_ofmap, axis=1)
                psum_alter_ofmap=quantizer_output.right_shift_back(psum_alter_ofmap)
                psum_alter_ofmap=quantizer_output.right_shift_back(psum_alter_ofmap)
                
            # ifmap fault injection
            if FI_ifmap:
                faultbit_ifmap=tf.constant(fault_dict['faultbit_ifmap'])
                psum_alter_ifmap=tf.multiply(ifmap_alloc_w, faultbit_ifmap)
                
                psum_alter_ifmap=self.mac_math_alter_make(psum_alter_ifmap, 
                                                          polarity_ifmap, 
                                                          quantizer_output, 
                                                          sim_truncarry, 
                                                          ifmap_alloc_i, 
                                                          ifmap_alloc_w)
                
            # wght fault injection
            if FI_wght:
                faultbit_wght=tf.constant(fault_dict['faultbit_wght'])
                psum_alter_wght=tf.multiply(wght_alloc_i, faultbit_wght)
                
                psum_alter_wght=self.mac_math_alter_make(psum_alter_wght, 
                                                         polarity_wght, 
                                                         quantizer_output, 
                                                         sim_truncarry, 
                                                         wght_alloc_i, 
                                                         wght_alloc_w)
                
            psum_alter=tf.zeros(fault_dict['psum_idx_list_len'])

            if FI_ofmap:
                param_ofmap=tf.constant(param_ofmap)
                psum_alter=tf.tensor_scatter_nd_update(psum_alter, param_ofmap, psum_alter_ofmap)
            if FI_ifmap:
                param_ifmap=tf.constant(param_ifmap)
                psum_alter=tf.tensor_scatter_nd_update(psum_alter, param_ifmap, psum_alter_ifmap)
            if FI_wght:
                param_wght=tf.constant(param_wght)
                psum_alter=tf.tensor_scatter_nd_update(psum_alter, param_wght, psum_alter_wght)

            cnt_psidx=fault_dict['cnt_psidx']
            if cnt_psidx is not None:
                psum_alter=tf.split(psum_alter,cnt_psidx)
                for alteritem in psum_alter:
                    alteritem=tf.reduce_sum(alteritem)
                psum_alter=tf.stack(psum_alter)
                psum_alter=quantizer_output.quantize(psum_alter)
            
            # add psum_alter back to ofmap
            output=tf.add(fdoutput_alloc, psum_alter)
            output=tf.tensor_scatter_nd_update(ofmap,fd_coor,output)
            
        return output
    
    def inject_mac_math_fault_uni(self, ifmap, wght, ofmap, fault_dict, 
                                  quantizer=None, quant_mode=None, layer_type='Conv2D',
                                  ksizes=(3,3), padding='valid', dilation_rates=(1,1), 
                                  sim_truncarry=None):
        """ The fault injection mathematical model for used in Lyer Tensor computing.
            This function is only for fast generation.
            Fast generation means the all the fault information dict are refer to the same source defect 
            with the same fault bit, SA type and param.
            
            This function is for inject the preprocessed fault data in Keras Layer/Model call.
            Seperate the CPU and GPU processing. The preprocess function under mac_fault_injector class are for GPU process.
        
        Arguments
        ---------
        ifmap: Tensor. 
            Quantized Tensor. Layer input.
        wght: Tensor. 
            Quantized Tensor. Layer output.
        ofmap: Tensor. 
            The Tensor to be injected fault by math alteration. Quantized Tensor. Layer output.
        fault_dict: Dictionary or List. 
            The dictionary contain fault list information.
        quantizer: Class or List. 
            The quantizer class, one or in list [input, weight, output]. The quantizer class contain following quantize operation infromation.
            word_width: Variable. The fix-point representation of the parameter word length.
            fractional_bits: Variable. Number of fractional bits in a fix-point parameter
            rounding: String. Rounding method of quantization, augment must be one of 'nearest' , 'down', 'zero', 'stochastic'.
        quant_mode: String. Be either 'intrinsic' or 'hybrid'.
            The quantization mode of MAC. 
            'intrinsic' means truncate before accumulation. 'hybrid' means accumulate with the word length of multiplier output than truncate.
        layer_type: String. One of 'Conv2D', 'Dense', 'DepthwiseConv2D'.
            The type of layer this solver wants to convert partial sum index and mapping into.
        ksize: Tuple. Size 2.
            The kernel size (row, col).
        padding: String. 'same' or 'valid'. 
            The type of padding algorithm to use.
        dilation_rate: Tuple. Size 2.
            The dilation rate (row, col).
        sim_truncarry: Bool. 
            simulate the truncation carry during mac math fault injection. 
            The truncation carry is cause by the carry-out of thrown away fractional bits. 
            It could be 1 in addition or -1 in subtraction, when the unwanted fractional bits value overflow to the needed fractional bits.
            How ever this could cause huge time overhead for absolute bit-true model.
        
        Returns
        -------
        Tensor. 
            The amount of adjustment apply to output feature map of a DNN layer which represent the faulty behabvior of MAC unit.
        
        Reminder!!
        ----------
        padding: Padding is tie to layer setting. 
            If user treat padding as a seperate layer.
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
               
        fd_coor=fault_dict['fd_coor']
        fd_coor=tf.constant(fd_coor)
        fdoutput_alloc=tf.gather_nd(ofmap,fd_coor)
        
        # data allocation
        # (coor idx, num of psidx, psum idx)
        psum_idx_ofmap=tf.constant(fault_dict['psum_idx_ofmap'])
    
        fault_param=fault_dict['fault_param']
        fault_type=fault_dict['fault_type']
        fault_bit=fault_dict['fault_bit']
        
        if fault_param in ['ifmap_in','ifmap_out','wght_in','wght_out']:               
            if padding=='same' and layer_type!='Dense':
                ifmap=self._padding_ifmap(ifmap, ksizes, dilation_rates)

            psum_idx_ifmap=tf.constant(fault_dict['psum_idx_ifmap'])
            psum_idx_wght=tf.constant(fault_dict['psum_idx_wght'])
            
            ifmap_alloc=tf.gather_nd(ifmap,psum_idx_ifmap)
            wght_alloc=tf.gather_nd(wght,psum_idx_wght) 
            
            ifmap_alloc=quantizer_input.left_shift_2int(ifmap_alloc)
            wght_alloc=quantizer_weight.left_shift_2int(wght_alloc)
        
        ofmap_alloc=tf.gather_nd(ofmap,psum_idx_ofmap)
        ofmap_alloc=quantizer_output.left_shift_2int(ofmap_alloc)

        # check polarity
        if fault_param=='ifmap_in' or fault_param=='ifmap_out':
            FI_param = ifmap_alloc
        elif fault_param=='wght_in' or fault_param=='wght_out':
            FI_param = wght_alloc
        elif fault_param=='psum_in' or fault_param=='psum_out':
            FI_param = ofmap_alloc
            
        modulator=fault_dict['modulator']
        signbit=fault_dict['signbit']
        
        polarity=self._polarity_check(fault_type, FI_param, modulator, signbit)
                    
        # fault injection
        if fault_param=='ifmap_in' or fault_param=='ifmap_out' or fault_param=='wght_in' or fault_param=='wght_out':
            # fault injection of ifmap and wght => mac math FI
            if fault_param=='ifmap_in' or fault_param=='ifmap_out':
                psum_alter= tf.multiply(wght_alloc,tf.constant(2**fault_bit))
            elif fault_param=='wght_in' or fault_param=='wght_out':
                psum_alter= tf.multiply(ifmap_alloc,tf.constant(2**fault_bit))
            
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
            psum_alter=tf.multiply(polarity,tf.constant(2**fault_bit))
            
            psum_alter=tf.reduce_sum(psum_alter, axis=1)
            psum_alter=quantizer_output.right_shift_back(psum_alter)
            psum_alter=quantizer_output.right_shift_back(psum_alter)

        # add psum_alter back to ofmap
        output=tf.add(fdoutput_alloc, psum_alter)
        output=tf.tensor_scatter_nd_update(ofmap,fd_coor,output)
            
        return output
        
    def inject_mac_math_fault_scatter(self, ifmap, wght, ofmap, fault_dict, 
                                      quantizer=None, quant_mode=None, layer_type='Conv2D',
                                      ksizes=(3,3), padding='valid', dilation_rates=(1,1), 
                                      sim_truncarry=None):
        """ The fault injection mathematical model for used in Layer Tensor computing
            Include both fast generation (fast_gen is True) and scattered generation (fast_gen is False).
            Scattered generation means the faults are from distinct defect sources that need to be split into 
            different groups and generate respectively. Therefore it's slower.
            
            This function is for inject the preprocessed fault data in Keras Layer/Model call.
            Seperate the CPU and GPU processing. The preprocess function under mac_fault_injector class are for GPU process.
        
        Arguments
        ---------
        ifmap: Tensor. 
            Quantized Tensor. Layer input.
        wght: Tensor. 
            Quantized Tensor. Layer output.
        ofmap: Tensor. 
            The Tensor to be injected fault by math alteration. Quantized Tensor. Layer output.
        fault_dict: Dictionary or List. 
            The dictionary contain fault list information.
        quantizer: Class or List. 
            The quantizer class, one or in list [input, weight, output]. The quantizer class contain following quantize operation infromation.
            word_width: Variable. The fix-point representation of the parameter word length.
            fractional_bits: Variable. Number of fractional bits in a fix-point parameter
            rounding: String. Rounding method of quantization, augment must be one of 'nearest' , 'down', 'zero', 'stochastic'.
        quant_mode: String. Be either 'intrinsic' or 'hybrid'.
            The quantization mode of MAC. 
            'intrinsic' means truncate before accumulation. 'hybrid' means accumulate with the word length of multiplier output than truncate.
        layer_type: String. One of 'Conv2D', 'Dense', 'DepthwiseConv2D'.
            The type of layer this solver wants to convert partial sum index and mapping into.
        ksize: Tuple. Size 2. 
            The kernel size (row, col).
        padding: String. 'same' or 'valid'. 
            The type of padding algorithm to use.
        dilation_rate: Tuple. Size 2. 
            The dilation rate (row, col).
        sim_truncarry: Bool. 
            simulate the truncation carry during mac math fault injection. 
            The truncation carry is cause by the carry-out of thrown away fractional bits. 
            It could be 1 in addition or -1 in subtraction, when the unwanted fractional bits value overflow to the needed fractional bits.
            How ever this could cause huge time overhead for absolute bit-true model.
        fast_gen: Bool. 
            Use fast generation or not. Fast generation has the same fault bit and SA type for all coordinates.
            The fault dictionay is form by fault data contamination.
        
        Returns
        -------
        Tensor. 
            The amount of adjustment apply to output feature map of a DNN layer which represent the faulty behabvior of MAC unit.
        
        Reminder!!
        ----------
        padding: Padding is tie to layer setting. 
            If user treat padding as a seperate layer.
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
                    
        fd_coor=tf.constant(fault_dict['fd_coor'])
        fdoutput_alloc=tf.gather_nd(ofmap,fd_coor)
        
        param_ofmap=fault_dict['param_ofmap']
        param_ifmap=fault_dict['param_ifmap']
        param_wght=fault_dict['param_wght']
        FI_ofmap=len(param_ofmap)>0
        FI_ifmap=len(param_ifmap)>0
        FI_wght=len(param_wght)>0
        
        # ofmap fault
        if FI_ofmap:
            # data gathering
            idx_ofmap=tf.constant(fault_dict['idx_ofmap'])
            
            ofmap_alloc=tf.gather_nd(ofmap,idx_ofmap)
            ofmap_alloc=quantizer_output.left_shift_2int(ofmap_alloc)
            
            # check polarity
            modulator_ofmap=fault_dict['modulator_ofmap']
            polarity_ofmap=self._polarity_check_type_grouping(ofmap_alloc, *modulator_ofmap)
            
        # ifmap fault
        if FI_ifmap:
            # data gathering            
            if padding=='same' and layer_type!='Dense':
                ifmap=self._padding_ifmap(ifmap, ksizes, dilation_rates)
            
            idx_ifmap_ifmap=tf.constant(fault_dict['idx_ifmap_ifmap'])
            idx_ifmap_wght=tf.constant(fault_dict['idx_ifmap_wght'])
            
            ifmap_alloc_i=tf.gather_nd(ifmap,idx_ifmap_ifmap)
            ifmap_alloc_i=quantizer_input.left_shift_2int(ifmap_alloc_i)
            ifmap_alloc_w=tf.gather_nd(wght,idx_ifmap_wght)
            ifmap_alloc_w=quantizer_input.left_shift_2int(ifmap_alloc_w)
            
            # check polarity
            modulator_ifmap=fault_dict['modulator_ifmap']
            polarity_ifmap=self._polarity_check_type_grouping(ifmap_alloc_i, *modulator_ifmap)
        
        # wght fault
        if FI_wght:
            if padding=='same' and layer_type!='Dense':
                if not FI_ifmap:
                    ifmap=self._padding_ifmap(ifmap, ksizes, dilation_rates)
                    
            idx_wght_wght=tf.constant(fault_dict['idx_wght_wght'])
            idx_wght_ifmap=tf.constant(fault_dict['idx_wght_ifmap'])
                    
            wght_alloc_w=tf.gather_nd(wght,idx_wght_wght) 
            wght_alloc_w=quantizer_weight.left_shift_2int(wght_alloc_w)
            wght_alloc_i=tf.gather_nd(ifmap,idx_wght_ifmap) 
            wght_alloc_i=quantizer_weight.left_shift_2int(wght_alloc_i)

            # check polarity
            modulator_wght=fault_dict['modulator_wght']
            polarity_wght=self._polarity_check_type_grouping(wght_alloc_w, *modulator_wght)
            
        # fault injection
        
        # ofmap fault injection
        if FI_ofmap:
            faultbit_ofmap=tf.constant(fault_dict['faultbit_ofmap'])
            psum_alter_ofmap=tf.multiply(polarity_ofmap, faultbit_ofmap)
            
            psum_alter_ofmap=tf.reduce_sum(psum_alter_ofmap, axis=1)
            psum_alter_ofmap=quantizer_output.right_shift_back(psum_alter_ofmap)
            psum_alter_ofmap=quantizer_output.right_shift_back(psum_alter_ofmap)
            
        # ifmap fault injection
        if FI_ifmap:
            faultbit_ifmap=tf.constant(fault_dict['faultbit_ifmap'])
            psum_alter_ifmap=tf.multiply(ifmap_alloc_w, faultbit_ifmap)
            
            psum_alter_ifmap=self.mac_math_alter_make(psum_alter_ifmap, 
                                                      polarity_ifmap, 
                                                      quantizer_output, 
                                                      sim_truncarry, 
                                                      ifmap_alloc_i, 
                                                      ifmap_alloc_w)
            
        # wght fault injection
        if FI_wght:
            faultbit_wght=tf.constant(fault_dict['faultbit_wght'])
            psum_alter_wght=tf.multiply(wght_alloc_i, faultbit_wght)
            
            psum_alter_wght=self.mac_math_alter_make(psum_alter_wght, 
                                                     polarity_wght, 
                                                     quantizer_output, 
                                                     sim_truncarry, 
                                                     wght_alloc_i, 
                                                     wght_alloc_w)
            
        psum_alter=tf.zeros(fault_dict['psum_idx_list_len'])

        if FI_ofmap:
            param_ofmap=tf.constant(param_ofmap)
            psum_alter=tf.tensor_scatter_nd_update(psum_alter, param_ofmap, psum_alter_ofmap)
        if FI_ifmap:
            param_ifmap=tf.constant(param_ifmap)
            psum_alter=tf.tensor_scatter_nd_update(psum_alter, param_ifmap, psum_alter_ifmap)
        if FI_wght:
            param_wght=tf.constant(param_wght)
            psum_alter=tf.tensor_scatter_nd_update(psum_alter, param_wght, psum_alter_wght)

        cnt_psidx=fault_dict['cnt_psidx']
        if cnt_psidx is not None:
            psum_alter=tf.split(psum_alter,cnt_psidx)
            for alteritem in psum_alter:
                alteritem=tf.reduce_sum(alteritem)
            psum_alter=tf.stack(psum_alter)
            psum_alter=quantizer_output.quantize(psum_alter)
        
        # add psum_alter back to ofmap
        output=tf.add(fdoutput_alloc, psum_alter)
        output=tf.tensor_scatter_nd_update(ofmap,fd_coor,output)
            
        return output
    
    def inject_mac_noise_fault_tensor(self, ofmap, fault_dict, fast_gen=None):
        """ Fault injection mac output Gaussian noise model.
            Generate a Gaussian noise mask for layer ofmap. 
            Using the result of PE dataflow model fault dictionary.
            Make amplifier array for Gaussian noise mask.
            Where no fault ofmap pixels were filted to 0, other faulty pixels will be
            amplified correspond to the SA type, fault bit order and the number of faulty
            partial sum indexes.
            
            Fast generation means the all the fault information dict are refer to the same source defect 
            with the same fault bit, SA type and param.
            Scattered generation means the faults are from distinct defect sources that need to be split into 
            different groups and generate respectively. Therefore it's slower.
            
            This function is for inject the preprocessed fault data in Keras Layer/Model call.
            Seperate the CPU and GPU processing. The preprocess function under mac_fault_injector class are for GPU process.
        
        Arguments
        ---------
        ofmap: Tensor. 
            The Tensor to be injected fault by math alteration. Quantized Tensor. Layer output.
        fault_dict: Dictionary or List. 
            The dictionary contain fault list information.
        fast_gen: Bool. 
            Use fast generation or not. Fast generation has the same fault bit and SA type for all coordinates.
            The fault dictionay is form by fault data contamination.
        
        Returns
        -------
        Tensor. 
            The amount of adjustment apply to output feature map of a DNN layer which represent the faulty behabvior of MAC unit.
        
        """                            
        if self.fast_gen:
            # data allocation
            stddev_amp_ofmap=fault_dict['stddev_amp_ofmap']
            stddev_amp_ofmap=tf.constant(stddev_amp_ofmap)
            gaussian_noise_mask=tf.random.normal(ofmap.shape)
            gaussian_noise_mask=tf.multiply(gaussian_noise_mask,stddev_amp_ofmap)
            output=tf.add(ofmap, gaussian_noise_mask)

        else: # slow loop gen
            stddev_amp_ofmap=fault_dict['stddev_amp_ofmap']
            stddev_amp_ofmap=tf.constant(stddev_amp_ofmap)
            gaussian_noise_mask=tf.random.normal(ofmap.shape)
            gaussian_noise_mask=tf.multiply(gaussian_noise_mask,stddev_amp_ofmap)
            output=tf.add(ofmap, gaussian_noise_mask)
                        
        return output
    
    def inject_mac_noise_fault_uni(self, ofmap, fault_dict):
        """ Fault injection mac output Gaussian noise model.
            Generate a Gaussian noise mask for layer ofmap. 
            Using the result of PE dataflow model fault dictionary.
            Make amplifier array for Gaussian noise mask.
            Where no fault ofmap pixels were filted to 0, other faulty pixels will be
            amplified correspond to the SA type, fault bit order and the number of faulty
            partial sum indexes.
            
            Fast generation means the all the fault information dict are refer to the same source defect 
            with the same fault bit, SA type and param.
            
            This function is for inject the preprocessed fault data in Keras Layer/Model call.
            Seperate the CPU and GPU processing. The preprocess function under mac_fault_injector class are for GPU process.
        
        Arguments
        ---------
        ofmap: Tensor. 
            The Tensor to be injected fault by math alteration. Quantized Tensor. Layer output.
        fault_dict: Dictionary or List. 
            The dictionary contain fault list information.
        
        Returns
        -------
        Tensor. 
            The amount of adjustment apply to output feature map of a DNN layer which represent the faulty behabvior of MAC unit.
        
        """
        stddev_amp_ofmap=fault_dict['stddev_amp_ofmap']
        stddev_amp_ofmap=tf.constant(stddev_amp_ofmap)
        gaussian_noise_mask=tf.random.normal(ofmap.shape)
        gaussian_noise_mask=tf.multiply(gaussian_noise_mask,stddev_amp_ofmap)
        output=tf.add(ofmap, gaussian_noise_mask)
        
        return output
        
    def inject_mac_noise_fault_scatter(self, ofmap, fault_dict):
        """ Fault injection mac output Gaussian noise model.
            Generate a Gaussian noise mask for layer ofmap. 
            Using the result of PE dataflow model fault dictionary.
            Make amplifier array for Gaussian noise mask.
            Where no fault ofmap pixels were filted to 0, other faulty pixels will be
            amplified correspond to the SA type, fault bit order and the number of faulty
            partial sum indexes.
            
            Fast generation means the all the fault information dict are refer to the same source defect 
            with the same fault bit, SA type and param.
            Scattered generation means the faults are from distinct defect sources that need to be split into 
            different groups and generate respectively. Therefore it's slower.
        
            This function is for inject the preprocessed fault data in Keras Layer/Model call.
            Seperate the CPU and GPU processing. The preprocess function under mac_fault_injector class are for GPU process.
        
        Arguments
        ---------
        ofmap: Tensor. 
            The Tensor to be injected fault by math alteration. Quantized Tensor. Layer output.
        fault_dict: Dictionary or List. 
            The dictionary contain fault list information.
        
        Returns
        -------
        Tensor. 
            The amount of adjustment apply to output feature map of a DNN layer which represent the faulty behabvior of MAC unit.
        
        """
        stddev_amp_ofmap=fault_dict['stddev_amp_ofmap']
        stddev_amp_ofmap=tf.constant(stddev_amp_ofmap)
        gaussian_noise_mask=tf.random.normal(ofmap.shape)
        gaussian_noise_mask=tf.multiply(gaussian_noise_mask,stddev_amp_ofmap)
        output=tf.add(ofmap, gaussian_noise_mask)
        
        return output
        
    def __call__(self, ofmap, fault_dict=None, ifmap=None, wght=None, 
                 noise_inject=None, sim_truncarry=None, fast_gen=None,
                 quantizer=None, quant_mode=None, layer_type='Conv2D',
                 ksizes=(3,3), padding='valid', dilation_rates=(1,1), 
                 **kwargs):
        """ The function caller for decide which fault injection method will be used.
        
        Parameters
        ----------
        ofmap: Tensor. 
            The Tensor to be injected fault by math alteration. Quantized Tensor. Layer output.
        fault_dict: Dictionary or List. 
            The dictionary contain fault list information.
        ifmap: Tensor. optional
            Quantized Tensor. Layer input.
        wght: Tensor. optional
            Quantized Tensor. Layer output.
        noise_inject : Bool, optional
            Use the Gaussian noise fault injection method or not. 
            Noise mean using Gaussion distribution model to simulate the input and weight for mac fault. 
            The default is None.
        sim_truncarry: Bool. optional
            Simulate the truncation carry during mac math fault injection. 
            The truncation carry is cause by the carry-out of thrown away fractional bits. 
            It could be 1 in addition or -1 in subtraction, when the unwanted fractional bits value overflow to the needed fractional bits.
            How ever this could cause huge time overhead for absolute bit-true model.
        fast_gen: Bool. optional
            Use fast generation or not. Fast generation has the same fault bit and SA type for all coordinates.
            The fault dictionay is form by fault data contamination.
        quantizer: Class or List. 
            | The quantizer class, one or in list [input, weight, output]. The quantizer class contain following quantize operation infromation.
            | word_width: Variable. The fix-point representation of the parameter word length.
            | fractional_bits: Variable. Number of fractional bits in a fix-point parameter
            | rounding: String. Rounding method of quantization, augment must be one of 'nearest' , 'down', 'zero', 'stochastic'.
        quant_mode: String. Be either 'intrinsic' or 'hybrid'.
            | The quantization mode of MAC.
            | 'intrinsic' means truncate before accumulation. 
            | 'hybrid' means accumulate with the word length of multiplier output than truncate.
        layer_type: String. One of 'Conv2D', 'Dense', 'DepthwiseConv2D'.
            The type of layer this solver wants to convert partial sum index and mapping into.
        ksize: Tuple. Size 2. 
            The kernel size (row, col).
        padding: String. 'same' or 'valid'. 
            The type of padding algorithm to use.
        dilation_rate: Tuple. Size 2. 
            The dilation rate (row, col).
        **kwargs : Other additional arguments
            If needed.

        Returns
        -------
        Tensor. 
            The amount of adjustment apply to output feature map of a DNN layer which represent the faulty behabvior of MAC unit.
            
        Reminder!!
        ----------
        padding: Padding is tie to layer setting. 
            If user treat padding as a seperate layer.
            For example x=layers.ZeroPadding2D(padding=(1, 1), name='conv1_pad')(inputs)
            The tile setting must be padding='valid' which means no padding.
            Or else there might be fault.
        """
        if fault_dict==None:
            fault_dict=self.fault_dict
        if noise_inject is None:
            noise_inject=self.noise_inject
        if fast_gen is None:
            fast_gen=self.fast_gen
        
        if noise_inject:
            if fast_gen:
                output=self.inject_mac_noise_fault_uni(ofmap, fault_dict, **kwargs)
            else:
                output=self.inject_mac_math_fault_scatter(ofmap, fault_dict, **kwargs)
        else:
            if fast_gen:
                output=self.inject_mac_math_fault_uni(ifmap, wght, ofmap, fault_dict,
                                                      quantizer=quantizer, quant_mode=quant_mode, layer_type=layer_type,
                                                      ksizes=ksizes, padding=padding, dilation_rates=dilation_rates,
                                                      sim_truncarry=sim_truncarry,
                                                      **kwargs)
            else:
                output=self.inject_mac_math_fault_scatter(ifmap, wght, ofmap, fault_dict,
                                                          quantizer=quantizer, quant_mode=quant_mode, layer_type=layer_type,
                                                          ksizes=ksizes, padding=padding, dilation_rates=dilation_rates,
                                                          sim_truncarry=sim_truncarry,
                                                          **kwargs)
        
        return output
        
