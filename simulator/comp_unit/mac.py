# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 10:55:00 2020

@author: Yung-Yu Tsai

The MAC unit description for fault transfer to TF computation
"""

import numpy as np
import json
from ..layers.quantized_ops import quantizer

class mac_unit:
    """ The mac unit information holder class. For describe PE I/O interconnection and math of faulty behavior.
        Also serve as preprocess library for the fault dictionary data before Keras Layer/Model call.
        Seperate the CPU and GPU processing. The preprocess function under mac_unit class are for CPU process.
    
    Arguments
    ---------
    quantizers: Class or String. 
        The quantize library of simulator. Or the file path to MAC unit setup (.json) file.
        
        Default ifmap input/output, weight input/output, partial sum input/output are the same configuration.
        If using setup (.json) file, this will include all the following arguments input.
        
        Quantizer Parameters:
            word_width: Variable. 
                The fix-point representation of the parameter word length.
            fractional_bits: Variable. 
                Number of fractional bits in a fix-point parameter
            rounding: String. 
                Rounding method of quantization, augment must be one of 'nearest' , 'down', 'zero', 'stochastic'.

    quant_mode: String. Be either 'intrinsic' or 'hybrid'.
        | The quantization mode of MAC. 
        | 'intrinsic' means truncate before accumulation. 
        | 'hybrid' means accumulate with the word length of multiplier output than truncate.
    ifmap_io: Dictionary. 
        Input feature map data I/O description.
    wght_io: Dictionary. 
        Weight data I/O description.
    psum_io: Dictionary. 
        Partial sum data I/O description. Partial sum fault prapagation is through MAC operation.
    noise_inject: Bool.
        Using mac noise fault injection method or not.
    sim_truncarry: Bool. 
        Simulate the truncation carry during mac math fault injection. 
        The truncation carry is cause by the carry-out of thrown away fractional bits. 
        It could be 1 in addition or -1 in subtraction, when the unwanted fractional bits value overflow to the needed fractional bits.
        How ever this could cause huge time overhead for absolute bit-true model.
    psumfault_handle: String
        | The method of handling fault on 'psum_in' and 'psum_out'.
        | 'single': treat all the partial sum alter generate by fault on the same ofmap pixel as one fault. 
        |    Only one stuck at fault is applied.
        | 'direct_sum': Sums all the partial sum alter generate by fault on the same ofmap pixel. 
        |    This may exaggerate or underestimate the fault effect. 
        |    Since the accumulated partial sum in each clock cycle is not visible in layer tensor.
        | 'rand_sum': Randomly give the polarity to all the partial sum alter generate by fault on the same ofmap pixel. 
        |    Assume that the 0 and 1 appearence are the same in partial sum of each clock cycle which is not visible in layer tensor. 
        |    Simulate the randomness of actual polarity. Mitigate the exaggerated or underestimated the fault effect
        |
        | When psum_io is 'io_pair' default psumfault_handle is 'rand_sum'.
        | Else psum_io is 'one_way_out' default psumfault_handle is 'single'.
    fast_gen: Bool. 
        Use fast generation or not. Fast generation has the same fault bit and SA type for all coordinates.
        The fault dictionay is form by fault data contamination.
    amp_factor_fmap: Float. 
        The adjustment term for Gaussian standard deviation of the ifmap value noise simulation.
    amp_factor_wght: Float. 
        The adjustment term for Gaussian standard deviation of the weight value noise simulation.
        
        
    I/O description: (Dictionary)
    -----------------------------
    >>> data_io={'type':'io_pair', 'dimension':'PE_x', 'direction':'forward'}
        
    type: The type of I/O. 
        | 'io_pair' the data flow from a primary input to a primary output which forms a I/O pair. The I/O pair become a systolic stream in PE array at some point.
        | 'one_way_in' the data is a primary input of MAC unit. No further propagation at any computation process.
        | 'one_way_out' the data is a primary output of MAC unit. No direct data propagation chain lead to this output.
    
    dimension: One of None, 'PE_x', 'PE_y'.
        The dimension of I/O pair connect in PE array. 
    
    direction: the direction of I/O pair deliver data on the given dimension. 
        | 'forward' data prapagate along the index of PEs. 
        | 'backward' data prapagate by the reversed order of PE index.
        
        
    preprocess_data description: (Dictionary)
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
    >>> preprocess_data={'stddev_amp_ofmap': 4D Ndarray,
    ...                  'mean_sum_ofmap': 4D Ndarray} 
    ... #the standard deviation amplifier mask, same shape as target ofmap
    ... #the mean value moving mask, same shape as target ofmap

    | mac noise fault injection scatter fault
    >>> preprocess_data={'stddev_amp_ofmap': 4D Ndarray, 
    ...                  'mean_sum_ofmap': 4D Ndarray} 
    ... #the standard deviation amplifier mask, same shape as target ofmap
    ... #the mean value moving mask, same shape as target ofmap

    """
    def __init__(self, quantizers, quant_mode='hybrid', 
                 ifmap_io=None, wght_io=None, psum_io=None, 
                 noise_inject=True, sim_truncarry=False, psumfault_handle=None, fast_gen=True,
                 amp_factor_fmap=1.0, amp_factor_wght=1.0):
        """ Class initialization """
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
        
        self.noise_inject=noise_inject
        self.sim_truncarry=sim_truncarry
        self.fast_gen=fast_gen
        self.amp_factor_fmap=amp_factor_fmap
        self.amp_factor_wght=amp_factor_wght
        if psumfault_handle is None:
            if psum_io['type']=='io_pair':
                psumfault_handle='rand_sum'
            else:
                psumfault_handle='single'
        else:
            if psumfault_handle not in ['single','direct_sum','rand_sum']:
                raise ValueError('The psumfault_handle must be one of \'single\',\'direct_sum\',\'rand_sum\'. ')
            self.psumfault_handle=psumfault_handle
        
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
        
        Arguments
        ---------
        param: String. One of ['ifmap_in', 'ifmap_out', 'wght_in', 'wght_out', 'psum_in', 'psum_out']. 
            The type of parameter has fault. 
            
        Returns
        -------
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
        
        Arguments
        ---------
        param: String. 
            The type of parameter has fault. One of ['ifmap_in', 'ifmap_out', 'wght_in', 'wght_out', 'psum_in', 'psum_out']. 
        fault_loc: Ndarray or List. 
            PE dataflow model coordinate represent as the fault location.
        array_shape: Tuple or List. 
            The shape of PE array.
        
        Returns
        -------
        Converted coordinates. 
            Multiple coordinate return in 2D ndarray.
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
        
    def _polarity_check(self, fault_bit, wl):
        """ Get the polarity of parameter 
            
        """
        modulator=np.left_shift(1,fault_bit,dtype=np.int32)
                
        if isinstance(fault_bit,int):
            if fault_bit==wl-1:
                signbit=True
            else:
                signbit=False
        else:
            signbit=np.equal(fault_bit,wl-1)
            signbit=np.add(np.multiply(signbit,-2,dtype=np.int32),1)
                        
        return modulator, signbit
    
    def _padding_idx(self, index, ksizes, dilation_rates):
        """ Preproccess ifmap data and index for padding situation """
        dilated_ksize_row_edge = (ksizes[0] + (ksizes[0]-1) * (dilation_rates[0] - 1))//2
        dilated_ksize_col_edge = (ksizes[1] + (ksizes[1]-1) * (dilation_rates[1] - 1))//2
        index[:,:,1]=np.add(index[:,:,1],dilated_ksize_row_edge)
        index[:,:,2]=np.add(index[:,:,2],dilated_ksize_col_edge)
                        
        return index
    
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
        
        return searched_SA0.astype(np.int32), searched_SA1.astype(np.int32), searched_flip.astype(np.int32)
    
    def _polarity_check_type_grouping(self, data_shape, fault_bit, data_idx, type0, type1, typef, wlpolar):
        """ Polarity check with type grouping and return the arranged combined polarity """
        # type grouping
        type0_idx, type1_idx, typef_idx=self._param_find_fault_type(data_idx,type0,type1,typef)
        
        # allocate type subgroup data & check polarity
        if len(type0_idx)>0:
            faultbit_type0=np.tile(fault_bit[type0_idx],[1,data_shape[1]])
            modulator0, signbit0=self._polarity_check(faultbit_type0, wlpolar)
        else:
            modulator0=None
            signbit0=None
            
        if len(type1_idx)>0:
            faultbit_type1=np.tile(fault_bit[type1_idx],[1,data_shape[1]])
            modulator1, signbit1=self._polarity_check(faultbit_type1, wlpolar)
        else:
            modulator1=None
            signbit1=None
            
        if len(typef_idx)>0:
            faultbit_typef=np.tile(fault_bit[typef_idx],[1,data_shape[1]])
            modulatorf, signbitf=self._polarity_check(faultbit_typef, wlpolar)
        else:
            modulatorf=None
            signbitf=None
            
        return [type0_idx, modulator0, signbit0, type1_idx, modulator1, signbit1, typef_idx, modulatorf, signbitf]
    
    def _fault_bit_ext4mult(self, fault_bit, num_psum_idx):
        """ Extend fault bit for tf.multiply
            Add a number of psum index axis and make 2's exponential
        """
        fault_bit=np.expand_dims(fault_bit,-1)
        fault_bit=np.tile(fault_bit,[1,num_psum_idx])
        fault_bit=np.power(2,fault_bit)
        
        return fault_bit.astype(np.int32)
    
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

    def preprocess_mac_math_fault_tensor(self, fault_dict, 
                                         quantizer=None, quant_mode=None, layer_type='Conv2D',
                                         ksizes=(3,3), padding='valid', dilation_rates=(1,1), 
                                         sim_truncarry=None, psumfault_handle=None, fast_gen=None):
        """ The fault injection mathematical model for used in Layer Tensor computing.
            Include both fast generation (fast_gen is True) and scattered generation (fast_gen is False).
            Fast generation means the all the fault information dict are refer to the same source defect 
            with the same fault bit, SA type and param.
            Scattered generation means the faults are from distinct defect sources that need to be split into 
            different groups and generate respectively. Therefore it's slower.
            
            This function is for preprocess the fault dictionary data before Keras Layer/Model call.
            Seperate the CPU and GPU processing. The preprocess function under mac_unit class are for CPU process.
            Recommand using inject_mac_math_fault_uni or inject_mac_math_fault_scatter if you are sure about your
            fault dictionry defect source is unique or not.
        
        Arguments
        ---------
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
        preprocess_data: Dictionary. 
            The Dictionary contains the configuration and data for Tensor operation part.
        
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
        if psumfault_handle is None:
            psumfault_handle=self.psumfault_handle
        if fast_gen is not None:
            self.fast_gen=fast_gen
            
        preprocess_data=dict()
            
        order_get_psidx_o,order_get_psidx_w,order_get_psidx_i=self._layer_coor_order(layer_type)
        
        fd_coor=np.array(list(fault_dict.keys()))
        fd_value=np.array(list(fault_dict.values()))
        preprocess_data['fd_coor']=fd_coor
        
        if self.fast_gen:
            # data allocation
            # (coor idx, num of psidx, psum idx)
            psum_idx_list=np.array([info['psum_idx'] for info in fd_value])
            psum_idx_ofmap=psum_idx_list[:,:,order_get_psidx_o]
        
            fault_param=fd_value[0]['param']
            fault_type=fd_value[0]['SA_type']
            fault_bit=fd_value[0]['SA_bit']
            
            preprocess_data['fault_param']=fault_param
            preprocess_data['fault_type']=fault_type
            preprocess_data['fault_bit']=fault_bit
            
            if fault_param in ['ifmap_in','ifmap_out','wght_in','wght_out']:    
                psum_idx_wght=psum_idx_list[:,:,order_get_psidx_w]
                psum_idx_ifmap=psum_idx_list[:,:,order_get_psidx_i]
            
                if padding=='same' and layer_type!='Dense':
                    psum_idx_ifmap=self._padding_idx(psum_idx_ifmap, ksizes, dilation_rates)
    
                preprocess_data['psum_idx_ifmap']=psum_idx_ifmap
                preprocess_data['psum_idx_wght']=psum_idx_wght
                preprocess_data['psum_idx_ofmap']=psum_idx_ofmap
            else:
                if psumfault_handle=='single':
                    preprocess_data['psum_idx_ofmap']=fd_coor
                else:
                    preprocess_data['psum_idx_ofmap']=psum_idx_ofmap
                
            # check polarity
            if fault_param=='ifmap_in' or fault_param=='ifmap_out':
                wlpolar = quantizer_input.nb
            elif fault_param=='wght_in' or fault_param=='wght_out':
                wlpolar = quantizer_weight.nb
            elif fault_param=='psum_in' or fault_param=='psum_out':
                if self.quant_mode=='intrinsic':
                    wlpolar = quantizer_output.nb
                elif self.quant_mode=='hybrid':
                    wlpolar = quantizer_input.nb+quantizer_weight.nb
                        
            modulator, signbit=self._polarity_check(fault_bit, wlpolar)
            
            preprocess_data['modulator']=modulator
            preprocess_data['signbit']=signbit
                
            
        else: # slow loop gen
            # loop data extraction
            if isinstance(fd_value[0]['id'],int):
                psum_idx_list,cnt_psidx,fault_bit,param_ifmap,param_wght,param_ofmap,type0,type1,typef=self._fault_value_extract_loop(fd_value, repetitive=False)
            
            elif isinstance(fd_value[0]['id'],list):
                psum_idx_list,cnt_psidx,fault_bit,param_ifmap,param_wght,param_ofmap,type0,type1,typef=self._fault_value_extract_loop(fd_value, repetitive=True)
            
            FI_ofmap=len(param_ofmap)>0
            FI_ifmap=len(param_ifmap)>0
            FI_wght=len(param_wght)>0
            
            # ofmap fault
            if FI_ofmap:
                # data gathering
                psum_idx_ofmap=psum_idx_list[param_ofmap]
                idx_ofmap=psum_idx_ofmap[:,:,order_get_psidx_o]
                faultbit_ofmap=fault_bit[param_ofmap]
                
                if psumfault_handle=='single':
                    preprocess_data['idx_ofmap']=fd_coor
                else:
                    preprocess_data['idx_ofmap']=idx_ofmap
                                
                # check polarity
                if self.quant_mode=='intrinsic':
                    wlpolar = quantizer_output.nb
                elif self.quant_mode=='hybrid':
                    wlpolar = quantizer_input.nb+quantizer_weight.nb   
                    
                modulator_ofmap=self._polarity_check_type_grouping(idx_ofmap.shape,
                                                                   faultbit_ofmap,
                                                                   param_ofmap,
                                                                   type0,type1,typef,
                                                                   wlpolar)
                preprocess_data['modulator_ofmap']=modulator_ofmap
                
            # ifmap fault
            if FI_ifmap>0:
                # data gathering
                psum_idx_ifmap=psum_idx_list[param_ifmap]
                idx_ifmap_ifmap=psum_idx_ifmap[:,:,order_get_psidx_i]
                idx_ifmap_wght=psum_idx_ifmap[:,:,order_get_psidx_w]
                faultbit_ifmap=fault_bit[param_ifmap]
            
                if padding=='same' and layer_type!='Dense':
                    idx_ifmap_ifmap=self._padding_idx(idx_ifmap_ifmap, ksizes, dilation_rates)
                
                preprocess_data['idx_ifmap_ifmap']=idx_ifmap_ifmap
                preprocess_data['idx_ifmap_wght']=idx_ifmap_wght
                                
                # check polarity
                wlpolar = quantizer_input.nb
                
                modulator_ifmap=self._polarity_check_type_grouping(idx_ifmap_ifmap.shape,
                                                                   faultbit_ifmap,
                                                                   param_ifmap,
                                                                   type0,type1,typef,
                                                                   wlpolar)
                preprocess_data['modulator_ifmap']=modulator_ifmap
            
            # wght fault
            if FI_wght>0:
                # data gathering
                psum_idx_wght=psum_idx_list[param_wght]
                idx_wght_wght=psum_idx_wght[:,:,order_get_psidx_w]
                idx_wght_ifmap=psum_idx_wght[:,:,order_get_psidx_i]
                faultbit_wght=fault_bit[param_wght]
                
                if padding=='same' and layer_type!='Dense':
                    idx_wght_ifmap=self._padding_idx(idx_wght_ifmap, ksizes, dilation_rates)
                        
                preprocess_data['idx_wght_wght']=idx_wght_wght
                preprocess_data['idx_wght_ifmap']=idx_wght_ifmap
                        
                # check polarity
                wlpolar = quantizer_weight.nb
                
                modulator_wght=self._polarity_check_type_grouping(idx_wght_wght.shape,
                                                                  faultbit_wght,
                                                                  param_wght,
                                                                  type0,type1,typef,
                                                                  wlpolar)
                preprocess_data['modulator_wght']=modulator_wght
                
            # fault injection
            
            # ofmap fault injection
            if FI_ofmap:
                faultbit_ofmap=self._fault_bit_ext4mult(faultbit_ofmap, idx_ofmap.shape[1])
                preprocess_data['faultbit_ofmap']=faultbit_ofmap
                
            # ifmap fault injection
            if FI_ifmap>0:
                faultbit_ifmap=self._fault_bit_ext4mult(faultbit_ifmap, idx_ifmap_ifmap.shape[1])
                preprocess_data['faultbit_ifmap']=faultbit_ifmap
                
            # wght fault injection
            if FI_wght:
                faultbit_wght=self._fault_bit_ext4mult(faultbit_wght, idx_wght_wght.shape[1])

            if FI_ofmap:
                param_ofmap=np.expand_dims(param_ofmap,-1)
                preprocess_data['param_ofmap']=param_ofmap
            if FI_ifmap:
                param_ifmap=np.expand_dims(param_ifmap,-1)
                preprocess_data['param_ifmap']=param_ifmap
            if FI_wght:
                param_wght=np.expand_dims(param_wght,-1)
                preprocess_data['param_wght']=param_wght
                
            preprocess_data['cnt_psidx']=cnt_psidx
            preprocess_data['psum_idx_list_len']=len(psum_idx_list)
            
        return preprocess_data
    
    def preprocess_mac_math_fault_uni(self, fault_dict, 
                                      quantizer=None, quant_mode=None, layer_type='Conv2D',
                                      ksizes=(3,3), padding='valid', dilation_rates=(1,1), 
                                      sim_truncarry=None, psumfault_handle=None):
        """ The fault injection mathematical model for used in Lyer Tensor computing.
            This function is only for fast generation.
            Fast generation means the all the fault information dict are refer to the same source defect 
            with the same fault bit, SA type and param.
            
            This function is for preprocess the fault dictionary data before Keras Layer/Model call.
            Seperate the CPU and GPU processing. The preprocess function under mac_unit class are for CPU process.
        
        Arguments
        ---------
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
        preprocess_data: Dictionary. 
            The Dictionary contains the configuration and data for Tensor operation part.
        
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
        if psumfault_handle is None:
            psumfault_handle=self.psumfault_handle
            
        preprocess_data=dict()
            
        order_get_psidx_o,order_get_psidx_w,order_get_psidx_i=self._layer_coor_order(layer_type)
        
        fd_coor=np.array(list(fault_dict.keys()))
        fd_value=np.array(list(fault_dict.values()))
        preprocess_data['fd_coor']=fd_coor
        
        # data allocation
        # (coor idx, num of psidx, psum idx)
        psum_idx_list=np.array([info['psum_idx'] for info in fd_value])
        psum_idx_ofmap=psum_idx_list[:,:,order_get_psidx_o]
    
        fault_param=fd_value[0]['param']
        fault_type=fd_value[0]['SA_type']
        fault_bit=fd_value[0]['SA_bit']
        
        preprocess_data['fault_param']=fault_param
        preprocess_data['fault_type']=fault_type
        preprocess_data['fault_bit']=fault_bit
        
        if fault_param in ['ifmap_in','ifmap_out','wght_in','wght_out']:    
            psum_idx_wght=psum_idx_list[:,:,order_get_psidx_w]
            psum_idx_ifmap=psum_idx_list[:,:,order_get_psidx_i]
        
            if padding=='same' and layer_type!='Dense':
                psum_idx_ifmap=self._padding_idx(psum_idx_ifmap, ksizes, dilation_rates)

            preprocess_data['psum_idx_ifmap']=psum_idx_ifmap
            preprocess_data['psum_idx_wght']=psum_idx_wght
            preprocess_data['psum_idx_ofmap']=psum_idx_ofmap
        else:
            if psumfault_handle=='single':
                preprocess_data['psum_idx_ofmap']=fd_coor
            else:
                preprocess_data['psum_idx_ofmap']=psum_idx_ofmap
            
        # check polarity
        if fault_param=='ifmap_in' or fault_param=='ifmap_out':
            wlpolar = quantizer_input.nb
        elif fault_param=='wght_in' or fault_param=='wght_out':
            wlpolar = quantizer_weight.nb
        elif fault_param=='psum_in' or fault_param=='psum_out':
            if self.quant_mode=='intrinsic':
                wlpolar = quantizer_output.nb
            elif self.quant_mode=='hybrid':
                wlpolar = quantizer_input.nb+quantizer_weight.nb
                    
        modulator, signbit=self._polarity_check(fault_bit, wlpolar)
        
        preprocess_data['modulator']=modulator
        preprocess_data['signbit']=signbit
                
        return preprocess_data
    
    def preprocess_mac_math_fault_scatter(self, fault_dict, 
                                          quantizer=None, quant_mode=None, layer_type='Conv2D',
                                          ksizes=(3,3), padding='valid', dilation_rates=(1,1), 
                                          sim_truncarry=None, psumfault_handle=None):
        """ The fault injection mathematical model for used in Layer Tensor computing
            Include both fast generation (fast_gen is True) and scattered generation (fast_gen is False).
            Scattered generation means the faults are from distinct defect sources that need to be split into 
            different groups and generate respectively. Therefore it's slower.
            
            This function is for preprocess the fault dictionary data before Keras Layer/Model call.
            Seperate the CPU and GPU processing. The preprocess function under mac_unit class are for CPU process.
        
        Arguments
        ---------
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
        preprocess_data: Dictionary. 
            The Dictionary contains the configuration and data for Tensor operation part.
        
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
        if psumfault_handle is None:
            psumfault_handle=self.psumfault_handle
            
        preprocess_data=dict()
            
        order_get_psidx_o,order_get_psidx_w,order_get_psidx_i=self._layer_coor_order(layer_type)
        
        fd_coor=np.array(list(fault_dict.keys()))
        fd_value=np.array(list(fault_dict.values()))
        preprocess_data['fd_coor']=fd_coor
        
        # loop data extraction
        if isinstance(fd_value[0]['id'],int):
            psum_idx_list,cnt_psidx,fault_bit,param_ifmap,param_wght,param_ofmap,type0,type1,typef=self._fault_value_extract_loop(fd_value, repetitive=False)
        
        elif isinstance(fd_value[0]['id'],list):
            psum_idx_list,cnt_psidx,fault_bit,param_ifmap,param_wght,param_ofmap,type0,type1,typef=self._fault_value_extract_loop(fd_value, repetitive=True)
        
        FI_ofmap=len(param_ofmap)>0
        FI_ifmap=len(param_ifmap)>0
        FI_wght=len(param_wght)>0
        
        # ofmap fault
        if FI_ofmap:
            # data gathering
            psum_idx_ofmap=psum_idx_list[param_ofmap]
            idx_ofmap=psum_idx_ofmap[:,:,order_get_psidx_o]
            faultbit_ofmap=fault_bit[param_ofmap]
            
            if psumfault_handle=='single':
                preprocess_data['idx_ofmap']=fd_coor
            else:
                preprocess_data['idx_ofmap']=idx_ofmap
                            
            # check polarity
            if self.quant_mode=='intrinsic':
                wlpolar = quantizer_output.nb
            elif self.quant_mode=='hybrid':
                wlpolar = quantizer_input.nb+quantizer_weight.nb   
                
            modulator_ofmap=self._polarity_check_type_grouping(idx_ofmap.shape,
                                                               faultbit_ofmap,
                                                               param_ofmap,
                                                               type0,type1,typef,
                                                               wlpolar)
            preprocess_data['modulator_ofmap']=modulator_ofmap
            
        # ifmap fault
        if FI_ifmap>0:
            # data gathering
            psum_idx_ifmap=psum_idx_list[param_ifmap]
            idx_ifmap_ifmap=psum_idx_ifmap[:,:,order_get_psidx_i]
            idx_ifmap_wght=psum_idx_ifmap[:,:,order_get_psidx_w]
            faultbit_ifmap=fault_bit[param_ifmap]
        
            if padding=='same' and layer_type!='Dense':
                idx_ifmap_ifmap=self._padding_idx(idx_ifmap_ifmap, ksizes, dilation_rates)
            
            preprocess_data['idx_ifmap_ifmap']=idx_ifmap_ifmap
            preprocess_data['idx_ifmap_wght']=idx_ifmap_wght
                            
            # check polarity
            wlpolar = quantizer_input.nb
            
            modulator_ifmap=self._polarity_check_type_grouping(idx_ifmap_ifmap.shape,
                                                               faultbit_ifmap,
                                                               param_ifmap,
                                                               type0,type1,typef,
                                                               wlpolar)
            preprocess_data['modulator_ifmap']=modulator_ifmap
        
        # wght fault
        if FI_wght>0:
            # data gathering
            psum_idx_wght=psum_idx_list[param_wght]
            idx_wght_wght=psum_idx_wght[:,:,order_get_psidx_w]
            idx_wght_ifmap=psum_idx_wght[:,:,order_get_psidx_i]
            faultbit_wght=fault_bit[param_wght]
            
            if padding=='same' and layer_type!='Dense':
                idx_wght_ifmap=self._padding_idx(idx_wght_ifmap, ksizes, dilation_rates)
                    
            preprocess_data['idx_wght_wght']=idx_wght_wght
            preprocess_data['idx_wght_ifmap']=idx_wght_ifmap
                    
            # check polarity
            wlpolar = quantizer_weight.nb
            
            modulator_wght=self._polarity_check_type_grouping(idx_wght_wght.shape,
                                                              faultbit_wght,
                                                              param_wght,
                                                              type0,type1,typef,
                                                              wlpolar)
            preprocess_data['modulator_wght']=modulator_wght
            
        # fault injection
        
        # ofmap fault injection
        if FI_ofmap:
            faultbit_ofmap=self._fault_bit_ext4mult(faultbit_ofmap, idx_ofmap.shape[1])
            preprocess_data['faultbit_ofmap']=faultbit_ofmap
            
        # ifmap fault injection
        if FI_ifmap>0:
            faultbit_ifmap=self._fault_bit_ext4mult(faultbit_ifmap, idx_ifmap_ifmap.shape[1])
            preprocess_data['faultbit_ifmap']=faultbit_ifmap
            
        # wght fault injection
        if FI_wght:
            faultbit_wght=self._fault_bit_ext4mult(faultbit_wght, idx_wght_wght.shape[1])
            preprocess_data['faultbit_wght']=faultbit_wght

        if FI_ofmap:
            param_ofmap=np.expand_dims(param_ofmap,-1)
            preprocess_data['param_ofmap']=param_ofmap
        if FI_ifmap:
            param_ifmap=np.expand_dims(param_ifmap,-1)
            preprocess_data['param_ifmap']=param_ifmap
        if FI_wght:
            param_wght=np.expand_dims(param_wght,-1)
            preprocess_data['param_wght']=param_wght
            
        preprocess_data['cnt_psidx']=cnt_psidx
        preprocess_data['psum_idx_list_len']=len(psum_idx_list)
            
        return preprocess_data

    def _fault_noise_extract_loop(self, fault_value, repetitive=False):
        """ Extract data from fault dictionary values 
            The fault value is in np.array(list(fault_dict.values())) format
            This is only for slow loop generation
        """
        fault_bit=list()
        param_ifmap=list()
        param_wght=list()
        param_ofmap=list()
        type0=list()
        type1=list()
        typef=list()
        
        if not repetitive:
            for i,info in enumerate(fault_value):
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
            psum_idx_amp=np.ones(len(fault_value),dtype=np.float32)
            psum_idx_mean=np.ones(len(fault_value),dtype=np.float32)
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
                fault_bit.append(info['SA_bit'])
                fault_param.append(info['param'])
                fault_type.append(info['SA_type'])
            
            psum_idx_amp=np.ones(np.sum(cnt_psidx),dtype=np.float32)
            psum_idx_mean=np.zeros(np.sum(cnt_psidx),dtype=np.float32)
            cnt_psidx=np.cumsum(cnt_psidx)[:-1]

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
        
        if len(type0)>0:
            np.multiply.at(psum_idx_amp, type0, np.divide(1,np.sqrt(2),dtype=np.float32))
            np.multiply.at(psum_idx_mean, type0, 0.5)
        if len(type1)>0:
            np.multiply.at(psum_idx_amp, type1, np.divide(1,np.sqrt(2),dtype=np.float32))
            np.multiply.at(psum_idx_mean, type1, 0.5)

        return psum_idx_amp,psum_idx_mean,cnt_psidx,fault_bit,param_ifmap,param_wght,param_ofmap

    def preprocess_mac_noise_fault_tensor(self, fault_dict, ofmap_shape, 
                                          dist_stats_fmap=None, dist_stats_wght=None,
                                          amp_factor_fmap=1.0, amp_factor_wght=1.0,
                                          quantizer=None, 
                                          fast_gen=None):
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
            
            This function is for preprocess the fault dictionary data before Keras Layer/Model call.
            Seperate the CPU and GPU processing. The preprocess function under mac_unit class are for CPU process.
        
        Arguments
        ---------
        fault_dict: Dictionary or List. 
            The dictionary contain fault list information.
        ofmap_shape: Tuple. 
            The ofmap shape for fault injection. The shape is needed for generate the mask amplifier.
        dist_stats_fmap : Dictionary.
            Statistic dctionary for input feature maps.
        dist_stats_wght : List of Dictionary.
            Statistic dctionary for kernel and bias.
        amp_factor_fmap: Float. 
            The adjustment term for Gaussian standard deviation of the ifmap value noise simulation.
        amp_factor_wght: Float. 
            The adjustment term for Gaussian standard deviation of the weight value noise simulation.
        quantizer: Class or List. 
            The quantizer class, one or in list [input, weight, output]. The quantizer class contain following quantize operation infromation.
            word_width: Variable. The fix-point representation of the parameter word length.
            fractional_bits: Variable. Number of fractional bits in a fix-point parameter
            rounding: String. Rounding method of quantization, augment must be one of 'nearest' , 'down', 'zero', 'stochastic'.
        fast_gen: Bool. 
            Use fast generation or not. Fast generation has the same fault bit and SA type for all coordinates.
            The fault dictionay is form by fault data contamination.
        
        Returns
        -------
        preprocess_data: Dictionary. 
            The Dictionary contains the configuration and data for Tensor operation part.
        
        """
        if quantizer is None:
            if isinstance(self.quantizer,list) and len(self.quantizer)==3:
                quantizer_output =self.quantizer[2]
            else:
                quantizer_output =self.quantizer
        else:
            if isinstance(quantizer,list) and len(quantizer)==3:
                quantizer_output =quantizer[2]
            else:
                quantizer_output =quantizer
                            
        if amp_factor_fmap==1.0:
            amp_factor_fmap=self.amp_factor_fmap
        if amp_factor_wght==1.0:
            amp_factor_wght=self.amp_factor_wght
        mean_factor_fmap=0.0
        mean_factor_wght=0.0
        if dist_stats_fmap is not None:
            amp_factor_fmap=np.float32(dist_stats_fmap['std_dev'])
            mean_factor_fmap=np.float32(dist_stats_fmap['mean'])
        if dist_stats_wght is not None:
            amp_factor_wght=np.float32(dist_stats_wght['std_dev'])
            mean_factor_wght=np.float32(dist_stats_wght['mean'])
            
        preprocess_data=dict()
            
        fd_coor=np.array(list(fault_dict.keys()))
        fd_value=np.array(list(fault_dict.values()))

        if self.fast_gen:
            # data allocation
            psum_idx_cnt=np.array([len(info['psum_idx']) for info in fd_value])
            
            fault_param=fd_value[0]['param']
            fault_type=fd_value[0]['SA_type']
            fault_bit=fd_value[0]['SA_bit']
            
            # the scaling for fault bit order
            fault_order=np.power(2.,np.subtract(fault_bit,quantizer_output.fb),dtype=np.float32)
            
            # check polarity
            if fault_type!='flip':
                psum_idx_cnt=np.divide(psum_idx_cnt,2)
                
            # amplify for multiple psum index (faults)
            stddev_amp=np.sqrt(psum_idx_cnt,dtype=np.float32)
            
            # fault injection
            if fault_param=='ifmap_in' or fault_param=='ifmap_out':
                # give the weight's value amplifier
                stddev_amp=np.multiply(stddev_amp,amp_factor_wght)
                stddev_amp=np.multiply(stddev_amp,fault_order)
                mean_sum=np.multiply(mean_factor_wght,psum_idx_cnt)
                mean_sum=np.multiply(mean_sum,fault_order)
                
            elif fault_param=='wght_in' or fault_param=='wght_out':
                # give the feature map's value amplifier
                stddev_amp=np.multiply(stddev_amp,amp_factor_fmap)
                stddev_amp=np.multiply(stddev_amp,fault_order)
                mean_sum=np.multiply(mean_factor_fmap,psum_idx_cnt)
                mean_sum=np.multiply(mean_sum,fault_order)
                # relu effect redution (the percentage of zero data)
                if dist_stats_fmap is not None:
                    quantile=dist_stats_fmap['quantile']
                    relu_redu=np.float32(np.divide(np.subtract(np.sum(quantile>0),0.5),len(quantile)-1))
                else:
                    relu_redu=np.float32(1)
                stddev_amp=np.multiply(stddev_amp,np.sqrt(relu_redu))
                mean_sum=np.multiply(mean_sum,relu_redu)
                
                           
            # fault injection of ofmap
            elif fault_param=='psum_in' or fault_param=='psum_out':
                 stddev_amp=np.multiply(stddev_amp,fault_order)
                 mean_sum=np.float32(0)
    
            # add psum_alter back to ofmap
            stddev_amp_ofmap=np.zeros(ofmap_shape,dtype=np.float32)
            np.add.at(stddev_amp_ofmap,tuple([*fd_coor.T]),stddev_amp)
            mean_sum_ofmap=np.zeros(ofmap_shape,dtype=np.float32)
            np.add.at(mean_sum_ofmap,tuple([*fd_coor.T]),mean_sum)
            
            preprocess_data['stddev_amp_ofmap']=stddev_amp_ofmap
            preprocess_data['mean_sum_ofmap']=mean_sum_ofmap

        else: # slow loop gen
            # loop data extraction
            if isinstance(fd_value[0]['id'],int):
                stddev_amp,mean_sum,cnt_psidx,fault_bit,param_ifmap,param_wght,param_ofmap=self._fault_noise_extract_loop(fd_value, repetitive=False)
            
            elif isinstance(fd_value[0]['id'],list):
                stddev_amp,mean_sum,cnt_psidx,fault_bit,param_ifmap,param_wght,param_ofmap=self._fault_noise_extract_loop(fd_value, repetitive=True)
                
            # the scaling for fault bit order
            fault_order=np.power(2.,np.subtract(fault_bit,quantizer_output.fb),dtype=np.float32)
    
            # fault injection
    
            # ofmap fault
            if len(param_ofmap)>0:
                np.multiply.at(stddev_amp, param_ofmap, fault_order)
                np.multiply.at(mean_sum, param_ofmap, 0.0)
                
            # ifmap fault
            if len(param_ifmap)>0:
                # give the weight's value amplifier
                ifmap_amp=np.multiply(fault_order,amp_factor_wght)
                np.multiply.at(stddev_amp, param_ifmap, ifmap_amp)
                ifmap_sum=np.multiply(fault_order,mean_factor_wght)
                np.multiply.at(mean_sum, param_ifmap, ifmap_sum)        
            
            # wght fault
            if len(param_wght)>0:
                # give the feature map's value amplifier
                wght_amp=np.multiply(fault_order,amp_factor_fmap)
                np.multiply.at(stddev_amp, param_wght, wght_amp)
                wght_sum=np.multiply(fault_order,mean_factor_fmap)
                np.multiply.at(mean_sum, param_wght, wght_sum)
                # relu effect redution (the percentage of zero data)
                if dist_stats_fmap is not None:
                    quantile=dist_stats_fmap['quantile']
                    relu_redu=np.float32(np.divide(np.subtract(np.sum(quantile>0),0.5),len(quantile)-1))
                else:
                    relu_redu=np.float32(1)
                np.multiply.at(stddev_amp, param_wght, np.sqrt(relu_redu))
                np.multiply.at(mean_sum, param_wght, relu_redu)
                
            if cnt_psidx is not None:
                stddev_amp=np.split(stddev_amp,cnt_psidx)
                for stddevpixel in stddev_amp:
                    nsum=len(stddevpixel)
                    stddevpixel=np.sqrt(np.sum(np.power(stddevpixel,2,dtype=np.float32)),dtype=np.float32)
                    stddevpixel=np.multiply(stddevpixel,nsum)
                stddev_amp=np.array(stddev_amp)
                
                mean_sum=np.split(mean_sum,cnt_psidx)
                for meanpixel in mean_sum:
                    meanpixel=np.sum(meanpixel)
                mean_sum=np.array(mean_sum)
    
            # add psum_alter back to ofmap
            stddev_amp_ofmap=np.zeros(ofmap_shape,dtype=np.float32)
            np.add.at(stddev_amp_ofmap,tuple([*fd_coor.T]),stddev_amp)
            mean_sum_ofmap=np.zeros(ofmap_shape,dtype=np.float32)
            np.add.at(mean_sum_ofmap,tuple([*fd_coor.T]),mean_sum)
            
            preprocess_data['stddev_amp_ofmap']=stddev_amp_ofmap
            preprocess_data['mean_sum_ofmap']=mean_sum_ofmap
                        
        return preprocess_data
    
    def preprocess_mac_noise_fault_uni(self, fault_dict, ofmap_shape, 
                                       dist_stats_fmap=None, dist_stats_wght=None,
                                       amp_factor_fmap=1.0, amp_factor_wght=1.0,
                                       quantizer=None):
        """ Fault injection mac output Gaussian noise model.
            Generate a Gaussian noise mask for layer ofmap. 
            Using the result of PE dataflow model fault dictionary.
            Make amplifier array for Gaussian noise mask.
            Where no fault ofmap pixels were filted to 0, other faulty pixels will be
            amplified correspond to the SA type, fault bit order and the number of faulty
            partial sum indexes.
            
            Fast generation means the all the fault information dict are refer to the same source defect 
            with the same fault bit, SA type and param.
            
            This function is for preprocess the fault dictionary data before Keras Layer/Model call.
            Seperate the CPU and GPU processing. The preprocess function under mac_unit class are for CPU process.
        
        Arguments
        ---------
        fault_dict: Dictionary or List. 
            The dictionary contain fault list information.
        ofmap_shape: Tuple. 
            The ofmap shape for fault injection. The shape is needed for generate the mask amplifier.
        dist_stats_fmap : Dictionary.
            Statistic dctionary for input feature maps.
        dist_stats_wght : List of Dictionary.
            Statistic dctionary for kernel and bias.
        amp_factor_fmap: Float. 
            The adjustment term for Gaussian standard deviation of the ifmap value noise simulation.
        amp_factor_wght: Float. 
            The adjustment term for Gaussian standard deviation of the weight value noise simulation.
        quantizer: Class or List. 
            The quantizer class, one or in list [input, weight, output]. The quantizer class contain following quantize operation infromation.
            word_width: Variable. The fix-point representation of the parameter word length.
            fractional_bits: Variable. Number of fractional bits in a fix-point parameter
            rounding: String. Rounding method of quantization, augment must be one of 'nearest' , 'down', 'zero', 'stochastic'.
        
        Returns
        -------
        preprocess_data: Dictionary. 
            The Dictionary contains the configuration and data for Tensor operation part.
        
        """
        if quantizer is None:
            if isinstance(self.quantizer,list) and len(self.quantizer)==3:
                quantizer_output =self.quantizer[2]
            else:
                quantizer_output =self.quantizer
        else:
            if isinstance(quantizer,list) and len(quantizer)==3:
                quantizer_output =quantizer[2]
            else:
                quantizer_output =quantizer
                            
        # mean & std setup
        if amp_factor_fmap==1.0:
            amp_factor_fmap=self.amp_factor_fmap
        if amp_factor_wght==1.0:
            amp_factor_wght=self.amp_factor_wght
        mean_factor_fmap=0.0
        mean_factor_wght=0.0
        if dist_stats_fmap is not None:
            amp_factor_fmap=np.float32(dist_stats_fmap['std_dev'])
            mean_factor_fmap=np.float32(dist_stats_fmap['mean'])
        if dist_stats_wght is not None:
            amp_factor_wght=np.float32(dist_stats_wght[0]['std_dev'])
            mean_factor_wght=np.float32(dist_stats_wght[0]['mean'])
            
        preprocess_data=dict()
            
        fd_coor=np.array(list(fault_dict.keys()))
        fd_value=np.array(list(fault_dict.values()))

        # data allocation
        psum_idx_cnt=np.array([len(info['psum_idx']) for info in fd_value])
        
        fault_param=fd_value[0]['param']
        fault_type=fd_value[0]['SA_type']
        fault_bit=fd_value[0]['SA_bit']
        
        fault_order=np.power(2.,np.subtract(fault_bit,quantizer_output.fb),dtype=np.float32)
        
        # check polarity
        if fault_type!='flip':
            psum_idx_cnt=np.divide(psum_idx_cnt,2)
            
        # amplify for multiple psum index (faults)
        stddev_amp=np.sqrt(psum_idx_cnt,dtype=np.float32)
        
        # fault injection
        if fault_param=='ifmap_in' or fault_param=='ifmap_out':
            # give the weight's value amplifier
            stddev_amp=np.multiply(stddev_amp,amp_factor_wght)
            stddev_amp=np.multiply(stddev_amp,fault_order)
            mean_sum=np.multiply(mean_factor_wght,psum_idx_cnt)
            mean_sum=np.multiply(mean_sum,fault_order)
            
        elif fault_param=='wght_in' or fault_param=='wght_out':
            # give the feature map's value amplifier
            stddev_amp=np.multiply(stddev_amp,amp_factor_fmap)
            stddev_amp=np.multiply(stddev_amp,fault_order)
            mean_sum=np.multiply(mean_factor_fmap,psum_idx_cnt)
            mean_sum=np.multiply(mean_sum,fault_order)
            # relu effect redution (the percentage of zero data)
            if dist_stats_fmap is not None:
                quantile=dist_stats_fmap['quantile']
                relu_redu=np.float32(np.divide(np.subtract(np.sum(quantile>0),0.5),len(quantile)-1))
            else:
                relu_redu=np.float32(1)
            stddev_amp=np.multiply(stddev_amp,np.sqrt(relu_redu))
            mean_sum=np.multiply(mean_sum,relu_redu)
            
                       
        # fault injection of ofmap
        elif fault_param=='psum_in' or fault_param=='psum_out':
             stddev_amp=np.multiply(stddev_amp,fault_order)
             mean_sum=np.float32(0)

        # add psum_alter back to ofmap
        stddev_amp_ofmap=np.zeros(ofmap_shape,dtype=np.float32)
        np.add.at(stddev_amp_ofmap,tuple([*fd_coor.T]),stddev_amp)
        mean_sum_ofmap=np.zeros(ofmap_shape,dtype=np.float32)
        np.add.at(mean_sum_ofmap,tuple([*fd_coor.T]),mean_sum)
        
        preprocess_data['stddev_amp_ofmap']=stddev_amp_ofmap
        preprocess_data['mean_sum_ofmap']=mean_sum_ofmap
        
        return preprocess_data
    
    def preprocess_mac_noise_fault_scatter(self, fault_dict, ofmap_shape, 
                                           dist_stats_fmap=None, dist_stats_wght=None,
                                           amp_factor_fmap=1.0, amp_factor_wght=1.0,
                                           quantizer=None):
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
        
            This function is for preprocess the fault dictionary data before Keras Layer/Model call.
            Seperate the CPU and GPU processing. The preprocess function under mac_unit class are for CPU process.
        
        Arguments
        ---------
        fault_dict: Dictionary or List. 
            The dictionary contain fault list information.
        ofmap_shape: Tuple. 
            The ofmap shape for fault injection. The shape is needed for generate the mask amplifier.
        dist_stats_fmap : Dictionary.
            Statistic dctionary for input feature maps.
        dist_stats_wght : List of Dictionary.
            Statistic dctionary for kernel and bias.
        amp_factor_fmap: Float. 
            The adjustment term for Gaussian standard deviation of the ifmap value noise simulation.
        amp_factor_wght: Float. 
            The adjustment term for Gaussian standard deviation of the weight value noise simulation.
        quantizer: Class or List. 
            The quantizer class, one or in list [input, weight, output]. The quantizer class contain following quantize operation infromation.
            word_width: Variable. The fix-point representation of the parameter word length.
            fractional_bits: Variable. Number of fractional bits in a fix-point parameter
            rounding: String. Rounding method of quantization, augment must be one of 'nearest' , 'down', 'zero', 'stochastic'.
        
        Returns
        -------
        preprocess_data: Dictionary. 
            The Dictionary contains the configuration and data for Tensor operation part.
        
        """
        if quantizer is None:
            if isinstance(self.quantizer,list) and len(self.quantizer)==3:
                quantizer_output =self.quantizer[2]
            else:
                quantizer_output =self.quantizer
        else:
            if isinstance(quantizer,list) and len(quantizer)==3:
                quantizer_output =quantizer[2]
            else:
                quantizer_output =quantizer
                            
        if amp_factor_fmap==1.0:
            amp_factor_fmap=self.amp_factor_fmap
        if amp_factor_wght==1.0:
            amp_factor_wght=self.amp_factor_wght
        mean_factor_fmap=0.0
        mean_factor_wght=0.0
        if dist_stats_fmap is not None:
            amp_factor_fmap=np.float32(dist_stats_fmap['std_dev'])
            mean_factor_fmap=np.float32(dist_stats_fmap['mean'])
        if dist_stats_wght is not None:
            amp_factor_wght=np.float32(dist_stats_wght['std_dev'])
            mean_factor_wght=np.float32(dist_stats_wght['mean'])
            
        preprocess_data=dict()
            
        fd_coor=np.array(list(fault_dict.keys()))
        fd_value=np.array(list(fault_dict.values()))

        # loop data extraction
        if isinstance(fd_value[0]['id'],int):
            stddev_amp,mean_sum,cnt_psidx,fault_bit,param_ifmap,param_wght,param_ofmap=self._fault_noise_extract_loop(fd_value, repetitive=False)
        
        elif isinstance(fd_value[0]['id'],list):
            stddev_amp,mean_sum,cnt_psidx,fault_bit,param_ifmap,param_wght,param_ofmap=self._fault_noise_extract_loop(fd_value, repetitive=True)
            
        # the scaling for fault bit order
        fault_order=np.power(2.,np.subtract(fault_bit,quantizer_output.fb),dtype=np.float32)

        # fault injection

        # ofmap fault
        if len(param_ofmap)>0:
            np.multiply.at(stddev_amp, param_ofmap, fault_order)
            np.multiply.at(mean_sum, param_ofmap, 0.0)
            
        # ifmap fault
        if len(param_ifmap)>0:
            # give the weight's value amplifier
            ifmap_amp=np.multiply(fault_order,amp_factor_wght)
            np.multiply.at(stddev_amp, param_ifmap, ifmap_amp)
            ifmap_sum=np.multiply(fault_order,mean_factor_wght)
            np.multiply.at(mean_sum, param_ifmap, ifmap_sum)        
        
        # wght fault
        if len(param_wght)>0:
            # give the feature map's value amplifier
            wght_amp=np.multiply(fault_order,amp_factor_fmap)
            np.multiply.at(stddev_amp, param_wght, wght_amp)
            wght_sum=np.multiply(fault_order,mean_factor_fmap)
            np.multiply.at(mean_sum, param_wght, wght_sum)
            # relu effect redution (the percentage of zero data)
            if dist_stats_fmap is not None:
                quantile=dist_stats_fmap['quantile']
                relu_redu=np.float32(np.divide(np.subtract(np.sum(quantile>0),0.5),len(quantile)-1))
            else:
                relu_redu=np.float32(1)
            np.multiply.at(stddev_amp, param_wght, np.sqrt(relu_redu))
            np.multiply.at(mean_sum, param_wght, relu_redu)
            
        if cnt_psidx is not None:
            stddev_amp=np.split(stddev_amp,cnt_psidx)
            for stddevpixel in stddev_amp:
                nsum=len(stddevpixel)
                stddevpixel=np.sqrt(np.sum(np.power(stddevpixel,2,dtype=np.float32)),dtype=np.float32)
                stddevpixel=np.multiply(stddevpixel,nsum)
            stddev_amp=np.array(stddev_amp)
            
            mean_sum=np.split(mean_sum,cnt_psidx)
            for meanpixel in mean_sum:
                meanpixel=np.sum(meanpixel)
            mean_sum=np.array(mean_sum)

        # add psum_alter back to ofmap
        stddev_amp_ofmap=np.zeros(ofmap_shape,dtype=np.float32)
        np.add.at(stddev_amp_ofmap,tuple([*fd_coor.T]),stddev_amp)
        mean_sum_ofmap=np.zeros(ofmap_shape,dtype=np.float32)
        np.add.at(mean_sum_ofmap,tuple([*fd_coor.T]),mean_sum)
        
        preprocess_data['stddev_amp_ofmap']=stddev_amp_ofmap
        preprocess_data['mean_sum_ofmap']=mean_sum_ofmap
                        
        return preprocess_data

    def preprocess_mac_fault_caller(self, fault_dict, ofmap_shape=None,
                                    noise_inject=None, sim_truncarry=None, fast_gen=None,
                                    quantizer=None, quant_mode=None, layer_type='Conv2D',
                                    ksizes=(3,3), padding='valid', dilation_rates=(1,1), 
                                    dist_stats_fmap=None, dist_stats_wght=None,
                                    amp_factor_fmap=1.0, amp_factor_wght=1.0,
                                    **kwargs):
        """ The function caller for decide which fault injection method will be used.
            The function called is for preprocess the fault dictionary data before Keras Layer/Model call.
            Seperate the CPU and GPU processing. The preprocess function under mac_unit class are for CPU process.

        
        Parameters
        ----------
        fault_dict: Dictionary or List. 
            The dictionary contain fault list information.
        ofmap_shape: Tuple. 
            The ofmap shape for fault injection. The shape is needed for generate the mask amplifier.
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
        dist_stats_fmap : Dictionary.
            Statistic dctionary for input feature maps.
        dist_stats_wght : List of Dictionary.
            Statistic dctionary for kernel and bias.
        amp_factor_fmap: Float. 
            The adjustment term for Gaussian standard deviation of the ifmap value noise simulation.
        amp_factor_wght: Float. 
            The adjustment term for Gaussian standard deviation of the weight value noise simulation.
        **kwargs : Other additional arguments
            If needed.

        Returns
        -------
        preprocess_data: Dictionary. 
            The Dictionary contains the configuration and data for Tensor operation part.
            
        Reminder!!
        ----------
        padding: Padding is tie to layer setting. 
            If user treat padding as a seperate layer.
            For example x=layers.ZeroPadding2D(padding=(1, 1), name='conv1_pad')(inputs)
            The tile setting must be padding='valid' which means no padding.
            Or else there might be fault.
        """
        if noise_inject is None:
            noise_inject=self.noise_inject
        if fast_gen is None:
            fast_gen=self.fast_gen
        
        if noise_inject:
            if ofmap_shape is None:
                raise ValueError('Ofmap shape is mandatory argument in noise inject mode.')
                
            if fast_gen:
                preprocess_data=self.preprocess_mac_noise_fault_uni(fault_dict, ofmap_shape, 
                                                                    dist_stats_fmap=dist_stats_fmap,
                                                                    dist_stats_wght=dist_stats_wght,
                                                                    amp_factor_fmap=amp_factor_fmap,
                                                                    amp_factor_wght=amp_factor_wght,
                                                                    quantizer=quantizer)
            else:
                preprocess_data=self.preprocess_mac_noise_fault_scatter(fault_dict, ofmap_shape, 
                                                                        dist_stats_fmap=dist_stats_fmap,
                                                                        dist_stats_wght=dist_stats_wght,
                                                                        amp_factor_fmap=amp_factor_fmap,
                                                                        amp_factor_wght=amp_factor_wght,
                                                                        quantizer=quantizer)
        else:
            if fast_gen:
                preprocess_data=self.preprocess_mac_math_fault_uni(fault_dict,
                                                                   quantizer=quantizer, quant_mode=quant_mode, layer_type=layer_type,
                                                                   ksizes=ksizes, padding=padding, dilation_rates=dilation_rates,
                                                                   sim_truncarry=sim_truncarry)
            else:
                preprocess_data=self.preprocess_mac_math_fault_scatter(fault_dict,
                                                                       quantizer=quantizer, quant_mode=quant_mode, layer_type=layer_type,
                                                                       ksizes=ksizes, padding=padding, dilation_rates=dilation_rates,
                                                                       sim_truncarry=sim_truncarry)
        
        return preprocess_data
    
    def consistency_check(self, quant_mode, quantizer):
        """ Check the consistency between MAC unit setup and layer quantization setup """
        if self.quant_mode!=quant_mode:
            raise ValueError('The quantization mode of Layer and MAC unit must be the same!!\nGot Layer %s, MAC %s'%(self.quant_mode,quant_mode))
        if type(self.quantizer)!=type(quantizer):
            raise TypeError('The type of quantizer of Layer and MAC unit should be the same!!\n Class quantizer for one quantize set up, List for [input, weight, output] respectively.')
        if self.quantizer!=quantizer:
            raise AttributeError('The attributes of Layer quantizer and MAC unit quantizer are different!!')

   
def preprocess_layer_mac_fault(layer, mac_unit_, layer_mac_fault_dict,
                               layer_fmap_dist_stat=None, layer_wght_dist_stat=None, **kwargs):
    """ Layer wise handle mac fault injection preprocess part before model/layer call
        Seperate the CPU and GPU processing. The preprocess function under mac_unit class are for CPU process.

    Parameters
    ----------
    layer : tensorflow.keras.Layer
        The model for generate modulator. Get the layer shape info in model.
    mac_unit_ : Class mac_unit.
        The mac unit information holder class. For describe PE I/O interconnection and math of faulty behavior.
    layer_mac_fault_dict : Dictionary
        Fault dctionary for output feature maps.
        The layers have no weight and MAC operation are setting its fault dictionary to None.
    layer_fmap_dist_stat : Dictionary
        Statistic dctionary for input feature maps.
        The layers have no weight and MAC operation are setting its statistic dictionary to None.
    layer_wght_dist_stat : List of Dictionary.
        Statistic dctionary for kernel and bias.
        The layers have no weight and MAC operation are setting its statistic dictionary to None.
    Returns
    -------
    preprocess_data: Dictionary.
        The Dictionary contains the configuration and data for Tensor operation part.
        The layers have no weight and MAC operation are setting its fault dictionary to None.

    """
    if layer_mac_fault_dict is None:
        preprocess_data=None
    else:
        layer_name=layer.name
        layer_name=layer_name.lower()
        
        if 'dense' in layer_name:
            preprocess_data=mac_unit_.preprocess_mac_fault_caller(layer_mac_fault_dict, 
                                                                  ofmap_shape=layer.output_shape,
                                                                  layer_type='Dense',
                                                                  dist_stats_fmap=layer_fmap_dist_stat, 
                                                                  dist_stats_wght=layer_wght_dist_stat,
                                                                  **kwargs)
        elif 'conv' in layer_name and 'depth' not in layer_name:
            preprocess_data=mac_unit_.preprocess_mac_fault_caller(layer_mac_fault_dict, 
                                                                  ofmap_shape=layer.output_shape,
                                                                  layer_type='Conv2D',
                                                                  ksizes=layer.kernel_size, 
                                                                  padding=layer.padding, 
                                                                  dilation_rates=layer.dilation_rate, 
                                                                  dist_stats_fmap=layer_fmap_dist_stat, 
                                                                  dist_stats_wght=layer_wght_dist_stat,
                                                                  **kwargs)
        elif 'conv' in layer_name and 'depth' in layer_name:
            preprocess_data=mac_unit_.preprocess_mac_fault_caller(layer_mac_fault_dict, 
                                                                  ofmap_shape=layer.output_shape,
                                                                  layer_type='DepthwiseConv2D',
                                                                  ksizes=layer.kernel_size, 
                                                                  padding=layer.padding, 
                                                                  dilation_rates=layer.dilation_rate,
                                                                  dist_stats_fmap=layer_fmap_dist_stat, 
                                                                  dist_stats_wght=layer_wght_dist_stat,
                                                                  **kwargs)
        else:
            raise TypeError('Layer is neigher Dense, Conv2D or DepthwiseConv2D. Layer mac fault dict should be None.')
            
    return preprocess_data

def preprocess_model_mac_fault(model, mac_unit_, model_mac_fault_dict_list, 
                               model_fmap_dist_stat_list=None, model_wght_dist_stat_list=None, **kwargs):
    """ Model wise handle mac fault injection preprocess part before model/layer call
        Seperate the CPU and GPU processing. The preprocess function under mac_unit class are for CPU process.

    Parameters
    ----------
    model : tensorflow.keras.Model
        The model for generate modulator. Get the layer shape info in model.
    mac_unit_ : Class mac_unit.
        The mac unit information holder class. For describe PE I/O interconnection and math of faulty behavior.
    model_mac_fault_dict_list : List of Dictionary
        Fault dctionary for output feature maps.
        The list are the same order as the Keras model layer list. Each Dictionary in List is for its corresponding layer.
        The layers have no weight and MAC operation are setting its fault dictionary to None.
    model_wght_dist_stat_list : List of Dictionary
        The model weight distribution statistic list. Each Dictionary in List is for its corresponding layer.
        The layers have no weight and MAC operation are setting its fault dictionary to None.
    model_fmap_dist_stat_list : List of Dictionary
        The model feature map distribution statistic list. Each Dictionary in List is for its corresponding layer.
        The layers have no weight and MAC operation are setting its fault dictionary to None.

    Returns
    -------
    model_preprocess_data: List of Dictionary.
        The list are the same order as the Keras model layer list. Each Dictionary in List is for its corresponding layer.
        The Dictionary contains the configuration and data for Tensor operation part.
        The layers have no weight and MAC operation are setting its fault dictionary to None.

    """
    model_depth=len(model.layers)
    model_preprocess_data_list=[None for _ in range(model_depth)]
    if model_wght_dist_stat_list is not None and len(model_wght_dist_stat_list)!=model_depth:
        raise AttributeError('The length of model_wght_dist_stat_list should be the same as number of layers in model.')
    if model_fmap_dist_stat_list is not None and len(model_fmap_dist_stat_list)!=model_depth:
        raise AttributeError('The length of model_fmap_dist_stat_list should be the same as number of layers in model.')

    for layer_num in range(1,model_depth):
        if model_wght_dist_stat_list is None:
            layer_wght_dist_stat=None
        else:
            layer_wght_dist_stat=model_wght_dist_stat_list[layer_num]
        if model_fmap_dist_stat_list is None:
            layer_fmap_dist_stat=None
        else:
            layer_fmap_dist_stat=model_fmap_dist_stat_list[layer_num]

        preprocess_data=preprocess_layer_mac_fault(model.layers[layer_num],
                                                   mac_unit_,
                                                   model_mac_fault_dict_list[layer_num], 
                                                   layer_fmap_dist_stat=layer_fmap_dist_stat,
                                                   layer_wght_dist_stat=layer_wght_dist_stat,
                                                   **kwargs)
        
        model_preprocess_data_list[layer_num]=preprocess_data
        
    return model_preprocess_data_list
        
