# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 16:22:01 2019

@author: Yung-Yu Tsai

DNN tiling for computation unit fault mapping
"""

import numpy as np
from simulator.memory.tile import tile,tile_FC

class tile_PE(tile):
    def __init__(self, tile_shape, is_fmap, required_axes=[], axis_prior=[]):
        """The tile of a DNN feature map or weights

        # Arguments
            tile_shape: Tuple. The shape of tile.
            Tm: Integer. The size of tile on the input feature map dimension (weight) or channel dimention (feature map).
            Tn: Integer. The size of tile on the output feature map dimension (weight) or batch dimention (feature map).
            Tr: Integer. The size of tile on the kernel row dimension or feature map row dimention.
            Tc: Integer. The size of tile on the kernel column dimension or feature map column dimention.
            is_fmap: Bool. The tile is feature map tile or weight tile.
            required_axes: List of Strings. The axis of direction in PE array i.e. 'PE_x', 'PE_y', 't_clk'. 
                These axes are the dimension in PE array dataflow model for tile mapping.
                The order in List is the priority for data mapping in PE array.
            col_prior: Dictionary of List of Strings. The priority of PE mapping in each PE dimension. Consist of 'Tm', 'Tn', 'Tr', 'Tc'.
                The dictionary structure is {'PE_x':['Tm', 'Tn', 'Tr', 'Tc'],
                                             'PE_y':['Tm', 'Tn', 'Tr', 'Tc'],
                                             't_clk':['Tm', 'Tn', 'Tr', 'Tc']}
    
        """
        if not isinstance(is_fmap,bool):
            raise ValueError('Augment is_fmap must be True (feature map tile) or False (weight tile)')
        if len(tile_shape) != 4:
            raise ValueError('The augment tile_shape must be in Tuple dtype and have length 4 but got length %d'%len(tile_shape))
        if is_fmap:    
            self.Tm=tile_shape[3]
            self.Tn=tile_shape[0]
            self.Tr=tile_shape[1]
            self.Tc=tile_shape[2]
        else:
            self.Tm=tile_shape[2]
            self.Tn=tile_shape[3]
            self.Tr=tile_shape[0]
            self.Tc=tile_shape[1]
        self.is_fmap=is_fmap
        self.required_axes=required_axes
        self.axis_prior=axis_prior
        self.axis_element=['PE_x','PE_y','t_clk']
        self.prior_element=['Tm','Tn','Tr','Tc']
        
        self.fault_dict=dict()
        self.use_bias=False
        self.bias_fault_dict=dict()
        self.shape_len=4
        
        self.expansion=False
        self.expand_method=None
        self.expand_shape=None
        self.slice_shape=None
        self.slice_permute=None
        self.fault_dict_expand=dict()
        
        self.print_detail=True
        
    def check_prior(self):
        if not isinstance(self.required_axes,list):
            raise ValueError('The augment required_axes must be in list dtype.')
            
        for axis in self.required_axes:
            if axis not in self.axis_element:
                raise ValueError('The augment required_axes must be in list %s'%(str(self.axis_element)))
        
            if not isinstance(self.axis_prior[axis],list) or len(self.axis_prior[axis])!=self.shape_len:
                raise ValueError('The augment axis_prior must be in list dtype and have length %d but got length %d'%(self.shape_len,len(self.axis_prior[axis])))
    
            for i in range(self.shape_len):
                if self.axis_prior[axis][i] not in self.prior_element:
                    raise ValueError('The augment axis_prior must be in list %s'%(str(self.prior_element)))

    def expand_data(self, method, expect_shape, slice_dims, slices_permute):
        """ Data expansion before put into PE array. Usually used for ifmap and weight reuse. 
            The data may  be cut into many pieces than fit into PE. Different slices calculate in different clock cycle.
        
        # Arguments
            method: String. The expansion method of tile.
            expect_shape: Tuple. The expected shape to be expand into.
            slice_dims: Tuple. Indicates the dimension of slices in the expect_shape. A tuple (a,b) size 2, which means
                the expect_shape[a:b] is the part of slice dimensions. The other parts are for time multiplexed permute.
            slices_permute: Tuple. Indicates how to permute the time multiplexed part of expect_shape. Tuple (c,d,e) means
                the order of time multiplexed part dimension to permute. Variable c,d,e are the axis index of expect_shape.
                
        """
        self.expansion=True
        self.expand_method=method
        self.expand_shape=expect_shape
        
        if not isinstance(slice_dims,tuple) or len(slice_dims)!=2:
            raise TypeError('slice_dims must be in type tuple and length 2, get length %d'%(len(slice_dims)))
        self.slice_shape=expect_shape[slice_dims[0]:slice_dims[1]]
        self.slice_permute=slices_permute
        
        
    