# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 16:22:01 2019

@author: Yung-Yu Tsai

DNN tiling for computation unit fault mapping
"""

import numpy as np
from simulator.memory.tile import tile,tile_FC

class tile_PE(tile):
    def __init__(self, tile_shape, is_fmap, required_axes=[], axis_prior=[], **kwargs):
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
            axis_prior: Dictionary of List of Strings. The priority of PE mapping in each PE dimension. Consist of 'Tm', 'Tn', 'Tr', 'Tc'.
                The dictionary structure is {'PE_x':['Tm', 'Tn', 'Tr', 'Tc'],
                                             'PE_y':['Tm', 'Tn', 'Tr', 'Tc'],
                                             't_clk':['Tm', 'Tn', 'Tr', 'Tc']}
    
        """
        super(tile_PE, self).__init__(tile_shape, is_fmap, **kwargs)
        self.tile_shape=tile_shape
        self.required_axes=required_axes
        self.axis_prior=axis_prior
        self.axis_element=['PE_x','PE_y','t_clk']
        self.prior_element=['Tm','Tn','Tr','Tc']
                
        self.expansion=False
        self.expand_method=None
        self.expand_shape=None
        self.slice_shape=None
        self.slice_permute=None
        self.fault_dict_expand=dict()
                
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

    def reshape_ravel_idx(self,index,source_shape,source_prior,target_shape,target_prior):
        """ Convert index to differet shapes for tile data expansion. Unravel index to a numtag than ravel to another index.
        
        # Arguments
            index: Tuple or 2D ndarray. The index(coordinate) of source_shape which will be transform to target_shape index.
            source_shape: Tuple. The shape of source array before tranformation.
            source_prior: List or Tuple of Integer. The list for unravel priority of source_shape dimensions. The list is the dimension index.
            target_shape: Tuple. The shape of target array for tranformation to.
            target_prior: List or Tuple of Integer. The list for ravel priority of target_shape dimensions. The list is the dimension index.

        """
        if len(source_shape)!=len(source_prior):
            raise ValueError('The length of source_shape must equals to source_prior, but got %d and %d.'%(len(source_shape),len(source_prior)))
        
        restore_index=np.zeros((len(target_shape),),dtype=int)
        for i in range(len(target_shape)):
            restore_index[target_prior[i]]=i

        if isinstance(index,tuple) or (isinstance(index,np.ndarray) and len(index.shape)==1):
            if len(index)!=len(source_shape):
                raise ValueError('The length of coordinate Tuple in tile must be %d but got %d.'%(self.shape_len,len(index)))

            numtag=np.ravel_multi_index(np.array(index)[source_prior],np.array(source_shape)[source_prior])
            
            coor=np.unravel_index(numtag,np.array(target_shape)[target_prior])
            coor=np.array(coor)[restore_index]
            
            return tuple(coor)
                        
        elif isinstance(index,np.ndarray):
            if index.shape[-1]!=self.shape_len:
                raise ValueError('The length of coordinate Tuple in tile must be %d but got %d.'%(self.shape_len,index.shape[-1]))
                
            numtag=np.ravel_multi_index(index.T[source_prior],np.array(source_shape)[source_prior])
                    
            coor=np.unravel_index(numtag,np.array(target_shape)[target_prior])
            coor=np.array(coor)[restore_index]

            return coor.T

    def expand_data(self, method, expect_shape, slice_dims, slices_permute):
        """ Data expansion before put into PE array. Usually used for ifmap and weight reuse. 
            The data may  be cut into many pieces than fit into PE. Different slices calculate in different clock cycle.
        
        # Arguments
            method: String. The expansion method of tile. Either be 'reshape' or 'extract_patches'. Method 'reshape' means 
                change data shape without any duplication in array. Method 'extract_patches' is the tf.extract_image_patches 
                method to get ifmap patches.
                
            expect_shape: Tuple. The expected shape to be expand into.
            
            slice_dims: Tuple. Indicates the dimension of slices in the expect_shape. A tuple must be the same size as 
                expect_shape, tuple (0,0,3,3) means the expect_shape dimension 2,3 are part of slice dimensions. The 
                0,1 dimension parts are for time multiplexed permute so the dimension size is 0.
                
            slices_permute: Tuple. Indicates how to permute the time multiplexed part of expect_shape. Tuple (a,b,c,d) means
                the order of time multiplexed part dimension to permute. Variable a,b,c,d are the axis index of expect_shape.
                
        """
        self.expansion=True
        self.expand_method=method
        self.expand_shape=expect_shape
        
        if not isinstance(slice_dims,tuple) or len(slice_dims)!=len(self.expand_shape):
            raise TypeError('slice_dims must be in length %d, but get length %d'%(len(self.expand_shape),len(slice_dims)))
        self.slice_shape=np.array(slice_dims)
        self.slice_shape=tuple(self.slice_shape[self.slice_shape>0])
        self.slice_permute=slices_permute
        
        
    