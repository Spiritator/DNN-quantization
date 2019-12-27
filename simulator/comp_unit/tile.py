# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 16:22:01 2019

@author: Yung-Yu Tsai

DNN tiling for computation unit fault mapping
"""

import numpy as np
from simulator.memory.tile import tile,tile_FC

class tile_PE(tile):
    def __init__(self, tile_shape, is_fmap, wl=32, row_prior=[], col_prior=[]):
        """The tile of a DNN feature map or weights

        # Arguments
            tile_shape: Tuple. The shape of tile.
            Tm: Integer. The size of tile on the input feature map dimension (weight) or channel dimention (feature map).
            Tn: Integer. The size of tile on the output feature map dimension (weight) or batch dimention (feature map).
            Tr: Integer. The size of tile on the kernel row dimension or feature map row dimention.
            Tc: Integer. The size of tile on the kernel column dimension or feature map column dimention.
            is_fmap: Bool. The tile is feature map tile or weight tile.
            wl: Integer. The word length of DNN model parameter.
            row_prior: List of Strings. The priority of memory mapping in the memory row dimension. Consist of 'Tm', 'Tn', 'Tr', 'Tc'.
            col_prior: List of Strings. The priority of memory mapping in the memory column dimension. Consist of 'Tm', 'Tn', 'Tr', 'Tc'.
    
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
        self.wl=wl
        self.row_prior=row_prior
        self.col_prior=col_prior
        self.prior_element=['Tm','Tn','Tr','Tc']
        self.slice_head_list=None
        self.slice_head_order=None
        self.fault_dict=dict()
        self.tile_size=None
        self.use_bias=False
        self.bias_fault_dict=dict()
        self.bias_range=None
        self.shape_len=4
        self.base_coor=None
        self.print_detail=True
