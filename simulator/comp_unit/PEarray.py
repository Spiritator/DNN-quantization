# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 16:30:04 2019

@author: Yung-Yu Tsai

Processing element array setting for compuation unit fault mapping
"""

import numpy as np

class array_axis:
    """
    The dimension in PE array which contains the dataflow information.
    
    """
    
    def __init__():
        """
        # Arguments
            
        
        """

    
class PEarray:
    """
    The PE array functional model for computation unit fault tolerance analysis.
    The 2D PE array consist of three basic dimension 'PE_x', 'PE_y' and 't_clk' 
    which means the row direction, column direction and clock cycles required to 
    complete a tile computation.
    The 3D PE array dataflow model is for mapping the tile data to computaion unit.
    
    """
    def __init__(self, n_x, n_y, n_clk=None, ofmap_tile=None, wght_tile=None, ifmap_tile=None):
        """
        # Arguments
            n_x: Integer. Number of PEs in a row.
            n_y: Integer. Number of PEs in a column.
            n_clk: Integer. Number of clock cycles for a tile to process.
            fault_num: Integer. Number of faults in array.
            fault_dict: Dictionary. The fault information {location : fault type}
                    example: fault_dict = {(PE_x1,PE_y1):{‘param’:‘ifmap_in’,
                                                          ‘global’:False,
                                                          ‘SA_type’:’flip’,
                                                          ‘SA_bit’:3} ,
                                           (PE_x2,PE_y2):{‘param’:‘psum_out’,
                                                          ‘global’:True,
                                                          ‘SA_type’:’0’,
                                                          ‘SA_bit’:5} ,
                                           }
                   'param' must be one of [ifmap_in, ifmap_out, wght_in, wght_out, psum_in, psum_out ]
                                           
            ofmap_tile: Class. The tile_PE class for PE array fault tolerance analysis. Output feature maps tile.
            wght_tile: Class. The tile_PE class for PE array fault tolerance analysis. Weights feature maps tile.
            ifmap_tile: Class. The tile_PE class for PE array fault tolerance analysis. Iutput feature maps tile.

        """
        self.n_x=n_x
        self.n_y=n_y
        self.n_clk=n_clk
        self.fault_num=None
        self.fault_dict=dict()
        self.ifmap_tile=ifmap_tile
        self.wght_tile=wght_tile
        self.ofmap_tile=ofmap_tile
        
    def mapping_ravel_idx(self,index,source_shape,source_prior,target_shape,target_prior):
        """ Convert index to differet shapes for tile data expansion. Unravel index to a numtag than ravel to another index.
        
        # Arguments
            index: Tuple or 2D ndarray. The index(coordinate) of source_shape which will be transform to target_shape index.
                2D ndarray (a,b) where a for list of coordinates, b for coordinate dimensions i.e. (16,4) there are 16 coordinates with 4 dimensions.
            source_shape: Tuple. The shape of source array before tranformation.
            source_prior: List or Tuple of Integer. The list for unravel priority of source_shape dimensions. The list is the dimension index.
            target_shape: Tuple. The shape of target array for tranformation to.
            target_prior: List or Tuple of Integer. The list for ravel priority of target_shape dimensions. The list is the dimension index.
        
        # Returns
            Converted coordinate. Single coordinate return in Tuple, multiple coordinate return in 2D ndarray.
        """
        if len(source_shape)!=len(source_prior):
            raise ValueError('The length of source_shape must equals to source_prior, but got %d and %d.'%(len(source_shape),len(source_prior)))
        
        restore_index=np.zeros((len(target_shape),),dtype=int)
        for i in range(len(target_shape)):
            restore_index[target_prior[i]]=i

        if isinstance(index,tuple) or (isinstance(index,np.ndarray) and len(index.shape)==1):
            if len(index)!=len(source_shape):
                raise ValueError('The length of coordinate Tuple in tile must be %d but got %d.'%(len(source_shape),len(index)))

            numtag=np.ravel_multi_index(np.array(index)[source_prior],np.array(source_shape)[source_prior])
            
            coor=np.unravel_index(numtag,np.array(target_shape)[target_prior])
            coor=np.array(coor)[restore_index]
            
            return tuple(coor)
                        
        elif isinstance(index,np.ndarray):
            if index.shape[-1]!=len(source_shape):
                raise ValueError('The length of coordinate Tuple in tile must be %d but got %d.'%(len(source_shape),index.shape[-1]))
                
            numtag=np.ravel_multi_index(index.T[source_prior],np.array(source_shape)[source_prior])
                    
            coor=np.unravel_index(numtag,np.array(target_shape)[target_prior])
            coor=np.array(coor)[restore_index]

            return coor.T
        
        else:
            raise TypeError('index for transformation must be either tuple or 2D numpy array.')
            
    def mapping_tile(self, tile):
        """ Mapping a tile onto PE array dataflow model. Direct tranformation in this function. No data duplication
        
        """
        pass

# TODO
# gen fault list
# read in tiles
# how the data flow        
# Tilted data      
# PE stall (shift in or other)        
# double buffer no stall
