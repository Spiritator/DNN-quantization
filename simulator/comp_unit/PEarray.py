# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 16:30:04 2019

@author: Yung-Yu Tsai

Processing element array setting for compuation unit fault mapping
"""

import numpy as np
    
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
        if len(target_shape)!=len(target_prior):
            raise ValueError('The length of target_shape must equals to target_prior, but got %d and %d.'%(len(target_shape),len(target_prior)))

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
            
    def tilt_idx(self, index, axis, direction, shape=None, shift=1):
        """ Make index tilted for systolic array input
        
        # Arguments
            index: Tuple or 2D ndarray. The index(coordinate) of source_shape which will be transform to target_shape index.
                2D ndarray (a,b) where a for list of coordinates, b for coordinate dimensions i.e. (16,4) there are 16 coordinates with 4 dimensions.
            shape: Tuple of Integer. The shape of data which the index represents. Needed argument for negative shift.
            axis: Integer. The axis wanted to be tilted.
            direction: Integer. The axis of direaction that are tilted to.
            shift: Integer. The amount of shifting for tilted representation. Positive number for tilt forward, negative for tilted backward.
        
        # Returns
            Converted coordinate. Single coordinate return in Tuple, multiple coordinate return in 2D ndarray.        
        """
        new_index=np.copy(index)
        if shift<0:
            new_index[:,direction]+=np.subtract(shape[axis]-1, new_index[:,axis])*(-shift)
        else:
            new_index[:,direction]+=new_index[:,axis]*shift
        return new_index
    
    def estimate_clk(self, mapping_shape, non_clk_PE_shape):
        """ Estimate the needed number of clock cycle by shape of mapping data
        
        """
        return int(np.ceil(np.prod(mapping_shape)/np.prod(non_clk_PE_shape)))
    
    def get_PE_prior(self, prior_list, tile_shape):
        """ Organize PE mapping shape and prior
        
        """
        map_shape_pe=list()
        mpp_ind=dict()
        mpp_cnt=-1
        map_prior_pe=list()
        
        if 'PE_x' in prior_list:
            map_shape_pe.append(self.n_x)
            mpp_cnt+=1
            mpp_ind['PE_x']=mpp_cnt
        if 'PE_y' in prior_list:
            map_shape_pe.append(self.n_y)
            mpp_cnt+=1
            mpp_ind['PE_y']=mpp_cnt
        
        if 't_clk' in prior_list:   
            mpp_cnt+=1
            mpp_ind['t_clk']=mpp_cnt
            if self.n_clk is None:
                map_shape_pe.append(self.estimate_clk(tile_shape,map_shape_pe))
            else:
                map_shape_pe.append(self.n_clk)
                
        for prior in prior_list:
            map_prior_pe.append(mpp_ind[prior])
            
        return map_shape_pe,map_prior_pe
        
    def mapping_tile_stationary(self, parameter, tile, PE_required_axes_prior=None, tile_mapping_prior=None):
        """ Mapping a tile onto PE array dataflow model. Direct transformation in this function. No data duplication.
            This stationary mapping place data to PE on a certain clock cycle.
        
        # Arguments
            parameter: String. The parameter being mapped to, must be 'ofmap', 'wght' or 'ifmap'.
            tile: Class. The tile_PE class for PE array fault tolerance analysis. The tile about to be mapped.
            PE_required_axes_prior: List of Strings. The axis of direction in PE array i.e. 'PE_x', 'PE_y', 't_clk'. 
                These axes are the dimension in PE array dataflow model for tile mapping.
                The order in List is the priority for data mapping in PE array.
            tile_mapping_prior: List or Tuple of Integer. The list for ravel priority of tile slice_shape dimensions. The list is the dimension index.
        
        # Returns
            Converted fault dictionary. Keys are PE dataflow model coordinates. Items are fault info dictionarys.
        """
        if PE_required_axes_prior is not None:
            tile.PE_required_axes_prior=PE_required_axes_prior
        if tile_mapping_prior is not None:
            tile.tile_mapping_prior=tile_mapping_prior
        
        tile.check_prior()
        
        if tile.expansion:
            tile_shape=tile.slice_shape
        else:
            tile_shape=tile.tile_shape
            
        map_shape_pe,map_prior_pe=self.get_PE_prior(tile.PE_required_axes_prior, tile_shape)

        if tile.expansion:
            orig_coors=np.array(list(tile.fault_dict_expand.keys()))
            mapped_coors=self.mapping_ravel_idx(orig_coors,
                                                source_shape=tile.slice_shape,
                                                source_prior=tile.tile_mapping_prior,
                                                target_shape=map_shape_pe,
                                                target_prior=map_prior_pe)
            
            mapped_coors_fd=list(zip(*mapped_coors.T))
            fault_info=list(tile.fault_dict_expand.values())
            new_fault_dict=dict(zip(mapped_coors_fd,fault_info))
        else:
            orig_coors=np.array(list(tile.fault_dict.keys()))
            mapped_coors=self.mapping_ravel_idx(orig_coors,
                                                source_shape=tile.tile_shape,
                                                source_prior=tile.tile_mapping_prior,
                                                target_shape=map_shape_pe,
                                                target_prior=map_prior_pe)
            
            mapped_coors_fd=list(zip(*mapped_coors.T))
            fault_info=list(tile.fault_dict.values())
            new_fault_dict=dict(zip(mapped_coors_fd,fault_info))

        if parameter=='ofmap':
            self.ofmap_fault_dict=new_fault_dict
            self.mapping_shape_ofmap=map_shape_pe
        elif parameter=='ifmap':
            self.ifmap_fault_dict=new_fault_dict
            self.mapping_shape_ifmap=map_shape_pe
        elif parameter=='wght':
            self.wght_fault_dict=new_fault_dict
            self.mapping_shape_wght=map_shape_pe
            
        return new_fault_dict

    def mapping_tile_streaming(self, parameter, tile, PE_required_axes_prior=None, tile_mapping_prior=None):
        """ Mapping a tile onto PE array dataflow model. Direct transformation in this function. No data duplication.
            This streaming mapping lets data stream through PE and record the fault traces.
        
        # Arguments
            parameter: String. The parameter being mapped to, must be 'ofmap', 'wght' or 'ifmap'.
            tile: Class. The tile_PE class for PE array fault tolerance analysis. The tile about to be mapped.
            PE_required_axes_prior: List of Strings. The axis of direction in PE array i.e. 'PE_x', 'PE_y', 't_clk'. 
                These axes are the dimension in PE array dataflow model for tile mapping.
                The order in List is the priority for data mapping in PE array.
            tile_mapping_prior: List or Tuple of Integer. The list for ravel priority of tile slice_shape dimensions. The list is the dimension index.
        
        """
        
        
# TODO
# gen fault list
# read in tiles
# how the data flow        
# Tilted data      
# PE stall (shift in or other)        
# double buffer no stall
