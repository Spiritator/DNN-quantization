# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 16:30:04 2019

@author: Yung-Yu Tsai

Processing element array setting for compuation unit fault mapping
"""

import numpy as np

class axis_info:
    """
    The axis information holder class. For hold and calling PE mapping parameters.
    """
    def __init__(self, 
                 PE_required_axes_prior=None, 
                 tile_mapping_prior=None,
                 PE_fix_axis=None,
                 indice=None,
                 PE_broadcast_axis=None,
                 tile_stream_axis=None,
                 tile_direction=None,
                 PE_stream_axis=None,
                 PE_direction=None):
        self.PE_required_axes_prior=PE_required_axes_prior
        self.tile_mapping_prior=tile_mapping_prior
        self.PE_fix_axis=PE_fix_axis
        self.indice=indice
        self.PE_broadcast_axis=PE_broadcast_axis
        self.tile_direction=tile_direction
        self.PE_stream_axis=PE_stream_axis
        self.PE_direction=PE_direction

        
class PEflow:
    """
    The PE flow description class. For information gathering and PE dataflow setup.
    A PEflow represent a data tile (one of ofmap, weight, ifmap)
    """
    def __init__(self, 
                 permute_info=None, 
                 fixed_info=None, 
                 broadcast_info=None, 
                 streaming_info=None, 
                 repeat=0, 
                 duplicate=0, 
                 pack_size=1,
                 stall_latency=0):
        """
        PE axis flow type
            'permute': permute data long axis. 
            'fixed': data fix in certain index on this axis.
            'broadcast': data being broadcast to all entries in this axis. 
            'streaming': data being streamed in in this axis.
        
        info must be feed in by dicitionary format
            ex: info_x = {'PE_required_axes_prior':['t_clk','PE_x'],
                          'tile_mapping_prior':[2,1,0]}
        
        info description
            'permute': 
                PE_required_axes_prior: List of Strings. The axis of direction in PE array i.e. 'PE_x', 'PE_y', 't_clk'. 
                    These axes are the dimension in PE array dataflow model for tile mapping.
                    The order in List is the priority for data mapping in PE array.
                tile_mapping_prior: List or Tuple of Integer. The list for ravel priority of slice_shape dimensions. The list is the dimension index.

            'fixed': 
                PE_fix_axis: String or List of Strings. The dimension of target_shape that are being fix to. i.e. 'PE_x', 'PE_y', 't_clk'. 
                indice: Integer or List of Integer. The indice of the targeted dimension that represent the location of fix data. If multiple dimensions are fixed indice_fix must align with fix_dims.

            'broadcast': 
                PE_broadcast_axis: String or List of Strings. The dimension of target_shape that are being broadcast to. i.e. 'PE_x', 'PE_y', 't_clk'. 
                                
            'streaming': 
                PE_stream_axis: String. The axis index whose dimension is the sweep going on PE. i.e. 'PE_x' or 'PE_y'.
                tile_direction: String. 'forward' or 'backward' the direction of data flow in. Stream starts from the 0 index and increment, or else starts from last index and decrement.
                PE_direction: String. 'forward' or 'backward' the direction of window sweeping. 

        
        # Arguments
            permute_info: Dictionary. The infomation of permute flow. Must in the format describe above.
            fixed_info: Dictionary. The infomation of fixed flow. Must in the format describe above.
            broadcast_info: Dictionary. The infomation of broadcast flow. Must in the format describe above.
            streaming_info: Dictionary. The infomation of streaming flow. Must in the format describe above.
            repeat: Integer. The times for pre-mapped tile repeat element wise on t_clk axis. For mapping clock cycle.
            duplicate: Integer. The times for pre-mapped tile duplicate entirely on t_clk axis. For mapping clock cycle.
            pack_size: Integer. The number of slices of pre-mapped tile data in a slice-pack.
            stall_latency: Integer. The clock cycles need to wait till data get ready. 
                Or the clock cycles need to wait for other data going through PE array. 
                All clock cycles combined.
        """
        if permute_info is None:
            self.permute_info=None
        else:
            self.permute_info=axis_info( **permute_info)
            
        if fixed_info is None:
            self.fixed_info=None
        else:
            self.fixed_info=axis_info( **fixed_info)
        
        if broadcast_info is None:
            self.broadcast_info=None
        else:
            self.broadcast_info=axis_info( **broadcast_info)
        
        if streaming_info is None:
            self.streaming_info=None
        else:
            self.streaming_info=axis_info( **streaming_info)
            
        self.repeat=repeat
        self.duplicate=duplicate
        self.pack_size=pack_size
        self.stall_latency=stall_latency
        self.axis_element=['PE_x','PE_y','t_clk']
        
    def check_prior(self, data_shape):
        if (not isinstance(self.permute_info.PE_required_axes_prior,list)) and (not isinstance(self.permute_info.PE_required_axes_prior,str)):
            raise TypeError('The augment PE_required_axes must be String or List of Strings dtype.')
            
        for axis in self.permute_info.PE_required_axes_prior:
            if axis not in self.axis_element:
                raise ValueError('The augment PE_required_axes must be in list %s'%(str(self.axis_element)))
                        
        if len(data_shape)!=len(self.permute_info.tile_mapping_prior):
            raise ValueError('The length of tile_mapping_prior must equals to data shape, but got %d and %d.'%(len(self.permute_info.tile_mapping_prior),len(data_shape)))

    def check_fix(self):
        if isinstance(self.fixed_info.PE_fix_axis,str):
            if self.fixed_info.PE_fix_axis not in self.axis_element:
                raise ValueError('The augment PE_dix_dims must be in list %s'%(str(self.axis_element)))
        elif isinstance(self.fixed_info.PE_fix_axis,list):
            for dim in self.fixed_info.PE_fix_axis:
                if dim not in self.axis_element:
                    raise ValueError('The augment PE_dix_dims must be in list %s'%(str(self.axis_element)))
        else:
            raise TypeError('PE_fix_axis must either be integer or list of integer.')

    def check_broadcast(self):
        if isinstance(self.broadcast_info.PE_broadcast_axis,str):
            if self.broadcast_info.PE_broadcast_axis not in self.axis_element:
                raise ValueError('The augment PE_broadcast_axis must be in list %s'%(str(self.axis_element)))
        elif isinstance(self.broadcast_info.PE_broadcast_axis,list):
            for dim in self.broadcast_info.PE_broadcast_axis:
                if dim not in self.axis_element:
                    raise ValueError('The augment PE_broadcast_axis must be in list %s'%(str(self.axis_element)))
        else:
            raise TypeError('PE_broadcast_axis must either be integer or list of integer.')
            
    def check_streaming(self):
        if isinstance(self.streaming_info.PE_stream_axis,str):
            if self.streaming_info.PE_stream_axis not in self.axis_element:
                raise ValueError('The augment PE_stream_axis must be in list %s'%(str(self.axis_element)))
        elif isinstance(self.streaming_info.PE_stream_axis,list):
            for dim in self.streaming_info.PE_stream_axis:
                if dim not in self.axis_element:
                    raise ValueError('The augment PE_stream_axis must be in list %s'%(str(self.axis_element)))
        else:
            raise TypeError('PE_stream_axis must either be integer or list of integer.')


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
        self.setup_ready=False
        self.n_x=n_x
        self.n_y=n_y
        self.n_clk=n_clk
        self.fault_num=None
        self.fault_dict=dict()
        self.ifmap_tile=ifmap_tile
        self.wght_tile=wght_tile
        self.ofmap_tile=ofmap_tile
        self.used_axes=list()
        self.tmp_clk=None
        
    def setup_dataflow(self, 
                       o_permute_info=None, o_fixed_info=None, o_broadcast_info=None, o_streaming_info=None, o_repeat=0, o_duplicate=0, o_pack_size=1, o_stall_latency=0,
                       w_permute_info=None, w_fixed_info=None, w_broadcast_info=None, w_streaming_info=None, w_repeat=0, w_duplicate=0, w_pack_size=1, w_stall_latency=0,
                       i_permute_info=None, i_fixed_info=None, i_broadcast_info=None, i_streaming_info=None, i_repeat=0, i_duplicate=0, i_pack_size=1, i_stall_latency=0,
                       p_permute_info=None, p_fixed_info=None, p_broadcast_info=None, p_streaming_info=None, p_repeat=0, p_duplicate=0, p_pack_size=1, p_stall_latency=0,
                       b_permute_info=None, b_fixed_info=None, b_broadcast_info=None, b_streaming_info=None, b_repeat=0, b_duplicate=0, b_pack_size=1, b_stall_latency=0):
        """ Setup dataflow of ofmap, weight, ifmap. Read in PE dataflow arguments for mapping.
            o_* for output feature map
            w_* for weight kernel
            i_* for input feature map
            p_* for partial sum (the same shape as ofmap tile)
            b_* for weight bias
        
        # Arguments
            permute_info: Dictionary. The infomation of permute flow. Must in the format describe above.
            fixed_info: Dictionary. The infomation of fixed flow. Must in the format describe above.
            broadcast_info: Dictionary. The infomation of broadcast flow. Must in the format describe above.
            streaming_info: Dictionary. The infomation of streaming flow. Must in the format describe above.
            repeat: Integer. The times for pre-mapped tile repeat element wise on t_clk axis. For mapping clock cycle.
            duplicate: Integer. The times for pre-mapped tile duplicate entirely on t_clk axis. For mapping clock cycle.
            stall: Integer. The clock cycles need to wait till data get ready.
            latency: Integer. The clock cycles need to wait for other data going through PE array.

        """
        self.setup_ready=True
        self.ofmap_flow=PEflow(o_permute_info, o_fixed_info, o_broadcast_info, o_streaming_info, o_repeat, o_duplicate, o_pack_size, o_stall_latency)
        self.wght_flow=PEflow(w_permute_info, w_fixed_info, w_broadcast_info, w_streaming_info, w_repeat, w_duplicate, w_pack_size, w_stall_latency)
        self.ifmap_flow=PEflow(i_permute_info, i_fixed_info, i_broadcast_info, i_streaming_info, i_repeat, i_duplicate, i_pack_size, i_stall_latency)
        self.psum_flow=PEflow(p_permute_info, p_fixed_info, p_broadcast_info, p_streaming_info, p_repeat, p_duplicate, p_pack_size, p_stall_latency)
        self.bias_flow=PEflow(b_permute_info, b_fixed_info, b_broadcast_info, b_streaming_info, b_repeat, b_duplicate, b_pack_size, b_stall_latency)
        
    def estimate_clk(self, mapping_shape, non_clk_PE_shape):
        """ Estimate the needed number of clock cycle by shape of mapping data
        
        """
        return int(np.ceil(np.prod(mapping_shape)/np.prod(non_clk_PE_shape)))
    
    def get_PE_prior(self, prior_list, tile_shape, keep_slice=False):
        """ Organize PE mapping permute shape and prior
        
        """
        if isinstance(prior_list,str):
            prior_list=[prior_list]
        self.used_axes+=prior_list
            
        map_shape_pe=list()
        mpp_ind=dict()
        mpp_cnt=-1
        map_prior_pe=list()
                
        if 'PE_y' in prior_list:
            map_shape_pe.append(self.n_y)
            mpp_cnt+=1
            mpp_ind['PE_y']=mpp_cnt
        if 'PE_x' in prior_list:
            map_shape_pe.append(self.n_x)
            mpp_cnt+=1
            mpp_ind['PE_x']=mpp_cnt
        
        if keep_slice:
            map_shape_pe.append(tile_shape[-1])
            
            if 't_clk' in prior_list:   
                mpp_cnt+=1
                mpp_ind['t_clk']=mpp_cnt
                if self.n_clk is None:
                    self.tmp_clk=self.estimate_clk(tile_shape,map_shape_pe)
                    map_shape_pe.insert(-1,self.tmp_clk)
                else:
                    map_shape_pe.insert(-1,self.n_clk)
                    
            map_prior_pe.append(mpp_cnt+1)
                    
        else:        
            if 't_clk' in prior_list:   
                mpp_cnt+=1
                mpp_ind['t_clk']=mpp_cnt
                if self.n_clk is None:
                    self.n_clk=self.estimate_clk(tile_shape,map_shape_pe)
                    map_shape_pe.append(self.n_clk)
                else:
                    map_shape_pe.append(self.n_clk)
        
        for prior in prior_list:
            map_prior_pe.append(mpp_ind[prior])
            
        return map_shape_pe,map_prior_pe
    
    def get_fix_arange(self,fix_dims,tile_shape,keep_slice=False):
        """ Organize PE mapping fixed shape and arange
        
        """
        if isinstance(fix_dims,str):
            fix_dims=[fix_dims]

        map_shape_pe=list()
        map_fixdims=list()
        mpp_cnt=-1
                
        if 'PE_y' in fix_dims:
            map_shape_pe.append(self.n_y)
            mpp_cnt+=1
            map_fixdims.append(mpp_cnt)
        elif 'PE_y' in self.used_axes:
            map_shape_pe.append(self.n_y)
            mpp_cnt+=1
            
        if 'PE_x' in fix_dims:
            map_shape_pe.append(self.n_x)
            mpp_cnt+=1
            map_fixdims.append(mpp_cnt)
        elif 'PE_x' in self.used_axes:
            map_shape_pe.append(self.n_x)
            mpp_cnt+=1

        if keep_slice:
            map_shape_pe.append(tile_shape[-1])
            
            if 't_clk' in fix_dims:
                if self.n_clk is None:
                    if self.tmp_clk is None:
                        self.tmp_clk=self.estimate_clk(tile_shape,map_shape_pe)
                        map_shape_pe.insert(-1,self.tmp_clk)
                    else:
                        map_shape_pe.insert(-1,self.tmp_clk)
                else:
                    map_shape_pe.insert(-1,self.n_clk)
                mpp_cnt+=1
                map_fixdims.append(mpp_cnt)
            elif 't_clk' in self.used_axes:
                if self.n_clk is None:
                    if self.tmp_clk is None:
                        self.tmp_clk=self.estimate_clk(tile_shape,map_shape_pe)
                        map_shape_pe.insert(-1,self.tmp_clk)
                    else:
                        map_shape_pe.insert(-1,self.tmp_clk)
                else:
                    map_shape_pe.insert(-1,self.n_clk)
                mpp_cnt+=1

        else:     
            if 't_clk' in fix_dims:
                if self.n_clk is None:
                    if self.tmp_clk is None:
                        self.tmp_clk=self.estimate_clk(tile_shape,map_shape_pe)
                        map_shape_pe.append(self.tmp_clk)
                    else:
                        map_shape_pe.append(self.tmp_clk)
                else:
                    map_shape_pe.append(self.n_clk)
                mpp_cnt+=1
                map_fixdims.append(mpp_cnt)
            elif 't_clk' in self.used_axes:
                if self.n_clk is None:
                    if self.tmp_clk is None:
                        self.tmp_clk=self.estimate_clk(tile_shape,map_shape_pe)
                        map_shape_pe.append(-1,self.tmp_clk)
                    else:
                        map_shape_pe.append(self.tmp_clk)
                else:
                    map_shape_pe.append(self.n_clk)
                mpp_cnt+=1
        
        self.used_axes+=fix_dims
        
        if keep_slice:
            map_arange=np.arange(len(self.used_axes)+1)
        else:
            map_arange=np.arange(len(self.used_axes))
            
        for i in map_fixdims:
            map_arange=np.delete(map_arange,i)
        
        if len(map_fixdims)==1:
            map_fixdims=map_fixdims[0]
        
        return map_fixdims,map_shape_pe,map_arange
    
    def get_broadcast_arange(self,broadcast_dims,tile_shape,keep_slice=False):
        """ Organize PE mapping broadcast shape and arange
        
        """
        if isinstance(broadcast_dims,str):
            broadcast_dims=[broadcast_dims]

        map_shape_pe=list()
        map_broaddims=list()
        mpp_cnt=-1
                
        if 'PE_y' in broadcast_dims:
            map_shape_pe.append(self.n_y)
            mpp_cnt+=1
            map_broaddims.append(mpp_cnt)
        elif 'PE_y' in self.used_axes:
            map_shape_pe.append(self.n_y)
            mpp_cnt+=1
            
        if 'PE_x' in broadcast_dims:
            map_shape_pe.append(self.n_x)
            mpp_cnt+=1
            map_broaddims.append(mpp_cnt)
        elif 'PE_x' in self.used_axes:
            map_shape_pe.append(self.n_x)
            mpp_cnt+=1

        if keep_slice:
            map_shape_pe.append(tile_shape[-1])
            
            if 't_clk' in broadcast_dims:
                if self.n_clk is None:
                    if self.tmp_clk is None:
                        self.tmp_clk=self.estimate_clk(tile_shape,map_shape_pe)
                        map_shape_pe.insert(-1,self.tmp_clk)
                    else:
                        map_shape_pe.insert(-1,self.tmp_clk)
                else:
                    map_shape_pe.insert(-1,self.n_clk)
                mpp_cnt+=1
                map_broaddims.append(mpp_cnt)
            elif 't_clk' in self.used_axes:
                if self.n_clk is None:
                    if self.tmp_clk is None:
                        self.tmp_clk=self.estimate_clk(tile_shape,map_shape_pe)
                        map_shape_pe.insert(-1,self.tmp_clk)
                    else:
                        map_shape_pe.insert(-1,self.tmp_clk)
                else:
                    map_shape_pe.insert(-1,self.n_clk)
                mpp_cnt+=1

        else:     
            if 't_clk' in broadcast_dims:
                if self.n_clk is None:
                    if self.tmp_clk is None:
                        self.tmp_clk=self.estimate_clk(tile_shape,map_shape_pe)
                        map_shape_pe.append(self.tmp_clk)
                    else:
                        map_shape_pe.append(self.tmp_clk)
                else:
                    map_shape_pe.append(self.n_clk)
                mpp_cnt+=1
                map_broaddims.append(mpp_cnt)
            elif 't_clk' in self.used_axes:
                if self.n_clk is None:
                    if self.tmp_clk is None:
                        self.tmp_clk=self.estimate_clk(tile_shape,map_shape_pe)
                        map_shape_pe.append(-1,self.tmp_clk)
                    else:
                        map_shape_pe.append(self.tmp_clk)
                else:
                    map_shape_pe.append(self.n_clk)
                mpp_cnt+=1
        
        self.used_axes+=broadcast_dims
        
        if keep_slice:
            map_arange=np.arange(len(self.used_axes)+1)
        else:
            map_arange=np.arange(len(self.used_axes))
            
        map_shape_data=np.copy(map_shape_pe)
            
        for i in map_broaddims:
            map_shape_data=np.delete(map_shape_data,i)
            map_arange=np.delete(map_arange,i)
        
        if len(map_broaddims)==1:
            map_broaddims=map_broaddims[0]
        
        return map_shape_data,map_shape_pe,map_broaddims,map_arange
    
    def get_streaming_arange(self,stream_dim,tile_shape,keep_slice=False):
        """ Organize PE mapping streaming shape and arange
        
        """
        
        map_shape_pe=list()
        mpp_cnt=-1
                
        if 'PE_y' == stream_dim:
            map_shape_pe.append(self.n_y)
            mpp_cnt+=1
            map_streamdim=mpp_cnt
        elif 'PE_y' in self.used_axes:
            map_shape_pe.append(self.n_y)
            mpp_cnt+=1
            
        if 'PE_x' == stream_dim:
            map_shape_pe.append(self.n_x)
            mpp_cnt+=1
            map_streamdim=mpp_cnt
        elif 'PE_x' in self.used_axes:
            map_shape_pe.append(self.n_x)
            mpp_cnt+=1

        if keep_slice:
            map_shape_pe.append(tile_shape[-1])
            if self.n_clk is None:
                if self.tmp_clk is None:
                    self.tmp_clk=self.estimate_clk(tile_shape,map_shape_pe)
                    map_shape_pe.insert(-1,self.tmp_clk)
                else:
                    map_shape_pe.insert(-1,self.tmp_clk)
            else:
                map_shape_pe.insert(-1,self.n_clk)
        else:     
            if self.n_clk is None:
                if self.tmp_clk is None:
                    self.tmp_clk=self.estimate_clk(tile_shape,map_shape_pe)
                    map_shape_pe.append(self.tmp_clk)
                else:
                    map_shape_pe.append(self.tmp_clk)
            else:
                map_shape_pe.append(self.n_clk)
        
        mpp_cnt+=1
        map_streamclk=mpp_cnt
        
        self.used_axes+=[stream_dim]
        
        if keep_slice:
            map_arange=np.arange(len(self.used_axes)+1)
        else:
            map_arange=np.arange(len(self.used_axes))
            
        map_shape_data=np.copy(map_shape_pe)
            
        map_shape_data=np.delete(map_shape_data,map_streamdim)
        map_arange=np.delete(map_arange,map_streamdim)
        
        return map_shape_data,map_streamclk,map_shape_pe,map_streamdim,map_arange
    
    def permute_ravel_idx(self,index, source_shape, source_prior, target_shape, target_prior):
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
    
    def stream_capture_idx(self, index, 
                           data_shape, data_stream_axis,  
                           window_shape, window_stream_axis, 
                           data_flow_direction='forward', window_flow_direction='forward',
                           axis_arange=None, get_cond_idx=False):
        """ Convert index from an array to the capture of thream through a window (PE array).
            The captured shot is the clock cycle that fault index run through.
            
            window_shape must have 1 more dimension than data_shape, that is let data stream through window.
            The one more dimension is time dimension which represent the capture shot on that clock cycle.
            
        # Arguments
            index: Tuple or 2D ndarray. The index(coordinate) of source_shape which will be transform to target_shape index.
                2D ndarray (a,b) where a for list of coordinates, b for coordinate dimensions i.e. (16,4) there are 16 coordinates with 4 dimensions.
            data_shape: Tuple. The shape of data array being streamed in.
            data_stream_axis: Integer. The axis index whose dimension is the flow going.
            data_flow_direction: String. 'forward' or 'backward' the direction of data flow in. 
                Stream starts from the 0 index and increment, or else starts from last index and decrement.
            window_shape: Tuple. The shape of window sweep on data. The last dimention is the time dimension that stacks captures.
            window_stream_axis: String. The axis index whose dimention is the sweep going.
            window_flow_direction: String. 'forward' or 'backward' the direction of window sweeping. 
            axis_arange: List of Integer. How the data_shape axis aranged in window_shape i.e. [1,2,0] put data_shape axis 1,2,0 to window_shape axis 0,1,2 respectively.
            get_cond_idx: Bool. Return condition index or not.
            
        # Returns
            Converted coordinate. Single coordinate return in Tuple, multiple coordinate return in 2D ndarray.
        """
        if isinstance(index,tuple):
            index=np.reshape(np.array(index),[1,-1])
        elif isinstance(index,np.ndarray):
            pass
        else:
            raise TypeError('index for transformation must be either tuple or 2D numpy array.')
            
        if len(window_shape)-1!=len(data_shape):
            raise ValueError('window_shape must have 1 more dimension than data_shape, but got window_shape %s and data_shape %s'%(str(window_shape),str(data_shape)))
            
        if data_flow_direction=='forward':
            idx_capture_clk=np.expand_dims(index[:,data_stream_axis],1)
        elif data_flow_direction=='backward':
            idx_capture_clk=np.expand_dims(np.subtract(data_shape[data_stream_axis]-1,index[:,data_stream_axis]),1)    
        else:
            raise ValueError('data_flow_direction must be \'forward\' or \'backward\'.')
            
        idx_capture_clk=np.tile(idx_capture_clk,[1,window_shape[window_stream_axis]])
                
        if window_flow_direction=='forward':
            base_coor_shift=np.expand_dims(np.arange(window_shape[window_stream_axis]),0)
            base_coor_shift=np.tile(base_coor_shift,[len(index),1])
            idx_capture_clk=np.add(idx_capture_clk,base_coor_shift)
            
        elif window_flow_direction=='backward':
            base_coor_shift=np.expand_dims(np.flip(np.arange(window_shape[window_stream_axis])),0)
            base_coor_shift=np.tile(base_coor_shift,[len(index),1])
            idx_capture_clk=np.add(idx_capture_clk,np.flip(base_coor_shift,1))
        else:
            raise ValueError('window_flow_direction must be \'forward\' or \'backward\'.')        
        
        if axis_arange is None:
            axis_arange=list()
            for i in range(len(data_shape)):
                if i==window_stream_axis:
                    axis_arange.append(data_stream_axis)
                else:
                    if i==data_stream_axis:
                        axis_arange.append(i+1)
                    else:
                        axis_arange.append(i)
        
        caped_index=np.zeros([len(index)*window_shape[window_stream_axis],len(window_shape)],dtype=int)
        for i,ax in enumerate(axis_arange):
            if ax==data_stream_axis:
                caped_index[:,ax]=np.reshape(base_coor_shift,[1,-1])
            else: 
                caped_index[:,ax]=np.repeat(index[:,i],window_shape[window_stream_axis])
                
        caped_index[:,-1]=np.reshape(idx_capture_clk,[1,-1])
        
        if get_cond_idx:
            return caped_index, np.repeat(np.arange(len(index)),window_shape[window_stream_axis])
        else:
            return caped_index
        
    def broadcast_idx(self, index, data_shape, target_shape, broadcast_dims,
                      axis_arange=None, get_cond_idx=False):
        """ Broadcast certain indexes of an array to all element in a given dimension. 
            The dimension of data_shape should be smaller than target_shape, there need to be space for broadcast.
        
        # Arguments
            index: Tuple or 2D ndarray. The index(coordinate) of source_shape which will be transform to target_shape index.
                2D ndarray (a,b) where a for list of coordinates, b for coordinate dimensions i.e. (16,4) there are 16 coordinates with 4 dimensions.
            data_shape: Tuple. The shape of data array being broadcasted.
            target_shape: Tuple. The shape of data array broadcast to.
            broadcast_dims: Integer or List of Integer. The dimension indexes of target_shape that are being broadcast to.
            axis_arange: List of Integer. How the data_shape axis aranged in target_shape i.e. [1,2,0] put data_shape axis 1,2,0 to target_shape axis 0,1,2 respectively.
            get_cond_idx: Bool. Return condition index or not.
            
        # Returns
            Converted coordinate. Single coordinate return in Tuple, multiple coordinate return in 2D ndarray.
            
        """
        if not len(data_shape)<len(target_shape):
            raise ValueError('Dimension data_shape must be smaller than target_shape. There need to be space for broadcast.')
        if isinstance(broadcast_dims,int):
            if (len(data_shape)<len(target_shape)-1) and (axis_arange is None):
                raise AttributeError('For the condition index shape (data_shape) is more than two order smaller than target_shape, the argument axis_arange is mandatory elemet. Got data_shape %d dims and target_shape %d dims.'%(len(data_shape),len(target_shape)))
        elif isinstance(broadcast_dims,list):
            if (len(data_shape)<len(target_shape)-len(broadcast_dims)) and (axis_arange is None):
                raise AttributeError('For the condition index shape (data_shape) is more than %d order smaller than target_shape, the argument axis_arange is mandatory elemet. Got data_shape %d dims and target_shape %d dims.'%(len(broadcast_dims)+1,len(data_shape),len(target_shape)))
        else:
            raise TypeError('broadcast_dims must either be integer or list of integer.')
        
        if isinstance(index,tuple):
            index=np.reshape(np.array(index),[1,-1])
        elif isinstance(index,np.ndarray):
            if len(data_shape)==1:
                index=np.reshape(index,[-1,1])
        else:
            raise TypeError('index for transformation must be either tuple or 2D numpy array.')
        
        if isinstance(broadcast_dims,int):
            idx_broadcast=np.repeat(index,target_shape[broadcast_dims],0)
            idx_leaf=np.tile(np.arange(target_shape[broadcast_dims]),len(index))
            cond_idx=np.repeat(np.arange(len(index)),target_shape[broadcast_dims])
        
        else:   
            idx_leaf=list()
            for dims in broadcast_dims:
                idx_leaf.append(target_shape[dims])
            idx_broadcast=np.repeat(index,np.prod(idx_leaf),0)
            cond_idx=np.repeat(np.arange(len(index)),np.prod(idx_leaf))
            idx_leaf=np.array(list(np.ndindex(*idx_leaf)))
            idx_leaf=np.tile(idx_leaf,[len(index),1])
                      
        if axis_arange is None:
            if isinstance(broadcast_dims,int):
                axis_arange=list()
                for i in range(len(target_shape)):
                    if i != broadcast_dims:
                        axis_arange.append(i)
            else:
                axis_arange=list()
                for i in range(len(target_shape)):
                    if i not in broadcast_dims:
                        axis_arange.append(i)
        
        broaded_index=np.zeros([len(idx_broadcast),len(target_shape)],dtype=int)
        
        if isinstance(broadcast_dims,int):
            broaded_index[:,broadcast_dims]=idx_leaf
        else:
            for i,ax in enumerate(broadcast_dims):
                broaded_index[:,ax]=idx_leaf[:,i]
        
        for i,ax in enumerate(axis_arange):
            broaded_index[:,ax]=idx_broadcast[:,i]
            
        if get_cond_idx:
            return broaded_index, cond_idx
        else:
            return broaded_index
        
    def fixed_idx(self, index, indice_fix, fix_dims, target_shape, axis_arange=None):
        """ Make certain dimension of data index fix on certain axis.
            In this condition the data only exist on specific index of this dimension.
        
        # Arguments
            index: Tuple or 2D ndarray. The index(coordinate) of source_shape which will be transform to target_shape index.
                2D ndarray (a,b) where a for list of coordinates, b for coordinate dimensions i.e. (16,4) there are 16 coordinates with 4 dimensions.
            indice_fix: Integer or List of Integer. The indice of the targeted dimension that represent the location of fix data. If multiple dimensions are fixed indice_fix must align with fix_dims.
            fix_dims: Integer or List of Integer. The dimension indexes of target_shape that are being fix to.
            target_shape: Tuple. The shape of data array fix to.
            axis_arange: List of Integer. How the data_shape axis aranged in target_shape i.e. [1,2,0] put data_shape axis 1,2,0 to target_shape axis 0,1,2 respectively.
            
        # Returns
            Converted coordinate. Single coordinate return in Tuple, multiple coordinate return in 2D ndarray.

        """
        if isinstance(fix_dims,int):
            if index.shape[1]+1!=len(target_shape):
                raise AttributeError('target_shape must be one more dimension than index where the fix dimension expand to.')
        elif isinstance(fix_dims,list):
            if index.shape[1]+len(fix_dims)!=len(target_shape):
                raise AttributeError('target_shape must be %d more dimension than index where the fix dimension expand to.'%len(fix_dims))
        else:
            raise TypeError('fix_dims must either be integer or list of integer.')
            
        if isinstance(index,tuple):
            index=np.reshape(np.array(index),[1,-1])
        elif isinstance(index,np.ndarray):
            if len(index.shape)==1:
                index=np.reshape(index,[-1,1])
        else:
            raise TypeError('index for transformation must be either tuple or 2D numpy array.')
        
        if isinstance(fix_dims,int):
            if indice_fix<0:
                indice_fix=target_shape[fix_dims]+indice_fix
            fixidx=np.multiply(np.ones(len(index),dtype=int),indice_fix)
        else:
            fixidx=list()
            for i,dims in enumerate(fix_dims):
                if indice_fix[i]<0:
                    indice_fix[i]=target_shape[dims]+indice_fix[i]
                fixidx.append(np.multiply(np.ones(len(index),dtype=int),indice_fix[i]))
                
        if axis_arange is None:
            if isinstance(fix_dims,int):
                axis_arange=list()
                for i in range(len(target_shape)):
                    if i != fix_dims:
                        axis_arange.append(i)
            else:
                axis_arange=list()
                for i in range(len(target_shape)):
                    if i not in fix_dims:
                        axis_arange.append(i)

        fixed_index=np.zeros([len(index),len(target_shape)],dtype=int)
        
        if isinstance(fix_dims,int):
            fixed_index[:,fix_dims]=fixidx
        else:
            for i,ax in enumerate(fix_dims):
                fixed_index[:,ax]=fixidx[i]
        
        for i,ax in enumerate(axis_arange):
            fixed_index[:,ax]=index[:,i]

        return fixed_index
    
    def serialize_slices(self, fault_dict, mapping_shape, slice_n_clk=None, pack_size=1, t_clk_dims=-2, slice_dims=-1):
        """ Serialize slice dimension into t_clk dimension. Converge the slice order on PE dataflow model.
        
        """
        index=np.array(list(fault_dict.keys()))
        fault_value=list(fault_dict.values())
        
        if slice_n_clk is None:
            slice_n_clk=mapping_shape[t_clk_dims]
        
        slice_num=mapping_shape[slice_dims]
        
        if t_clk_dims<0:
            t_clk_dims=index.shape[1]+t_clk_dims
        if slice_dims<0:
            slice_dims=index.shape[1]+slice_dims
        
        PE_shape_idx=np.delete(index,[t_clk_dims,slice_dims],axis=1)
        clk_idx=index[:,t_clk_dims]
        slice_idx=index[:,slice_dims]
        
        mapping_shape=list(np.delete(mapping_shape,[t_clk_dims,slice_dims]))
        
        if pack_size>1:
            slice_rmd=np.remainder(slice_idx,pack_size)
            slice_idx=np.floor_divide(slice_idx,pack_size)
            clk_idx=np.add(np.multiply(slice_rmd,slice_n_clk),clk_idx)
            new_index=np.append(PE_shape_idx,np.reshape(clk_idx,[len(clk_idx),1]),1)
            new_index=np.append(new_index,np.reshape(slice_idx,[len(slice_idx),1]),1)
            
            mapping_shape.append(slice_n_clk*pack_size)
            mapping_shape.append(int(np.ceil(slice_num/pack_size)))
            
        else:
            clk_idx=np.add(np.multiply(slice_idx,slice_n_clk),clk_idx)
            new_index=np.append(PE_shape_idx,np.reshape(clk_idx,[len(clk_idx),1]),1)
            
            mapping_shape.append(slice_n_clk*slice_num)
        
        index_fd=list(zip(*new_index.T))
        new_fault_dict=dict(zip(index_fd,fault_value))
        return new_fault_dict,mapping_shape
    
    def insert_stalllatency(self, fault_dict, stalllatency, mapping_shape, t_clk_dims=-2):
        """ Insert stall and latency to fault dictionary t_clk axis.
        
        """
        index=np.array(list(fault_dict.keys()))
        fault_value=list(fault_dict.values())

        index[:,t_clk_dims]=np.add(index[:,t_clk_dims],stalllatency)
        mapping_shape[t_clk_dims]+=stalllatency
        
        index_fd=list(zip(*index.T))
        new_fault_dict=dict(zip(index_fd,fault_value))

        return new_fault_dict,mapping_shape
        
    def premapping_tile(self, parameter):
        """ Pre-mapping a tile onto PE array dataflow model. Need setup dataflow config in advance.
            All three parameter ofmap, weight, ifmap are setup with specific axis config.
            Each axis on PE array are assign with dataflow mode (one of 'permute', 'fixed', 'broadcast', 'streaming').
            The pre-mapping phase will tackle axes in following order. 'permute' -> 'fixed' -> 'broadcast' -> 'streaming'
        
        # Arguments
            parameter: String. The parameter being mapped to, must be 'ofmap', 'wght' or 'ifmap'.
            tile: Class. The tile_PE class for PE array fault tolerance analysis. The tile about to be mapped.
            flow: Class. The PEflow class for tile mapping on PE array. The flow describe how the tile are mapped.
        
        # Returns
            Converted fault dictionary. Keys are PE dataflow model coordinates. Items are fault info dictionarys.
        """
        if not self.setup_ready:
            raise AttributeError('The dataflow setup is not ready!')
        
        if parameter=='ofmap':
            tile=self.ofmap_tile
            flow=self.ofmap_flow
        elif parameter=='ifmap':
            tile=self.ifmap_tile
            flow=self.ifmap_flow
        elif parameter=='wght':
            tile=self.wght_tile
            flow=self.wght_flow
        else:
            raise ValueError('parameter should be one of \'ifmap\', \'wght\', \'ofmap\'.')

        
        if tile.tilting:
            tile_shape=tile.tilted_slice_shape
        else:
            if tile.expansion:
                tile_shape=tile.slice_shape
            else:
                tile_shape=tile.tile_shape+(1,)
                
        if tile.expansion:
            mapped_coors=np.array(list(tile.fault_dict_expand.keys()))
            fault_value=list(tile.fault_dict_expand.values())
        else:
            mapped_coors=np.array(list(tile.fault_dict.keys()))
            mapped_coors=np.append(mapped_coors,np.zeros([len(mapped_coors),1],dtype=int),axis=1)
            fault_value=list(tile.fault_dict.values())
            
        if parameter=='ofmap':
            PEparam={'param':'psum_out'}
        elif parameter=='ifmap':
            PEparam={'param':'ifmap_in'}
        elif parameter=='wght':
            PEparam={'param':'wght_in'}
        
        for value in fault_value:
            value.update(PEparam)
               
        self.used_axes=list()
        
        # permute
        if flow.permute_info is not None:
            flow.check_prior(tile_shape)
            
            map_shape_pe,map_prior_pe=self.get_PE_prior(flow.permute_info.PE_required_axes_prior, 
                                                        tile_shape, 
                                                        keep_slice=True)
    
            mapped_coors=self.permute_ravel_idx(mapped_coors,
                                                source_shape=tile_shape,
                                                source_prior=flow.permute_info.tile_mapping_prior,
                                                target_shape=map_shape_pe,
                                                target_prior=map_prior_pe)
        
        # fixed
        if flow.fixed_info is not None:
            flow.check_fix()
            map_fixdims,map_shape_pe,map_arange=self.get_fix_arange(flow.fixed_info.PE_fix_axis, 
                                                                    tile_shape, 
                                                                    keep_slice=True)
            
            mapped_coors=self.fixed_idx(mapped_coors, 
                                        indice_fix=flow.fixed_info.indice, 
                                        fix_dims=map_fixdims, 
                                        target_shape=map_shape_pe, 
                                        axis_arange=map_arange)
        
        # broadcast
        if flow.broadcast_info is not None:
            flow.check_broadcast()
            map_shape_data,map_shape_pe,map_broaddims,map_arange=self.get_broadcast_arange(flow.broadcast_info.PE_broadcast_axis, 
                                                                                           tile_shape, 
                                                                                           keep_slice=True)
            
            mapped_coors,cond_idx=self.broadcast_idx(mapped_coors, 
                                                     data_shape=map_shape_data,
                                                     target_shape=map_prior_pe, 
                                                     broadcast_dims=map_broaddims,
                                                     axis_arange=map_arange, 
                                                     get_cond_idx=True)
            
            fault_value=[fault_value[i] for i in cond_idx]
          
        # streaming
        if flow.streaming_info is not None:
            flow.check_streaming()
            map_shape_data,map_streamclk,map_shape_pe,map_streamdim,map_arange\
            =self.get_streaming_arange(flow.streaming_info.PE_stream_axis, 
                                       tile_shape, 
                                       keep_slice=True)
            
            mapped_coors,cond_idx=self.stream_capture_idx(mapped_coors, 
                                                          data_shape=map_shape_data, 
                                                          data_stream_axis=map_streamclk,
                                                          window_shape=map_shape_pe, 
                                                          window_stream_axis=map_streamdim, 
                                                          data_flow_direction=flow.streaming_info.tile_direction, 
                                                          window_flow_direction=flow.streaming_info.PE_direction,
                                                          axis_arange=map_arange, 
                                                          get_cond_idx=True)

            fault_value=[fault_value[i] for i in cond_idx]
            

        mapped_coors_fd=list(zip(*mapped_coors.T))
        new_fault_dict=dict(zip(mapped_coors_fd,fault_value))

        if parameter=='ofmap':
            self.ofmap_map_fd=new_fault_dict
            self.shape_ofmap_mapping=map_shape_pe
        elif parameter=='ifmap':
            self.ifmap_map_fd=new_fault_dict
            self.shape_ifmap_mapping=map_shape_pe
        elif parameter=='wght':
            self.wght_map_fd=new_fault_dict
            self.shape_wght_mapping=map_shape_pe
            
        return new_fault_dict
    
    def duplicate_mapping(self, parameter):
        """ Duplicate pre-mapped tile which is on PE array dataflow model. Need setup dataflow config in advance.
            There might need multiple iteration for a ofmap tile to complete it computation. 
            Maybe for calculating different channels or accumulate partial sum.
            Repeat means the times for pre-mapped tile repeat element wise on t_clk axis. For mapping clock cycle.
            Duplicate means the times for pre-mapped tile duplicate entirely on t_clk axis. For mapping clock cycle.
            
        
        # Arguments
            parameter: String. The parameter being mapped to, must be 'ofmap', 'wght' or 'ifmap'.
        
        # Returns
            Converted fault dictionary. Keys are PE dataflow model coordinates. Items are fault info dictionarys.
        """
        if not self.setup_ready:
            raise AttributeError('The dataflow setup is not ready!')
        
        if parameter=='ofmap':
            fault_dict=self.ofmap_map_fd
            flow=self.ofmap_flow
            cutset_num=self.shape_ofmap_mapping[-1]
        elif parameter=='ifmap':
            fault_dict=self.ifmap_map_fd
            flow=self.ifmap_flow
            cutset_num=self.shape_ifmap_mapping[-1]
        elif parameter=='wght':
            fault_dict=self.wght_map_fd
            flow=self.wght_flow
            cutset_num=self.shape_wght_mapping[-1]
        else:
            raise ValueError('parameter should be one of \'ifmap\', \'wght\', \'ofmap\'.')

        duped_coors=np.array(list(fault_dict.keys()))
        fault_value=np.array(list(fault_dict.values()))
        
        # repeat
        if flow.repeat>0:
            slices_mod=np.tile(np.arange(flow.repeat),len(duped_coors))
            
            duped_coors=np.repeat(duped_coors,flow.repeat,0)
            fault_value=np.repeat(fault_value,flow.repeat,0)
            
            slices_idx=duped_coors[:,-1]
            slices_idx=np.add(np.multiply(slices_idx,flow.repeat),slices_mod)
            
            duped_coors[:,-1]=slices_idx
            cutset_num*=flow.repeat

        # duplicate
        if flow.duplicate>0:
            slices_mod=np.repeat(np.arange(flow.duplicate),len(duped_coors))
            
            duped_coors=np.tile(duped_coors,[flow.duplicate,1])
            fault_value=np.tile(fault_value,flow.duplicate)
            
            slices_idx=duped_coors[:,-1]
            slices_idx=np.add(np.multiply(slices_mod,cutset_num),slices_idx)
            
            duped_coors[:,-1]=slices_idx
            cutset_num*=flow.duplicate
        
        duped_coors_fd=list(zip(*duped_coors.T))
        new_fault_dict=dict(zip(duped_coors_fd,fault_value))
        
        if parameter=='ofmap':
            self.ofmap_map_fd=new_fault_dict
            self.shape_ofmap_mapping[-1]=cutset_num
        elif parameter=='ifmap':
            self.ifmap_map_fd=new_fault_dict
            self.shape_ifmap_mapping[-1]=cutset_num
        elif parameter=='wght':
            self.wght_map_fd=new_fault_dict
            self.shape_wght_mapping[-1]=cutset_num
            
        return new_fault_dict

    def align_slice_pack(self):
        """ Align pre-mapped and duplicated fault dictionarys which is mapped on PE array dataflow model. Need setup dataflow config in advance.
            All the fault dictionary are in the correct location within slice. Forming slice pack to slign the timing of each slices to complete tile computation.
            Insert stall and latency for actual PE dataflow for each slice pack. Finally, combines all mapped tile fault dictionary onto PE dataflow model.
        
        # Arguments
            parameter: String. The parameter being mapped to, must be 'ofmap', 'wght' or 'ifmap'.
        
        # Returns
            Converted and combined fault dictionary. Keys are PE dataflow model coordinates. Items are fault info dictionarys.
        """
        if not self.setup_ready:
            raise AttributeError('The dataflow setup is not ready!')
        
        # form slice pack
        if self.ifmap_flow.pack_size>1:
            self.ifmap_map_fd,self.shape_ifmap_mapping=self.serialize_slices(self.ifmap_map_fd, 
                                                                             mapping_shape=self.shape_ifmap_mapping,
                                                                             pack_size=self.ifmap_flow.pack_size)

        if self.wght_flow.pack_size>1:
            self.wght_map_fd,self.shape_wght_mapping=self.serialize_slices(self.wght_map_fd, 
                                                                           mapping_shape=self.shape_wght_mapping,
                                                                           pack_size=self.wght_flow.pack_size)

        if self.ofmap_flow.pack_size>1:
            self.ofmap_map_fd,self.shape_ofmap_mapping=self.serialize_slices(self.ofmap_map_fd, 
                                                                             mapping_shape=self.shape_ofmap_mapping,
                                                                             pack_size=self.ofmap_flow.pack_size)            
            
        # insert stall & latency
        if self.ifmap_flow.stall_latency>0:
            self.ifmap_map_fd,self.shape_ifmap_mapping=self.insert_stalllatency(self.ifmap_map_fd, 
                                                                                self.ifmap_flow.stall_latency, 
                                                                                self.shape_ifmap_mapping)
            
        if self.wght_flow.stall_latency>0:
            self.wght_map_fd,self.shape_wght_mapping=self.insert_stalllatency(self.wght_map_fd, 
                                                                              self.wght_flow.stall_latency, 
                                                                              self.shape_wght_mapping)
            
        if self.ofmap_flow.stall_latency>0:
            self.ofmap_map_fd,self.shape_ofmap_mapping=self.insert_stalllatency(self.ofmap_map_fd, 
                                                                                self.ofmap_flow.stall_latency, 
                                                                                self.shape_ofmap_mapping)
            
        # align clock cycle
        slice_num=[self.shape_ifmap_mapping[-1], self.shape_ofmap_mapping[-1], self.shape_wght_mapping[-1]]
        if not slice_num[1:] == slice_num[:-1]:
            print('WARNING: The number of slices of ifmap, ofmap, weight should be the same but got %s'%str(slice_num))
            
        pack_clk=max(self.shape_ofmap_mapping[-2],self.shape_wght_mapping[-2],self.shape_ifmap_mapping[-2])
        
        self.n_clk=pack_clk*max(slice_num)
        
        self.fault_dict.update(self.serialize_slices(self.ofmap_map_fd,self.shape_ifmap_mapping,slice_n_clk=pack_clk)[0])
        self.fault_dict.update(self.serialize_slices(self.wght_map_fd,self.shape_wght_mapping,slice_n_clk=pack_clk)[0])
        self.fault_dict.update(self.serialize_slices(self.ifmap_map_fd,self.shape_ifmap_mapping,slice_n_clk=pack_clk)[0])
        self.fault_num=len(self.fault_dict)
        
        return self.fault_dict
    
    def clear(self):
        """ Clear fault dictionary of PE dataflow model """
        self.fault_num=None
        self.fault_dict=dict()
        
    def clear_map_config(self):
        """ Clear mapping setup of PE dataflow model """
        self.setup_ready=False
        self.ofmap_flow=None
        self.wght_flow=None
        self.ifmap_flow=None
        self.used_axes=list()
        self.tmp_clk=None

        
# TODO
# gen fault list
# how the data flow        
# form fault information
# describe psum & bias
# 
