# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 16:22:01 2019

@author: Yung-Yu Tsai

DNN tiling for computation unit fault mapping
"""

import numpy as np
import tqdm as tqdm

from simulator.memory.tile import tile,tile_FC

class tile_PE(tile):
    def __init__(self, tile_shape, is_fmap, **kwargs):
        """The tile of a DNN feature map or weights

        # Arguments
            tile_shape: Tuple. The shape of tile.
            Tm: Integer. The size of tile on the input feature map dimension (weight) or channel dimention (feature map).
            Tn: Integer. The size of tile on the output feature map dimension (weight) or batch dimention (feature map).
            Tr: Integer. The size of tile on the kernel row dimension or feature map row dimention.
            Tc: Integer. The size of tile on the kernel column dimension or feature map column dimention.
            is_fmap: Bool. The tile is feature map tile or weight tile.
    
        """
        super(tile_PE, self).__init__(tile_shape, is_fmap, **kwargs)
        self.tile_shape=tile_shape
                
        self.expansion=False
        self.expand_method=None
        self.expand_shape=None
        self.expand_prior_orig=None
        self.expand_prior_targ=None
        self.slice_shape=None
        self.slicing_dims=None
        self.slices_permute=None
        self.slices_cutset=None
        self.tilting=False
        self.tilted_slice_shape=None
        self.reshape_patches=False
        self.fault_dict_rehsaped=dict()
        self.fault_dict_expand=dict()
        
        if is_fmap:
            self.psum_fault_dict=dict()
            self.psum_fault_dict_expand=dict()
        else:
            self.bias_fault_dict=dict()
            self.bias_fault_dict_expand=dict()
                                    
    def conv_output_length(self, input_length, filter_size, padding, stride, dilation=1, edge_fill=False):
        """Determines output length of a convolution given input length.
    
        # Arguments
            input_length: integer.
            filter_size: integer.
            padding: one of `"same"`, `"valid"`, `"full"`.
            stride: integer.
            dilation: dilation rate, integer.
            edge_fill: Bool. Fill the edge(right, down) kernel exceed part with zero and count as ofmap pixel.

        # Returns
            The output length (integer).
        """
        if input_length is None:
            return None
        assert padding in {'same', 'valid', 'full', 'causal'}
        dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
        if padding == 'same':
            output_length = input_length
        elif padding == 'valid':
            output_length = input_length - dilated_filter_size + 1
        elif padding == 'causal':
            output_length = input_length
        elif padding == 'full':
            output_length = input_length + dilated_filter_size - 1
            
        if edge_fill:
            return int(np.ceil((output_length + stride - 1) / stride))
        else:
            return (output_length + stride - 1) // stride

    def reshape_ravel_idx(self,index,source_shape,source_prior,target_shape,target_prior):
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
            
        source_prior=np.argsort(np.array(source_prior))[::-1]
        target_prior=np.argsort(np.array(target_prior))[::-1]
            
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
            
    def get_slices_cutset(self, orig_shape, slicing_dims):
        div_dims=np.array([],dtype=int)
        for dim in slicing_dims:
            if dim==0:
                div_dims=np.append(div_dims,1)
            else:
                div_dims=np.append(div_dims,dim)
                
        orig_shape=np.array(orig_shape)
        cutset=np.ceil(np.divide(orig_shape,div_dims))
        cutset=cutset.astype(int)
        return cutset,div_dims
            
    def slice_permute_idx(self, index, orig_shape, slicing_dims, slices_permute):
        """ Index transformation for slice array and permute it into new axis. For slice within tile of PE mapping flow.
        
        # Arguments
        index: Tuple or 2D ndarray. The index(coordinate) of source_shape which will be transform to target_shape index.
                2D ndarray (a,b) where a for list of coordinates, b for coordinate dimensions i.e. (16,4) there are 16 coordinates with 4 dimensions.

        orig_shape: Tuple. The shape of orignal array were going to be sliced.
        
        slicing_dims: Tuple. Indicates the dimension of slices in the expect_shape. A tuple must be the same size as 
                expect_shape, tuple (0,0,3,3) means the expect_shape dimension 2,3 are part of slice dimensions. The 
                0,1 dimension parts are for time multiplexed permute so the dimension size is 0.
                
        slices_permute: List or Tuple of Integer.. Indicates how to permute the time multiplexed part of expect_shape. Tuple (a,b,c,d) means
                the order of time multiplexed part dimension to permute. Variable a,b,c,d are the axis index of expect_shape.
                
        # Returns
            Converted coordinate. Single coordinate return in Tuple, multiple coordinate return in 2D ndarray.
        """        
        if isinstance(index,tuple):
            index=np.array(index)
        elif isinstance(index,np.ndarray):
            pass
        else:
            raise TypeError('index for transformation must be either tuple or 2D numpy array.')

        cutset,div_dims=self.get_slices_cutset(orig_shape, slicing_dims)
                
        slices_permute=np.argsort(np.array(slices_permute))[::-1]
        
        permute_dims=np.floor_divide(index,div_dims)
        mapping_dims=np.remainder(index,div_dims)
        if len(index.shape)==1:
            mapping_dims=mapping_dims[np.argwhere(np.array(slicing_dims)>0)]
            tclk=np.ravel_multi_index(permute_dims[slices_permute],cutset[slices_permute])
            if len(mapping_dims.shape)==1:
                mapping_dims=np.expand_dims(mapping_dims,-1)
            mapping_dims=np.append(mapping_dims,tclk)
        else:
            mapping_dims=mapping_dims[:,np.squeeze(np.argwhere(np.array(slicing_dims)>0))]
            tclk=np.ravel_multi_index(permute_dims.T[slices_permute],cutset[slices_permute])
            if len(mapping_dims.shape)==1:
                mapping_dims=np.expand_dims(mapping_dims,-1)
            mapping_dims=np.append(mapping_dims,np.expand_dims(tclk,-1),axis=-1)
            
        return mapping_dims

    def assemble_slice_idx(self, index, orig_shape, slicing_dims, slices_permute):
        """ Index transformation for assemble sliced array and remove the slice axis. For reverse slice within tile of PE mapping flow.
        
        # Arguments
        index: Tuple or 2D ndarray. The index(coordinate) of source_shape which will be transform to target_shape index.
                2D ndarray (a,b) where a for list of coordinates, b for coordinate dimensions i.e. (16,4) there are 16 coordinates with 4 dimensions.

        orig_shape: Tuple. The shape of orignal array were going to be sliced.
        
        slicing_dims: Tuple. Indicates the dimension of slices in the expect_shape. A tuple must be the same size as 
                expect_shape, tuple (0,0,3,3) means the expect_shape dimension 2,3 are part of slice dimensions. The 
                0,1 dimension parts are for time multiplexed permute so the dimension size is 0.
                
        slices_permute: List or Tuple of Integer.. Indicates how to permute the time multiplexed part of expect_shape. Tuple (a,b,c,d) means
                the order of time multiplexed part dimension to permute. Variable a,b,c,d are the axis index of expect_shape.
                
        # Returns
            Converted coordinate. Single coordinate return in Tuple, multiple coordinate return in 2D ndarray.
        """        
        if isinstance(index,tuple):
            index=np.array(index)
        elif isinstance(index,np.ndarray):
            pass
        else:
            raise TypeError('index for transformation must be either tuple or 2D numpy array.')

        cutset,div_dims=self.get_slices_cutset(orig_shape, slicing_dims)
        
        slices_permute=np.argsort(np.array(slices_permute))[::-1]
        
        restore_index=np.zeros((len(slices_permute),),dtype=int)
        for i in range(len(slices_permute)):
            restore_index[slices_permute[i]]=i
        
        if len(index.shape)==1:
            tclk=index[-1]
            mapping_dims=index[:-1]
            permute_dims=np.unravel_index(tclk,cutset[slices_permute])
            permute_dims=np.array(permute_dims)[restore_index]
            assembled_idx=np.zeros(len(orig_shape),dtype=int)
            assembled_idx[np.squeeze(np.argwhere(np.array(slicing_dims)>0))]=mapping_dims
        else:
            tclk=index[:,-1]
            mapping_dims=index[:,:-1]
            if len(mapping_dims.shape)==1:
                mapping_dims=np.expand_dims(mapping_dims,-1)
            permute_dims=np.unravel_index(tclk,cutset[slices_permute])
            permute_dims=np.array(permute_dims)[restore_index]
            permute_dims=permute_dims.T
            assembled_idx=np.zeros([len(index),len(orig_shape)],dtype=int)
            assembled_idx[:,np.squeeze(np.argwhere(np.array(slicing_dims)>0))]=mapping_dims
            
        permute_dims=np.multiply(permute_dims,div_dims)
        assembled_idx=np.add(assembled_idx,permute_dims)
            
        return assembled_idx
    
    def get_extracted_shape(self, fmap_shape, ksizes, strides=(1,1,1,1), dilation_rates=(1,1,1,1), padding='valid', edge_fill=False):
        new_dim_row = self.conv_output_length(
            fmap_shape[1],
            ksizes[1],
            padding=padding,
            stride=strides[1],
            dilation=dilation_rates[1],
            edge_fill=edge_fill)
        
        new_dim_col = self.conv_output_length(
            fmap_shape[2],
            ksizes[2],
            padding=padding,
            stride=strides[2],
            dilation=dilation_rates[2],
            edge_fill=edge_fill)
        
        extracted_shape=(fmap_shape[0], new_dim_row, new_dim_col, fmap_shape[3]*ksizes[1]*ksizes[2])
        
        return extracted_shape
    
    def extract_patches_idx(self, index, fmap_shape, ksizes, strides=(1,1,1,1), dilation_rates=(1,1,1,1), padding='valid', edge_fill=False, patches_unravel=[0,1,2], get_cond_idx=False):
        """ Index transformation for tf.extract_image_patches for ifmap expansion. 
            This was used for convert 4D ifmap tile to ofmap column and row with all the partial product input fmap.
            [batch, row, column, # of kernel 2D * # of ifmap channel]
            
        # Arguments
            index: Tuple or 2D ndarray. The index(coordinate) of source_shape which will be transform to target_shape index.
                2D ndarray (a,b) where a for list of coordinates, b for coordinate dimensions i.e. (16,4) there are 16 coordinates with 4 dimensions.
            fmap_shape: Tuple. The shape of orignal fmap were going to be extracted.
            ksize: List of Integer. Length >= 4. The size of the sliding window for each dimension of images. [1, row, col, 1]
            strides: List of Integer. Length >= 4. How far the centers of two consecutive patches are in the images. [1, stride_rows, stride_cols, 1]
            dilation_rates: List of Integer. Length >= 4. Must be: [1, rate_rows, rate_cols, 1]. This is the input stride, 
                specifying how far two consecutive patch samples are in the input.
            padding: String. 'same' or 'valid'. The type of padding algorithm to use.
            edge_fill: Bool. When the kernel window partially exceed the edge(right, bottom) of feature map, whether to fill 
                the exceeded area with zero and count as an ofmap pixel or not.
            patches_unravel: List of Integer. The order of [row, col, channel] unravel into 1 dimmension default [0,1,2]. 
                [row, col, channel] are the needed data to accumulate output a pixel.
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
                    
        dilated_ksize_row = ksizes[1] + (ksizes[1]-1) * (dilation_rates[1] - 1)
        dilated_ksize_col = ksizes[2] + (ksizes[2]-1) * (dilation_rates[2] - 1)
                
        idx_patches_candidate=np.expand_dims(index[:,1:3],1)
        idx_patches_candidate=np.tile(idx_patches_candidate,[1,ksizes[1]*ksizes[2],1])
        
        base_kcoor_reduction=list(np.ndindex(ksizes[1:3]))
        base_kcoor_reduction.reverse()
        base_kcoor_reduction=np.multiply(base_kcoor_reduction,dilation_rates[1:3])

        idx_patches_candidate=np.subtract(idx_patches_candidate,np.expand_dims(base_kcoor_reduction,0))
        
        # condition strides > 1
        if strides[1]>1 or strides[2]>1:
            if padding=='valid':
                cond_arg=idx_patches_candidate[:,:,0]%strides[1]==0 #row
                cond_tmp=idx_patches_candidate[:,:,1]%strides[2]==0 #col
            elif padding=='same':
                cond_arg=np.subtract(idx_patches_candidate[:,:,0],np.floor_divide(dilated_ksize_row,2))%strides[1]==0 #row
                cond_tmp=np.subtract(idx_patches_candidate[:,:,1],np.floor_divide(dilated_ksize_col,2))%strides[2]==0 #col
                
            cond_arg=np.bitwise_and(cond_arg,cond_tmp) 
                
        else: # stride = 1
            cond_arg=np.ones(idx_patches_candidate.shape[0:-1],dtype=bool)
            
        # condition edge
        if padding=='valid':
            cond_tmp=idx_patches_candidate[:,:,0]>=0 # left edge
            cond_arg=np.bitwise_and(cond_arg,cond_tmp) 
            
            cond_tmp=idx_patches_candidate[:,:,1]>=0 # up edge
            cond_arg=np.bitwise_and(cond_arg,cond_tmp) 
            
            if not edge_fill:
                cond_tmp=idx_patches_candidate[:,:,0]<(fmap_shape[1]-(dilated_ksize_row-1)) # right edge
                cond_arg=np.bitwise_and(cond_arg,cond_tmp) 
                
                cond_tmp=idx_patches_candidate[:,:,1]<(fmap_shape[2]-(dilated_ksize_col-1)) # down edge
                cond_arg=np.bitwise_and(cond_arg,cond_tmp) 
                
        elif padding=='same':
            cond_tmp=idx_patches_candidate[:,:,0]>=0-(np.floor_divide(dilated_ksize_row,2)) # left edge
            cond_arg=np.bitwise_and(cond_arg,cond_tmp) 
            
            cond_tmp=idx_patches_candidate[:,:,1]>=0-(np.floor_divide(dilated_ksize_col,2)) # up edge
            cond_arg=np.bitwise_and(cond_arg,cond_tmp) 
            
            if not edge_fill:
                cond_tmp=idx_patches_candidate[:,:,0]<(fmap_shape[1]-(np.floor_divide(dilated_ksize_row,2))) # right edge
                cond_arg=np.bitwise_and(cond_arg,cond_tmp) 
                
                cond_tmp=idx_patches_candidate[:,:,1]<(fmap_shape[2]-(np.floor_divide(dilated_ksize_col,2))) # down edge
                cond_arg=np.bitwise_and(cond_arg,cond_tmp) 
                
        else:
            raise ValueError('padding must be either \'valid\' or \'same\'.')
            
        extracted_2D=idx_patches_candidate[cond_arg]
        if padding=='same':
            extracted_2D=np.add(extracted_2D,[[np.floor_divide(dilated_ksize_row,2),np.floor_divide(dilated_ksize_col,2)]])
        if strides[1]>1 or strides[2]>1:
            extracted_2D=np.floor_divide(extracted_2D,[strides[1:3]])
        cond_idx=np.argwhere(cond_arg)
        
        extracted_batch=index[:,0][cond_idx[:,0]]
        extracted_batch=np.expand_dims(extracted_batch,-1)
        
        extracted_psum=index[:,-1][cond_idx[:,0]]
        extracted_psum=np.expand_dims(extracted_psum,0)
        
        patches_rcm=np.unravel_index(np.subtract(ksizes[1]*ksizes[2]-1,cond_idx[:,1]),ksizes[1:3])
        patches_rcm=np.concatenate([patches_rcm,extracted_psum],axis=0)
        
        patches_unravel=np.argsort(np.array(patches_unravel))[::-1]
        extracted_psum=np.ravel_multi_index(patches_rcm[patches_unravel],np.array([ksizes[1],ksizes[2],fmap_shape[3]])[patches_unravel])
        extracted_psum=np.expand_dims(extracted_psum,-1)
        #extracted_psum=np.add(np.subtract(np.multiply(extracted_psum,ksizes[1]*ksizes[2]),cond_idx[:,1]),ksizes[1]*ksizes[2]-1) # rework
        
        extracted_index=np.concatenate([extracted_batch,extracted_2D,extracted_psum],axis=1)

        if get_cond_idx:
            return extracted_index, cond_idx
        else:
            return extracted_index
    
    def return_patches_idx(self, index, fmap_shape, ksizes, strides=(1,1,1,1), dilation_rates=(1,1,1,1), padding='valid', edge_fill=False, patches_unravel=[0,1,2]):
        """ Index transformation for reverse tf.extract_image_patches for ifmap shrink in PE to Tile mapping. 
            This was used for convert ofmap column and row with all the partial product input fmap to 4D ifmap tile.
            [batch, row, column, # of kernel 2D * # of ifmap channel] -> [batch, row, column, in channel]
            
        # Arguments
            index: Tuple or 2D ndarray. The index(coordinate) of source_shape which will be transform to target_shape index.
                2D ndarray (a,b) where a for list of coordinates, b for coordinate dimensions i.e. (16,4) there are 16 coordinates with 4 dimensions.
            fmap_shape: Tuple. The shape of orignal fmap were going to be extracted.
            ksize: List of Integer. Length >= 4. The size of the sliding window for each dimension of images. [1, row, col, 1]
            strides: List of Integer. Length >= 4. How far the centers of two consecutive patches are in the images. [1, stride_rows, stride_cols, 1]
            dilation_rates: List of Integer. Length >= 4. Must be: [1, rate_rows, rate_cols, 1]. This is the input stride, 
                specifying how far two consecutive patch samples are in the input.
            padding: String. 'same' or 'valid'. The type of padding algorithm to use.
            edge_fill: Bool. When the kernel window partially exceed the edge(right, bottom) of feature map, whether to fill 
                the exceeded area with zero and count as an ofmap pixel or not.
            patches_unravel: List of Integer. The order of [row, col, channel] unravel into 1 dimmension default [0,1,2]. 
                [row, col, channel] are the needed data to accumulate output a pixel.
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
                    
        dilated_ksize_row = ksizes[1] + (ksizes[1]-1) * (dilation_rates[1] - 1)
        dilated_ksize_col = ksizes[2] + (ksizes[2]-1) * (dilation_rates[2] - 1)
        
        batch_idx=index[:,0]
        returned_2D=index[:,1:3]
        extracted_psum=index[:,-1]
        
        patches_unravel=np.argsort(np.array(patches_unravel))[::-1]
        restore_index=np.zeros((3,),dtype=int)
        for i in range(3):
            restore_index[patches_unravel[i]]=i
            
        extracted_psum=np.unravel_index(extracted_psum,np.array([ksizes[1],ksizes[2],fmap_shape[3]])[patches_unravel])
        extracted_psum=np.array(extracted_psum)[restore_index]
        
        idx_patches_candidate=extracted_psum[0:2]
        idx_patches_candidate=np.ravel_multi_index(idx_patches_candidate,ksizes[1:3])
#        idx_patches_candidate=np.remainder(extracted_psum,ksizes[1]*ksizes[2])

        ifchannel_idx=extracted_psum[-1]
#        ifchannel_idx=np.floor_divide(extracted_psum,ksizes[1]*ksizes[2]) # rework
        
        if strides[1]>1 or strides[2]>1:
            returned_2D=np.multiply(returned_2D,[strides[1:3]])
        if padding=='same':
            returned_2D=np.subtract(returned_2D,[[np.floor_divide(dilated_ksize_row,2),np.floor_divide(dilated_ksize_col,2)]])

        base_kcoor_reduction=list(np.ndindex(ksizes[1:3]))
        base_kcoor_reduction=np.multiply(base_kcoor_reduction,dilation_rates[1:3])

        returned_2D=np.add(returned_2D,base_kcoor_reduction[idx_patches_candidate])
        
        batch_idx=np.expand_dims(batch_idx,-1)
        ifchannel_idx=np.expand_dims(ifchannel_idx,-1)
        returned_index=np.concatenate([batch_idx,returned_2D,ifchannel_idx],axis=1)

        return returned_index
        
    def tilt_idx(self, index, axis, direction, shape, shift=1):
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
            Shape of tilted index array. Tuple.
        """
        new_index=np.copy(index)
        if shift<0:
            new_index[:,direction]+=np.subtract(shape[axis]-1, new_index[:,axis])*(-shift)
            new_shape=list(shape)
            new_shape[direction]+=new_shape[axis]*(-shift)
        else:
            new_index[:,direction]+=new_index[:,axis]*shift
            new_shape=list(shape)
            new_shape[direction]+=(new_shape[axis]-1)*shift
        return new_index,tuple(new_shape)
    
    def untilt_idx(self, index, axis, direction, shape, shift=1):
        """ Undo tilting operation for systolic array input index 
        
        # Arguments
            index: Tuple or 2D ndarray. The index(coordinate) of source_shape which will be transform to target_shape index.
                2D ndarray (a,b) where a for list of coordinates, b for coordinate dimensions i.e. (16,4) there are 16 coordinates with 4 dimensions.
            shape: Tuple of Integer. The shape of data which the index represents. Needed argument for negative shift.
            axis: Integer. The axis wanted to be untilted.
            direction: Integer. The axis of direction that were tilted to.
            shift: Integer. The amount of shifting for tilted representation. Positive number for tilt forward, negative for tilted backward.
        
        # Returns
            Converted coordinate. Single coordinate return in Tuple, multiple coordinate return in 2D ndarray.        
            Shape of untilted index array. Tuple.
        """
        new_index=np.copy(index)
        if shift<0:
            new_index[:,direction]-=np.subtract(shape[axis]-1, new_index[:,axis])*(-shift)
            new_shape=list(shape)
            new_shape[direction]-=new_shape[axis]*(-shift)
        else:
            new_index[:,direction]-=new_index[:,axis]*shift
            new_shape=list(shape)
            new_shape[direction]-=(new_shape[axis]-1)*shift
        return new_index,tuple(new_shape)

    def expand_reshape_data(self, orig_prior, expect_shape, reshape_prior, slicing_dims, slices_permute,
                            tilting=False, tilt_axis=None, tilt_direction=None, tilt_shift=1,
                            dataflow_pre_plan=False):
        """ Data expansion before put into PE array. 
            The data may be cut into many pieces then fit into PE. Different slices calculate in different clock cycle.
            Method 'reshape' means change data shape without any duplication in array.
        
        # Arguments              
            orig_prior: List or Tuple of Integer or String. The list for unravel priority of tile dimensions. 
                The integer list is the dimension index. The string list is consist of 'Tm', 'Tn', 'Tr', 'Tc'.
                
            expect_shape: Tuple. The expected shape to be expand into.
            
            reshape_prior: List or Tuple of Integer. The list for ravel priority of expect_shape dimensions. The list is the dimension index. 
            
            slicing_dims: Tuple. Indicates the dimension of slices in the expect_shape. A tuple must be the same size as 
                expect_shape, tuple (0,0,3,3) means the expect_shape dimension 2,3 are part of slice dimensions. The 
                0,1 dimension parts are for time multiplexed permute so the dimension size is 0.
                
            slices_permute: Tuple. Indicates how to permute the time multiplexed part of expect_shape. Tuple (a,b,c,d) means
                the order of time multiplexed part dimension to permute. Variable a,b,c,d are the axis index of expect_shape.
            
            tilting: Bool. Tilt the index or not. For PE systolic array input.
            tilt_axis: Integer. The axis wanted to be tilted.
            tilt_direction: Integer. The axis of direaction that are tilted to.
            tilt_shift: Integer. The amount of shifting for tilted representation. Positive number for tilt forward, negative for tilted backward.
            
            dataflow_pre_plan: Bool. Plan the dataflow model ahead. If True there will be no actual Tile to PEarray fault dictionary list transformation.
                Only save the expansion configuration for later PEarray to Tile transform.
                
        # Returns
            Converted fault dictionary.

        """
        self.expansion=True
        self.expand_method='reshape'
        
        if len(orig_prior)!=len(self.tile_shape):
            raise ValueError('orig_prior must be in length %d, but get length %d.'%(len(self.tile_shape),len(orig_prior)))
        if isinstance(orig_prior[0],int):
            self.expand_prior_orig=orig_prior
        elif isinstance(orig_prior[0],str):
            for i in range(len(self.tile_shape)):
                if orig_prior[i] not in self.prior_element:
                    raise ValueError('The augment orig_prior must be in list %s'%(str(self.prior_element)))
            self.expand_prior_orig=self.get_tile_dims_prior(orig_prior)
        
        self.expand_shape=expect_shape
        
        if len(reshape_prior)!=len(self.expand_shape):
            raise TypeError('reshape_prior must be in length %d, but get length %d'%(len(self.expand_shape),len(reshape_prior)))
        self.expand_prior_targ=reshape_prior
        
        if len(slicing_dims)!=len(self.expand_shape):
            raise TypeError('slicing_dims must be in length %d, but get length %d'%(len(self.expand_shape),len(slicing_dims)))
        self.slicing_dims=slicing_dims
        self.slice_shape=np.array(slicing_dims)
        self.slice_shape=tuple(self.slice_shape[self.slice_shape>0])
        self.slices_cutset,_=self.get_slices_cutset(self.expand_shape, self.slicing_dims)
        self.slice_shape=self.slice_shape+(np.prod(self.slices_cutset),)
        
        if len(slices_permute)!=len(self.expand_shape):
            raise TypeError('slices_permute must be in length %d, but get length %d'%(len(self.expand_shape),len(slices_permute)))
        self.slices_permute=slices_permute
        
        if not dataflow_pre_plan:
            orig_coors=np.array(list(self.fault_dict.keys()))
            reshaped_coors=self.reshape_ravel_idx(orig_coors,
                                                  source_shape=self.tile_shape,
                                                  source_prior=self.expand_prior_orig,
                                                  target_shape=self.expand_shape,
                                                  target_prior=self.expand_prior_targ)
            
            reshaped_coors_fd=list(zip(*reshaped_coors.T))
            fault_info=list(self.fault_dict.values())
            self.fault_dict_rehsaped=dict(zip(reshaped_coors_fd,fault_info))
        
            permuted_coors=self.slice_permute_idx(reshaped_coors,
                                                  orig_shape=self.expand_shape,
                                                  slicing_dims=self.slicing_dims,
                                                  slices_permute=self.slices_permute)
        
        if tilting:
            self.tilting=True
            self.tilt_axis=tilt_axis
            self.tilt_direction=tilt_direction
            self.tilt_shift=tilt_shift
            
            if not dataflow_pre_plan:
                permuted_coors,self.tilted_slice_shape=self.tilt_idx(permuted_coors,
                                                                     self.tilt_axis,
                                                                     self.tilt_direction,
                                                                     self.slice_shape,
                                                                     self.tilt_shift)
            else:
                _,self.tilted_slice_shape=self.tilt_idx(np.zeros([1,len(self.slice_shape)],dtype=int),
                                                        self.tilt_axis,
                                                        self.tilt_direction,
                                                        self.slice_shape,
                                                        self.tilt_shift)
                    
        if not dataflow_pre_plan:
            permuted_coor_fd=list(zip(*permuted_coors.T))
            self.fault_dict_expand=dict(zip(permuted_coor_fd,fault_info))
        
            return self.fault_dict_expand
        
        else:
            return None
        
    def shrink_reshape_data(self,psum=False):
        """ Reverse data expansion of PE array dataflow model. 
            Re-assemble the cut tile slices.
            Method 'reshape' means change data shape without any reduction in array.
        
        # Arguments              
            psum: Bool. Indicate the transformation is for partial sum or not.
                
        # Returns
            Converted fault dictionary.

        """
        if not psum:
            if len(self.fault_dict_expand)==0:
                return dict()
            permuted_coors=np.array(list(self.fault_dict_expand.keys()))
            fault_info=np.array(list(self.fault_dict_expand.values()))
        else:
            if len(self.psum_fault_dict_expand)==0:
                return dict()
            permuted_coors=np.array(list(self.psum_fault_dict_expand.keys()))
            fault_info=np.array(list(self.psum_fault_dict_expand.values()))
        
        if self.tilting:            
            permuted_coors,_=self.untilt_idx(permuted_coors,
                                             self.tilt_axis,
                                             self.tilt_direction,
                                             self.tilted_slice_shape,
                                             self.tilt_shift)
            # pop inalid t_clk
            permuted_coors,cond_idx=self.pop_outlier_idx(permuted_coors,self.slice_shape,get_cond_idx=True)
            fault_info=fault_info[cond_idx]
        
        reshaped_coors=self.assemble_slice_idx(permuted_coors,
                                               orig_shape=self.expand_shape,
                                               slicing_dims=self.slicing_dims,
                                               slices_permute=self.slices_permute)
        # pop under-utilized area on PE
        reshaped_coors,cond_idx=self.pop_outlier_idx(reshaped_coors,self.expand_shape,get_cond_idx=True)
        fault_info=fault_info[cond_idx]
        
        if not psum:
            reshaped_coors_fd=list(zip(*reshaped_coors.T))
            self.fault_dict_rehsaped=dict(zip(reshaped_coors_fd,fault_info))

        orig_coors=self.reshape_ravel_idx(reshaped_coors,
                                          source_shape=self.expand_shape,
                                          source_prior=self.expand_prior_targ,
                                          target_shape=self.tile_shape,
                                          target_prior=self.expand_prior_orig)
                    
        orig_coor_fd=list(zip(*orig_coors.T))
        new_fault_dict=dict(zip(orig_coor_fd,fault_info))
        if not psum:
            self.fault_dict=new_fault_dict
        else:
            self.psum_fault_dict=new_fault_dict
    
        return new_fault_dict
            
    def expand_extract_patches(self, ksizes, strides=(1,1,1,1), dilation_rates=(1,1,1,1), padding='valid', edge_fill=False, patches_unravel=[0,1,2],
            reshape_patches=False, patches_prior=None, expect_shape=None, reshape_prior=None, slicing_dims=None, slices_permute=None,
            tilting=False, tilt_axis=None, tilt_direction=None, tilt_shift=1,
            dataflow_pre_plan=False):
        """ Data expansion before put into PE array. Usually used for ifmap reuse. 
            The data may be cut into many pieces than fit into PE. Different slices calculate in different clock cycle.
            Method 'extract patches' means extract feature patches to output feature maps corresponding kernel multiply.
            This method contains data duplication.
            reference: https://www.tensorflow.org/api_docs/python/tf/image/extract_patches
        
        # Arguments     
            ksize: List of Integer. Length >= 4. The size of the sliding window for each dimension of images. [1, row, col, 1]
            
            strides: List of Integer. Length >= 4. How far the centers of two consecutive patches are in the images. [1, stride_rows, stride_cols, 1]
            
            dilation_rates: List of Integer. Length >= 4. Must be: [1, rate_rows, rate_cols, 1]. This is the input stride, 
                specifying how far two consecutive patch samples are in the input.
                
            padding: String. 'same' or 'valid'. The type of padding algorithm to use.
            
            edge_fill: Bool. When the kernel window partially exceed the edge(right, bottom) of feature map, whether to fill 
                the exceeded area with zero and count as an ofmap pixel or not.
                
            patches_unravel: List of Integer. The order of [row, col, channel] unravel into 1 dimmension default [0,1,2]. 
                [row, col, channel] are the needed data to accumulate output a pixel.
         
            reshape_patches: Bool. Reshape the result of extract patches or not.
            
            patches_prior: List or Tuple of Integer or String. The list for unravel priority of patches dimensions. 
                The integer list is the dimension index. The string list is consist of 'Tm', 'Tn', 'Tr', 'Tc'.
                
            expect_shape: Tuple. The expected shape to be expand into.
            
            reshape_prior: List or Tuple of Integer. The list for ravel priority of expect_shape dimensions. The list is the dimension index. 
            
            slicing_dims: Tuple. Indicates the dimension of slices in the expect_shape. A tuple must be the same size as 
                expect_shape, tuple (0,0,3,3) means the expect_shape dimension 2,3 are part of slice dimensions. The 
                0,1 dimension parts are for time multiplexed permute so the dimension size is 0.
                
            slices_permute: Tuple. Indicates how to permute the time multiplexed part of expect_shape. Tuple (a,b,c,d) means
                the order of time multiplexed part dimension to permute. Variable a,b,c,d are the axis index of expect_shape.
                
            tilting: Bool. Tilt the index or not. For PE systolic array input.
            tilt_axis: Integer. The axis wanted to be tilted.
            tilt_direction: Integer. The axis of direaction that are tilted to.
            tilt_shift: Integer. The amount of shifting for tilted representation. Positive number for tilt forward, negative for tilted backward.
        
            dataflow_pre_plan: Bool. Plan the dataflow model ahead. If True there will be no actual Tile to PEarray fault dictionary list transformation.
                Only save the expansion configuration for later PEarray to Tile transform.

        # Returns
            Converted fault dictionary.
        """
        self.expansion=True
        self.expand_method='extract_patches'
        self.reshape_patches=reshape_patches
        
        if len(ksizes)!=4:
            raise ValueError('ksize length must be 4, but got %d'%len(ksizes))
        if len(strides)!=4:
            raise ValueError('strides length must be 4, but got %d'%len(strides))
        if len(dilation_rates)!=4:
            raise ValueError('dilation rate length must be 4, but got %d'%len(dilation_rates))
            
        if dataflow_pre_plan:
            self.ksizes=tuple(ksizes)
            self.strides=strides
            self.dilation_rates=dilation_rates
            self.padding=padding
            self.edge_fill=edge_fill
            self.patches_unravel=patches_unravel
        
        extracted_shape=self.get_extracted_shape(fmap_shape=self.tile_shape,
                                                 ksizes=ksizes,
                                                 strides=strides,
                                                 dilation_rates=dilation_rates,
                                                 padding=padding,
                                                 edge_fill=edge_fill)
        self.extracted_shape=extracted_shape
        
        if reshape_patches:
            if len(patches_prior)!=len(extracted_shape):
                raise ValueError('patches_prior must be in length %d, but get length %d.'%(len(self.tile_shape),len(patches_prior)))
            self.expand_prior_orig=patches_prior
            self.expand_shape=expect_shape
            
            if len(reshape_prior)!=len(self.expand_shape):
                raise TypeError('slicing_dims must be in length %d, but get length %d'%(len(self.expand_shape),len(reshape_prior)))
            self.expand_prior_targ=reshape_prior
        
        else:
            self.expand_shape=extracted_shape
                
        if len(slicing_dims)!=len(self.expand_shape):
            raise TypeError('slicing_dims must be in length %d, but get length %d'%(len(self.expand_shape),len(slicing_dims)))
        self.slicing_dims=slicing_dims
        self.slice_shape=np.array(slicing_dims)
        self.slice_shape=tuple(self.slice_shape[self.slice_shape>0])
        self.slices_cutset,_=self.get_slices_cutset(self.expand_shape, self.slicing_dims)
        self.slice_shape=self.slice_shape+(np.prod(self.slices_cutset),)
        
        if len(slices_permute)!=len(self.expand_shape):
            raise TypeError('slices_permute must be in length %d, but get length %d'%(len(self.expand_shape),len(slices_permute)))
        self.slices_permute=slices_permute
        
        if not dataflow_pre_plan:
            orig_coors=np.array(list(self.fault_dict.keys()))
            fault_info=list(self.fault_dict.values())
            
            extracted_coors,cond_idx=self.extract_patches_idx(orig_coors,
                                                              fmap_shape=self.tile_shape,
                                                              ksizes=ksizes,
                                                              strides=strides,
                                                              dilation_rates=dilation_rates,
                                                              padding=padding,
                                                              edge_fill=edge_fill,
                                                              patches_unravel=patches_unravel,
                                                              get_cond_idx=True)
            fault_info=[fault_info[i] for i in cond_idx[:,0]]
            
            if reshape_patches:
                reshaped_coors=self.reshape_ravel_idx(extracted_coors,
                                                      source_shape=self.extracted_shape,
                                                      source_prior=self.expand_prior_orig,
                                                      target_shape=self.expand_shape,
                                                      target_prior=self.expand_prior_targ)
                
                reshaped_coors_fd=list(zip(*reshaped_coors.T))
            else:
                reshaped_coors_fd=list(zip(*extracted_coors.T))
            
            self.fault_dict_rehsaped=dict(zip(reshaped_coors_fd,fault_info))
            
            permuted_coors=self.slice_permute_idx(reshaped_coors,
                                                  orig_shape=self.expand_shape,
                                                  slicing_dims=self.slicing_dims,
                                                  slices_permute=self.slices_permute)
        
        if tilting:
            self.tilting=True
            self.tilt_axis=tilt_axis
            self.tilt_direction=tilt_direction
            self.tilt_shift=tilt_shift
            
            if not dataflow_pre_plan:
                permuted_coors,self.tilted_slice_shape=self.tilt_idx(permuted_coors,
                                                                     self.tilt_axis,
                                                                     self.tilt_direction,
                                                                     self.slice_shape,
                                                                     self.tilt_shift)
            else:
                _,self.tilted_slice_shape=self.tilt_idx(np.zeros([1,len(self.slice_shape)],dtype=int),
                                                        self.tilt_axis,
                                                        self.tilt_direction,
                                                        self.slice_shape,
                                                        self.tilt_shift)

        
        if not dataflow_pre_plan:
            permuted_coor_fd=list(zip(*permuted_coors.T))
            self.fault_dict_expand=dict(zip(permuted_coor_fd,fault_info))
            
            return self.fault_dict_expand
        
        else:
            return None
        
    def shrink_return_patches(self,psum=False,fast_gen=False):
        """ Reverse data expansion before put into PE array. Usually used for ifmap reduction. 
            Re-assemble the cut tile alices
            Method 'return patches' means return the extracted feature patches to ifmap tile.
            This method contains data reduction.
            reference: https://www.tensorflow.org/api_docs/python/tf/image/extract_patches
        
        # Arguments              
            psum: Bool. Indicate the transformation is for partial sum or not.

        # Returns
            Converted fault dictionary.
        """  
        if not psum:
            permuted_coors=np.array(list(self.fault_dict_expand.keys()))
            fault_info=np.array(list(self.fault_dict_expand.values()))
        else:
            permuted_coors=np.array(list(self.psum_fault_dict_expand.keys()))
            fault_info=np.array(list(self.psum_fault_dict_expand.values()))
        
        if self.tilting:
            permuted_coors,_=self.untilt_idx(permuted_coors,
                                             self.tilt_axis,
                                             self.tilt_direction,
                                             self.slice_shape,
                                             self.tilt_shift)
            
            # pop inalid t_clk
            permuted_coors,cond_idx=self.pop_outlier_idx(permuted_coors,self.slice_shape,get_cond_idx=True)
            fault_info=fault_info[cond_idx]
            
        reshaped_coors=self.assemble_slice_idx(permuted_coors,
                                               orig_shape=self.expand_shape,
                                               slicing_dims=self.slicing_dims,
                                               slices_permute=self.slices_permute)
        
        # pop under-utilized area on PE
        reshaped_coors,cond_idx=self.pop_outlier_idx(reshaped_coors,self.expand_shape,get_cond_idx=True)
        fault_info=fault_info[cond_idx]
        
        if not psum:
            reshaped_coors_fd=list(zip(*reshaped_coors.T))
            self.fault_dict_rehsaped=dict(zip(reshaped_coors_fd,fault_info))
        
        if self.reshape_patches:
            extracted_coors=self.reshape_ravel_idx(reshaped_coors,
                                                   source_shape=self.expand_shape,
                                                   source_prior=self.expand_prior_targ,
                                                   target_shape=self.extracted_shape,
                                                   target_prior=self.expand_prior_orig)
        else:
            extracted_coors=reshaped_coors
        
        orig_coors=self.return_patches_idx(extracted_coors,
                                           fmap_shape=self.tile_shape,
                                           ksizes=self.ksizes,
                                           strides=self.strides,
                                           dilation_rates=self.dilation_rates,
                                           padding=self.padding,
                                           edge_fill=self.edge_fill,
                                           patches_unravel=self.patches_unravel)
        
        # pop inalid edge condition that doesn't exist in tile
#        orig_coors,cond_idx=self.pop_outlier_idx(orig_coors,list(self.tile_shape),get_cond_idx=True)
#        fault_info=fault_info[cond_idx]
        # keep the edge condition for correspond io solving, also for tile2layer. Problem solve by fault injection.
        
        # deal with repetitive orig_coors
        orig_coors,uni_idx,rep_idx,cnt_idx=np.unique(orig_coors,return_index=True,return_inverse=True,return_counts=True,axis=0)
        
        if len(uni_idx)==len(rep_idx):
            fault_info=fault_info[uni_idx]
        else:
            if fast_gen:
                id_list=np.array([value['id'] for value in fault_info])
                
                id_list=id_list[np.argsort(rep_idx)]
                cnt_idx=np.cumsum(cnt_idx)[:-1]
                id_list=np.split(id_list,cnt_idx)
                
                fault_info=fault_info[uni_idx]
                for i in range(len(uni_idx)):
                    fault_info[i]['id']=id_list[i].flatten()
            else:
                id_list_rep=[list() for _ in range(len(uni_idx))]
                type_list_rep=[list() for _ in range(len(uni_idx))]
                bit_list_rep=[list() for _ in range(len(uni_idx))]
                param_list_rep=[list() for _ in range(len(uni_idx))]
                
                for i,repid in enumerate(rep_idx):
                    if isinstance(fault_info[i]['id'],int):
                        id_list_rep[repid].append(fault_info[i]['id'])
                    else:
                        id_list_rep[repid]+=fault_info[i]['id']
                        
                    if isinstance(fault_info[i]['SA_type'],str):
                        type_list_rep[repid].append(fault_info[i]['SA_type'])
                    else:
                        type_list_rep[repid]+=fault_info[i]['SA_type']
                        
                    if isinstance(fault_info[i]['SA_bit'],int):
                        bit_list_rep[repid].append(fault_info[i]['SA_bit'])
                    else:
                        bit_list_rep[repid]+=fault_info[i]['SA_bit']
                        
                    if isinstance(fault_info[i]['param'],str):
                        param_list_rep[repid].append(fault_info[i]['param'])
                    else:
                        param_list_rep[repid]+=fault_info[i]['param']
            
                fault_info=fault_info[uni_idx]
                for i in range(len(uni_idx)):
                    fault_info[i]['id']=id_list_rep[i]
                    fault_info[i]['SA_type']=type_list_rep[i]
                    fault_info[i]['SA_bit']=bit_list_rep[i]
                    fault_info[i]['param']=param_list_rep[i]

        orig_coor_fd=list(zip(*orig_coors.T))
        new_fault_dict=dict(zip(orig_coor_fd,fault_info))
        if not psum:
            self.fault_dict=new_fault_dict
        else:
            self.psum_fault_dict=new_fault_dict
        
        return new_fault_dict
            
    def expand_slice_bias(self, bias_slice_width, dataflow_pre_plan=False):
        """ Data expansion before put into PE array. 
            The data are being cut into many pieces then fit into PE. Different slices calculate in different clock cycle.
        
        # Arguments                            
            bias_slice_width: Integer. The expected slice width to be expand into. 
            
            dataflow_pre_plan: Bool. Plan the dataflow model ahead. If True there will be no actual Tile to PEarray fault dictionary list transformation.
                Only save the expansion configuration for later PEarray to Tile transform.
                            
        # Returns
            Converted fault dictionary.

        """
        if self.is_fmap:
            raise TypeError('This is feature maps tile, no bias!')
        self.use_bias=True
        self.expansion=True
            
        self.bias_slice_shape=(bias_slice_width, int(np.ceil(self.Tn/bias_slice_width)))
        
        if not dataflow_pre_plan:
            orig_coors=np.array(list(self.bias_fault_dict.keys()))
            fault_info=list(self.bias_fault_dict.values())
            
            sliced_coors=np.concatenate([np.remainder(orig_coors,bias_slice_width),np.floor_divide(orig_coors,bias_slice_width)],axis=1)
            
            sliced_coors_fd=list(zip(*sliced_coors.T))
            self.bias_fault_dict_expand=dict(zip(sliced_coors_fd,fault_info))
            
            return self.bias_fault_dict_expand
        
        else:
            self.bias_fault_dict_expand=dict()
            return None
        
    def shrink_slice_bias(self):
        """ Data shrink for PE array to Tile mapping.
            The data are being cut into many pieces then fit into PE. Different slices calculate in different clock cycle.
        
        # Arguments                            
                            
        # Returns
            Converted fault dictionary.

        """
        if self.is_fmap:
            raise TypeError('This is feature maps tile, no bias!')           
        
        if not self.use_bias:
            raise AttributeError('not use bias')
        
        if len(self.bias_fault_dict_expand)==0:
            return dict()
        
        sliced_coors=np.array(list(self.bias_fault_dict_expand.keys()))
        fault_info=list(self.bias_fault_dict_expand.values())
        
        bias_idx=sliced_coors[:,0]
        slice_idx=sliced_coors[:,1]
        orig_coors=np.add(np.multiply(slice_idx,self.bias_slice_shape[0]),bias_idx)
        
        orig_coors_fd=list(zip(*np.expand_dims(orig_coors,0)))
        self.bias_fault_dict=dict(zip(orig_coors_fd,fault_info))
        
        return self.bias_fault_dict
        
    def pop_outlier_idx(self, index, shape, get_cond_idx=False):
        """ Remove coordinates in fault dictionary that lies outside of current shape.
            Only used in PEarray to Tile mapping. Due to time expand on fault list generation.
            In Tile to PEarray mapping, coordinates outside current shape might be invalid configuration.
        
        """        
        index_bound=np.floor_divide(index,shape)
        cond_arg=np.max(index_bound,axis=1)<1
        cond_tmp=np.min(index_bound,axis=1)>=0
        cond_arg=np.bitwise_and(cond_arg,cond_tmp)        
        
        poped_index=index[cond_arg]
        
        if get_cond_idx:
            return poped_index,cond_arg
        else:
            return poped_index
    
    def clear(self):
        """ Clear fault dictionary of tile """
        self.fault_dict=dict()
        self.fault_dict_rehsaped=dict()
        self.fault_dict_expand=dict()
        self.psum_fault_dict=dict()
        self.psum_fault_dict_expand=dict()
        self.bias_fault_dict=dict()
        self.bias_fault_dict_expand=dict()
        
    def clear_expansion(self):
        """ Clear expansion shapes and priorities of tile """
        self.expand_shape=None
        self.expand_prior_orig=None
        self.expand_prior_targ=None
        self.slice_shape=None
        self.slicing_dims=None
        self.slices_permute=None
        self.slices_cutset=None
        self.tilted_slice_shape=None
        self.bias_slice_shape=None

class io_data_solver:
    """
    The PE dataflow mapping fault list data solving class. 
    For solveing the PE fault to each corresponding data on ifmap, weight, ofmap.
    Produces a fault dictionary of the layer ofmap shape which contains fault information.
    As well as fault id and parrtial sum index.
    """
    def __init__(self, ofmap_tile, wght_tile, ifmap_tile, fault_num=None):
        self.ofmap_tile=ofmap_tile
        self.wght_tile=wght_tile
        self.ifmap_tile=ifmap_tile
        self.fault_num=fault_num
        
    def _state_setting(self, faultvalue):
        """
        Distinguish and set the state of an data type
        """
        if len(faultvalue)>0:
            if isinstance(faultvalue[0]['id'],np.ndarray):
                state='fastgen'
                idlist=np.array([info['id'] for info in faultvalue])
                maxx=np.max(np.concatenate(idlist))
                if idlist.dtype==np.object:
                    idl_cnt=np.array([len(i) for i in idlist])
                    idl_cnt=np.cumsum(idl_cnt)-1
                    idlist=np.concatenate(idlist)
                    idlist=(idlist,idl_cnt)
            elif isinstance(faultvalue[0]['id'],int):
                state='normal'
                idlist=[info['id'] for info in faultvalue]
                maxx=max(idlist)
            elif isinstance(faultvalue[0]['id'],list):
                state='repetitive'
                idlist=[info['id'] for info in faultvalue]
                maxx=max([max(idl) for idl in idlist])
        else:
            idlist=list()
            state=None
            maxx=-1
        
        return idlist,state,maxx
    
    def _state_idxget(self,idlist,faultcoors,faultvalue,state,faultid,paramin):
        """
        Extract the data coordinate index and fault parameter by fault id
        """
        if state is None:
            idx=None
            param=None
            faultindex=None
            
        elif state=='fastgen':
            if isinstance(idlist,np.ndarray):
                idx=np.argwhere(idlist==faultid)
                if len(idx)==0:
                    idx=None
                    param=None
                    faultindex=None
                else:
                    idx=idx[0][0]
                    param=faultvalue[idx]['param']
                    faultindex=faultcoors[idx]
                    
            elif isinstance(idlist,tuple):
                idl_cnt=idlist[1]
                idlist=idlist[0]
                
                idx=np.argwhere(idlist==faultid)
                if len(idx)==0:
                    idx=None
                    param=None
                    faultindex=None
                else:
                    idx=idx[0][0]
                    idx=np.searchsorted(idl_cnt,idx)
                    param=faultvalue[idx]['param']
                    faultindex=faultcoors[idx]
            
        elif state=='normal':
            try:
                idx=idlist.index(faultid)
                param=faultvalue[idx]['param']
                faultindex=faultcoors[idx]
            except ValueError:
                idx=None
                param=None
                faultindex=None
                
        elif state=='repetitive':
            idx=None
            param=None
            faultindex=None
            for ii,idl in enumerate(idlist):
                if faultid in idl:
                    ii2=idl.index(faultid)
                    idx=[ii,ii2]
                    param=faultvalue[ii]['param'][ii2]
                    faultindex=faultcoors[ii]
        
        if paramin is not None:
            param=paramin
        
        return idx,param,faultindex
    
    def _state_make_new_fd(self,state,faultid,paramin,opindex,windex,iindex,faultvalue,idx,newfd):
        """
        Add new fault information to new fault dict
        Individually append (slow loop version)
        """
        try:
            # psum index (Batch, Output Channel, Ofmap Row, Ofmap Column, Ifmap Channel, Kernel Row, Kernel Column, Ifmap Row, Ifmap Column)
            psidx=tuple(np.concatenate([opindex[[0,3,1,2]],windex[[2,0,1]],iindex[[1,2]]]))
            if state is None:
                pass
            elif state=='fastgen':
                try:
                    newfv=newfd[tuple(opindex)]
                    newfv['psum_idx'].append(psidx)
                except KeyError:
                    newfv=faultvalue[idx].copy()
                    newfv.update({'psum_idx':[psidx]})
                    newfd[tuple(opindex)]=newfv
            elif state=='normal':
                newfv=faultvalue[idx].copy()
                newfv.update({'psum_idx':psidx})
                newfd[tuple(opindex)]=newfv
            elif state=='repetitive':
                try:
                    newfv=newfd[tuple(opindex)]
                    newfv['SA_bit'].append(faultvalue[idx[0]]['SA_bit'][idx[1]])
                    newfv['SA_type'].append(faultvalue[idx[0]]['SA_type'][idx[1]])
                    newfv['param'].append(paramin)
                    newfv['psum_idx'].append(psidx)
                    newfv['id'].append(faultid)
                except KeyError:
                    newfv={'SA_bit':[faultvalue[idx[0]]['SA_bit'][idx[1]]],
                           'SA_type':[faultvalue[idx[0]]['SA_type'][idx[1]]],
                           'param':[paramin],
                           'psum_idx':[psidx],
                           'id':[faultid]}
                    newfd[tuple(opindex)]=newfv
        except TypeError:
            pass
        
    def _get_base_data_id(self, data_id):
        """
        Solve base data fault id list for find other data coordinate
        """
        if isinstance(data_id,np.ndarray):
            shape_cnt=data_id.shape
            if len(shape_cnt)>1:
                data_idf=data_id.flatten()
        elif isinstance(data_id,tuple):
            shape_cnt=data_id[1]
            data_idf=data_id[0]
        
        return data_idf,shape_cnt
        
    def _get_data_coor_by_id(self, data_id, search_id, data_coors):
        """
        Get data coordinate by giving fault id list
        """
        if isinstance(data_id,np.ndarray):
            datashape=data_id.shape
            if len(datashape)>1:
                data_idf=data_id.flatten()
                sorter=np.argsort(data_idf)
                search=np.searchsorted(data_idf,search_id,sorter=sorter)
                search=sorter[search]
                search=np.floor_divide(search,datashape[1])
                found_index=data_coors[search]
            else:
                sorter=np.argsort(data_id)
                search=np.searchsorted(data_id,search_id,sorter=sorter)
                search=sorter[search]
                found_index=data_coors[search]
        elif isinstance(data_id,tuple):
            data_idl_cnt=data_id[1]
            data_idf=data_id[0]
            sorter=np.argsort(data_idf)
            search=np.searchsorted(data_idf,search_id,sorter=sorter)
            search=sorter[search]
            search=np.searchsorted(data_idl_cnt,search)
            found_index=data_coors[search]
            
        return found_index

    def fast_gen_new_fd(self, save2tile=False, print_detail=False):
        """
        Extract the data coordinate index and fault parameter by fault id
        Add new fault information to new fault dict
        Numpy generation (fast version)
        """
        #TODO
        # WHAT IF the searchsort result are repeatitive, HOW TO filt no index condition
        
        
        if self.pstate!='fastgen' or self.wstate!='fastgen' or self.istate!='fastgen':
            raise ValueError('All psum_state, wght_state, ifmap_state are must be \'fast_gen\' to run fast generation method.')        
            
        new_solved_fd=dict()
            
        if not save2tile:
            if print_detail:
                print('    GenFD (1/5): Solve Partial Sum Coordinates...',end=' ')     
            # solve psum, use as basis for fault id search
            search_id, shape_cnt=self._get_base_data_id(self.psum_id)
            
            if print_detail:
                print('\r    GenFD (2/5): Solve Input Feature Map Coordinates...',end=' ') 
            # solve ifmap
            ifmap_index=self._get_data_coor_by_id(self.ifmap_id, search_id, self.ifmap_coors)
            
            if print_detail:
                print('\r    GenFD (3/5): Solve Weight Coordinates...\t\t',end=' ') 
            # solve weight
            wght_index=self._get_data_coor_by_id(self.wght_id, search_id, self.wght_coors)
            
        else:
            if print_detail:
                print('    GenFD (1/5): Solve Base Data Coordinates...',end=' ') 
            
            param=self.psum_vl[0]['param']
            if param=='ifmap_in' or param=='ifmap_out':
                search_id, shape_cnt=self._get_base_data_id(self.ifmap_id)
            elif param=='wght_in' or param=='wght_out':
                search_id, shape_cnt=self._get_base_data_id(self.wght_id)
            elif param=='psum_in' or param=='psum_out':
                search_id, shape_cnt=self._get_base_data_id(self.psum_id)
                
            if param=='ifmap_in' or param=='ifmap_out':
                if print_detail:
                    print('\r    GenFD (2/5): Solve Output Feature Map Coordinates...',end=' ') 
                # solve ifmap
                outpsum_index=self._get_data_coor_by_id(self.psum_id, search_id, self.psum_coors)
                if print_detail:
                    print('\r    GenFD (3/5): Solve Weight Coordinates...\t\t',end=' ') 
                # solve weight
                wght_index=self._get_data_coor_by_id(self.wght_id, search_id, self.wght_coors)
                
            elif param=='wght_in' or param=='wght_out':
                if print_detail:
                    print('\r    GenFD (2/5): Solve Input Feature Map Coordinates...',end=' ') 
                # solve ifmap
                ifmap_index=self._get_data_coor_by_id(self.ifmap_id, search_id, self.ifmap_coors)
                if print_detail:
                    print('\r    GenFD (3/5): Solve Output Feature Map Coordinates...',end=' ') 
                # solve weight
                outpsum_index=self._get_data_coor_by_id(self.psum_id, search_id, self.psum_coors)
                
            elif param=='psum_in' or param=='psum_out':
                if print_detail:
                    print('\r    GenFD (2/5): Solve Input Feature Map Coordinates...',end=' ') 
                # solve ifmap
                ifmap_index=self._get_data_coor_by_id(self.ifmap_id, search_id, self.ifmap_coors)
                if print_detail:
                    print('\r    GenFD (3/5): Solve Weight Coordinates...\t\t',end=' ') 
                # solve weight
                wght_index=self._get_data_coor_by_id(self.wght_id, search_id, self.wght_coors)
        
        if print_detail:
            print('\r    GenFD (4/5): Build Partial Sum Indexes...\t\t',end=' ')         
        
        # build psum_idx
        if not save2tile:
            based_coors=self.psum_coors
            based_vl=self.psum_vl
        else:
            if param=='ifmap_in' or param=='ifmap_out':
                based_coors=self.ifmap_coors
                based_vl=self.ifmap_vl
            elif param=='wght_in' or param=='wght_out':
                based_coors=self.wght_coors
                based_vl=self.wght_vl
            elif param=='psum_in' or param=='psum_out':
                based_coors=self.psum_coors
                based_vl=self.psum_vl
            
        if isinstance(shape_cnt,tuple):
            if len(shape_cnt)>1:
                outpsum_index=np.repeat(based_coors,shape_cnt[1],axis=0)
        elif isinstance(shape_cnt,np.ndarray):
            cnt0=shape_cnt[0]+1
            idlrep=shape_cnt[1:]-shape_cnt[:-1]
            idlrep=np.concatenate([[cnt0],idlrep])
            idlconstruct=np.repeat(np.arange(len(idlrep)),idlrep)
            outpsum_index=based_coors[idlconstruct]
            
        psum_index=np.concatenate([outpsum_index[:,[0,3,1,2]],wght_index[:,[2,0,1]],ifmap_index[:,[1,2]]],axis=1)
        if isinstance(shape_cnt,tuple):
            if len(shape_cnt)>1:
                psum_index=np.split(psum_index,shape_cnt[0])
        elif isinstance(shape_cnt,np.ndarray):
            psum_index=np.split(psum_index,idlrep)

        if print_detail:
            print('\r    GenFD (5/5): Make Solved Fault Dictionary...\t\t',end=' ')         

        for i,opidx in enumerate(based_coors):
            newfv=based_vl[i].copy()
            newfv.update({'psum_idx':psum_index[i]})
            new_solved_fd[tuple(opidx)]=newfv
            
        if not save2tile:
            return new_solved_fd
        else:
            if param=='ifmap_in' or param=='ifmap_out':
                fd_assigner=(new_solved_fd,dict(),dict(),dict())
            elif param=='wght_in' or param=='wght_out':
                fd_assigner=(dict(),new_solved_fd,dict(),dict())
            elif param=='psum_in' or param=='psum_out':
                fd_assigner=(dict(),dict(),dict(),new_solved_fd)

            self.ifmap_tile.fault_dict, self.wght_tile.fault_dict, self.wght_tile.bias_fault_dict, self.ofmap_tile.fault_dict = fd_assigner
            return new_solved_fd

    def loop_gen_new_fd(self, save2tile=False, print_detail=False):
        """
        Extract the data coordinate index and fault parameter by fault id
        Add new fault information to new fault dict
        For loop generation (slower version, for user raw input PE fault dictionary)
        """
        if not save2tile:
            new_solved_fd=dict()
        else:
            new_ifmap_fd=dict()
            new_wght_fd=dict()
            new_ofmap_fd=dict()
            new_bias_fd=dict()
        
        if print_detail:
            pbar=tqdm.tqdm(desc='\tSolved Fault', total=self.fault_num, leave=False)
        
        for i in range(self.fault_num):
            param=None
            
            if save2tile:
                oidx,param,ofmap_index=self._state_idxget(self.ofmap_id, self.ofmap_coors, self.ofmap_vl, self.ostate,i,param)
                bidx,param,bias_index=self._state_idxget(self.bias_id, self.bias_coors, self.bias_vl, self.bstate,i,param)
            pidx,param,psum_index=self._state_idxget(self.psum_id, self.psum_coors, self.psum_vl, self.pstate,i,param)
            widx,param,wght_index=self._state_idxget(self.wght_id, self.wght_coors, self.wght_vl, self.wstate,i,param)
            iidx,param,ifmap_index=self._state_idxget(self.ifmap_id, self.ifmap_coors, self.ifmap_vl, self.istate,i,param)
            
            # partial sum index (batch, Tn, TrO, TcO, Tm, TrK, TcK, TrI, TcI)
            if param is not None:
                if not save2tile:
                    self._state_make_new_fd(self.pstate,i,param,psum_index,wght_index,ifmap_index,self.psum_vl,pidx,new_solved_fd)
                else:
                    if (param=='ifmap_in' or param=='ifmap_out') and iidx is not None:
                        self._state_make_new_fd(self.istate,i,param,psum_index,wght_index,ifmap_index,self.ifmap_vl,iidx,new_ifmap_fd)
                    
                    elif (param=='wght_in' or param=='wght_out') and widx is not None:
                        self._state_make_new_fd(self.wstate,i,param,psum_index,wght_index,ifmap_index,self.wght_vl,widx,new_wght_fd)
                                        
                    elif param=='psum_in' and pidx is not None:
                        if bidx is None:                                   
                            self._state_make_new_fd(self.pstate,i,param,psum_index,wght_index,ifmap_index,self.psum_vl,pidx,new_ofmap_fd)
                        else:
                            new_bias_fd[tuple(bias_index)]=self.bias_vl[bidx]
                                                        
                    elif param=='psum_out' and pidx is not None:
                        if oidx is None:
                            self._state_make_new_fd(self.pstate,i,param,psum_index,wght_index,ifmap_index,self.psum_vl,pidx,new_ofmap_fd)
                            new_ofmap_fd[tuple(psum_index)].update({'ofmap':False})
                        else:
                            self._state_make_new_fd(self.ostate,i,param,ofmap_index,wght_index,ifmap_index,self.ofmap_vl,oidx,new_ofmap_fd)                   
                            new_ofmap_fd[tuple(ofmap_index)].update({'ofmap':True})
                        
            if print_detail:
                pbar.update()
            
        if print_detail:
            pbar.close()
            
        if not save2tile:
            return new_solved_fd
        else:
            self.ofmap_tile.fault_dict=new_ofmap_fd
            self.ifmap_tile.fault_dict=new_ifmap_fd
            self.wght_tile.fault_dict=new_wght_fd
            self.wght_tile.bias_fault_dict=new_bias_fd
            return new_ifmap_fd, new_wght_fd, new_bias_fd, new_ofmap_fd
        
    def solve_correspond_io(self, save2tile=False, print_detail=False):
        """ Solving the PE array to Tile mapping fault dictionarys.
            Regarding ofmap, ifmap, weight, partial sum, bias fault dictionarys, 
            and find the relation between them. Give fault info (psum index).
        
        # Arguments                   
            save2tile: Bool. If true, save the solving result fault dict to repective tile.
                Else false, return the fault dictionary of the layer ofmap shape which contains solved information.
            print_detail: Bool. Print the solving process.
        # Returns
            Solved fault dictionary.

        """
        if save2tile:
            ofmap_fd=self.ofmap_tile.fault_dict
            bias_fd=self.wght_tile.bias_fault_dict
        ifmap_fd=self.ifmap_tile.fault_dict
        wght_fd=self.wght_tile.fault_dict
        psum_fd=self.ofmap_tile.psum_fault_dict
        
        if save2tile:
            self.ofmap_coors=np.array(list(ofmap_fd.keys()))
            self.bias_coors =np.array(list(bias_fd.keys()))
        self.ifmap_coors=np.array(list(ifmap_fd.keys()))
        self.wght_coors =np.array(list(wght_fd.keys()))
        self.psum_coors =np.array(list(psum_fd.keys()))
 
        if save2tile:       
            self.ofmap_vl=np.array(list(ofmap_fd.values()))
            self.bias_vl =np.array(list(bias_fd.values()))
        self.ifmap_vl=np.array(list(ifmap_fd.values()))
        self.wght_vl =np.array(list(wght_fd.values()))
        self.psum_vl =np.array(list(psum_fd.values()))
       
        if save2tile:       
            self.ofmap_id,self.ostate,maxo=self._state_setting(self.ofmap_vl)
            self.bias_id,self.bstate,maxb=self._state_setting(self.bias_vl)
        self.ifmap_id,self.istate,maxi=self._state_setting(self.ifmap_vl)
        self.wght_id,self.wstate,maxw=self._state_setting(self.wght_vl)
        self.psum_id,self.pstate,maxp=self._state_setting(self.psum_vl)
        
        if self.fault_num==None:
            if not save2tile:
                self.fault_num=max([maxi,maxw,maxp])+1
            else:
                self.fault_num=max([maxo,maxi,maxw,maxp,maxb])+1
            
        # solving by fast gen method
        if self.pstate=='fastgen' and self.wstate=='fastgen' and self.istate=='fastgen':
            fault_dict_solved=self.fast_gen_new_fd(save2tile,print_detail)
        else:
            fault_dict_solved=self.loop_gen_new_fd(save2tile,print_detail)
            
        self.fault_dict_solved=fault_dict_solved
        return fault_dict_solved
 
    def _pop_outlier_idx(self, index, shape, get_cond_idx=False):
        """ Remove coordinates in fault dictionary that lies outside of current shape."""        
        index_bound=np.floor_divide(index,shape)
        cond_arg=np.max(index_bound,axis=1)<1
        
        if np.all(cond_arg):
            if get_cond_idx:
                return index,cond_arg
            else:
                return index
        
        poped_index=index[cond_arg]
        
        if get_cond_idx:
            return poped_index,cond_arg
        else:
            return poped_index
    
    def _gen_base_coor(self, tile_shape, layer_shape, is_ifmap=None):
        """ Generate layer base coordinates for tile to layer duplication """
        restore_multiple=np.divide(layer_shape,tile_shape,dtype=np.float32)
        restore_multiple=np.ceil(restore_multiple)
        restore_multiple=restore_multiple.astype(np.int32)
        
        if is_ifmap is not None:
            if is_ifmap:
                restore_multiple=restore_multiple[[0,3,2,1]]
                reorder_shape=np.array(tile_shape)[[0,3,2,1]]
            else:
                restore_multiple=restore_multiple[[3,2,1,0]]
                reorder_shape=np.array(tile_shape)[[3,2,1,0]]
        else:
            reorder_shape=layer_shape
                
        base_coor=list(np.ndindex(*restore_multiple))            
        base_coor=np.multiply(base_coor,np.tile(reorder_shape,[len(base_coor),1]))
           
        if is_ifmap is not None:
            if is_ifmap:
                restore_multiple=restore_multiple[[0,3,2,1]]
                base_coor=base_coor[:,[0,3,2,1]]
            else:
                restore_multiple=restore_multiple[[3,2,1,0]]
                base_coor=base_coor[:,[3,2,1,0]]
                        
        return base_coor, restore_multiple
    
    def tile2layer(self, fault_dict=None, based_tile='ofmap', layer=None, layer_input_shape=None, layer_weight_shape=None, layer_output_shape=None):
        """ Restore the fault dictionary from tile to entire layer
            This tile2layer combines all ifmap, weight, ofmap tile.
            Solves tile2layer for both fault dict coordinates and partial sum indexes.
            By consider the duplication cause by tiling cut in input channel or kernel 2D.

        # Arguments
            fault_dict: Dictionary. The fault dictionary be duplicate expnand to layer. Contains fault information with partial sum indexes.
            based_tile: String. The tile which the coordinates of fault dictionary indicate to. Must be one of 'ofmap','wght','ifmap'.
            layer: Class. Keras Layer class, for extract the layer input, weight, output shapes
            layer_input_shape: Tuple. The shape of a layer input parameter were divided into tile.
            layer_weight_shape: Tuple. The shape of a layer weight parameter were divided into tile.
            layer_output_shape: Tuple. The shape of a layer output parameter were divided into tile.
        
        # Returns
            The fault information Dictionary of a layer parameter (feature maps or weights).
        """
        # the combined inter tile tile2layer
        if based_tile not in ['ofmap','wght','ifmap']:
            raise ValueError('based_tile must be one of \'ofmap\',\'wght\',\'ifmap\'.')
        
        if fault_dict is None:
            fault_dict=self.fault_dict_solved
            
        if len(fault_dict)==0:
            return dict()
        
        if layer is not None:
            layer_input_shape=layer.input_shape
            layer_output_shape=layer.output_shape
            layer_weight_shape=[weight_shape.shape for weight_shape in layer.get_weights()]
        
        # unpack partial sum index
        fd_coor=np.array(list(fault_dict.keys()))
        fd_value=np.array(list(fault_dict.values()))
        psum_idx=np.array([info['psum_idx'] for info in fd_value])
        
        if isinstance(fd_value[0]['id'],np.ndarray):
            state='fastgen'
            if psum_idx.dtype!=np.object:
                psidx_cnt=psum_idx.shape
                psum_idx=np.concatenate(psum_idx)
            else:
                psidx_cnt=np.array([len(i) for i in psum_idx])
                psum_idx=np.concatenate(psum_idx)
        elif isinstance(fd_value[0]['id'],int):
            state='normal'
        elif isinstance(fd_value[0]['id'],list):
            state='repetitive'
            psidx_cnt=np.array([len(i) for i in psum_idx])
            psum_idx=np.concatenate(psum_idx)
            
        self.num_fault_coor=len(fd_coor)
        self.num_psum_idx=len(psum_idx)
                
        # get base coors
        base_coor_o, restore_multiple_o=self._gen_base_coor(self.ofmap_tile.tile_shape, layer_output_shape, is_ifmap=True)
        base_coor_w, restore_multiple_w=self._gen_base_coor(self.wght_tile.tile_shape, layer_weight_shape[0], is_ifmap=False)
        base_coor_i, restore_multiple_i=self._gen_base_coor(self.ifmap_tile.tile_shape, layer_input_shape, is_ifmap=True)
        
        # consistency check
        if restore_multiple_i[0]!=restore_multiple_o[0] or restore_multiple_i[3]!=restore_multiple_w[2] or restore_multiple_o[3]!=restore_multiple_o[3]:
            raise ValueError('The tile shape is inconsistent! \nInput (batch, row, col, channel)=%s \nWeight (row,col,in-channel,out-channel)=%s \nOutput (batch,row,col,channel)=%s'%(str(self.ifmap_tile.tile_shape),str(self.wght_tile.tile_shape),str(self.ofmap_tile.tile_shape)))
        # verify row col of ofmap to ifmap tile
        if self.ifmap_tile.extracted_shape[1]!=self.ofmap_tile.tile_shape[1] or self.ifmap_tile.extracted_shape[2]!=self.ofmap_tile.tile_shape[2]:
            raise ValueError('The row, col shape of ifmap tile and ofmap tile does not match! \n ifmap (row,col)=%s  ofmap (row,col)=$s'%(str(self.ifmap_tile.extracted_shape[1:3]),str(self.ofmap_tile.tile_shape[1:3])))
               
        # ofmap form 
        base_coor_o=np.repeat(base_coor_o,restore_multiple_w[2],axis=0) # repeatition for tile level psum
        if restore_multiple_w[0]*restore_multiple_w[1]>1: # duplicate if tile split on kernel 2D
            base_coor_o=np.repeat(base_coor_o, restore_multiple_w[0]*restore_multiple_w[1], axis=0)
        # match w->o
        base_coor_w=np.split(base_coor_w,restore_multiple_w[3]) # split output channel
        base_coor_w=np.repeat(base_coor_w,restore_multiple_o[1]*restore_multiple_o[2],axis=0) # duplicate for ofmap 2D tiles
        base_coor_w=np.concatenate(base_coor_w)
        base_coor_w=np.tile(base_coor_w,[restore_multiple_o[0],1]) # duplicate for batch
        # match i->o
        base_coor_i=np.split(base_coor_i,restore_multiple_i[0]) # split batch
        base_coor_i=np.stack(base_coor_i) # new batch dims
        base_coor_i=np.split(base_coor_i,restore_multiple_i[3],axis=1) # split input channel
        base_coor_i=np.stack(base_coor_i) # new input channel psum dims
        base_coor_i=np.transpose(base_coor_i,[1,2,0,3]) # reorder axes for interleave input channel psum
        base_coor_i=np.tile(base_coor_i,[1,restore_multiple_o[3],1,1]) # duplicate for ofmap channel tile cut
        base_coor_i=np.reshape(base_coor_i,[-1,4]) # serialize (batch, ofmap 2D, input channel psum)
        if restore_multiple_w[0]*restore_multiple_w[1]>1: # duplicate if tile split on kernel 2D
            base_coor_i=np.repeat(base_coor_i, restore_multiple_w[0]*restore_multiple_w[1], axis=0)
        
        # the base partial sum indexes
        base_coor_psum_idx=np.concatenate([base_coor_o[:,[0,3,1,2]],base_coor_w[:,[2,0,1]],base_coor_i[:,[1,2]]],axis=1)
        self.num_base_coor=len(base_coor_psum_idx)

        # based tile layer fault coors
        if based_tile=='ofmap':
            layer_base_coor=base_coor_o
        elif based_tile=='wght':
            layer_base_coor=base_coor_w
        elif based_tile=='ifmap':
            layer_base_coor=base_coor_i
            
        layer_fault_coor=list()
        for i in range(4):
            layer_fault_coor.append(np.add.outer(layer_base_coor[:,i],fd_coor[:,i]))
        layer_fault_coor=np.stack(layer_fault_coor,axis=-1)
        layer_fault_coor=np.reshape(layer_fault_coor,[-1,4])


        # partial sum indexes to layer fault coors
        layer_psum_idx=list()
        for i in range(9):
            layer_psum_idx.append(np.add.outer(base_coor_psum_idx[:,i],psum_idx[:,i]))
        layer_psum_idx=np.stack(layer_psum_idx,axis=-1)
        layer_psum_idx=np.reshape(layer_psum_idx,[-1,9])
        
        # remove outlier partial sum index
        try:
            if self.ifmap_tile.padding=='same':
                layer_input_shape_row=layer_input_shape[1] + (self.ifmap_tile.ksizes[1] + (self.ifmap_tile.ksizes[1]-1) * (self.ifmap_tile.dilation_rates[1] - 1))//2
                layer_input_shape_col=layer_input_shape[2] + (self.ifmap_tile.ksizes[2] + (self.ifmap_tile.ksizes[2]-1) * (self.ifmap_tile.dilation_rates[2] - 1))//2
            else:
                layer_input_shape_row=layer_input_shape[1] 
                layer_input_shape_col=layer_input_shape[2] 
        except AttributeError:
            layer_input_shape_row=layer_input_shape[1] 
            layer_input_shape_col=layer_input_shape[2] 
            
        psum_idx_shape=[[layer_output_shape[0],layer_output_shape[3],layer_output_shape[1],layer_output_shape[2],layer_weight_shape[0][2],layer_weight_shape[0][0],layer_weight_shape[0][1],layer_input_shape_row,layer_input_shape_col]]
        layer_psum_idx,psidx_cond=self._pop_outlier_idx(layer_psum_idx, psum_idx_shape, get_cond_idx=True)
        self.num_layer_psum_idx=len(layer_psum_idx)
        
        # split serial layer partial sum index for each for coordinate
        if np.all(psidx_cond):
            if state=='fastgen':
                if isinstance(psidx_cnt,tuple):
                    layer_psum_idx=np.split(layer_psum_idx,psidx_cnt[0]*self.num_base_coor)
                else:
                    psidx_cnt=np.tile(psidx_cnt,self.num_base_coor)
                    psidx_cnt=np.cumsum(psidx_cnt)[:-1]
                    layer_psum_idx=np.split(layer_psum_idx,psidx_cnt)
            elif state=='normal':
                pass
            elif state=='repetitive':
                psidx_cnt=np.tile(psidx_cnt,self.num_base_coor)
                psidx_cnt=np.cumsum(psidx_cnt)[:-1]
                layer_psum_idx=np.split(layer_psum_idx,psidx_cnt)
        else:
            psidx_cond=np.bitwise_not(psidx_cond)
            if state=='fastgen':
                if isinstance(psidx_cnt,tuple):
                    psidx_cond=np.reshape(psidx_cond,[psidx_cnt[0]*self.num_base_coor,-1])
                    psidx_cond=np.sum(psidx_cond,axis=1)
                    psidx_cnt=np.full_like(psidx_cond,psidx_cnt[1])
                    psidx_cnt=np.subtract(psidx_cnt,psidx_cond)
                    psidx_cnt=np.cumsum(psidx_cnt)[:-1]
                    layer_psum_idx=np.split(layer_psum_idx,psidx_cnt)
                else:
                    psidx_cnt=np.tile(psidx_cnt,self.num_base_coor)
                    psidx_cnt_search=np.cumsum(psidx_cnt)-1
                    psidx_cond=np.argwhere(psidx_cond)
                    psidx_cond=np.searchsorted(psidx_cnt_search,psidx_cond)
                    psidx_cond=np.unique(psidx_cond,return_counts=True)
                    psidx_cnt[psidx_cond[0]]=np.subtract(psidx_cnt[psidx_cond[0]],psidx_cond[1])
                    psidx_cnt=np.cumsum(psidx_cnt)[:-1]
                    layer_psum_idx=np.split(layer_psum_idx,psidx_cnt)
                    
            elif state=='normal':
                pass
            elif state=='repetitive':
                psidx_cnt=np.tile(psidx_cnt,self.num_base_coor)
                psidx_cnt_search=np.cumsum(psidx_cnt)-1
                psidx_cond=np.argwhere(psidx_cond)
                psidx_cond=np.searchsorted(psidx_cnt_search,psidx_cond)
                psidx_cond=np.unique(psidx_cond,return_counts=True)
                psidx_cnt[psidx_cond[0]]=np.subtract(psidx_cnt[psidx_cond[0]],psidx_cond[1])
                psidx_cnt=np.cumsum(psidx_cnt)[:-1]
                layer_psum_idx=np.split(layer_psum_idx,psidx_cnt)

        layer_psum_idx=np.array(layer_psum_idx)
        
        # remove outlier fault coors
        layer_fault_coor,fc_cond=self._pop_outlier_idx(layer_fault_coor, layer_output_shape, get_cond_idx=True)
        if not np.all(fc_cond):
            if state=='fastgen' or state=='repetitive':
                layer_psum_idx=layer_psum_idx[fc_cond]
            elif state=='normal':
                if len(layer_psum_idx)==len(layer_fault_coor):
                    empty_fc_cond=None
                else:
                    empty_fc_cond=np.bitwise_and(fc_cond,psidx_cond)
                    empty_fc_cond=psidx_cond[empty_fc_cond]
                    empty_fc_cond=np.bitwise_not(empty_fc_cond)
                    layer_fault_coor=layer_fault_coor[empty_fc_cond]
                
        # deal with repetitive layer fault coors
        layer_fault_coor,uni_idx,rep_idx,cnt_idx=np.unique(layer_fault_coor,return_index=True,return_inverse=True,return_counts=True,axis=0)
        self.num_layer_fault_coor=len(layer_fault_coor)
        
        if not np.all(fc_cond):
            if state=='normal':
                if empty_fc_cond is not None:
                    empty_fc_cond=np.subtract(np.cumsum(empty_fc_cond),1)
                    uni_idx=np.searchsorted(empty_fc_cond,uni_idx)
            fc_cond=np.subtract(np.cumsum(fc_cond),1)
            uni_idx=np.searchsorted(fc_cond,uni_idx)

        # collapse duplicate coors
        if len(uni_idx)==len(rep_idx):
            new_fd_value=fd_value[np.remainder(uni_idx,self.num_fault_coor)]
        else:
            if self.pstate=='fastgen' and self.wstate=='fastgen' and self.istate=='fastgen':
                sorter=np.argsort(rep_idx)
                cnt_idx=np.cumsum(cnt_idx)[:-1]
                
                layer_psum_idx=layer_psum_idx[sorter]
                layer_psum_idx=np.split(layer_psum_idx,cnt_idx)
                
                #fd_value=fd_value[np.remainder(uni_idx,self.num_base_coor)]
                
                new_fd_value=list()
                for i,uidx in enumerate(uni_idx):
                    new_fv=fd_value[np.remainder(uidx,self.num_fault_coor)].copy()
                    new_fv['psum_idx']=np.concatenate(layer_psum_idx[i])
                    new_fd_value.append(new_fv)
            else:
                if state=='normal':
                    sorter=np.argsort(rep_idx)
                    cnt_idx=np.cumsum(cnt_idx)[:-1]
                    
                    layer_psum_idx=layer_psum_idx[sorter]
                    layer_psum_idx=np.split(layer_psum_idx,cnt_idx)
                                      
                    new_fd_value=list()
                    for i,uidx in enumerate(uni_idx):
                        new_fv=fd_value[np.remainder(uidx,self.num_fault_coor)].copy() 
                        new_fv['psum_idx']=layer_psum_idx[i]
                        new_fd_value.append(new_fv)

                else:
                    psum_idx_rep=[list() for _ in range(len(uni_idx))]
                    id_list_rep=[list() for _ in range(len(uni_idx))]
                    type_list_rep=[list() for _ in range(len(uni_idx))]
                    bit_list_rep=[list() for _ in range(len(uni_idx))]
                    param_list_rep=[list() for _ in range(len(uni_idx))]
                    
                    for i,repid in enumerate(rep_idx):
                        orig_i=np.remainder(i,self.num_fault_coor)
                        
                        psum_idx_rep[repid].append(layer_psum_idx[i])
    
                        if isinstance(fd_value[orig_i]['id'],int):
                            id_list_rep[repid].append(fd_value[orig_i]['id'])
                        else:
                            id_list_rep[repid]+=fd_value[orig_i]['id']
                        
                        if isinstance(fd_value[orig_i]['SA_type'],str):
                            type_list_rep[repid].append(fd_value[orig_i]['SA_type'])
                        else:
                            type_list_rep[repid]+=fd_value[orig_i]['SA_type']
                        
                        if isinstance(fd_value[orig_i]['SA_bit'],int):
                            bit_list_rep[repid].append(fd_value[orig_i]['SA_bit'])
                        else:
                            bit_list_rep[repid]+=fd_value[orig_i]['SA_bit']
                            
                        if isinstance(fd_value[orig_i]['param'],str):
                            param_list_rep[repid].append(fd_value[orig_i]['param'])
                        else:
                            param_list_rep[repid]+=fd_value[orig_i]['param']
                
                    new_fd_value=list()
                    for i,uidx in enumerate(uni_idx):
                        new_fv={'psum_idx':psum_idx_rep[i],
                                'id':id_list_rep[i],
                                'SA_type':type_list_rep[i],
                                'SA_bit':bit_list_rep[i],
                                'param':param_list_rep[i]}
                        new_fd_value.append(new_fv)

        layer_fault_coor_fd=list(zip(*layer_fault_coor.T))
        layer_fault_dict=dict(zip(layer_fault_coor_fd,new_fd_value))
        
        return layer_fault_dict
    
    def report_layer_map(self):
        return {'num_base_coor':self.num_base_coor,
                'num_fault_coor':self.num_fault_coor,
                'num_psum_idx':self.num_psum_idx,
                'num_layer_fault_coor':self.num_layer_fault_coor,
                'num_layer_psum_idx':self.num_layer_psum_idx}

    def clear(self):
        """
        Clear solved fault dict and mapping data generate along the solving process for next generation.
        """
        self.fault_num=None
        self.fault_dict_solved=dict()
        self.ofmap_coors=None
        self.bias_coors =None
        self.ifmap_coors=None
        self.wght_coors =None
        self.psum_coors =None
        self.ofmap_vl=None
        self.bias_vl =None
        self.ifmap_vl=None
        self.wght_vl =None
        self.psum_vl =None
        self.ofmap_id=None
        self.bias_id=None
        self.ifmap_id=None
        self.wght_id=None
        self.psum_id=None
        self.ostate=None
        self.bstate=None
        self.istate=None
        self.wstate=None
        self.pstate=None
        
    def clear_layer(self):
        """
        Clear tile to layer mapping configuration
        """
        self.num_base_coor=0
        self.num_layer_fault_coor=0
        self.num_layer_psum_idx=0
        self.num_psum_idx=0

# this function is depricated   
def _solve_correspond_io(ofmap_tile, wght_tile, ifmap_tile, fault_num=None, print_detail=False):
    """ Solving the PE array to Tile mapping fault dictionarys.
        Regarding ofmap, ifmap, weight, partial sum, bias fault dictionarys, 
        and find the relation between them. Give fault info (psum index).
    
    """
    print('Warning: this function is depricated, use class io_data_solver instead!')
    
    ofmap_fd=ofmap_tile.fault_dict
    ifmap_fd=ifmap_tile.fault_dict
    wght_fd=wght_tile.fault_dict
    psum_fd=ofmap_tile.psum_fault_dict
    bias_fd=wght_tile.bias_fault_dict
    
    #ofmap_coors=np.array(list(ofmap_fd.keys()))
    ifmap_coors=np.array(list(ifmap_fd.keys()))
    wght_coors =np.array(list(wght_fd.keys()))
    psum_coors =np.array(list(psum_fd.keys()))
    #bias_coors =np.array(list(bias_fd.keys()))
    
    ofmap_vl=np.array(list(ofmap_fd.values()))
    ifmap_vl=np.array(list(ifmap_fd.values()))
    wght_vl =np.array(list(wght_fd.values()))
    psum_vl =np.array(list(psum_fd.values()))
    bias_vl =np.array(list(bias_fd.values()))
    
    def state_setting(faultvalue):
        if len(faultvalue)>0:
            if isinstance(faultvalue[0]['id'],np.ndarray):
                state='fastgen'
                idlist=np.array([info['id'] for info in faultvalue])
                maxx=np.max(np.concatenate(idlist))
                if idlist.dtype==np.object:
                    idl_cnt=np.array([len(i) for i in idlist])
                    idl_cnt=np.cumsum(idl_cnt)-1
                    idlist=np.concatenate(idlist)
                    idlist=(idlist,idl_cnt)
            elif isinstance(faultvalue[0]['id'],int):
                state='normal'
                idlist=[info['id'] for info in faultvalue]
                maxx=max(idlist)
            elif isinstance(faultvalue[0]['id'],list):
                state='repetitive'
                idlist=[info['id'] for info in faultvalue]
                maxx=max([max(idl) for idl in idlist])
        else:
            idlist=list()
            state=None
            maxx=-1
        
        return idlist,state,maxx
    
    ofmap_id,ostate,maxo=state_setting(ofmap_vl)
    ifmap_id,istate,maxi=state_setting(ifmap_vl)
    wght_id,wstate,maxw=state_setting(wght_vl)
    psum_id,pstate,maxp=state_setting(psum_vl)
    bias_id,bstate,maxb=state_setting(bias_vl)
    
    if fault_num==None:
        fault_num=max([maxo,maxi,maxw,maxp,maxb])+1
        
    new_solved_fd=dict()
    
    def state_idxget(idlist,faultcoors,faultvalue,state,faultid,paramin):
        if state is None:
            idx=None
            param=None
            faultindex=None
            
        elif state=='fastgen':
            if isinstance(idlist,np.ndarray):
                idx=np.argwhere(idlist==faultid)
                if len(idx)==0:
                    idx=None
                    param=None
                    faultindex=None
                else:
                    idx=idx[0][0]
                    param=faultvalue[idx]['param']
                    faultindex=faultcoors[idx]
                    
            elif isinstance(idlist,tuple):
                idl_cnt=idlist[1]
                idlist=idlist[0]
                
                idx=np.argwhere(idlist==faultid)
                if len(idx)==0:
                    idx=None
                    param=None
                    faultindex=None
                else:
                    idx=idx[0][0]
                    idx=np.searchsorted(idl_cnt,idx)
                    param=faultvalue[idx]['param']
                    faultindex=faultcoors[idx]
            
        elif state=='normal':
            try:
                idx=idlist.index(i)
                param=faultvalue[idx]['param']
                faultindex=faultcoors[idx]
            except ValueError:
                idx=None
                param=None
                faultindex=None
                
        elif state=='repetitive':
            idx=None
            param=None
            faultindex=None
            for ii,idl in enumerate(idlist):
                if faultid in idl:
                    ii2=idl.index(faultid)
                    idx=[ii,ii2]
                    param=faultvalue[ii]['param'][ii2]
                    faultindex=faultcoors[ii]
        
        if paramin is not None:
            param=paramin
        
        return idx,param,faultindex
    
    def state_make_new_fd(state,faultid,paramin,opindex,windex,iindex,faultvalue,idx,newfd):
        try:
            # psum index (Batch, Output Channel, Ofmap Row, Ofmap Column, Ifmap Channel, Kernel Row, Kernel Column, Ifmap Row, Ifmap Column)
            psidx=tuple(np.concatenate([opindex[[0,3,1,2]],windex[[2,0,1]],iindex[[1,2]]]))
            if state is None:
                pass
            elif state=='fastgen':
                try:
                    newfv=newfd[tuple(opindex)]
                    newfv['psum_idx'].append(psidx)
                except KeyError:
                    newfv=faultvalue[idx].copy()
                    newfv.update({'psum_idx':[psidx]})
                    newfd[tuple(opindex)]=newfv
            elif state=='normal':
                newfv=faultvalue[idx].copy()
                newfv.update({'psum_idx':psidx})
                newfd[tuple(opindex)]=newfv
            elif state=='repetitive':
                try:
                    newfv=newfd[tuple(opindex)]
                    newfv['SA_bit'].append(faultvalue[idx[0]]['SA_bit'][idx[1]])
                    newfv['SA_type'].append(faultvalue[idx[0]]['SA_type'][idx[1]])
                    newfv['param'].append(paramin)
                    newfv['psum_idx'].append(psidx)
                    newfv['id'].append(faultid)
                except KeyError:
                    newfv={'SA_bit':[faultvalue[idx[0]]['SA_bit'][idx[1]]],
                           'SA_type':[faultvalue[idx[0]]['SA_type'][idx[1]]],
                           'param':[paramin],
                           'psum_idx':[psidx],
                           'id':[faultid]}
                    newfd[tuple(opindex)]=newfv
        except TypeError:
            pass
        

    #TODO 
    # the fast access 'fast gen' method
    if pstate=='fastgen' and wstate=='fastgen' and istate=='fastgen':
        # solve psum, use as basis for fault id search
        if isinstance(psum_id,np.ndarray):
            pshape_cnt=psum_id.shape
            if len(pshape_cnt)>1:
                psum_id=psum_id.flatten()
        elif isinstance(psum_id,tuple):
            pshape_cnt=psum_id[1]
            psum_id=psum_id[0]
        
        # solve ifmap
        if isinstance(ifmap_id,np.ndarray):
            ishape=ifmap_id.shape
            if len(ishape)>1:
                ifmap_id=ifmap_id.flatten()
                sorter_i=np.argsort(ifmap_id)
                search_i=np.searchsorted(ifmap_id,psum_id,sorter=sorter_i)
                search_i=sorter_i[search_i]
                search_i=np.floor_divide(search_i,ishape[1])
                ifmap_index=ifmap_coors[search_i]
            else:
                sorter_i=np.argsort(ifmap_id)
                search_i=np.searchsorted(ifmap_id,psum_id,sorter=sorter_i)
                search_i=sorter_i[search_i]
                ifmap_index=ifmap_coors[search_i]
        elif isinstance(ifmap_id,tuple):
            ifmap_idl_cnt=ifmap_id[1]
            ifmap_id=ifmap_id[0]
            sorter_i=np.argsort(ifmap_id)
            search_i=np.searchsorted(ifmap_id,psum_id,sorter=sorter_i)
            search_i=sorter_i[search_i]
            search_i=np.searchsorted(ifmap_idl_cnt,search_i)
            ifmap_index=ifmap_coors[search_i]
            
        # solve weight
        if isinstance(wght_id,np.ndarray):
            wshape=wght_id.shape
            if len(wshape)>1:
                wght_id=wght_id.flatten()
                sorter_w=np.argsort(wght_id)
                search_w=np.searchsorted(wght_id,psum_id,sorter=sorter_w)
                search_w=sorter_w[search_w]
                search_w=np.floor_divide(search_w,wshape[1])
                wght_index=wght_coors[search_w]
            else:
                sorter_w=np.argsort(wght_id)
                search_w=np.searchsorted(wght_id,psum_id,sorter=sorter_w)
                search_w=sorter_w[search_w]
                wght_index=wght_coors[search_w]
        elif isinstance(wght_id,tuple):
            wght_idl_cnt=wght_id[1]
            wght_id=wght_id[0]
            sorter_w=np.argsort(wght_id)
            search_w=np.searchsorted(wght_id,psum_id,sorter=sorter_w)
            search_w=sorter_w[search_w]
            search_w=np.searchsorted(wght_idl_cnt,search_w)
            wght_index=wght_coors[search_w]
        
        # build psum_idx
        if isinstance(pshape_cnt,tuple):
            if len(pshape_cnt)>1:
                outpsum_index=np.repeat(psum_coors,pshape_cnt[1],axis=0)
        elif isinstance(pshape_cnt,np.ndarray):
            cnt0=pshape_cnt[0]+1
            idlrep=pshape_cnt[1:]-pshape_cnt[:-1]
            idlrep=np.concatenate([[cnt0],idlrep])
            idlconstruct=np.repeat(np.arange(len(idlrep)),idlrep)
            outpsum_index=psum_coors[idlconstruct]
            
        psum_index=np.concatenate([outpsum_index[:,[0,3,1,2]],wght_index[:,[2,0,1]],ifmap_index[:,[1,2]]],axis=1)
        if isinstance(pshape_cnt,tuple):
            if len(pshape_cnt)>1:
                psum_index=np.split(psum_index,pshape_cnt[0])
        elif isinstance(pshape_cnt,np.ndarray):
            psum_index=np.split(psum_index,idlrep)

        for i,opidx in enumerate(psum_coors):
            newfv=psum_vl[i].copy()
            newfv.update({'psum_idx':psum_index[i]})
            new_solved_fd[tuple(opidx)]=newfv
            
        return new_solved_fd
    
    if print_detail:
        pbar=tqdm.tqdm(desc='\tSolved Fault', total=fault_num, leave=False)
    
    for i in range(fault_num):
        param=None
        
        pidx,param,psum_index=state_idxget(psum_id,psum_coors,psum_vl,pstate,i,param)
        widx,param,wght_index=state_idxget(wght_id,wght_coors,wght_vl,wstate,i,param)
        iidx,param,ifmap_index=state_idxget(ifmap_id,ifmap_coors,ifmap_vl,istate,i,param)
        #oidx,param,ofmap_index=state_idxget(ofmap_id,ofmap_coors,ofmap_vl,ostate,i,param)
        #bidx,param,bias_index=state_idxget(bias_id,bias_coors,bias_vl,bstate,i,param)
        
        # partial sum index (batch, Tn, TrO, TcO, Tm, TrK, TcK, TrI, TcI)
        if param is not None:
            state_make_new_fd(pstate,i,param,psum_index,wght_index,ifmap_index,psum_vl,pidx,new_solved_fd)
            
#            if param=='ifmap_in' and iidx is not None:
#                state_make_new_fd(istate,i,param,psum_index,wght_index,ifmap_vl,iidx,new_ifmap_fd,ifmap_index)
#                            
#            elif param=='ifmap_out' and iidx is not None:
#                state_make_new_fd(istate,i,param,psum_index,wght_index,ifmap_vl,iidx,new_ifmap_fd,ifmap_index)
#            
#            elif param=='wght_in' and widx is not None:
#                state_make_new_fd(wstate,i,param,psum_index,wght_index,wght_vl,widx,new_wght_fd,wght_index)
#                            
#            elif param=='wght_out'and widx is not None:
#                state_make_new_fd(wstate,i,param,psum_index,wght_index,wght_vl,widx,new_wght_fd,wght_index)
#                                
#            elif param=='psum_in' and pidx is not None:
#                if bidx is None:                                   
#                    state_make_new_fd(pstate,i,param,psum_index,wght_index,psum_vl,pidx,new_ofmap_fd,psum_index)
#                else:
#                    new_bias_fd[tuple(bias_index)]=bias_vl[bidx]
                            
#                try:
#                    if pstate=='repetitive':
#                        pidxx=pidx[0]
#                    else:
#                        pidxx=pidx
#
#                    if psum_vl[pidxx]['edge']:
#                        if bidx is None:
#                            psum_index=np.ravel_multi_index(np.array(psum_index)[[0,3,2,1]],np.array(ofmap_tile.tile_shape)[[0,3,2,1]])
#                            psum_index-=1
#                            if psum_index<0:
#                                continue
#                            
#                            psum_index=np.unravel_index(psum_index,np.array(ofmap_tile.tile_shape)[[0,3,2,1]])
#                            psum_index=np.array(psum_index)[[0,3,2,1]]
#                            
#                            wght_index=np.ravel_multi_index(np.array(wght_index)[[3,2,1,0]],np.array(wght_tile.tile_shape)[[3,2,1,0]])
#                            wght_index-=1
#                            if wght_index<0:
#                                continue
#                            
#                            wght_index=np.unravel_index(wght_index,np.array(ofmap_tile.tile_shape)[[3,2,1,0]])
#                            wght_index=np.array(wght_index)[[3,2,1,0]]
#                            
#                        
#                            state_make_new_fd(pstate,i,param,psum_index,wght_index,psum_vl,pidx,new_ofmap_fd,psum_index)
#                            
#                        else:
#                            new_bias_fd[tuple(bias_index)]=bias_vl[bidx]
#                except:
#                    state_make_new_fd(pstate,i,param,psum_index,wght_index,psum_vl,pidx,new_ofmap_fd,psum_index)
#            
#            elif param=='psum_out' and pidx is not None:
#                if oidx is None:
#                    state_make_new_fd(pstate,i,param,psum_index,wght_index,psum_vl,pidx,new_ofmap_fd,psum_index)
#                    new_ofmap_fd[tuple(psum_index)].update({'ofmap':False})
#                else:
#                    state_make_new_fd(ostate,i,param,ofmap_index,wght_index,ofmap_vl,oidx,new_ofmap_fd,ofmap_index)                   
#                    new_ofmap_fd[tuple(ofmap_index)].update({'ofmap':True})
                    
        if print_detail:
            pbar.update()
        
    if print_detail:
        pbar.close()
    
    return new_solved_fd
            