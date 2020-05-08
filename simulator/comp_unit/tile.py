# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 16:22:01 2019

@author: Yung-Yu Tsai

DNN tiling for computation unit fault mapping
"""

import numpy as np
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
    
    def extract_patches_idx(self, index, fmap_shape, ksizes, strides=(1,1,1,1), dilation_rates=(1,1,1,1), padding='valid', edge_fill=False, get_cond_idx=False):
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
        extracted_psum=np.add(np.subtract(np.multiply(extracted_psum,ksizes[1]*ksizes[2]),cond_idx[:,1]),ksizes[1]*ksizes[2]-1)
        extracted_psum=np.expand_dims(extracted_psum,-1)
        
        extracted_index=np.concatenate([extracted_batch,extracted_2D,extracted_psum],axis=1)

        if get_cond_idx:
            return extracted_index, cond_idx
        else:
            return extracted_index
    
    def return_patches_idx(self, index, fmap_shape, ksizes, strides=(1,1,1,1), dilation_rates=(1,1,1,1), padding='valid', edge_fill=False):
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
        
        idx_patches_candidate=np.remainder(extracted_psum,ksizes[1]*ksizes[2])
#        idx_patches_candidate=np.subtract(ksizes[1]*ksizes[2]-1,idx_patches_candidate)
        
        ifchannel_idx=np.floor_divide(extracted_psum,ksizes[1]*ksizes[2])
        
        if strides[1]>1 or strides[2]>1:
            returned_2D=np.multiply(returned_2D,[strides[1:3]])
        if padding=='same':
            returned_2D=np.subtract(returned_2D,[[np.floor_divide(dilated_ksize_row,2),np.floor_divide(dilated_ksize_col,2)]])

        base_kcoor_reduction=list(np.ndindex(ksizes[1:3]))
#        base_kcoor_reduction.reverse()
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
        
        if not isinstance(slicing_dims,tuple) or len(slicing_dims)!=len(self.expand_shape):
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
            
    def expand_extract_patches(self, ksizes, strides=(1,1,1,1), dilation_rates=(1,1,1,1), padding='valid', edge_fill=False, 
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
        self.reshape_patches=reshape_patches
        
        if len(ksizes)!=4:
            raise ValueError('ksize length must be 4, but got %d'%len(ksizes))
        if len(strides)!=4:
            raise ValueError('strides length must be 4, but got %d'%len(strides))
        if len(dilation_rates)!=4:
            raise ValueError('dilation rate length must be 4, but got %d'%len(dilation_rates))
            
        if dataflow_pre_plan:
            self.ksizes=ksizes
            self.strides=strides
            self.dilation_rates=dilation_rates
            self.padding=padding
            self.edge_fill=edge_fill
        
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
                
        if not isinstance(slicing_dims,tuple) or len(slicing_dims)!=len(self.expand_shape):
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
        
    def shrink_return_patches(self,psum=False):
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
                                           edge_fill=self.edge_fill)
        
        # pop inalid edge condition that doesn't exist in tile
        orig_coors,cond_idx=self.pop_outlier_idx(orig_coors,list(self.tile_shape),get_cond_idx=True)
        fault_info=fault_info[cond_idx]
        
        # deal with repeative orig_coors
        orig_coors,uni_idx,rep_idx,cnt_idx=np.unique(orig_coors,return_index=True,return_inverse=True,return_counts=True,axis=0)
        
        if len(uni_idx)==len(rep_idx):
            pass
        else:
            id_list_rep=[list() for _ in range(len(uni_idx))]
            bit_list_rep=[list() for _ in range(len(uni_idx))]
            param_list_rep=[list() for _ in range(len(uni_idx))]
            
            for i,repid in enumerate(rep_idx):
                if isinstance(fault_info[i]['id'],int):
                    id_list_rep[repid].append(fault_info[i]['id'])
                else:
                    id_list_rep[repid]+=fault_info[i]['id']
                    
                if isinstance(fault_info[i]['SA_bit'],int):
                    bit_list_rep[repid].append(fault_info[i]['SA_bit'])
                else:
                    bit_list_rep[repid]+=fault_info[i]['SA_bit']
                    
                if isinstance(fault_info[i]['param'],int):
                    param_list_rep[repid].append(fault_info[i]['param'])
                else:
                    param_list_rep[repid]+=fault_info[i]['param']
            
            fault_info=fault_info[uni_idx]
            for i in range(len(uni_idx)):
                fault_info[i]['id']=id_list_rep[i]
                fault_info[i]['SA_bit']=bit_list_rep[i]
                fault_info[i]['param']=param_list_rep[i]

        orig_coor_fd=list(zip(*orig_coors.T))
        new_fault_dict=dict(zip(orig_coor_fd,fault_info))
        if not psum:
            self.fault_dict=new_fault_dict
        else:
            self.psum_fault_dict=new_fault_dict
        
        return new_fault_dict
            
    def expand_slice_bias(self, slice_width, dataflow_pre_plan=False):
        """ Data expansion before put into PE array. 
            The data are being cut into many pieces then fit into PE. Different slices calculate in different clock cycle.
        
        # Arguments                            
            slice_width: Integer. The expected slice width to be expand into. 
            
            dataflow_pre_plan: Bool. Plan the dataflow model ahead. If True there will be no actual Tile to PEarray fault dictionary list transformation.
                Only save the expansion configuration for later PEarray to Tile transform.
                            
        # Returns
            Converted fault dictionary.

        """
        if self.is_fmap:
            raise TypeError('This is feature maps tile, no bias!')
        self.use_bias=True
        self.expansion=True
            
        self.bias_slice_shape=(slice_width, int(np.ceil(self.Tn/slice_width)))
        
        if not dataflow_pre_plan:
            orig_coors=np.array(list(self.bias_fault_dict.keys()))
            fault_info=list(self.bias_fault_dict.values())
            
            sliced_coors=np.concatenate([np.remainder(orig_coors,slice_width),np.floor_divide(orig_coors,slice_width)],axis=1)
            
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

     
def solve_correspond_io(ofmap_tile, wght_tile, ifmap_tile, fault_num=None, fast_gen=False):
    """ Solving the PE array to Tile mapping fault dictionarys.
        Regarding ofmap, ifmap, weight, partial sum, bias fault dictionarys, 
        and find the relation between them. Give fault info (psum index).
    
    """
    ofmap_fd=ofmap_tile.fault_dict
    ifmap_fd=ifmap_tile.fault_dict
    wght_fd=wght_tile.fault_dict
    psum_fd=ofmap_tile.psum_fault_dict
    bias_fd=wght_tile.bias_fault_dict
    
    ofmap_coors=np.array(list(ofmap_fd.keys()))
    ifmap_coors=np.array(list(ifmap_fd.keys()))
    wght_coors =np.array(list(wght_fd.keys()))
    psum_coors =np.array(list(psum_fd.keys()))
    bias_coors =np.array(list(bias_fd.keys()))
    
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
                maxx=np.max(idlist)
            elif isinstance(faultvalue[0]['id'],int):
                state='normal'
                idlist=[info['id'] for info in faultvalue]
                maxx=max(idlist)
            elif isinstance(faultvalue[0]['id'],list):
                state='repeative'
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
        
    new_ofmap_fd=dict()
    new_ifmap_fd=dict()
    new_wght_fd=dict()
    new_bias_fd=dict()
    
    def state_idxget(idlist,faultcoors,faultvalue,state,faultid,paramin):
        if state is None:
            idx=None
            param=None
            faultindex=None
            
        elif state=='fastgen':
            idx=np.argwhere(idlist==faultid)
            if len(idx)==0:
                idx=None
                param=None
                faultindex=None
            else:
                idx=idx[0][0]
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
                
        elif state=='repeative':
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
    
    def state_make_new_fd(state,faultid,paramin,opindex,windex,faultvalue,idx,newfd,dataindex):
        try:
            psidx=tuple(np.append(opindex[[0,3,1,2]],windex[[2,0,1]]))
            if state is None:
                pass
            elif state=='fastgen':
                try:
                    newfv=newfd[tuple(dataindex)]
                    newfv['psum_idx'].append(psidx)
                except KeyError:
                    newfv=faultvalue[idx].copy()
                    newfv.update({'psum_idx':[psidx]})
                    newfd[tuple(dataindex)]=newfv
            elif state=='normal':
                newfv=faultvalue[idx].copy()
                newfv.update({'psum_idx':psidx})
                newfd[tuple(dataindex)]=newfv
            elif state=='repeative':
                newfv={'SA_bit':faultvalue[idx[0]]['SA_bit'][idx[1]],
                       'SA_type':faultvalue[idx[0]]['SA_type'][idx[1]],
                       'param':paramin,
                       'psum_idx':psidx,
                       'id':faultid}
                newfd[tuple(dataindex)]=newfv
        except TypeError:
            pass
        
    for i in range(fault_num):
        param=None
        
        pidx,param,psum_index=state_idxget(psum_id,psum_coors,psum_vl,pstate,i,param)
        widx,param,wght_index=state_idxget(wght_id,wght_coors,wght_vl,wstate,i,param)
        iidx,param,ifmap_index=state_idxget(ifmap_id,ifmap_coors,ifmap_vl,istate,i,param)
        oidx,param,ofmap_index=state_idxget(ofmap_id,ofmap_coors,ofmap_vl,ostate,i,param)
        bidx,param,bias_index=state_idxget(bias_id,bias_coors,bias_vl,bstate,i,param)
        
        # partial sum index (batch, Tn, TrO, TcO, Tm, TrK, Tck)
        if param is not None:
            if param=='ifmap_in' and iidx is not None:
                state_make_new_fd(istate,i,param,psum_index,wght_index,ifmap_vl,iidx,new_ifmap_fd,ifmap_index)
                            
            elif param=='ifmap_out' and iidx is not None:
                try:
                    if istate=='repeative':
                        if ifmap_vl[iidx[0]]['edge']:
                            pass
                    else:
                        if ifmap_vl[iidx]['edge']:
                            pass
                except:
                    state_make_new_fd(istate,i,param,psum_index,wght_index,ifmap_vl,iidx,new_ifmap_fd,ifmap_index)
            
            elif param=='wght_in' and widx is not None:
                state_make_new_fd(wstate,i,param,psum_index,wght_index,wght_vl,widx,new_wght_fd,wght_index)
                            
            elif param=='wght_out'and widx is not None:
                try:
                    if wstate=='repeative':
                        if wght_vl[widx[0]]['edge']:
                            pass
                    else:
                        if wght_vl[widx]['edge']:
                            pass
                except:
                    state_make_new_fd(wstate,i,param,psum_index,wght_index,wght_vl,widx,new_wght_fd,wght_index)
                                
            elif param=='psum_in' and pidx is not None:
                try:
                    if pstate=='repeative':
                        pidxx=pidx[0]
                    else:
                        pidxx=pidx

                    if psum_vl[pidxx]['edge']:
                        if bidx is None:
                            psum_index=np.ravel_multi_index(np.array(psum_index)[[0,3,2,1]],np.array(ofmap_tile.tile_shape)[[0,3,2,1]])
                            psum_index-=1
                            if psum_index<0:
                                continue
                            
                            psum_index=np.unravel_index(psum_index,np.array(ofmap_tile.tile_shape)[[0,3,2,1]])
                            psum_index=np.array(psum_index)[[0,3,2,1]]
                            
                            wght_index=np.ravel_multi_index(np.array(wght_index)[[3,2,1,0]],np.array(wght_tile.tile_shape)[[3,2,1,0]])
                            wght_index-=1
                            if wght_index<0:
                                continue
                            
                            wght_index=np.unravel_index(wght_index,np.array(ofmap_tile.tile_shape)[[3,2,1,0]])
                            wght_index=np.array(wght_index)[[3,2,1,0]]
                            
                        
                            state_make_new_fd(pstate,i,param,psum_index,wght_index,psum_vl,pidx,new_ofmap_fd,psum_index)
                            
                        else:
                            new_bias_fd[tuple(bias_index)]=bias_vl[bidx]
                except:
                    state_make_new_fd(pstate,i,param,psum_index,wght_index,psum_vl,pidx,new_ofmap_fd,psum_index)
            
            elif param=='psum_out' and pidx is not None:
                if oidx is None:
                    state_make_new_fd(pstate,i,param,psum_index,wght_index,psum_vl,pidx,new_ofmap_fd,psum_index)
                    new_ofmap_fd[tuple(psum_index)].update({'ofmap':False})
                else:
                    state_make_new_fd(ostate,i,param,ofmap_index,wght_index,ofmap_vl,oidx,new_ofmap_fd,ofmap_index)                   
                    new_ofmap_fd[tuple(ofmap_index)].update({'ofmap':True})
    
    ofmap_tile.fault_dict=new_ofmap_fd
    ifmap_tile.fault_dict=new_ifmap_fd
    wght_tile.fault_dict=new_wght_fd
    wght_tile.bias_fault_dict=new_bias_fd
            