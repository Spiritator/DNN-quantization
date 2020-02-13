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
        self.expand_prior_orig=None
        self.expand_prior_targ=None
        self.slice_shape=None
        self.slicing_dims=None
        self.slice_permute=None
        self.permuted_shape=None
        self.reshape_patches=False
        self.fault_dict_rehsaped=dict()
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
            
    def get_permuted_shape(self, orig_shape, slicing_dims):
        div_dims=np.array([],dtype=int)
        for dim in slicing_dims:
            if dim==0:
                div_dims=np.append(div_dims,1)
            else:
                div_dims=np.append(div_dims,dim)
                
        orig_shape=np.array(orig_shape)
        permute_shape=np.ceil(np.divide(orig_shape,div_dims))
        permute_shape=permute_shape.astype(int)
        return permute_shape,div_dims
            
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

        permute_shape,div_dims=self.get_permuted_shape(orig_shape, slicing_dims)
        
        permute_dims=np.floor_divide(index,div_dims)
        mapping_dims=np.remainder(index,div_dims)
        if len(index.shape)==1:
            mapping_dims=mapping_dims[np.argwhere(np.array(slicing_dims)>0)]
            tclk=np.ravel_multi_index(permute_dims[slices_permute],permute_shape[slices_permute])
            if len(mapping_dims.shape)==1:
                mapping_dims=np.expand_dims(mapping_dims,-1)
            mapping_dims=np.append(mapping_dims,tclk)
        else:
            mapping_dims=mapping_dims[:,np.squeeze(np.argwhere(np.array(slicing_dims)>0))]
            tclk=np.ravel_multi_index(permute_dims.T[slices_permute],permute_shape[slices_permute])
            if len(mapping_dims.shape)==1:
                mapping_dims=np.expand_dims(mapping_dims,-1)
            mapping_dims=np.append(mapping_dims,np.expand_dims(tclk,-1),axis=-1)
            
        return mapping_dims
    
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
            cond_arg=idx_patches_candidate[:,:,0]%strides[1]==0 #row
            cond_tmp=idx_patches_candidate[:,:,1]%strides[2]==0 #col
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
            cond_tmp=idx_patches_candidate[:,:,0]>=0-(dilated_ksize_row-1) # left edge
            cond_arg=np.bitwise_and(cond_arg,cond_tmp) 
            
            cond_tmp=idx_patches_candidate[:,:,1]>=0-(dilated_ksize_col-1) # up edge
            cond_arg=np.bitwise_and(cond_arg,cond_tmp) 
            
            if not edge_fill:
                cond_tmp=idx_patches_candidate[:,:,0]<(fmap_shape[1]) # right edge
                cond_arg=np.bitwise_and(cond_arg,cond_tmp) 
                
                cond_tmp=idx_patches_candidate[:,:,1]<(fmap_shape[2]) # down edge
                cond_arg=np.bitwise_and(cond_arg,cond_tmp) 
                
        else:
            raise ValueError('padding must be either \'valid\' or \'same\'.')
            
        extracted_2D=idx_patches_candidate[cond_arg]
        if padding=='same':
            extracted_2D=np.add(extracted_2D,[[dilated_ksize_row-1,dilated_ksize_col-1]])
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

    def expand_reshape_data(self, orig_prior, expect_shape, reshape_prior, slicing_dims, slices_permute):
        """ Data expansion before put into PE array. Usually used for ifmap and weight reuse. 
            The data may be cut into many pieces than fit into PE. Different slices calculate in different clock cycle.
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
            raise TypeError('slicing_dims must be in length %d, but get length %d'%(len(self.expand_shape),len(reshape_prior)))
        self.expand_prior_targ=reshape_prior
        
        if not isinstance(slicing_dims,tuple) or len(slicing_dims)!=len(self.expand_shape):
            raise TypeError('slicing_dims must be in length %d, but get length %d'%(len(self.expand_shape),len(slicing_dims)))
        self.slicing_dims=slicing_dims
        self.slice_shape=np.array(slicing_dims)
        self.slice_shape=tuple(self.slice_shape[self.slice_shape>0])
        
        if len(slices_permute)!=len(self.expand_shape):
            raise TypeError('slice_permute must be in length %d, but get length %d'%(len(self.expand_shape),len(slices_permute)))
        self.slice_permute=slices_permute
        
        orig_coors=np.array(list(self.fault_dict.keys()))
        reshaped_coors=self.reshape_ravel_idx(orig_coors,
                                              source_shape=self.tile_shape,
                                              source_prior=self.expand_prior_orig,
                                              target_shape=self.expand_shape,
                                              target_prior=self.expand_prior_targ)
        
        reshaped_coors_fd=list(zip(*reshaped_coors.T))
        fault_info=list(self.fault_dict.values())
        self.fault_dict_rehsaped=dict(zip(reshaped_coors_fd,fault_info))
        
        self.permuted_shape,_=self.get_permuted_shape(reshaped_coors,slicing_dims)
        permuted_coors=self.slice_permute_idx(reshaped_coors,
                                              orig_shape=self.expand_shape,
                                              slicing_dims=slicing_dims,
                                              slices_permute=slices_permute)
        
        permuted_coor_fd=list(zip(*permuted_coors.T))
        self.fault_dict_expand=dict(zip(permuted_coor_fd,fault_info))
        
        return self.fault_dict_expand
    
    def expand_extract_patches(self, ksizes, strides=(1,1,1,1), dilation_rates=(1,1,1,1), padding='valid', edge_fill=False, 
            reshape_patches=False, patches_prior=None, expect_shape=None, reshape_prior=None, slicing_dims=None, slices_permute=None):
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
        
        if len(slices_permute)!=len(self.expand_shape):
            raise TypeError('slice_permute must be in length %d, but get length %d'%(len(self.expand_shape),len(slices_permute)))
        self.slice_permute=slices_permute
        
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
        
        self.permuted_shape,_=self.get_permuted_shape(reshaped_coors,slicing_dims)
        permuted_coors=self.slice_permute_idx(reshaped_coors,
                                              orig_shape=self.expand_shape,
                                              slicing_dims=slicing_dims,
                                              slices_permute=slices_permute)
        
        permuted_coor_fd=list(zip(*permuted_coors.T))
        self.fault_dict_expand=dict(zip(permuted_coor_fd,fault_info))
        
        return self.fault_dict_expand
        

        
# TODO
# Tilted problem        
    