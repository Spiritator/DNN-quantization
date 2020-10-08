# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 13:36:17 2019

@author: Yung-Yu Tsai

DNN tiling for memory fault mapping
"""

import numpy as np

class tile:
    """The tile of a DNN feature map or weights

    Arguments
    ---------
    tile_shape: Tuple. 
        The shape of tile.
    Tm: Integer. 
        The size of tile on the input feature map dimension (weight) or channel dimention (feature map).
    Tn: Integer. 
        The size of tile on the output feature map dimension (weight) or batch dimention (feature map).
    Tr: Integer. 
        The size of tile on the kernel row dimension or feature map row dimention.
    Tc: Integer. 
        The size of tile on the kernel column dimension or feature map column dimention.
    is_fmap: Bool. 
        The tile is feature map tile or weight tile.
    wl: Integer. 
        The word length of DNN model parameter.
    row_prior: List of Strings. 
        The priority of memory mapping in the memory row dimension. Consist of 'Tm', 'Tn', 'Tr', 'Tc'.
    col_prior: List of Strings.     
        The priority of memory mapping in the memory column dimension. Consist of 'Tm', 'Tn', 'Tr', 'Tc'.

    """
    def __init__(self, tile_shape, is_fmap, wl=32, row_prior=[], col_prior=[]):
        """ Tile initializer """
        if not isinstance(is_fmap,bool):
            raise TypeError('Argument is_fmap must be True (feature map tile) or False (weight tile)')
        if len(tile_shape) != 4:
            raise ValueError('The argument tile_shape must be in Tuple dtype and have length 4 but got length %d'%len(tile_shape))
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
        
    def check_prior(self):
        """ Check the given priorty argument is valid """
        if not isinstance(self.row_prior,list) or not isinstance(self.col_prior,list) or len(self.row_prior)!=self.shape_len or len(self.col_prior)!=self.shape_len:
            raise ValueError('The argument row_prior and col_prior must be in list dtype and have length %d but got length %d and %d'%(self.shape_len,len(self.row_prior),len(self.col_prior)))
                
        for i in range(self.shape_len):
            if self.row_prior[i] not in self.prior_element:
                raise ValueError('The argument row_prior must be in list %s'%(str(self.prior_element)))
            if self.col_prior[i] not in self.prior_element:
                raise ValueError('The argument col_prior must be in list %s'%(str(self.prior_element)))
    
    def priorexchange(self,prior):
        """ Transfer String priority axis to its corresponding size and axis index.
        
        Parameters
        ----------
        prior : String. One of 'Tm', 'Tn', 'Tr', 'Tc'.

        Returns
        -------
        Integer
            Size of prior.
        Integer
            Index of prior in shape.
        """
        if prior not in self.prior_element:
            raise ValueError('The argument row_prior must be in list %s'%(str(self.prior_element)))
            
        if self.is_fmap:
            axis_idex=[3,0,1,2]
        else:
            axis_idex=[2,3,0,1]
            
        if prior == 'Tm':
            return self.Tm,axis_idex[0]
        elif prior == 'Tn':
            return self.Tn,axis_idex[1]
        elif prior == 'Tr':
            return self.Tr,axis_idex[2]
        elif prior == 'Tc':
            return self.Tc,axis_idex[3]
        
    def get_tile_dims_prior(self,prior_list):
        """ Get the shape and its dimension order by the given prior list.
            Use for rehape data index.

        Parameters
        ----------
        prior_list : List of String
            The wanted new prior order.

        Returns
        -------
        Tuple of Intger
            The new shape.
        dims_reorder : List of Integer
            The dimesion order to original shape.
        """
        if self.is_fmap:
            axis_idex=[3,0,1,2]
        else:
            axis_idex=[2,3,0,1]

        dims_prior=list()
        dims_reorder=list()
        for prior in reversed(prior_list):
            if prior == 'Tm':
                dims_prior.append(self.Tm)
                dims_reorder.append(axis_idex[0])
            elif prior == 'Tn':
                dims_prior.append(self.Tn)
                dims_reorder.append(axis_idex[1])
            elif prior == 'Tr':
                dims_prior.append(self.Tr)
                dims_reorder.append(axis_idex[2])
            elif prior == 'Tc':
                dims_prior.append(self.Tc)
                dims_reorder.append(axis_idex[3])
                
        return tuple(dims_prior),dims_reorder
        
    def coor_tile_recursive_call(self,coor,prior_list,prior_index,increase=False):
        """ The moving call for moving coordinate a step on tile shape.
        
        Parameters
        ----------
        coor : Tuple of integer
            The coordinate on tile.
        prior_list : List of String
            The permutation of tile data shape.
        prior_index : List of Integer
            The prior reorder index.
        increase : Bool, optional
            Moving step is forward to the index or backward. The default is False.

        Returns
        -------
        coor : Tuple of integer
            The moved coordinate on tile.
        """
        T_size,T_index=self.priorexchange(prior_list[prior_index])
        if increase:
            if coor[T_index] < T_size-1:
                coor[T_index]+=1
            else:
                if prior_index==len(prior_list)-1:
                    raise ValueError('Index out of tile range !!')
                    
                coor[T_index]=0
                coor=self.coor_tile_recursive_call(coor,prior_list,prior_index+1,increase=True)
        else:
            if coor[T_index] > 0:
                coor[T_index]-=1
            else:
                if prior_index==len(prior_list)-1:
                    raise ValueError('Index out of tile range !!')
                    
                coor[T_index]=T_size-1
                coor=self.coor_tile_recursive_call(coor,prior_list,prior_index+1,increase=False)
            
        return coor
        
    def coor_tile_move(self,coor,bit,n_move,increase=False,row_mode=False):
        """ Moving coordinates on tile by given steps.

        Parameters
        ----------
        coor : Tuple of integer
            The coordinate on tile.
        bit : Integer
            The position (bit index) in a parameter.
        n_move : Integer
            The steps of movement.
        increase : Bool, optional
            Moving step is forward to the index or backward. The default is False.
        row_mode : Bool, optional
            The moving tile shape priory is base on row priority or not. The default is False.

        Returns
        -------
        Tuple of Integers
            The moved coordinate on tile.
        Integer
            The moved position (bit index) in a parameter.

        """
        if row_mode:
            prior_list=self.row_prior
        else:
            prior_list=self.col_prior

        coor_tmp=list(coor)
        bit_coor_tmp=self.wl-bit-1
        
        for n_count in range(n_move):
            #print(coor_tmp,bit_coor_tmp,n_count)
            if increase:
                if bit_coor_tmp < self.wl-1:
                    bit_coor_tmp+=1
                else:
                    bit_coor_tmp=0
                    coor_tmp=self.coor_tile_recursive_call(coor_tmp,prior_list,0,increase=True)
            else:
                if bit_coor_tmp > 0:
                    bit_coor_tmp-=1
                else:
                    bit_coor_tmp=self.wl-1
                    coor_tmp=self.coor_tile_recursive_call(coor_tmp,prior_list,0,increase=False)
                
        return tuple(coor_tmp),self.wl-bit_coor_tmp-1
    
    def check_tile_overflow(self,bitmap,addr=None):
        """ Check wether the address is out of the range of bitmap or not.

        Parameters
        ----------
        bitmap : Class
            The bitmap class for memory fault tolerance analysis.
        addr : Tuple or Ndarray. , optional
            The address of memory bitmap.

        Returns
        -------
        bool
            Is the address in range.
        """
        if addr is None:
            bitmap_size=bitmap.row*bitmap.col
        else:
            if isinstance(addr,tuple):
                bitmap_size=bitmap.get_numtag(addr)+1
            elif isinstance(addr,np.ndarray):
                bitmap_size=np.max(bitmap.get_numtag(addr))+1
            
            
        if self.tile_size is None:
            self.tile_size=self.Tm*self.Tn*self.Tr*self.Tc*self.wl
            
        if bitmap_size<self.tile_size:
            return True
        else:
            return False
        
    def check_within_bias_range(self,bitmap,addr=None):
        """ Check wether the address is out of the range of bitmap or not. Including the bias data.

        bitmap : Class
            The bitmap class for memory fault tolerance analysis.
        addr : Tuple or Ndarray. , optional
            The address of memory bitmap.

        Returns
        -------
        bool
            Is the address in range.
        """
        if addr is None:
            bitmap_size=bitmap.row*bitmap.col
        else:
            bitmap_size=bitmap.get_numtag(addr)+1
        
        if self.tile_size is None:
            self.tile_size=self.Tm*self.Tn*self.Tr*self.Tc*self.wl
            
        if self.bias_range is None:
            if self.use_bias:
                bias_size=self.Tn*self.wl
            else:
                bias_size=0
            
            self.bias_range=self.tile_size+bias_size
               
        if bitmap_size<self.bias_range:
            return True
        else:
            return False
    
    def build_slice_head(self,bitmap):
        """ Create the list of coodinate which are the first entry to the bitmap rows.
            For check the actual row index if there is difference cause by the row and column perspective change.

        Parameters
        ----------
        bitmap : Class
            The bitmap class for memory fault tolerance analysis.
        """
        slice_head_list=np.multiply(np.arange(bitmap.row),bitmap.col)
        if self.tile_size is None:
            self.tile_size=self.Tm*self.Tn*self.Tr*self.Tc*self.wl
        slice_head_list=slice_head_list[slice_head_list<self.tile_size]
        slice_head_list=self.numtag2coor(slice_head_list)[0]
        bit__=np.multiply(np.ones(len(slice_head_list),dtype=int),self.wl-1)
        self.slice_head_list=self.get_numtag(slice_head_list,bit__,row_mode=True)
        
        self.slice_head_order=np.argsort(self.slice_head_list)
        slice_head_list,bit_val=self.numtag2coor(self.slice_head_list,row_mode=True)
        if np.sum(bit_val)%(self.wl-1)!=0:
            raise ValueError('coordinate slice head is not the MSB of a word. There might be some error.')
        self.slice_head_list=slice_head_list

    
    def get_numtag(self,coor,bit,row_mode=False):
        """Get the bitmap and tile conversion index numtag.

        Arguments
        ---------
        coor: Tuple or 2D ndarray. 
            The tile coordinate wanted to be converted to memory address.
        bit: Integer or List of Integer or 1D Ndarray. 
            The bit location in a parameer.
        row_mode: True or False. 
            Using column or row priority to generate numtag.
    
        Returns
        -------
            | The numtag (Integer)
            | Numtag array (Ndarray)
        """                   
        if row_mode:
            prior_list=self.row_prior
        else:
            prior_list=self.col_prior

        dims_prior,axis_index=self.get_tile_dims_prior(prior_list)
        
        if (isinstance(coor,tuple) or (isinstance(coor,np.ndarray) and len(coor.shape)==1))and isinstance(bit,int):
            if len(coor)!=self.shape_len:
                raise ValueError('The length of coordinate Tuple in tile must be %d but got %d.'%(self.shape_len,len(coor)))

            numtag=np.ravel_multi_index(np.array(coor)[axis_index],dims_prior)
            numtag=numtag*self.wl+self.wl-bit-1
            
        elif isinstance(coor,np.ndarray) and (isinstance(bit,np.ndarray) or isinstance(bit,list)):
            if coor.shape[-1]!=self.shape_len:
                raise ValueError('The length of coordinate Tuple in tile must be %d but got %d.'%(self.shape_len,coor.shape[-1]))
                
            numtag=np.ravel_multi_index(coor.T[axis_index],dims_prior)
            numtag=np.multiply(numtag+1,self.wl)
            numtag=np.subtract(np.subtract(numtag,bit),1)
        
        return numtag
        
    def numtag2coor(self,numtag,row_mode=False):
        """Convert the numtag to its corresponding coordinate.

        Arguments
        ---------
        numtag: Integer or 1D ndarray. 
            The bitmap and tile conversion index numtag.
    
        Returns
            | The tile coordinate (Tuple), bit location (Integer)
            | The array of tile coordinates (2D Ndarray), array of bit locations (1D Ndarray)
        """
        if row_mode:
            prior_list=self.row_prior
        else:
            prior_list=self.col_prior

        dims_prior,axis_index=self.get_tile_dims_prior(prior_list)
        
        bit=np.remainder(numtag,self.wl)
        bit=np.subtract(self.wl-1,bit)
        numtag_tmp=np.floor_divide(numtag,self.wl)
        
        restore_index=np.zeros((self.shape_len,),dtype=int)
        for i in range(self.shape_len):
            restore_index[axis_index[i]]=i
        
        if isinstance(numtag,np.ndarray) and isinstance(bit,np.ndarray):
            coor=np.unravel_index(numtag_tmp,dims_prior)
            coor=np.array(coor)[restore_index]

            return coor.T,bit
        
        else:
            coor=np.unravel_index(numtag_tmp,dims_prior)
            coor=np.array(coor)[restore_index]
            
            return tuple(coor),bit
                        
    def tile2bitmap(self,coor,bit,bitmap):
        """Convert the tile coordinate to its corresponding address on memory bitmap.

        Arguments
        ---------
        coor: Tuple or 2D ndarray. 
            The tile coordinate wanted to be converted to memory address.
        bit: Integer or List of Integer or 1D Ndarray. 
            The bit location in a parameer.
        bitmap: Class. 
            The bitmap class for memory fault tolerance analysis.
    
        Returns
            | The memory address (Tuple) in the bitmap.
            | The array of memory addresses (2D Ndarray)
        """

        numtag=self.get_numtag(coor,bit)
        col_addr=np.remainder(numtag , bitmap.col)
            
        numtag_slice_head=np.subtract(numtag,col_addr)
        row_addr_psuedo=np.floor_divide(numtag_slice_head,bitmap.col)
        
        if self.slice_head_list is None or self.slice_head_order is None:
            self.build_slice_head(bitmap)
            
        row_addr=self.slice_head_order[row_addr_psuedo]
        
        if isinstance(row_addr,np.int64) and isinstance(col_addr,np.int64):
            return (row_addr,col_addr)
        elif isinstance(row_addr,np.ndarray) and isinstance(col_addr,np.ndarray):
            return np.transpose([row_addr,col_addr])

    def bitmap2tile(self,addr,bitmap):
        """Convert the address on memory bitmap to its corresponding tile coordinate.

        Arguments
        ---------
        addr: Tuple or Ndarray. 
            The address of memory wanted to be converted to tile coordinate.
        bitmap: Class. 
            The bitmap class for memory fault tolerance analysis.
    
        Returns
        -------
            | The tile coordinate (Tuple). The corresponding bit in parameter (Integer).
            | The array of tile coordinates (2D Ndarray), array of bit locations (1D Ndarray)
        """
        if isinstance(addr,tuple):
            if len(addr)!=2:
                raise ValueError('The length of address Tuple in memory must be 2 but got %d.'%(len(addr)))
            addr0=addr[0]
            addr1=addr[1]
            bit__=self.wl-1
        elif isinstance(addr,np.ndarray):
            if addr.shape[-1]!=2:
                raise ValueError('The length of address Tuple in memory must be 2 but got %d.'%(len(addr)))
            addr0=addr[:,0]
            addr1=addr[:,1]
            bit__=np.multiply(np.ones(addr.shape[0],dtype=int),self.wl-1)

                
        if self.slice_head_list is None or self.slice_head_order is None:
            self.build_slice_head(bitmap)            
        
        coor_head=self.slice_head_list[self.slice_head_order[addr0]]
        
        coor_numtag=self.get_numtag(coor_head,bit__)
        coor_numtag=np.add(coor_numtag,addr1)
        
        if isinstance(addr,tuple):
            if coor_numtag>=self.tile_size:
                if self.check_tile_overflow(bitmap,addr):
                    if self.print_detail:
                        print(addr)
                        print('Meet the condition of row of the end of tile is not the last row of data in memory. Due to the different priority setting of column and row. Data being repermutated.')
                    n_move=addr[1]-(self.tile_size-self.get_numtag(coor_head,self.wl-1))
                    coor_head=self.slice_head_list[self.slice_head_order[addr[0]+1]]
                    coor,bit=self.coor_tile_move(coor_head,bit__,n_move,increase=True)
                else:
                    raise ValueError('Index out of tile range !!')
            else:
                coor,bit=self.numtag2coor(coor_numtag)
        elif isinstance(addr,np.ndarray):
            coor_reorder_idx=np.reshape(np.argwhere(coor_numtag>=self.tile_size),-1)
            coor_numtag=coor_numtag[coor_numtag<self.tile_size]
        
            coor,bit=self.numtag2coor(coor_numtag)
        
            if len(coor_reorder_idx)>0:
                addr_reorder=addr[coor_reorder_idx]
                if self.check_tile_overflow(bitmap,addr_reorder):
                    if self.print_detail:
                        for addrr in addr_reorder:
                            print(addrr)
                            print('Meet the condition of row of the end of tile is not the last row of data in memory. Due to the different priority setting of column and row. Data being repermutated.')
                    reorder_diff=self.tile_size-self.get_numtag(self.slice_head_list[-2],self.wl-1)
                    n_move=np.subtract(addr_reorder[:,1],reorder_diff)
                    coor_head=self.slice_head_list[self.slice_head_order[np.add(addr_reorder[:,0],1)]]
                    coor_numtag_reorder=self.get_numtag(coor_head,bit__[:len(coor_reorder_idx)])
                    coor_reorder,bit_reorder=self.numtag2coor(coor_numtag_reorder)
                    coor=np.concatenate((coor,coor_reorder))
                    bit=np.concatenate((bit,bit_reorder))
                else:
                    raise ValueError('Index out of tile range !!')
        
        return coor,bit
        

    def fault_dict_bitmap2tile(self,bitmap,use_bias=None,row_prior=None,col_prior=None,fast_mode=False):
        """Mapping the fault on the memory bitmap to tile coordinate

        Arguments
        ---------
        bitmap: Class. 
            The bitmap class for memory fault tolerance analysis.
        use_bias: Bool.
            Use bias in weight tile or not.
        row_prior: List of Strings. 
            The priority of memory mapping in the memory row dimension. Consist of 'Tm', 'Tn', 'Tr', 'Tc'.
        col_prior: List of Strings. 
            The priority of memory mapping in the memory column dimension. Consist of 'Tm', 'Tn', 'Tr', 'Tc'.
        
        Returns
        -------
        The fault information Dictionary of tile.
        """
        if self.wl!=bitmap.wl and bitmap.wl is not None:
            raise ValueError('Word length of tile (%d) must be the same as bitmap (%d).'%(self.wl,bitmap.wl))
            
        if bitmap.col % self.wl != 0:
            raise ValueError('The memory column size %d does not fit word length %d.'%(bitmap.col,self.wl))
            
        if row_prior is not None:
            self.row_prior=row_prior
        if col_prior is not None:
            self.col_prior=col_prior
        self.check_prior()
        
        if self.is_fmap and use_bias:
            raise ValueError('Feature map tile with use_bias option True. Only weight tile can mapping with bias.')
        if use_bias is not None:
            self.use_bias=use_bias

        if self.check_tile_overflow(bitmap):
            raise ValueError('The tile is bigger than the memory !')
        
        if self.use_bias:
            if self.check_within_bias_range(bitmap):
                raise ValueError('The tile is bigger than the memory !')

        if fast_mode:
            addr=np.array(list(bitmap.fault_dict.keys()))
            fault_type=np.array(list(bitmap.fault_dict.values()))
            
            addr_numtag=bitmap.get_numtag(addr)
            if self.use_bias:
                addr_idx=np.reshape(np.argwhere(addr_numtag<self.bias_range),-1)
                addr=addr[addr_idx]
                fault_type=fault_type[addr_idx]
                addr_numtag=addr_numtag[addr_numtag<self.bias_range]
                addr_bias_idx=np.reshape(np.argwhere(addr_numtag>=self.tile_size),-1)
                addr_bias=addr[addr_bias_idx]
                fault_type_bias=fault_type[addr_bias_idx]
                numtag_bias=addr_numtag[addr_bias_idx]
            addr_idx=np.reshape(np.argwhere(addr_numtag<self.tile_size),-1)
            addr=addr[addr_idx]
            fault_type=fault_type[addr_idx]
            
            fault_coor,fault_bit=self.bitmap2tile(addr,bitmap)
            
            fault_coor=list(zip(*fault_coor.T))
            fault_info=[{'SA_type':typee,'SA_bit':bit} for typee,bit in zip(fault_type,fault_bit)]
            self.fault_dict=dict(zip(fault_coor,fault_info))
            
            if self.use_bias:
                if len(addr_bias)>0 and len(fault_type_bias)>0 and len(numtag_bias)>0:
                    if self.print_detail:
                        print('bias fault ',addr_bias)
                    numtag_bias=np.subtract(numtag_bias,self.tile_size-1)
                    bias_coor=np.floor_divide(numtag_bias,self.wl)
                    bias_bit=np.subtract(self.wl-1,np.remainder(numtag_bias,self.wl))
                    bias_info=[{'SA_type':typee,'SA_bit':bit} for typee,bit in zip(fault_type_bias,bias_bit)]
                    bias_coor=list(zip(*np.expand_dims(bias_coor,0)))
                    
                    self.bias_fault_dict=dict(zip(bias_coor,bias_info))
            
        else:
            for addr in bitmap.fault_dict.keys():
                if self.check_tile_overflow(bitmap,addr):
                    fault_type=bitmap.fault_dict[addr]
                    fault_coor,fault_bit=self.bitmap2tile(addr,bitmap)
                    
                    if fault_coor in self.fault_dict.keys():
                        if isinstance(self.fault_dict[fault_coor]['SA_bit'],list):
                            self.fault_dict[fault_coor]['SA_type'].append(fault_type)
                            self.fault_dict[fault_coor]['SA_bit'].append(fault_bit)
                        else:
                            self.fault_dict[fault_coor]['SA_type']=[self.fault_dict[fault_coor]['SA_type'],fault_type]
                            self.fault_dict[fault_coor]['SA_bit']=[self.fault_dict[fault_coor]['SA_bit'],fault_bit]
                    else:
                        self.fault_dict[fault_coor]={'SA_type':fault_type,
                                                     'SA_bit' : fault_bit}
                elif self.check_within_bias_range(bitmap,addr) and not self.is_fmap:
                    if self.print_detail:
                        print('bias fault %s'%str(addr))
                    fault_type=bitmap.fault_dict[addr]
                    bias_numtag=bitmap.get_numtag(addr)-self.tile_size+1
                    self.bias_fault_dict[(bias_numtag//self.wl,)]={'SA_type':fault_type,
                                                                   'SA_bit' :self.wl - bias_numtag % self.wl -1}
        
        if self.is_fmap:
            return self.fault_dict
        else:
            if len(self.bias_fault_dict) == 0:
                return [self.fault_dict,None]
            else:
                return [self.fault_dict,self.bias_fault_dict]
    
    def fault_dict_tile2bitmap(self,bitmap,use_bias=None,row_prior=None,col_prior=None):
        """Mapping the fault on the tile coordinate to memory bitmap 

        Arguments
        ---------
        bitmap: Class. 
            The bitmap class for memory fault tolerance analysis.
        row_prior: List of Strings. 
            The priority of memory mapping in the memory row dimension. Consist of 'Tm', 'Tn', 'Tr', 'Tc'.
        col_prior: List of Strings. 
            The priority of memory mapping in the memory column dimension. Consist of 'Tm', 'Tn', 'Tr', 'Tc'.
        
        Returns
        -------
        The fault information Dictionary of bitmap.
        """
        if self.wl!=bitmap.wl and bitmap.wl is not None:
            raise ValueError('Word length of tile (%d) must be the same as bitmap (%d).'%(self.wl,bitmap.wl))
            
        if bitmap.col % self.wl != 0:
            raise ValueError('The memory column size %d does not fit word length %d.'%(bitmap.col,self.wl))
            
        if row_prior is not None:
            self.row_prior=row_prior
        if col_prior is not None:
            self.col_prior=col_prior
        self.check_prior()
        
        if self.is_fmap and use_bias:
            raise ValueError('Feature map tile with use_bias option True. Only weight tile can mapping with bias.')
        if use_bias is not None:
            self.use_bias=use_bias
        
        if self.check_tile_overflow(bitmap):
            raise ValueError('The tile is bigger than the memory !')
            
        if len(self.fault_dict)!=0:
            coor=np.array(list(self.fault_dict.keys()))
            fault_type=np.array([typee['SA_type'] for typee in self.fault_dict.values()])
            fault_bit=np.array([typee['SA_bit'] for typee in self.fault_dict.values()])
            multi_fault_index=np.array([isinstance(bit_,list) for bit_ in fault_bit])
            
            if np.sum(multi_fault_index) > 0:
                multi_fault_index=np.reshape(np.argwhere(multi_fault_index>0),-1)
                multi_fault_num=np.array([len(fault_bit[idx]) for idx in multi_fault_index])
                multi_fault_num=[i for i in range(len(multi_fault_num)) for _ in range(multi_fault_num[i]-2)]
                multi_fault_index=np.insert(multi_fault_index,multi_fault_num,multi_fault_index[multi_fault_num])
                coor=np.insert(coor,multi_fault_index,coor[multi_fault_index],axis=0)
                fault_bit=np.hstack(fault_bit.flat)
                fault_type=np.hstack(fault_type.flat)
            
            fault_addr=self.tile2bitmap(coor,fault_bit,bitmap)
                        
            fault_addr=list(zip(*fault_addr.T))
            bitmap.fault_dict=dict(zip(fault_addr,fault_type))
                            
        if self.use_bias:
            if len(self.bias_fault_dict)==0:
                return bitmap.fault_dict
            
            kernel_end_coor=self.numtag2coor(self.tile_size-1)[0]
            kernel_end_addr=self.tile2bitmap(kernel_end_coor,0,bitmap)
            
            coor=np.array(list(self.bias_fault_dict.keys()))
            fault_type=np.array([typee['SA_type'] for typee in self.bias_fault_dict.values()])
            fault_bit=np.array([typee['SA_bit'] for typee in self.bias_fault_dict.values()])
            multi_fault_index=np.array([isinstance(bit_,list) for bit_ in fault_bit])
            
            if np.sum(multi_fault_index) > 0:
                multi_fault_index=np.reshape(np.argwhere(multi_fault_index>0),-1)
                multi_fault_num=np.array([len(fault_bit[idx]) for idx in multi_fault_index])
                multi_fault_num=[i for i in range(len(multi_fault_num)) for _ in range(multi_fault_num[i]-2)]
                multi_fault_index=np.insert(multi_fault_index,multi_fault_num,multi_fault_index[multi_fault_num])
                coor=np.insert(coor,multi_fault_index,coor[multi_fault_index],axis=0)
                fault_bit=np.hstack(fault_bit.flat)
                fault_type=np.hstack(fault_type.flat)
            
            n_move=np.add(np.subtract(np.multiply(coor,self.wl),np.reshape(fault_bit,(-1,1))),self.wl-1)
            fault_addr1=np.add(kernel_end_addr[1],n_move)
            fault_addr0=np.add(kernel_end_addr[0],np.floor_divide(fault_addr1,bitmap.col))
            fault_addr1=np.remainder(fault_addr1,bitmap.col)
            fault_addr=np.concatenate((fault_addr0,fault_addr1),axis=1)
            
            if len(fault_addr)<=1:
                bitmap.fault_dict[tuple(fault_addr[0])]=fault_type[0]
                if self.print_detail:
                    print('bias fault ',fault_addr)
            else:
                fault_addr=list(zip(*fault_addr.T))
                bias_fd_addr=dict(zip(fault_addr,fault_type))
                bitmap.fault_dict.update(bias_fd_addr)
                if self.print_detail:
                    print('bias fault ',bias_fd_addr)
                    
        return bitmap.fault_dict
    
    def fault_dict_tile2layer(self,layer_shape,use_bias=None):
        """Restore the fault dictionary from tile to entire layer

        Arguments
        ---------
        layer_shape: Tuple. 
            The shape of a layer parameter were divided into tile.
        use_bias: Bool.
            Use bias in weight tile or not.
        
        Returns
        -------
        The fault information Dictionary of a layer parameter (feature maps or weights).
        """
        if self.is_fmap and use_bias:
            raise ValueError('Feature map tile with use_bias option True. Only weight tile can mapping with bias.')
        if use_bias is not None:
            self.use_bias=use_bias
            
        layer_shape=list(layer_shape)
        
        if self.base_coor is None:
            if self.is_fmap:
                tile_shape=[self.Tn,self.Tr,self.Tc,self.Tm]
            else:
                tile_shape=[self.Tr,self.Tc,self.Tm,self.Tn]
                
            restore_multiple=np.floor_divide(layer_shape,tile_shape)
            
            base_coor=list(np.ndindex(*np.add(restore_multiple,[1,1,1,1])))            
                            
            base_coor=np.multiply(base_coor,np.tile(tile_shape,[len(base_coor),1]))
            
            self.base_coor=base_coor
        
        
        if len(self.fault_dict)!=0:
            tile_fault_coor=list(self.fault_dict.keys())
            layer_fault_coor=np.add(np.tile(self.base_coor,[len(tile_fault_coor),1]),np.tile(tile_fault_coor,[len(self.base_coor),1]))
            
            tile_fault_info=list(self.fault_dict.values())
            layer_fault_info=np.tile(tile_fault_info,[len(self.base_coor)])
    
            outside_element_index=list()
            for i in range(4):
                outside_element_index.append(np.argwhere(layer_fault_coor[:,i]>=layer_shape[i]))
            outside_element_index=np.reshape(np.concatenate(outside_element_index),[1,-1])
                
            layer_fault_coor=np.delete(layer_fault_coor,outside_element_index,0)
            layer_fault_info=np.delete(layer_fault_info,outside_element_index,0)
            
            layer_fault_coor=np.split(np.transpose(layer_fault_coor),4,axis=0)
            layer_fault_coor=[axis[0] for axis in layer_fault_coor]
            layer_fault_coor=list(zip(*layer_fault_coor))
            layer_fault_info=list(layer_fault_info)
            layer_fault_dict=dict(zip(layer_fault_coor,layer_fault_info))
        else:
            layer_fault_dict=dict()
                  
        
        layer_fault_dict_bias=dict()
                    
        if self.use_bias:
            if len(self.bias_fault_dict)==0:
                return [layer_fault_dict,layer_fault_dict_bias]
            
            bias_fault_coor=list(self.bias_fault_dict.keys())
            bias_fault_coor=np.reshape(bias_fault_coor,[-1])

            bias_restore_multiple=layer_shape[-1]//self.Tn+1
            bias_base_coor=np.multiply(np.arange(bias_restore_multiple),self.Tn)
            bias_fault_coor=np.reshape(np.add.outer(bias_base_coor,bias_fault_coor),[-1])
            #bias_fault_coor=np.add(np.tile(bias_base_coor,[1,len(bias_fault_coor)]),np.tile(bias_fault_coor,[1,bias_restore_multiple]))
            
            bias_fault_info=list(self.bias_fault_dict.values())
            bias_fault_info=np.tile(bias_fault_info,[bias_restore_multiple])

            bias_outside_index=np.argwhere(bias_fault_coor>=layer_shape[-1])
            
            bias_fault_coor=np.delete(bias_fault_coor,bias_outside_index)
            bias_fault_info=np.delete(bias_fault_info,bias_outside_index)
            bias_fault_coor=list(zip(*np.expand_dims(bias_fault_coor,0)))

            layer_fault_dict_bias=dict(zip(bias_fault_coor,bias_fault_info))
                    
            return [layer_fault_dict,layer_fault_dict_bias]

        return layer_fault_dict
    
    def gen_layer_fault_dict(self,layer_shape,bitmap,use_bias=None,col_prior=None,row_prior=None,fast_mode=False):
        """Generate the fault dictionary of a layer from bitmap fault dictionary

        Arguments
        ---------
        layer_shape: Tuple. 
            The shape of a layer parameter were divided into tile.
        bitmap: Class. 
            The bitmap class for memory fault tolerance analysis.
        use_bias: Bool.
            Use bias in weight tile or not.
        row_prior: List of Strings. 
            The priority of memory mapping in the memory row dimension. Consist of 'Tm', 'Tn', 'Tr', 'Tc'.
        col_prior: List of Strings. 
            The priority of memory mapping in the memory column dimension. Consist of 'Tm', 'Tn', 'Tr', 'Tc'.
        
        Returns
        -------
        The fault information Dictionary of a layer parameter (feature maps or weights).
        """

        if col_prior is not None:
            self.col_prior=col_prior
        if row_prior is not None:
            self.row_prior=row_prior
        if self.is_fmap and use_bias:
            raise ValueError('Feature map tile with use_bias option True. Only weight tile can mapping with bias.')
        if use_bias is not None:
            self.use_bias=use_bias
            
        self.fault_dict_bitmap2tile(bitmap,fast_mode=fast_mode)
    
        return self.fault_dict_tile2layer(layer_shape)
    
    def clear(self):
        """Clear the fault information of tile"""
        self.fault_dict=dict()
        self.bias_fault_dict=dict()
        
    def clear_bitmap(self):
        '''Clear bitmap information of tile'''
        self.slice_head_list=None
        self.slice_head_order=None
        
    def clear_layer(self):
        '''Clear layer information of tile'''
        self.base_coor=None
        
    
        
class tile_FC(tile):
    """The tile of a DNN feature map or weights

    Arguments
    tile_shape: Tuple. 
        The shape of tile.
    Tm: Integer. 
        The size of tile on the input neurons (weight) or channel dimention (feature map).
    Tn: Integer. 
        The size of tile on the output neurons (weight) or batch dimention (feature map).
    is_fmap: Bool. 
        The tile is feature map tile or weight tile.
    wl: Integer. 
        The word length of DNN model parameter.
    row_prior: List of Strings. 
        The priority of memory mapping in the memory row dimension. Consist of 'Tm', 'Tn'.
    col_prior: List of Strings. 
        The priority of memory mapping in the memory column dimension. Consist of 'Tm', 'Tn'.
    """
    def __init__(self, tile_shape, is_fmap, wl=32, row_prior=[], col_prior=[]):
        """ tile_FC initializer """
        if not isinstance(is_fmap,bool):
            raise TypeError('Argument is_fmap must be True (feature map tile) or False (weight tile)')
        if len(tile_shape) != 2:
            raise ValueError('The argument tile_shape must be in Tuple dtype and have length 4 but got length %d'%len(tile_shape))
        if is_fmap:    
            self.Tm=tile_shape[1]
            self.Tn=tile_shape[0]
        else:
            self.Tm=tile_shape[0]
            self.Tn=tile_shape[1]
        self.is_fmap=is_fmap
        self.wl=wl
        if len(row_prior) != 0:
            self.row_prior=row_prior
        else:
            self.row_prior=['Tm','Tn']
        if len(col_prior) != 0:
            self.col_prior=col_prior
        else:
            self.col_prior=['Tm','Tn']
        self.prior_element=['Tm','Tn']
        self.slice_head_list=None
        self.slice_head_order=None
        self.fault_dict=dict()
        self.tile_size=None
        self.use_bias=False
        self.bias_fault_dict=dict()
        self.bias_range=None
        self.base_coor=None
        self.shape_len=2
        self.print_detail=True
                        
    def priorexchange(self,prior):
        """ Transfer String priority axis to its corresponding size and axis index.
        
        Parameters
        ----------
        prior : String. One of 'Tm', 'Tn'.

        Returns
        -------
        Integer
            Size of prior.
        Integer
            Index of prior in shape.
        """
        if prior not in self.prior_element:
            raise ValueError('The argument row_prior must be in list %s'%(str(self.prior_element)))
            
        if self.is_fmap:
            axis_idex=[1,0]
        else:
            axis_idex=[0,1]
            
        if prior == 'Tm':
            return self.Tm,axis_idex[0]
        elif prior == 'Tn':
            return self.Tn,axis_idex[1]
    
    def get_tile_dims_prior(self,prior_list):
        """ Get the shape and its dimension order by the given prior list.
            Use for rehape data index.

        Parameters
        ----------
        prior_list : List of String
            The wanted new prior order.

        Returns
        -------
        Tuple of Intger
            The new shape.
        dims_reorder : List of Integer
            The dimesion order to original shape.
        """
        if self.is_fmap:
            axis_idex=[1,0]
        else:
            axis_idex=[0,1]

        dims_prior=list()
        dims_reorder=list()
        for prior in reversed(prior_list):
            if prior == 'Tm':
                dims_prior.append(self.Tm)
                dims_reorder.append(axis_idex[0])
            elif prior == 'Tn':
                dims_prior.append(self.Tn)
                dims_reorder.append(axis_idex[1])
                
        return tuple(dims_prior),dims_reorder
        
    def check_tile_overflow(self,bitmap,addr=None):
        """ Check wether the address is out of the range of bitmap or not.

        Parameters
        ----------
        bitmap : Class
            The bitmap class for memory fault tolerance analysis.
        addr : Tuple or Ndarray. , optional
            The address of memory bitmap.

        Returns
        -------
        bool
            Is the address in range.
        """
        if addr is None:
            bitmap_size=bitmap.row*bitmap.col
        else:
            bitmap_size=bitmap.get_numtag(addr)+1
            
            
        if self.tile_size is None:
            self.tile_size=self.Tm*self.Tn*self.wl
            
        if bitmap_size<self.tile_size:
            return True
        else:
            return False
        
    def check_within_bias_range(self,bitmap,addr=None):
        """ Check wether the address is out of the range of bitmap or not. Including the bias data.

        bitmap : Class
            The bitmap class for memory fault tolerance analysis.
        addr : Tuple or Ndarray. , optional
            The address of memory bitmap.

        Returns
        -------
        bool
            Is the address in range.
        """
        if addr is None:
            bitmap_size=bitmap.row*bitmap.col
        else:
            bitmap_size=bitmap.get_numtag(addr)+1
        
        if self.tile_size is None:
            self.tile_size=self.Tm*self.Tn*self.wl
            
        if self.bias_range is None:
            if self.use_bias:
                bias_size=self.Tn*self.wl
            else:
                bias_size=0
            
            self.bias_range=self.tile_size+bias_size
               
        if bitmap_size<self.bias_range:
            return True
        else:
            return False

    def fault_dict_tile2layer(self,layer_shape,use_bias=None):
        """Restore the fault dictionary from tile to entire layer

        Arguments
        ---------
        layer_shape: Tuple. 
            The shape of a layer parameter were divided into tile.
        use_bias: Bool.
            Use bias in weight tile or not.
        
        Returns
        -------
        The fault information Dictionary of a layer parameter (feature maps or weights).
        """
        if self.is_fmap and use_bias:
            raise ValueError('Feature map tile with use_bias option True. Only weight tile can mapping with bias.')
        if use_bias is not None:
            self.use_bias=use_bias
            
        layer_shape=list(layer_shape)
        
        if self.base_coor is None:
            if self.is_fmap:
                tile_shape=[self.Tn,self.Tm]
            else:
                tile_shape=[self.Tm,self.Tn]
                
            restore_multiple=np.floor_divide(layer_shape,tile_shape)
            
            base_coor=list(np.ndindex(*np.add(restore_multiple,[1,1])))            
                
            base_coor=np.multiply(base_coor,np.tile(tile_shape,[len(base_coor),1]))
            
            self.base_coor=base_coor
        
        if len(self.fault_dict)!=0:
            tile_fault_coor=list(self.fault_dict.keys())
            layer_fault_coor=np.add(np.tile(self.base_coor,[len(tile_fault_coor),1]),np.tile(tile_fault_coor,[len(self.base_coor),1]))
            
            tile_fault_info=list(self.fault_dict.values())
            layer_fault_info=np.tile(tile_fault_info,[len(self.base_coor)])
    
            outside_element_index=list()
            for i in range(2):
                outside_element_index.append(np.argwhere(layer_fault_coor[:,i]>=layer_shape[i]))
            outside_element_index=np.reshape(np.concatenate(outside_element_index),[1,-1])
                
            layer_fault_coor=np.delete(layer_fault_coor,outside_element_index,0)
            layer_fault_info=np.delete(layer_fault_info,outside_element_index,0)
            
            layer_fault_coor=np.split(np.transpose(layer_fault_coor),2,axis=0)
            layer_fault_coor=[axis[0] for axis in layer_fault_coor]
            layer_fault_coor=list(zip(*layer_fault_coor))
            layer_fault_info=list(layer_fault_info)
            layer_fault_dict=dict(zip(layer_fault_coor,layer_fault_info))
        else:
            layer_fault_dict=dict()
                  
        
        layer_fault_dict_bias=dict()
                    
        if self.use_bias:
            if len(self.bias_fault_dict)==0:
                return [layer_fault_dict,layer_fault_dict_bias]
            
            bias_fault_coor=list(self.bias_fault_dict.keys())
            bias_fault_coor=np.reshape(bias_fault_coor,[-1])

            bias_restore_multiple=layer_shape[-1]//self.Tn+1
            bias_base_coor=np.multiply(np.arange(bias_restore_multiple),self.Tn)
            bias_fault_coor=np.reshape(np.add.outer(bias_base_coor,bias_fault_coor),[-1])
            
            bias_fault_info=list(self.bias_fault_dict.values())
            bias_fault_info=np.tile(bias_fault_info,[bias_restore_multiple])

            bias_outside_index=np.argwhere(bias_fault_coor>=layer_shape[-1])
            
            bias_fault_coor=np.delete(bias_fault_coor,bias_outside_index)
            bias_fault_info=np.delete(bias_fault_info,bias_outside_index)
            bias_fault_coor=list(zip(*np.expand_dims(bias_fault_coor,0)))

            layer_fault_dict_bias=dict(zip(bias_fault_coor,bias_fault_info))
                    
            return [layer_fault_dict,layer_fault_dict_bias]

        return layer_fault_dict


        
        
        
def generate_layer_memory_mapping(layer,ifmap_buffer,wght_buffer,ofmap_buffer,ifmap_tile,wght_tile,ofmap_tile,print_detail=True,fast_mode=False,**kwargs):
    """Generate the fault dictionary list of a layer base on its memory mapping and buffer fault information.

    Arguments
    ---------
    layer: Keras.Layer. 
        The layer are being generate memory mapping.
    ifmap_buffer: Class (bitmap). 
        The bitmap class for memory fault tolerance analysis of input feature maps.
    ofmap_buffer: Class (bitmap). 
        The bitmap class for memory fault tolerance analysis of output feature maps.
    wght_buffer: Class (bitmap). 
        The bitmap class for memory fault tolerance analysis of weights.
    ifmap_tile: Class (tile or tile_FC). 
        The tile or tile_FC class for memory fault tolerance analysis of input feature maps.
    ofmap_tile: Class (tile or tile_FC). 
        The tile or tile_FC class for memory fault tolerance analysis of output feature maps.
    wght_tile: Class (tile or tile_FC). 
        The tile or tile_FC class for memory fault tolerance analysis of weight feature maps.
    print_detail: Bool. 
        Print generation detail or not.

    Returns
    -------
    The fault information Dictionary List.
    """
    
    if print_detail:
        print('\nMapping memory fault on layer ...')
    else:
        ifmap_tile.print_detail=False
        ofmap_tile.print_detail=False
        wght_tile.print_detail=False
    
    layer_input_shape=layer.input_shape
    layer_output_shape=layer.output_shape
    layer_weight_shape=[weight_shape.shape for weight_shape in layer.get_weights()]
    
    if len(layer_weight_shape)==0:
        if print_detail:
            print('    no weight layer Skipped!')
        return None, None, [None,None]
    
    # ifmap memory mapping
    if len(ifmap_buffer.fault_dict) == 0:
        if print_detail:
            print('The input feature map buffer has no fault information. Try bitmap.gen_bitmap_SA_fault_dict or assign fault information.\nProceed without inject fault.')
        ifmap_fault_dict=None
    else:
        ifmap_fault_dict=ifmap_tile.gen_layer_fault_dict(layer_input_shape,ifmap_buffer,fast_mode=fast_mode)
    
        if print_detail:
            print('    mapped layer ifmap %d faults'%(len(ifmap_fault_dict)))
    
    
    # ofmap memory mapping
    if len(ofmap_buffer.fault_dict) == 0:
        if print_detail:
            print('The output feature map buffer has no fault information. Try bitmap.gen_bitmap_SA_fault_dict or assign fault information.\nProceed without inject fault.')
        ofmap_fault_dict=None
    else:
        ofmap_fault_dict=ofmap_tile.gen_layer_fault_dict(layer_output_shape,ofmap_buffer,fast_mode=fast_mode)
    
        if print_detail:
            print('    mapped layer ofmap %d faults'%(len(ofmap_fault_dict)))
    
    # weight memory mapping
    if len(wght_buffer.fault_dict) == 0:
        if print_detail:
            print('The weights buffer has no fault information. Try bitmap.gen_bitmap_SA_fault_dict or assign fault information.\nProceed without inject fault.')
        weight_fault_dict=[None for i in layer_weight_shape]
    else:
        if len(layer_weight_shape)>1:
            use_bias=True
        else:
            use_bias=False
        
        weight_fault_dict=wght_tile.gen_layer_fault_dict(layer_weight_shape[0],wght_buffer,use_bias=use_bias,fast_mode=fast_mode)
        
        if print_detail:
            print('    mapped layer weight %s faults'%(str([len(weight_fault_dict[0]),len(weight_fault_dict[1])])))
                
    return ifmap_fault_dict, ofmap_fault_dict, weight_fault_dict





