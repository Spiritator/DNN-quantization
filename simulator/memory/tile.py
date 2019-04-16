# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 13:36:17 2019

@author: Yung-Yu Tsai

DNN tiling for memory fault mapping
"""

import numpy as np

class tile:
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
        
    def check_prior(self):
        if not isinstance(self.row_prior,list) or not isinstance(self.col_prior,list) or len(self.row_prior)!=self.shape_len or len(self.col_prior)!=self.shape_len:
            raise ValueError('The augment row_prior and col_prior must be in list dtype and have length %d but got length %d and %d'%(self.shape_len,len(self.row_prior),len(self.col_prior)))
                
        for i in range(self.shape_len):
            if self.row_prior[i] not in self.prior_element:
                raise ValueError('The augment row_prior must be in list %s'%(str(self.prior_element)))
            if self.col_prior[i] not in self.prior_element:
                raise ValueError('The augment col_prior must be in list %s'%(str(self.prior_element)))
    
    def priorexchange(self,prior):
        if prior not in self.prior_element:
            raise ValueError('The augment row_prior must be in list %s'%(str(self.prior_element)))
            
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
        
    def coor_tile_recursive_call(self,coor,prior_list,prior_index,increase=False):
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
        if addr is None:
            bitmap_size=bitmap.row*bitmap.col
        else:
            bitmap_size=bitmap.get_numtag(addr)+1
            
            
        if self.tile_size is None:
            self.tile_size=self.Tm*self.Tn*self.Tr*self.Tc*self.wl
            
        if bitmap_size<self.tile_size:
            return True
        else:
            return False
        
    def check_within_bias_range(self,bitmap,addr=None):
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
    
    def get_numtag(self,coor,bit,row_mode=False):
        """Get the bitmap and tile conversion index numtag.

        # Arguments
            coor: Tuple. The tile coordinate wanted to be converted to memory address.
            bit: Integer. The bit location in a parameer.
            row_mode: True or False. Using column or row priority to generate numtag.
    
        # Returns
            The numtag (Integer)
        """
        if len(coor)!=self.shape_len:
            raise ValueError('The length of coordinate Tuple in tile must be %d but got %d.'%(self.shape_len,len(coor)))
                   
        if row_mode:
            prior_list=self.row_prior
        else:
            prior_list=self.col_prior

        numtag=0
        coef_tmp=1
        for i in range(self.shape_len):
            T_size,T_index=self.priorexchange(prior_list[i])
            numtag+=coef_tmp*coor[T_index]
            coef_tmp*=T_size

        numtag=numtag*self.wl+self.wl-bit-1
        
        return numtag
        
    def numtag2coor(self,numtag,row_mode=False):
        """Convert the numtag to its corresponding coordinate.

        # Arguments
            numtag: Integer. The bitmap and tile conversion index numtag.
    
        # Returns
            The tile coordinate (Tuple)
        """
        if row_mode:
            prior_list=self.row_prior
        else:
            prior_list=self.col_prior

        
        coor=[0 for i in range(self.shape_len)]
        
        bit=self.wl-(numtag % self.wl)-1
        numtag_tmp=numtag//self.wl
        
        for i in reversed(range(self.shape_len)):
            T_size,T_index=self.priorexchange(prior_list[i])
            coef_tmp=1
            for j in reversed(range(i)):
                T_size,garbage=self.priorexchange(prior_list[j])
                coef_tmp*=T_size
            coor[T_index]=numtag_tmp//coef_tmp
            numtag_tmp=numtag_tmp % coef_tmp
        
        return tuple(coor),bit
    
    def tile2bitmap(self,coor,bit,bitmap):
        """Convert the tile coordinate to its corresponding address on memory bitmap.

        # Arguments
            coor: Tuple. The tile coordinate wanted to be converted to memory address.
            bit: Integer. The bit location in a parameer.
            bitmap: Class. The bitmap class for memory fault tolerance analysis.
    
        # Returns
            The memory address (Tuple) in the bitmap.
        """

        numtag=self.get_numtag(coor,bit)
        col_addr=numtag % bitmap.col
        coor_slice_head,bit_val=self.coor_tile_move(coor,bit,col_addr,increase=False)
        
        if bit_val is not self.wl-1:
            raise ValueError('coordinate slice head is not the MSB of a word. There might be some error.')
            
        numtag_slice_head=numtag-col_addr
        row_addr_psuedo=numtag_slice_head//bitmap.col
        
        if self.slice_head_list is None:
            slice_head_list=[self.numtag2coor(bitmap.col*i)[0] for i in range(bitmap.row)]
            self.slice_head_list=[self.get_numtag(head,self.wl-1,row_mode=True) for head in slice_head_list]
            
        if self.slice_head_order is None:
            self.slice_head_order=np.argsort(self.slice_head_list)
            slice_head_list=[]
            for numtag_head in self.slice_head_list:
                coor_head,bit_val=self.numtag2coor(numtag_head,row_mode=True)
                if bit_val is not self.wl-1:
                    raise ValueError('coordinate slice head is not the MSB of a word. There might be some error.')
                slice_head_list.append(coor_head)
                
            self.slice_head_list=slice_head_list

        row_addr=self.slice_head_order[row_addr_psuedo]
        
        return (row_addr,col_addr)

    def bitmap2tile(self,addr,bitmap):
        """Convert the address on memory bitmap to its corresponding tile coordinate.

        # Arguments
            addr: Tuple. The address of memory wanted to be converted to tile coordinate.
            bitmap: Class. The bitmap class for memory fault tolerance analysis.
    
        # Returns
            The tile coordinate (Tuple). The corresponding bit in parameter (Integer).
        """
        if len(addr)!=2:
            raise ValueError('The length of address Tuple in memory must be 2 but got %d.'%(len(addr)))

        if self.slice_head_list is None:
            slice_head_list=[self.numtag2coor(bitmap.col*i)[0] for i in range(bitmap.row)]
            self.slice_head_list=[self.get_numtag(head,self.wl-1,row_mode=True) for head in slice_head_list]
        
        if self.slice_head_order is None:
            self.slice_head_order=np.argsort(self.slice_head_list)
            slice_head_list=[]
            for numtag_head in self.slice_head_list:
                coor_head,bit_val=self.numtag2coor(numtag_head,row_mode=True)
                if bit_val is not self.wl-1:
                    raise ValueError('coordinate slice head is not the MSB of a word. There might be some error.')
                slice_head_list.append(coor_head)
                
            self.slice_head_list=slice_head_list
            
        
        coor_head=self.slice_head_list[self.slice_head_order[addr[0]]]
        try:
            coor,bit=self.coor_tile_move(coor_head,self.wl-1,addr[1],increase=True)
        except Exception as ValueError:
            if self.check_tile_overflow(bitmap,addr):
                print(addr)
                print('Meet the condition of row of the end of tile is not the last row of data in memory. Due to the different priority setting of column and row. Data being repermutated.')
                n_move=addr[1]-(self.tile_size-self.get_numtag(coor_head,self.wl-1))
                coor_head=self.slice_head_list[self.slice_head_order[addr[0]+1]]
                coor,bit=self.coor_tile_move(coor_head,self.wl-1,n_move,increase=True)
            else:
                raise ValueError('Index out of tile range !!')
        
        return coor,bit
        

    def fault_dict_bitmap2tile(self,bitmap,use_bias=None,row_prior=None,col_prior=None):
        """Mapping the fault on the memory bitmap to tile coordinate

        # Arguments
            bitmap: Class. The bitmap class for memory fault tolerance analysis.
            use_bias: Use bias in weight tile or not.
            row_prior: List of Strings. The priority of memory mapping in the memory row dimension. Consist of 'Tm', 'Tn', 'Tr', 'Tc'.
            col_prior: List of Strings. The priority of memory mapping in the memory column dimension. Consist of 'Tm', 'Tn', 'Tr', 'Tc'.
        
        # Returns
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
                print('bias fault %s'%str(addr))
                fault_type=bitmap.fault_dict[addr]
                bias_numtag=bitmap.get_numtag((addr))-self.tile_size
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

        # Arguments
            bitmap: Class. The bitmap class for memory fault tolerance analysis.
            row_prior: List of Strings. The priority of memory mapping in the memory row dimension. Consist of 'Tm', 'Tn', 'Tr', 'Tc'.
            col_prior: List of Strings. The priority of memory mapping in the memory column dimension. Consist of 'Tm', 'Tn', 'Tr', 'Tc'.
        
        # Returns
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
            if self.check_within_bias_range(bitmap):
                raise ValueError('The tile is bigger than the memory !')
            
        
        for coor in self.fault_dict.keys():                
            if not isinstance(self.fault_dict[coor]['SA_bit'],list):
                fault_type=self.fault_dict[coor]['SA_type']
                fault_addr=self.tile2bitmap(coor,self.fault_dict[coor]['SA_bit'],bitmap)
                
                bitmap.fault_dict[fault_addr]=fault_type
            else:
                for i in range(len(self.fault_dict[coor]['SA_bit'])):
                    fault_type=self.fault_dict[coor]['SA_type'][i]
                    fault_addr=self.tile2bitmap(coor,self.fault_dict[coor]['SA_bit'][i],bitmap)
                    
                    bitmap.fault_dict[fault_addr]=fault_type
                    
        if self.use_bias:
            #kernel_end_coor=[self.Tr-1,self.Tc-1,self.Tm-1,self.Tn-1]
            kernel_end_coor=self.numtag2coor(self.tile_size-1)[0]
            kernel_end_addr=self.tile2bitmap(kernel_end_coor,0,bitmap)
            for coor in self.bias_fault_dict.keys():
                fault_type=self.bias_fault_dict[coor]['SA_type']
                fault_addr=kernel_end_addr
                n_move=coor[0]*self.wl+self.wl-self.bias_fault_dict[coor]['SA_bit']
                fault_addr=list(fault_addr)
                fault_addr[1]+=n_move
                if fault_addr[1]>=bitmap.col:
                    fault_addr[0]+=fault_addr[1]//bitmap.col
                    fault_addr[1]=fault_addr[1]%bitmap.col
                fault_addr=tuple(fault_addr)
                print('bias fault %s'%str(fault_addr))
                
                bitmap.fault_dict[fault_addr]=fault_type
                
        return bitmap.fault_dict
    
    def fault_dict_tile2layer(self,layer_shape,use_bias=None):
        """Restore the fault dictionary from tile to entire layer

        # Arguments
            layer_shape: Tuple. The shape of a layer parameter were divided into tile.
            use_bias: Use bias in weight tile or not.
        
        # Returns
            The fault information Dictionary of a layer parameter (feature maps or weights).
        """
        if self.is_fmap and use_bias:
            raise ValueError('Feature map tile with use_bias option True. Only weight tile can mapping with bias.')
        if use_bias is not None:
            self.use_bias=use_bias
            
        layer_shape=list(layer_shape)
        if self.is_fmap:
            tile_shape=[self.Tn,self.Tr,self.Tc,self.Tm]
        else:
            tile_shape=[self.Tr,self.Tc,self.Tm,self.Tn]
            
        restore_multiple=np.floor_divide(layer_shape,tile_shape)
        
        base_coor=np.array([[0,0,0,0]])
        
        def gen_coor(coor,index):
            if coor[index] < restore_multiple[index]:
                coor[index]+=1
            else:
                if index != 0:
                    coor[index]=0
                    coor=gen_coor(coor,index-1)
            return coor
                
        coor_tmp=[0,0,0,0]
        
        for i in range(np.prod(np.add(restore_multiple,[1,1,1,1]))-1):
            coor_tmp=gen_coor(coor_tmp,3)
            base_coor=np.append(base_coor,[coor_tmp],axis=0)
            
        base_coor=np.multiply(base_coor,np.tile(tile_shape,[len(base_coor),1]))
        
        layer_fault_dict=dict()
        
        for i in range(len(base_coor)):
            for tile_fault_coor in self.fault_dict.keys():
                layer_fault_coor=np.add(base_coor[i],list(tile_fault_coor))
                if all(layer_shape>layer_fault_coor):
                    layer_fault_dict[tuple(layer_fault_coor)]=self.fault_dict[tile_fault_coor]
                  
        layer_fault_dict_bias=dict()
                    
        if self.use_bias:
            for i in range(layer_shape[-1]//self.Tn+1):
                for bias_fault_coor in self.bias_fault_dict.keys():
                    bias_fault_coor_layer=i*self.Tn+bias_fault_coor[0]
                    if bias_fault_coor_layer<layer_shape[-1]:
                        layer_fault_dict_bias[(bias_fault_coor_layer,)]=self.bias_fault_dict[bias_fault_coor]
                    
            return [layer_fault_dict,layer_fault_dict_bias]

        return layer_fault_dict
    
    def gen_layer_fault_dict(self,layer_shape,bitmap,use_bias=None,col_prior=None,row_prior=None):
        """Generate the fault dictionary of a layer from bitmap fault dictionary

        # Arguments
            layer_shape: Tuple. The shape of a layer parameter were divided into tile.
            bitmap: Class. The bitmap class for memory fault tolerance analysis.
            use_bias: Use bias in weight tile or not.
            row_prior: List of Strings. The priority of memory mapping in the memory row dimension. Consist of 'Tm', 'Tn', 'Tr', 'Tc'.
            col_prior: List of Strings. The priority of memory mapping in the memory column dimension. Consist of 'Tm', 'Tn', 'Tr', 'Tc'.
        
        # Returns
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
            
        self.fault_dict_bitmap2tile(bitmap)
    
        return self.fault_dict_tile2layer(layer_shape)
    
    def clear(self):
        """Clear the fault information of tile"""
        self.fault_dict=dict()
        self.bias_fault_dict=dict()
        
    
        
class tile_FC(tile):
    def __init__(self, tile_shape, is_fmap, wl=32, row_prior=[], col_prior=[]):
        """The tile of a DNN feature map or weights

        # Arguments
            tile_shape: Tuple. The shape of tile.
            Tm: Integer. The size of tile on the input neurons (weight) or channel dimention (feature map).
            Tn: Integer. The size of tile on the output neurons (weight) or batch dimention (feature map).
            is_fmap: Bool. The tile is feature map tile or weight tile.
            wl: Integer. The word length of DNN model parameter.
            row_prior: List of Strings. The priority of memory mapping in the memory row dimension. Consist of 'Tm', 'Tn'.
            col_prior: List of Strings. The priority of memory mapping in the memory column dimension. Consist of 'Tm', 'Tn'.
    
        """
        if not isinstance(is_fmap,bool):
            raise ValueError('Augment is_fmap must be True (feature map tile) or False (weight tile)')
        if len(tile_shape) != 2:
            raise ValueError('The augment tile_shape must be in Tuple dtype and have length 4 but got length %d'%len(tile_shape))
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
        self.shape_len=2
                        
    def priorexchange(self,prior):
        if prior not in self.prior_element:
            raise ValueError('The augment row_prior must be in list %s'%(str(self.prior_element)))
            
        if self.is_fmap:
            axis_idex=[1,0]
        else:
            axis_idex=[0,1]
            
        if prior == 'Tm':
            return self.Tm,axis_idex[0]
        elif prior == 'Tn':
            return self.Tn,axis_idex[1]
        
    def check_tile_overflow(self,bitmap,addr=None):
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

        # Arguments
            layer_shape: Tuple. The shape of a layer parameter were divided into tile.
            use_bias: Use bias in weight tile or not.
        
        # Returns
            The fault information Dictionary of a layer parameter (feature maps or weights).
        """
        if self.is_fmap and use_bias:
            raise ValueError('Feature map tile with use_bias option True. Only weight tile can mapping with bias.')
        if use_bias is not None:
            self.use_bias=use_bias
            
        layer_shape=list(layer_shape)
        if self.is_fmap:
            tile_shape=[self.Tn,self.Tm]
        else:
            tile_shape=[self.Tm,self.Tn]
            
        restore_multiple=np.floor_divide(layer_shape,tile_shape)
        
        base_coor=np.array([[0,0]])
        
        def gen_coor(coor,index):
            if coor[index] < restore_multiple[index]:
                coor[index]+=1
            else:
                if index != 0:
                    coor[index]=0
                    coor=gen_coor(coor,index-1)
            return coor
                
        coor_tmp=[0,0]
        
        for i in range(np.prod(np.add(restore_multiple,[1,1]))-1):
            coor_tmp=gen_coor(coor_tmp,1)
            base_coor=np.append(base_coor,[coor_tmp],axis=0)
            
        base_coor=np.multiply(base_coor,np.tile(tile_shape,[len(base_coor),1]))
        
        layer_fault_dict=dict()
        
        for i in range(len(base_coor)):
            for tile_fault_coor in self.fault_dict.keys():
                layer_fault_coor=np.add(base_coor[i],list(tile_fault_coor))
                if all(layer_shape>layer_fault_coor):
                    layer_fault_dict[tuple(layer_fault_coor)]=self.fault_dict[tile_fault_coor]
                  
        layer_fault_dict_bias=dict()
                    
        if self.use_bias:
            for i in range(layer_shape[-1]//self.Tn+1):
                for bias_fault_coor in self.bias_fault_dict.keys():
                    bias_fault_coor_layer=i*self.Tn+bias_fault_coor[0]
                    if bias_fault_coor_layer<layer_shape[-1]:
                        layer_fault_dict_bias[(bias_fault_coor_layer,)]=self.bias_fault_dict[bias_fault_coor]
                    
            return [layer_fault_dict,layer_fault_dict_bias]

        return layer_fault_dict


        
        
        
def generate_layer_memory_mapping(layer,ifmap_buffer,wght_buffer,ofmap_buffer,ifmap_tile,wght_tile,ofmap_tile,**kwargs):
    """Generate the fault dictionary list of a layer base on its memory mapping and buffer fault information.

    # Arguments
        layer: Keras.Layer. 
        ifmap_buffer: Class (bitmap). The bitmap class for memory fault tolerance analysis of input feature maps.
        ofmap_buffer: Class (bitmap). The bitmap class for memory fault tolerance analysis of output feature maps.
        wght_buffer: Class (bitmap). The bitmap class for memory fault tolerance analysis of weights.
        batch_size: Integer. The batch size of fault tolerance evaluation process.
        model_word_length: Integer. The word length of model parameters.

    # Returns
        The fault information Dictionary List.
    """
    
    print('\nMapping memory fault on layer ...')
    
    layer_input_shape=layer.input_shape
    layer_output_shape=layer.output_shape
    layer_weight_shape=[weight_shape.shape for weight_shape in layer.get_weights()]
    
    if len(layer_weight_shape)==0:
        print('    no weight layer Skipped!')
        return None, None, [None,None]
    
    # ifmap memory mapping
    if len(ifmap_buffer.fault_dict) == 0:
        print('The input feature map buffer has no fault information. Try bitmap.gen_bitmap_SA_fault_dict or assign fault information.\nProceed without inject fault.')
        ifmap_fault_dict=None
    else:
        ifmap_fault_dict=ifmap_tile.gen_layer_fault_dict(layer_input_shape,ifmap_buffer)
    
        print('    mapped layer ifmap %d faults'%(len(ifmap_fault_dict)))
    
    
    # ofmap memory mapping
    if len(ofmap_buffer.fault_dict) == 0:
        print('The output feature map buffer has no fault information. Try bitmap.gen_bitmap_SA_fault_dict or assign fault information.\nProceed without inject fault.')
        ofmap_fault_dict=None
    else:
        ofmap_fault_dict=ofmap_tile.gen_layer_fault_dict(layer_output_shape,ofmap_buffer)
    
        print('    mapped layer ofmap %d faults'%(len(ofmap_fault_dict)))
    
    # weight memory mapping
    if len(wght_buffer.fault_dict) == 0:
        print('The weights buffer has no fault information. Try bitmap.gen_bitmap_SA_fault_dict or assign fault information.\nProceed without inject fault.')
        weight_fault_dict=[None for i in layer_weight_shape]
    else:
        if len(layer_weight_shape)>1:
            use_bias=True
        else:
            use_bias=False
        
        weight_fault_dict=wght_tile.gen_layer_fault_dict(layer_weight_shape[0],wght_buffer,use_bias=use_bias)
        
        print('    mapped layer weight %s faults'%(str([len(weight_fault_dict[0]),len(weight_fault_dict[1])])))
                
    return ifmap_fault_dict, ofmap_fault_dict, weight_fault_dict





