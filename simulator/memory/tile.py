# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 13:36:17 2019

@author: Yung-Yu Tsai

DNN tiling for memory fault mapping
"""

import numpy as np

class tile:
    def __init__(self, Tm, Tn, Tr, Tc, is_fmap=True, wl=32, row_prior=[], col_prior=[]):
        """The tile of a DNN feature map or weights

        # Arguments
            Tm: Integer. The size of tile on the input feature map dimension (weight) or channel dimention (feature map).
            Tn: Integer. The size of tile on the output feature map dimension (weight) or batch dimention (feature map).
            Tr: Integer. The size of tile on the kernel row dimension or feature map row dimention.
            Tc: Integer. The size of tile on the kernel column dimension or feature map column dimention.
            wl: Integer. The word length of DNN model parameter.
            row_prior: List of Strings. The priority of memory mapping in the memory row dimension. Consist of 'Tm', 'Tn', 'Tr', 'Tc'.
            col_prior: List of Strings. The priority of memory mapping in the memory column dimension. Consist of 'Tm', 'Tn', 'Tr', 'Tc'.
    
        """
        self.Tm=Tm
        self.Tn=Tn
        self.Tr=Tr
        self.Tc=Tc
        self.is_fmap=is_fmap
        self.wl=wl
        self.row_prior=row_prior
        self.col_prior=col_prior
        self.prior_list=['Tm','Tn','Tr','Tc']
        self.slice_head_list=None
        
    def check_prior(self):
        if not isinstance(self.row_prior,list) or not isinstance(self.col_prior,list) or len(self.row_prior)!=4 or len(self.col_prior)!=4:
            raise ValueError('The augment row_prior and col_prior must be in list dtype and have length 4 but got length %d and %d'%(len(self.row_prior),len(self.col_prior)))
                
        for i in range(4):
            if self.row_prior[i] not in self.prior_list:
                raise ValueError('The augment row_prior must be in list %s'%(str(self.prior_list)))
            if self.col_prior[i] not in self.prior_list:
                raise ValueError('The augment col_prior must be in list %s'%(str(self.prior_list)))
    
    def priorexchange(self,prior):
        if prior not in self.prior_list:
            raise ValueError('The augment row_prior must be in list %s'%(str(self.prior_list)))
            
        if self.is_fmap:
            axis_idex=[3,0,1,2]
        else:
            axis_idex=[2,3,0,1]
            
        if prior is 'Tm':
            return self.Tm,axis_idex[0]
        elif prior is 'Tn':
            return self.Tn,axis_idex[1]
        elif prior is 'Tr':
            return self.Tr,axis_idex[2]
        elif prior is 'Tc':
            return self.Tc,axis_idex[3]
        
    def coor_tile_recursive_call(self,coor,prior_list,prior_index,increase=False):
        T_size,T_index=self.priorexchange(prior_list[prior_index])
        if increase:
            if coor[T_index] < T_size-1:
                coor[T_index]+=1
            else:
                if prior_index==len(prior_list)-1:
                    raise ValueError('Index out of range !!')
                    
                coor[T_index]=0
                coor=self.coor_tile_recursive_call(coor,prior_list,prior_index+1,increase=True)
        else:
            if coor[T_index] > 0:
                coor[T_index]-=1
            else:
                if prior_index==len(prior_list)-1:
                    raise ValueError('Index out of range !!')
                    
                coor[T_index]=T_size-1
                coor=self.coor_tile_recursive_call(coor,prior_list,prior_index+1,increase=False)
            
        return coor
        
    def coor_tile_move(self,coor,bit,n_move,increase=False,row_mode=False):
        if row_mode:
            prior_list=self.col_prior
        else:
            prior_list=self.row_prior

        coor_tmp=list(coor)
        bit_coor_tmp=self.wl-bit-1
        
        for n_count in reversed(range(n_move)):
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
    
    def get_numtag(self,coor,bit,row_mode=False):
        if len(coor)!=4:
            raise ValueError('The length of coordinate Tuple in tile must be 4 but got %d.'%(len(coor)))
                   
        if row_mode:
            prior_list=self.col_prior
        else:
            prior_list=self.row_prior

        numtag=0
        coef_tmp=1
        for i in range(4):
            T_size,T_index=self.priorexchange(prior_list[i])
            numtag+=coef_tmp*coor[T_index]
            coef_tmp*=T_size

        numtag=numtag*self.wl+self.wl-bit-1
        
        return numtag
        
    def numtag2coor(self,numtag,row_mode=False):
        if row_mode:
            prior_list=self.col_prior
        else:
            prior_list=self.row_prior

        
        coor=[0,0,0,0]
        
        bit=self.wl-(numtag % self.wl)-1
        numtag_tmp=numtag//self.wl
        
        for i in reversed(range(4)):
            T_size,T_index=self.priorexchange(prior_list[i])
            coef_tmp=1
            for j in reversed(range(i)):
                T_size,garbage=self.priorexchange(prior_list[j])
                coef_tmp*=T_size
            coor[T_index]=numtag_tmp//coef_tmp
            numtag_tmp=numtag_tmp % coef_tmp
        
        return tuple(coor),bit
    
    def tile2bitmap(self,coor,bit,bitmap):
        numtag=self.get_numtag(coor,bit)
        col_addr=numtag % bitmap.col
        coor_slice_head,bit_val=self.coor_tile_move(coor,bit,col_addr,increase=False)
        
        if bit_val is not self.wl-1:
            raise ValueError('coordinate slice head is not the MSB of a word. There might be some error.')
            
        numtag_slice_head=numtag-col_addr
        row_addr_psuedo=numtag_slice_head//bitmap.col
        
        if self.slice_head_list is None:
            slice_head_list=[self.numtag2coor(bitmap.col*i)[0] for i in range(bitmap.row-1)]
            self.slice_head_list=[self.get_numtag(head,self.wl-1,row_mode=True) for head in slice_head_list]
        row_addr=np.argsort(self.slice_head_list)[row_addr_psuedo]
        
        return (row_addr,col_addr)

    def bitmap2tile(self,addr,bitmap):
        if len(addr)!=2:
            raise ValueError('The length of address Tuple in memory must be 2 but got %d.'%(len(addr)))

        if self.slice_head_list is None:
            slice_head_list=[self.numtag2coor(bitmap.col*i)[0] for i in range(bitmap.row-1)]
            self.slice_head_list=[self.get_numtag(head,self.wl-1,row_mode=True) for head in slice_head_list]
        
        numtag_head=self.slice_head_list[np.argwhere(np.argsort(self.slice_head_list)==addr[0])[0,0]]
        coor_head,bit_val=self.numtag2coor(numtag_head,row_mode=True)
        coor,bit=self.coor_tile_move(coor_head,bit_val,addr[1],increase=True)
        
        return coor,bit
        

    def fault_dict_bitmap2tile(self,bitmap,row_prior=None,col_prior=None):
        """Mapping the fault on the memory bitmap to tile coordinate

        # Arguments
            bitmap: Class. The bitmap class for memory fault tolerance analysis.
            row_prior: List of Strings. The priority of memory mapping in the memory row dimension. Consist of 'Tm', 'Tn', 'Tr', 'Tc'.
            col_prior: List of Strings. The priority of memory mapping in the memory column dimension. Consist of 'Tm', 'Tn', 'Tr', 'Tc'.
        
        # Returns
            The fault information Dictionary of tile.
        """
        if self.wl!=bitmap.wl and bitmap.wl is not None:
            raise ValueError('Word length of tile (%d) must be the same as bitmap (%d).'%(self.wl,bitmap.wl))
            
        if bitmap.col % self.wl is not 0:
            raise ValueError('The memory column size %d does not fit word length %d.'%(bitmap.col,self.wl))
            
        if row_prior is not None:
            self.row_prior=row_prior
        if col_prior is not None:
            self.col_prior=col_prior
        self.check_prior()
        
        
        for addr in bitmap.fault_dict.keys():
            pass    

