# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 13:36:17 2019

@author: Yung-Yu Tsai

DNN tiling for memory fault mapping
"""

import numpy as np

class tile:
    def __init__(self, Tm, Tn, Tr, Tc, wl=32, row_prior=[], col_prior=[]):
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
        self.wl=wl
        self.row_prior=row_prior
        self.col_prior=col_prior
        
    def check_prior(self):
        if not isinstance(self.row_prior,list) or not isinstance(self.col_prior,list) or len(self.row_prior)!=4 or len(self.col_prior)!=4:
            raise ValueError('The augment row_prior and col_prior must be in list dtype and have length 4 but got length %d and %d'%(len(self.row_prior),len(self.col_prior)))
                
        prior_list=['Tm','Tn','Tr','Tc']
        for i in range(4):
            if self.row_prior[i] not in prior_list:
                raise ValueError('The augment row_prior must be in list %s'%(str(prior_list)))
            if self.col_prior[i] not in prior_list:
                raise ValueError('The augment col_prior must be in list %s'%(str(prior_list)))
        
    def get_numtag_bitmap(self,bitmap,addr):
        if len(addr)!=2:
            raise ValueError('The length of address Tuple in memory must be 2 but got %d.'%(len(addr)))
            
        return addr[0]*bitmap.col+addr[1]
    def get_numtag_tile(self,bitmap,coor):
        if len(coor)!=4:
            raise ValueError('The length of coordinate Tuple in tile must be 4 but got %d.'%(len(coor)))
           
        #row_addr=

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

