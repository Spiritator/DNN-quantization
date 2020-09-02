# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:54:08 2019

@author: Yung-Yu Tsai

Memory bitmap setting for memory fault mapping
"""

import numpy as np
    
class bitmap:
    """ The bitmap of a buffer for memory fault tolerance analysis.
    
    Arguments
    ---------
    row: Integer. 
        Number of rows in memory.
    col: Integer. 
        Number of columns in memory.
    wl: Integer. 
        The word length of memory
    fault_num: Integer. 
        Number of faults in memory.
    fault_dict: Dictionary. 
        The fault information {location : fault type}
    """

    def __init__(self, row, col, wl=None):
        """ Memory bitmap initializer """
        self.row=row
        self.col=col
        self.wl=wl
        self.fault_num=None
        self.fault_dict=dict()

    def fault_num_gen_mem(self, fault_rate):
        """ Genenerate the number of fault by fault rate
        
        Arguments
        ---------
        fault_rate: Float
            Bit error rate
        """
        self.fault_num=int(self.row * self.col * fault_rate)
    
    def addr_gen_mem(self,distribution='uniform',poisson_lam=None):
        """ Genenerate the fault location in a memory

        Arguments
        ---------
        distribution: String. 
            The distribution type of locaton in memory. Must be one of 'uniform', 'poisson', 'normal'.
        poisson_lam: Integer. 
            The lambda of poisson distribution.
    
        Returns
        -------
        The location index Tuple(Integer).
        """
        if distribution=='uniform':
            row_tmp=np.random.randint(self.row)
            col_tmp=np.random.randint(self.col)
        elif distribution=='poisson':
            if not isinstance(poisson_lam,tuple) or len(poisson_lam)!=2:
                raise TypeError('Poisson distribution lambda setting must be a tuple has length of 2 (row, col).')
            
            if isinstance(poisson_lam[0],int) and poisson_lam[0]>=0 and poisson_lam[0]<self.row:
                row_tmp=np.random.poisson(poisson_lam[0])
                while row_tmp>=self.row:
                    row_tmp=np.random.poisson(poisson_lam[0])
            else:
                raise ValueError('Poisson distribution Lambda must within feature map shape. Feature map shape %s but got lambda input %s'%(str((self.row,self.col)),str(poisson_lam)))
            
            if isinstance(poisson_lam[1],int) and poisson_lam[1]>=0 and poisson_lam[1]<self.col:
                col_tmp=np.random.poisson(poisson_lam[1])
                while col_tmp>=self.col:
                    col_tmp=np.random.poisson(poisson_lam[1])
            else:
                raise ValueError('Poisson distribution Lambda must within feature map shape. Feature map shape %s but got lambda input %s'%(str((self.row,self.col)),str(poisson_lam)))
    
        elif distribution=='normal':
            #TODO   
            # make the normal distribution generation
            pass 
        else:
            raise NameError('Invalid type of random generation distribution. Please choose between uniform, poisson, normal.')
        
        return (row_tmp,col_tmp)
    
    def addr_gen_mem_fast(self,fault_num,distribution='uniform',poisson_lam=None):
        """ Genenerate the fault location in a memory
            Faster generation may have repetitive fault addr.

        Arguments
        ---------
        distribution: String. 
            The distribution type of locaton in memory. Must be one of 'uniform', 'poisson', 'normal'.
        poisson_lam: Integer. 
            The lambda of poisson distribution.
    
        Returns
        -------
        The location index Tuple(Integer).
        """
        if distribution=='uniform':
            row_tmp=np.random.randint(self.row,size=fault_num)
            col_tmp=np.random.randint(self.col,size=fault_num)
        elif distribution=='poisson':
            if not isinstance(poisson_lam,tuple) or len(poisson_lam)!=2:
                raise TypeError('Poisson distribution lambda setting must be a tuple has length of 2 (row, col).')
            
            if isinstance(poisson_lam[0],int) and poisson_lam[0]>=0 and poisson_lam[0]<self.row:
                row_tmp=np.random.poisson(poisson_lam[0],size=fault_num)
                row_tmp=np.clip(row_tmp,0,self.row-1)
            else:
                raise ValueError('Poisson distribution Lambda must within feature map shape. Feature map shape %s but got lambda input %s'%(str((self.row,self.col)),str(poisson_lam)))
            
            if isinstance(poisson_lam[1],int) and poisson_lam[1]>=0 and poisson_lam[1]<self.col:
                col_tmp=np.random.poisson(poisson_lam[1],size=fault_num)
                col_tmp=np.clip(col_tmp,0,self.col-1)
            else:
                raise ValueError('Poisson distribution Lambda must within feature map shape. Feature map shape %s but got lambda input %s'%(str((self.row,self.col)),str(poisson_lam)))
    
        elif distribution=='normal':
            #TODO   
            # make the normal distribution generation
            pass 
        else:
            raise NameError('Invalid type of random generation distribution. Please choose between uniform, poisson, normal.')
        
        return zip(row_tmp,col_tmp)

    def gen_bitmap_SA_fault_dict(self,fault_rate,fast_gen=False,addr_distribution='uniform',addr_pois_lam=None,fault_type='flip',**kwargs):
        """ Generate the fault dictionary of memory base on its shape and with specific distibution type.

        Arguments
        ---------
        fault_rate: Float. 
            The probability of fault occurance in memory.
        addr_distribution: String. 
            The distribution type of address in memory. Must be one of 'uniform', 'poisson', 'normal'.
        addr_pois_lam: Integer. 
            The lambda of poisson distribution of memory address.
        fault_type: String. 
            The type of fault.
    
        Returns
        -------
        The fault information Dictionary. The number of fault generated Integer.
        """
        fault_count=0        
        fault_dict=dict()
        self.fault_num_gen_mem(fault_rate)
                
        if fast_gen:
            addr=self.addr_gen_mem_fast(self.fault_num,distribution=addr_distribution,poisson_lam=addr_pois_lam,**kwargs)
            fault_dict=dict(zip(addr,[fault_type for _ in range(self.fault_num)]))
        else:
            while fault_count<self.fault_num:
                addr=self.addr_gen_mem(distribution=addr_distribution,poisson_lam=addr_pois_lam,**kwargs)
                
                if addr in fault_dict.keys():
                    continue
                else:
                    fault_dict[addr]=fault_type
                    fault_count += 1
            
        self.fault_dict=fault_dict
        
        return fault_dict,self.fault_num
    
    def get_numtag(self,addr):
        """ Get the bitmap and tile conversion index numtag.

        Arguments
        ---------
        addr: Tuple. 
            The address of memory bit oriented representation. Length 2 i.e. 2D representation of memory. (row,col)
        addr: Ndarray. 
            N number of (row,col) addrs in ndarray of shape (N,2) in dtype integer.
    
        Returns
        -------
            | The numtag (Integer)
            | Numtag array (Ndarray)
        """
        if isinstance(addr,tuple):
            if len(addr)!=2:
                raise ValueError('The length of address Tuple in memory must be 2 but got %d.'%(len(addr)))
                
            return addr[0]*self.col+addr[1]
        elif isinstance(addr,np.ndarray):
            return np.ravel_multi_index(addr.T,(self.row,self.col))
        else:
            raise TypeError('addr must be eithr tuple of length 2 which represent a addr in memory (row,col) or ndarray of shape (n,2) which represent n number of (row,col) addrs.')
    
    def numtag2addr(self,numtag):
        """Convert the numtag to its corresponding address.

        Arguments
        ---------
        numtag: Integer. 
            The bitmap and tile conversion index numtag.
        numtag: Ndarray. 
            The bitmap and tile conversion index numtag array 1D.
    
        Returns
        -------
            | The memory address (Tuple)
            | Memory address array (Ndarray) shape (N,2)
        """
        if isinstance(numtag,int):                
            return (numtag//self.col, numtag % self.col)
        elif isinstance(numtag,np.ndarray) or isinstance(numtag,list):
            return np.transpose(np.unravel_index(numtag,(self.row,self.col)))
        else:
            raise TypeError('addr must be eithr tuple of length 2 which represent a addr in memory (row,col) or ndarray of shape (n,2) which represent n number of (row,col) addrs.')
    
    def clear(self):
        """Clear the fault information of tile"""
        self.fault_dict=dict()







