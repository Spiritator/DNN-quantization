# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 16:30:04 2019

@author: Yung-Yu Tsai

Processing element array setting for compuation unit fault mapping
"""

import numpy as np

class array_axis:
    """
    The dimension in PE array which contains the dataflow information.
    
    """
    
    def __init__():
        """
        # Arguments
            
        
        """

    
class PEarray:
    """
    The PE array functional model for computation unit fault tolerance analysis.
    The 2D PE array consist of three basic dimension 'PE_x', 'PE_y' and 't_clk' 
    which means the row direction, column direction and clock cycles required to 
    complete a tile computation.
    The 3D PE array dataflow model is for mapping the tile data to computaion unit.
    
    """

    def __init__(self, n_x, n_y, n_clk=None):
        """
        # Arguments
            n_x: Integer. Number of PEs in a row.
            n_y: Integer. Number of PEs in a column.
            n_clk: Integer. Number of clock cycles for a tile to process.
            fault_num: Integer. Number of faults in array.
            fault_dict: Dictionary. The fault information {location : fault type}
    
        """
        self.n_x=n_x
        self.n_y=n_y
        self.n_clk=n_clk
        self.fault_num=None
        self.fault_dict=dict()


