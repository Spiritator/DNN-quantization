# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 16:30:04 2019

@author: Yung-Yu Tsai

Processing element array setting for compuation unit fault mapping
"""

import numpy as np
    
class PEarray:
    """
    The PE array functional model for computation unit fault tolerance analysis.
    
    """

    def __init__(self, n_x, n_y, n_clk, wl=None):
        """
        # Arguments
            n_x: Integer. Number of PEs in a row.
            n_y: Integer. Number of PEs in a column.
            n_clk: Integer. Number of clock cycles for a tile to process.
            wl: Integer. The word length of memory
            fault_num: Integer. Number of faults in memory.
            fault_dict: Dictionary. The fault information {location : fault type}
    
        """
        self.n_x=n_x
        self.n_y=n_y
        self.n_clk=n_clk
        self.wl=wl
        self.fault_num=None
        self.fault_dict=dict()


