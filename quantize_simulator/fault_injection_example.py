# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 16:09:29 2018

@author: Yung-Yu Tsai

fault injection test
"""

import numpy as np
from testing.fault_injection import generate_single_stuck_at_fault

original_weihgt=np.arange(1,10)

fault_weight=generate_single_stuck_at_fault(original_weihgt,8,4,4,'0')

