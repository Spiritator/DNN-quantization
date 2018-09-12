# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 16:39:57 2018

@author: Yung-Yu Tsai

weight fault injection
"""

import keras
from keras.models import Sequential, Model
import numpy as np

def inject_stuck_at_fault(model,layer,x,y,channel,stuck_at):
    
