# -*- coding: utf-8 -*-
"""
Created on Wed May 22 16:59:00 2019

@author: Yung-Yu Tsai

Code for fault tolerance metrics plotting
"""

import os,csv,matplotlib
import numpy as np

def make_FT_report_csv(stat_dir,report_filename):
    stat_file_list=os.listdir(stat_dir)
    stat_files=dict()
    for fname in stat_file_list:
        stat,f_ext=os.path.splitext(fname)
        stat_files[float(stat)]=fname
        
        
    stat_data=dict()
    for key in sorted(stat_files.keys()):
        with open(os.path.join(stat_dir,stat_files[key]), 'r', newline='') as stat_csvfile:
            csvdata=csv.DictReader(stat_csvfile)
            for keyy in csvdata.keys():
                pass
            #TODO
                #metric=
