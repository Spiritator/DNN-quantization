# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 09:37:48 2019

@author: Yung-Yu Tsai

Plot FT metrics and make FT stastistic report
"""

from utils_tool.plot import make_FT_report_csv,plot_FT_analysis

#%%

stat_folder_dir='cifar10_4C2F_memory_fault_rate_small_full_fmap'
stat_data=make_FT_report_csv('../../test_result/'+stat_folder_dir,stat_folder_dir)
stat_data=plot_FT_analysis(stat_dir='../../test_result/'+stat_folder_dir)

