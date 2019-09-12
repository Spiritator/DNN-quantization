# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 09:37:48 2019

@author: Yung-Yu Tsai

Plot FT metrics and make FT stastistic report
"""

from simulator.utils_tool.plot import make_FT_report_csv,plot_FT_analysis,plot_FT_analysis_multiple,plot_FT_2D_heatmap
import os
import numpy as np

#%%
# plot 

stat_folder_dir='imagenet_mobilenet_memory_fault_rate_fmap'
stat_data=make_FT_report_csv('../test_result/'+stat_folder_dir,stat_folder_dir)
stat_data=plot_FT_analysis(stat_dir='../test_result/'+stat_folder_dir)

#%%
# plot multliple data line together

relative_dir='../test_result/'
stat_data_list=list()
stat_vs_folders=['mnist_lenet5_memory_fault_rate','cifar10_4C2F_memory_fault_rate_small','imagenet_mobilenet_memory_fault_rate','imagenet_resnet_memory_fault_rate']

for dirr in stat_vs_folders:
    stat_data_list.append(make_FT_report_csv(relative_dir+dirr,None,write_csv=False))

pic_save_dir='vs_memory_fault_rate_4net'

color_dict_list=[{'max':'lightblue','min':'lightblue','avg':'blue','var':'darkgray'},
                 {'max':'peachpuff','min':'peachpuff','avg':'red','var':'darkgray'},
                 {'max':'lightgreen','min':'lightgreen','avg':'green','var':'darkgray'},
                 {'max':'thistle','min':'thistle','avg':'purple','var':'darkgray'}]

label_list=['lenet 6.25KB','4C2F 25.6KB','mobilenet 260KB','resnet (S,16,12-8)']

plot_FT_analysis_multiple(stat_data_list,relative_dir+pic_save_dir,color_dict_list,label_list)


#%%
# plot 2D heat map for layer by layer FT or feature map center FT

relative_dir='../test_result/'
stat_folder_dir='mnist_lenet5_model_fault_rate_lbl'

# collect data
stat_data_var_dict=dict()

var_dir_list=os.listdir(relative_dir+stat_folder_dir)

for dirr in var_dir_list:
    stat_data_var_dict[dirr]=make_FT_report_csv(relative_dir+stat_folder_dir+'/'+dirr,None,write_csv=False)

# data transformation
metric_list=list(stat_data_var_dict[var_dir_list[0]][0.1].keys())
stat_data_metric_dict=dict()

statvd_len=[len(stat_data_var_dict[varr]) for varr in var_dir_list] # get biggest data layer
var_len=max(statvd_len)
argvarlen=statvd_len.index(var_len)

fr_list=list(stat_data_var_dict[var_dir_list[argvarlen]].keys())


