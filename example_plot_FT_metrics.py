# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 09:37:48 2019

@author: Yung-Yu Tsai

Plot FT metrics and make FT stastistic report
"""

from simulator.utils_tool.plot import make_FT_report,plot_FT_analysis,plot_FT_analysis_multiple,plot_FT_2D_heatmap,dict_format_lfms_to_ms2Dlf
import os
import numpy as np

#%% plot 

stat_folder_dir='imagenet_mobilenet_memory_fault_rate_fmap'
stat_data=make_FT_report('../test_result/'+stat_folder_dir,stat_folder_dir)
stat_data=plot_FT_analysis(stat_dir='../test_result/'+stat_folder_dir)

#%% plot multliple data line together

relative_dir='../test_result/'
stat_data_list=list()
stat_vs_folders=['mnist_lenet5_model_fault_rate','cifar10_4C2F_model_fault_rate','imagenet_mobilenet_model_fault_rate','imagenet_resnet_model_fault_rate']

for dirr in stat_vs_folders:
    stat_data_list.append(make_FT_report(relative_dir+dirr))

pic_save_dir='vs_model_fault_rate_4net'

color_dict_list=[{'max':'lightblue','min':'lightblue','avg':'blue','var':'darkgray'},
                 {'max':'peachpuff','min':'peachpuff','avg':'red','var':'darkgray'},
                 {'max':'lightgreen','min':'lightgreen','avg':'green','var':'darkgray'},
                 {'max':'thistle','min':'thistle','avg':'purple','var':'darkgray'}]

label_list=['lenet (S,8,3)','4C2F (S,10,6)','mobilenet (S,16,9)','resnet (S,16,12-8)']

plot_FT_analysis_multiple(stat_data_list,relative_dir+pic_save_dir,color_dict_list,label_list)
#plot_FT_analysis_multiple(stat_data_list,relative_dir+pic_save_dir,color_dict_list,label_list,font_size=14,legend_size=10,save_plot_format='eps')

#%% plot save .eps for latex

stat_folder_dir='mnist_lenet5_model_fault_rate_wght'
stat_data=make_FT_report('../test_result/'+stat_folder_dir,stat_folder_dir)
stat_data=plot_FT_analysis(stat_dir='../test_result/'+stat_folder_dir,font_size=14,save_plot_format='eps')

#%% plot save .eps for latex

relative_dir='../test_result/'
stat_data_list=list()
stat_vs_folders=['imagenet_mobilenet_model_fault_rate_wght','imagenet_mobilenet_model_fault_rate_fmap']

for dirr in stat_vs_folders:
    stat_data_list.append(make_FT_report(relative_dir+dirr))

pic_save_dir='vs_imagenet_mobilenet_model_fault_rate_fvw'

color_dict_list=[{'max':'lightblue','min':'lightblue','avg':'blue','var':'darkgray'},
                 {'max':'peachpuff','min':'peachpuff','avg':'red','var':'darkgray'},]

label_list=['weights','feature map']

plot_FT_analysis_multiple(stat_data_list,relative_dir+pic_save_dir,color_dict_list,label_list,font_size=14,save_plot_format='eps')


#%% plot 2D heat map for layer by layer FT or feature map center FT

relative_dir='../test_result/'
stat_folder_dir='imagenet_mobilenet_model_fault_rate_fmc'

# collect data
stat_data_var_dict=dict()

var_dir_list=os.listdir(relative_dir+stat_folder_dir)

if 'plot' in var_dir_list:
    var_dir_list.remove('plot')
    
#for i in range(len(var_dir_list)):
#    var_dir_list[i]=int(var_dir_list[i])
#var_dir_list.sort()
    
for dirr in var_dir_list:
    stat_data_var_dict[dirr]=make_FT_report(relative_dir+stat_folder_dir+'/'+str(dirr))

# data transformation
stat_data_metric_dict,fr_list=dict_format_lfms_to_ms2Dlf(stat_data_var_dict)

#plot_FT_2D_heatmap(stat_data_metric_dict,relative_dir+stat_folder_dir,fr_list,var_dir_list,
#                   'layer index','fault rate')

#plot_FT_2D_heatmap(stat_data_metric_dict,relative_dir+stat_folder_dir,fr_list,var_dir_list,
#                   'layer index','fault rate',
#                   aspect_ratio='equal',annotate=False,xtick_rot=-60,
#                   label_redu=2,grid_width=2)

#plot_FT_2D_heatmap(stat_data_metric_dict,relative_dir+stat_folder_dir,fr_list,var_dir_list,
#                   'layer index','fault rate',
#                   aspect_ratio=1.5,annotate=False,xtick_rot=-60,
#                   label_redu=3,grid_width=0.5)

#plot_FT_2D_heatmap(stat_data_metric_dict,relative_dir+stat_folder_dir,fr_list,var_dir_list,
#                   'concentration','fault rate',
#                   valfmt='{x:.2f}',aspect_ratio=0.5,grid_width=1)

plot_FT_2D_heatmap(stat_data_metric_dict,relative_dir+stat_folder_dir,fr_list,var_dir_list,
                   'concentration','fault rate',
                   valfmt='{x:.2f}',aspect_ratio=0.3,annotate=False,xtick_rot=-60,
                   label_redu=2,grid_width=1)
