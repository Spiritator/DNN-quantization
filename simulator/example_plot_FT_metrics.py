# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 09:37:48 2019

@author: Yung-Yu Tsai

Plot FT metrics and make FT stastistic report
"""

from utils_tool.plot import make_FT_report_csv,plot_FT_analysis,plot_FT_analysis_multiple

#%%
# plot 

stat_folder_dir='mnist_lenet5_model_fault_rate_16'
stat_data=make_FT_report_csv('../../test_result/'+stat_folder_dir,stat_folder_dir)
stat_data=plot_FT_analysis(stat_dir='../../test_result/'+stat_folder_dir)

#%%
# plot multliple data line together

relative_dir='../../test_result/'
stat_data_list=list()
stat_vs_folders=['cifar10_4C2F_model_fault_rate_fmap','cifar10_4C2F_model_fault_rate_10_fmap']

for dirr in stat_vs_folders:
    stat_data_list.append(make_FT_report_csv(relative_dir+dirr,None,write_csv=False))

pic_save_dir='vs_cifar10_4C2F_model_fault_rate_fmap_16v10'

color_dict_list=[{'max':'lightblue','min':'lightblue','avg':'blue','var':'darkgray'},
                 {'max':'peachpuff','min':'peachpuff','avg':'red','var':'darkgray'}]

label_list=['(S,16,12)','(S,10,6)']

plot_FT_analysis_multiple(stat_data_list,relative_dir+pic_save_dir,color_dict_list,label_list)

