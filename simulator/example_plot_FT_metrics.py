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
stat_vs_folders=['mnist_lenet5_model_fault_rate','mnist_lenet5_model_fault_rate_16']

for dirr in stat_vs_folders:
    stat_data_list.append(make_FT_report_csv(relative_dir+dirr,None,write_csv=False))

pic_save_dir='vs_mnist_lenet5_model_fault_rate_8v16'

color_dict_list=[{'max':'lightblue','min':'lightblue','avg':'blue','var':'darkgray'},
                 {'max':'peachpuff','min':'peachpuff','avg':'red','var':'darkgray'}]

label_list=['(S,8,3)','(S,16,8)']

plot_FT_analysis_multiple(stat_data_list,relative_dir+pic_save_dir,color_dict_list,label_list)

