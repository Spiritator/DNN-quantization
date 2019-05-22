# -*- coding: utf-8 -*-
"""
Created on Wed May 22 16:59:00 2019

@author: Yung-Yu Tsai

Code for fault tolerance metrics plotting
"""

import os,csv,matplotlib
import numpy as np

def make_FT_report_csv(stat_dir,report_filename,write_csv=True):
    stat_file_list=os.listdir(stat_dir)
    stat_files=dict()
    for fname in stat_file_list:
        stat,f_ext=os.path.splitext(fname)
        stat_files[float(stat)]=fname
        
        
    stat_data=dict()
    for key in sorted(stat_files.keys()):
        with open(os.path.join(stat_dir,stat_files[key]), 'r', newline='') as stat_csvfile:
            csvdata=csv.DictReader(stat_csvfile)
            analyzed_metrics=dict()
            metrics=csvdata.fieldnames
            for metric in metrics:
                analyzed_metrics[metric]=np.array([],dtype=np.float32)

            for row in csvdata:
                for metric in metrics:
                    analyzed_metrics[metric]=np.append(analyzed_metrics[metric],float(row[metric]))
            
            stat_data[key]=analyzed_metrics
    
    #analysis
    for key in stat_data.keys():
        for keyy in stat_data[key].keys():
            analyzed_metrics=dict()
            metric_arr=stat_data[key][keyy]
            analyzed_metrics['avg']=np.mean(metric_arr)
            analyzed_metrics['std_dev']=np.std(metric_arr)
            analyzed_metrics['max']=np.max(metric_arr)
            analyzed_metrics['min']=np.min(metric_arr)
            analyzed_metrics['var_up']=analyzed_metrics['avg']+analyzed_metrics['std_dev']
            analyzed_metrics['var_down']=analyzed_metrics['avg']-analyzed_metrics['std_dev']
            stat_data[key][keyy]=analyzed_metrics
            
    if write_csv:
        repo_dir=os.path.split(stat_dir)
        repo_dir=repo_dir[0]
        
        with open(os.path.join(repo_dir,report_filename+'.csv'), 'w', newline='') as repo_csvfile:
            for key in stat_data.keys():
                report_fieldnames=[key]+metrics
                writer=csv.DictWriter(repo_csvfile, fieldnames=report_fieldnames)
                writer.writeheader()
                for analysis in ['avg','std_dev','max','min','var_up','var_down']:
                    analysis_result_dict=dict()
                    analysis_result_dict[key]=analysis
                    for keyy in stat_data[key].keys():
                        analysis_result_dict[keyy]=stat_data[key][keyy][analysis]
                    writer.writerow(analysis_result_dict)

            
    return stat_data
                
