# -*- coding: utf-8 -*-
"""
Created on Wed May 22 16:59:00 2019

@author: Yung-Yu Tsai

Code for fault tolerance metrics plotting
"""

import os,csv
import matplotlib.pyplot as plt
import numpy as np

def _preprocess_float_fault_rate_text(fl_fr_text):
    if 'e-' in fl_fr_text:
        ed_id=fl_fr_text.find('e-')
        flfrnew=fl_fr_text[0]+fl_fr_text[ed_id:]
    else:
        flfrnew=fl_fr_text
    
    return flfrnew

def make_FT_report_csv(stat_dir,report_filename,write_csv=True):
    stat_file_list=os.listdir(stat_dir)
    stat_files=dict()
    for fname in stat_file_list:
        if fname.endswith('.csv'):
            stat,f_ext=os.path.splitext(fname)
            stat=_preprocess_float_fault_rate_text(stat)
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
            analyzed_metrics['var_up']=np.clip(analyzed_metrics['avg']+analyzed_metrics['std_dev'],0,analyzed_metrics['max'])
            analyzed_metrics['var_down']=np.clip(analyzed_metrics['avg']-analyzed_metrics['std_dev'],analyzed_metrics['min'],np.inf)
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

def plot_FT_analysis(stat_dir=None,report_filename=None):
    if stat_dir is None and report_filename is None:
        raise ValueError('Both augment stat_dir and report_filename are None! Choose one of them as data to draw analysis plot.')
        
    if stat_dir is None:
        with open(os.path.join(report_filename), 'r', newline='') as repo_csvfile:
            csvdata=csv.DictReader(repo_csvfile)
            stat_data=dict()
            sub_stat=dict()
            metrics=csvdata.fieldnames
            stat, metrics= metrics[0],metrics[1:]
            for metric in metrics:
                sub_stat[metric]=dict()
                
            stat_tmp=float(stat)
            for row in csvdata:
                if row[stat] in ['avg','std_dev','max','min','var_up','var_down']:
                    for metric in metrics:
                        sub_stat[metric][row[stat]]=float(row[metric])
                else:
                    stat_data[stat_tmp]=sub_stat
                    sub_stat=dict()
                    for metric in metrics:
                        sub_stat[metric]=dict()
                    stat_tmp=float(row[stat])
                    
            stat_data[stat_tmp]=sub_stat
            
    if report_filename is None:
        stat_data=make_FT_report_csv(stat_dir,None,write_csv=False)
        
        
    x=list(stat_data.keys())
    metrics=list(stat_data[x[0]].keys())

    for metric in metrics:
        fig = plt.figure()
        
        avg=[stat_data[xid][metric]['avg'] for xid in x]

        maxx=[stat_data[xid][metric]['max'] for xid in x]
        plt.plot(x, maxx, label='max', c='dodgerblue', linestyle='--', marker='.')
        
        minn=[stat_data[xid][metric]['min'] for xid in x]
        plt.plot(x, minn, label='min', c='dodgerblue', linestyle='-.', marker='.')
        
        var_up=[stat_data[xid][metric]['var_up'] for xid in x]
        var_down=[stat_data[xid][metric]['var_down'] for xid in x]
        
        plt.fill_between(x, var_up, var_down, facecolor = 'lightgray', label='variance')
        
        plt.plot(x, avg, label='average', c='darkblue', marker='.')
            
        plt.title(metric)
        plt.ylabel(metric)
        plt.xlabel('bit fault rate')
        plt.xscale('log')
        if 'loss' in metric:
            plt.legend(loc='lower right')
        elif 'acc' in metric:
            plt.legend(loc='upper right')
        else:
            plt.legend(loc='lower right')
        plt.tight_layout()
        if stat_dir is not None:
            pic_path=stat_dir+'/'+metric+'.png'
        elif report_filename is not None:
            pic_path=os.path.split(report_filename)[0]+'/'+metric+'.png'
        plt.savefig(pic_path,dpi=250)
    
    return stat_data

def plot_FT_analysis_multiple(stat_data_list,plot_save_dir,plot_color_list,label_list):
    '''
        plot_color_list: List of Dictionarys. 
                         Dictionary in format {'max':'color_of_max_line','min':'color_of_min_line','avg':'color_of_avg_line','var':'color_of_var_line'}. 
                         The dictionary values are string of the matplotlib.pyplot color scheme.
    
    '''
    if not isinstance(stat_data_list,list):
        raise TypeError('augment stat_data_list should be type list consist of dictionary of stat_data of a FT analysis.')
    if not isinstance(plot_color_list,list):
        raise TypeError('augment plot_color_list should be type list consist of dictionary of color sheme of a FT analysis in pyplot color format string.')
    if len(stat_data_list)!=len(plot_color_list):
        raise ValueError('stat_data_list not equal to plot_color_list please check your data.')        
        
    x=list(stat_data_list[0].keys())
    metrics=list(stat_data_list[0][x[0]].keys())

    for iterr,metric in enumerate(metrics):
        fig = plt.figure()
        
        for i in range(len(stat_data_list)):
            x=list(stat_data_list[i].keys())
            metrics_i=list(stat_data_list[i][x[0]].keys())
            
            avg=[stat_data_list[i][xid][metrics_i[iterr]]['avg'] for xid in x]
    
            maxx=[stat_data_list[i][xid][metrics_i[iterr]]['max'] for xid in x]
            plt.plot(x, maxx, c=plot_color_list[i]['max'], linestyle='-', marker='^')
            
            minn=[stat_data_list[i][xid][metrics_i[iterr]]['min'] for xid in x]
            plt.plot(x, minn, c=plot_color_list[i]['min'], linestyle='-', marker='v')
            
            var_up=[stat_data_list[i][xid][metrics_i[iterr]]['var_up'] for xid in x]
            var_down=[stat_data_list[i][xid][metrics_i[iterr]]['var_down'] for xid in x]
            
            plt.fill_between(x, var_up, var_down, facecolor = plot_color_list[i]['var'], alpha=0.5)
            
            plt.plot(x, avg, label=label_list[i], c=plot_color_list[i]['avg'], marker='.')
            
        plt.title(metric)
        plt.ylabel(metric)
        plt.xlabel('bit fault rate')
        plt.xscale('log')
        if 'loss' in metric:
            plt.legend(loc='lower right')
        elif 'acc' in metric:
            plt.legend(loc='upper right')
        else:
            plt.legend(loc='lower right')
        plt.tight_layout()
        pic_path=plot_save_dir+'/'+metric+'.png'
        plt.savefig(pic_path,dpi=250)
    
def plot_FT_2D_heatmap():
    pass
                
