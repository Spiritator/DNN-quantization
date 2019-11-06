# -*- coding: utf-8 -*-
"""
Created on Wed May 22 16:59:00 2019

@author: Yung-Yu Tsai

Code for fault tolerance metrics plotting
"""

import os,csv
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
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

def plot_FT_analysis(stat_dir=None,report_filename=None,save_plot_format='png'):
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
    
    if save_plot_format not in ['png','eps']:
        raise ValueError('Plot save file must be either png or eps format.')

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
            pic_path=stat_dir+'/'+metric+'.'+save_plot_format
        elif report_filename is not None:
            pic_path=os.path.split(report_filename)[0]+'/'+metric+'.'+save_plot_format
            
        if save_plot_format=='eps':
            plt.savefig(pic_path, format='eps')
        else:
            plt.savefig(pic_path,dpi=250)
    
    return stat_data

def plot_FT_analysis_multiple(stat_data_list,plot_save_dir,plot_color_list,label_list,save_plot_format='png'):
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

    if save_plot_format not in ['png','eps']:
        raise ValueError('Plot save file must be either png or eps format.')

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
        pic_path=plot_save_dir+'/'+metric+'.'+save_plot_format
        
        if save_plot_format=='eps':
            plt.savefig(pic_path, format='eps')
        else:
            plt.savefig(pic_path,dpi=250)
    
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", 
            aspect_ratio=0.4, xtick_rot=0, label_redu=None, grid_width=3, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, aspect=aspect_ratio, **kwargs)
    
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))

    if label_redu is not None:
        # reduced amount of tick label for better visualization on large heatmap
        ax.set_xticklabels([x if i%label_redu==0 else ' ' for i,x in enumerate(col_labels)])
        ax.set_yticklabels([x if i%label_redu==0 else ' ' for i,x in enumerate(row_labels)])
    else:
        # ... and label them with the respective list entries.
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=xtick_rot, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=grid_width)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = [0.0,1.0]
        threshold[0] = im.norm(data.max())*0.2
        threshold[1] = im.norm(data.max())*0.8

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold[1] or im.norm(data[i, j]) < threshold[0])])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def plot_FT_2D_heatmap(stat_data_dict, plot_save_dir, fr_list,var_list, 
                       xlabel, ylabel,
                       aspect_ratio=0.3, valfmt="{x:.3f}", annotate=True, 
                       xtick_rot=0, label_redu=None, grid_width=3,
                       save_plot_format='png'):
    
    if not os.path.isdir(plot_save_dir+'/plot'):
        os.mkdir(plot_save_dir+'/plot')
    
    if save_plot_format not in ['png','eps']:
        raise ValueError('Plot save file must be either png or eps format.')
    
    for mtrc in stat_data_dict.keys():
        for mtrcstat in stat_data_dict[mtrc].keys():
            FT_arr=stat_data_dict[mtrc][mtrcstat]
            
            fig, ax = plt.subplots()
            
            if 'acc' in mtrc and 'loss' not in mtrc:   
                if mtrcstat=='std_dev':
                    colorbar='RdYlGn_r'
                else:
                    colorbar='RdYlGn'
            else:
                colorbar='RdYlGn_r'
                
            im, cbar = heatmap(FT_arr, fr_list, var_list, ax=ax,
                               cmap=colorbar, cbarlabel=mtrc, 
                               aspect_ratio=aspect_ratio, xtick_rot=xtick_rot, 
                               label_redu=label_redu, grid_width=grid_width)
            
            if annotate:
                texts = annotate_heatmap(im, valfmt=valfmt)
            
            plt.title(mtrc+'  '+mtrcstat)
            plt.ylabel(ylabel)
            plt.xlabel(xlabel)
            fig.tight_layout()
            
            if not os.path.isdir(plot_save_dir+'/plot/'+mtrcstat):
                os.mkdir(plot_save_dir+'/plot/'+mtrcstat)
            
            pic_path=plot_save_dir+'/plot/'+mtrcstat+'/'+mtrc+'-'+mtrcstat+'.'+save_plot_format
            
            if save_plot_format=='eps':
                plt.savefig(pic_path, format='eps',bbox_inches='tight')
            else:
                plt.savefig(pic_path,dpi=250,bbox_inches='tight')
                
            plt.show()

                
    
def dict_format_lfms_to_ms2Dlf(stat_data_dict):
    """ 
        Convert FT data dictionay from format: 
            data_dict[layer][fault_rate][FT_metric][metric_statistics]
        to format:
            data_dict[FT_metric][metric_statistics][fault_rate_x_layer__2D_array]
    """
    var_dir_list=list(stat_data_dict.keys())
    # data transformation
    metric_list=list(stat_data_dict[var_dir_list[0]][0.1].keys())
    metric_stats=list(stat_data_dict[var_dir_list[0]][0.1]['loss'].keys())
    stat_data_metric_dict=dict()
    
    statvd_len=[len(stat_data_dict[varr]) for varr in var_dir_list] # get biggest data layer
    var_len=max(statvd_len)
    argvarlen=statvd_len.index(var_len)
    
    fr_list=list(stat_data_dict[var_dir_list[argvarlen]].keys())
    
    for mtrc in metric_list:
        sdmd_tmp=dict()
        for mtrcstat in metric_stats:
            if 'acc' in mtrc and 'loss' not in mtrc:
                if mtrcstat=='std_dev':
                    data_tmp=np.zeros((var_len,len(var_dir_list)))
                else:
                    data_tmp=np.ones((var_len,len(var_dir_list)),dtype=float)
            else:
                data_tmp=np.zeros((var_len,len(var_dir_list)))
            for i,layer in enumerate(var_dir_list):
                for j,fr in enumerate(fr_list):
                    if fr in stat_data_dict[layer].keys():
                        data_tmp[j,i]=stat_data_dict[layer][fr][mtrc][mtrcstat]
            sdmd_tmp[mtrcstat]=data_tmp
        stat_data_metric_dict[mtrc]=sdmd_tmp
        
    return stat_data_metric_dict,fr_list